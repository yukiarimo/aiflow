import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from . import utils
from .models import DurationDiscriminatorV2, MultiPeriodDiscriminator, MultiResolutionDiscriminator, SynthesizerTrn, slice_segments
from aiflow.models.yuna_speech.text import symbols
import random
import torch.utils.data

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
global_step = 0
run_start_step = 0
resume_skip_step = None
logged_ground_truth = False


def loss_ramp(warmup_steps):
	"""0 -> 1 over warmup_steps counted from the start of this process, not from global_step. Newly enabled or re-scoped loss terms need to phase in even when resuming at step 20k, so the ramp is anchored to the resume point rather than to absolute training progress."""
	warmup_steps = int(warmup_steps or 0)
	if warmup_steps <= 0:
		return 1.0
	return min(1.0, (global_step - run_start_step + 1) / warmup_steps)


def unwrap_ddp(module):
	return module.module if hasattr(module, "module") else module


def maybe_ddp(module, rank, n_gpus, find_unused_parameters=False):
	if n_gpus > 1:
		return DDP(module, device_ids=[rank], find_unused_parameters=find_unused_parameters)
	return module


def grad_norm_and_clip(parameters, max_norm=None):
	"""Global L2 grad norm, clipped in place when max_norm > 0. Returns the pre-clip norm."""
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	if not parameters:
		return 0.0
	limit = float(max_norm) if max_norm and float(max_norm) > 0 else float("inf")
	return float(torch.nn.utils.clip_grad_norm_(parameters, limit))


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
	"""Loads audio, speaker_id, language_id, pitch, and text."""
	def __init__(self, audiopaths_sid_text, hparams):
		self.hparams = hparams
		self.audiopaths_sid_text = utils.load_filepaths_and_text(audiopaths_sid_text)
		self.max_wav_value = hparams.data.max_wav_value
		self.sampling_rate = hparams.data.sampling_rate
		self.filter_length = hparams.data.filter_length
		self.hop_length = hparams.data.hop_length
		self.win_length = hparams.data.win_length
		self.n_mel_channels = getattr(hparams.data, "n_mel_channels", 128)
		self.min_text_len = getattr(hparams.data, "min_text_len", 1)
		self.max_text_len = getattr(hparams.data, "max_text_len", 20000)
		self.min_audio_len = getattr(hparams.data, "min_audio_len", 8192)
		self.speakers_map = getattr(hparams.data, "speakers", {}) or {}
		self.languages_map = getattr(hparams.data, "languages", {}) or {}
		self.pitch_dir = getattr(hparams.data, "pitch_dir", None)
		self.energy_dir = getattr(hparams.data, "energy_dir", None)
		self.qwen_dir = getattr(hparams.data, "qwen_dir", None)
		self.base_dir = getattr(hparams.data, "base_dir", None)
		self.f0_max = getattr(hparams.data, "f0_max", 800.0)
		self.use_spp = getattr(hparams.model, "use_spp", False)
		self.use_qwen_distill = getattr(hparams.train, "use_qwen_distill", False)
		random.seed(8)
		random.shuffle(self.audiopaths_sid_text)
		self._filter()

	def _filter(self):
		"""Filter text & store spec lengths."""
		audiopaths_sid_text_new, lengths = [], []

		for row in self.audiopaths_sid_text:
			if len(row) == 4:
				audiopath, sid, lid, text = row
			elif len(row) == 3:
				audiopath, sid, text = row
				lid = "0"
			else:
				continue

			if not os.path.isfile(audiopath): continue

			if self.min_text_len <= len(text) <= self.max_text_len:
				audiopaths_sid_text_new.append([audiopath, sid, lid, text])
				length = os.path.getsize(audiopath) // (2 * self.hop_length)

				if length < self.min_audio_len // self.hop_length: continue
				lengths.append(length)

		self.audiopaths_sid_text = audiopaths_sid_text_new
		self.lengths = lengths
		print(len(self.lengths))

	def get_audio_text_speaker_pair(self, audiopath_sid_text):
		audiopath, sid, lid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3]
		text_norm = utils.parse_phoneme_field(text)
		text = torch.LongTensor(text_norm)
		spec, wav = self.get_audio(audiopath)
		pitch = self.get_pitch(audiopath, spec.size(1))
		energy = self.get_energy(audiopath, spec.size(1))
		qwen_feat, qwen_len = self.get_qwen(audiopath)
		sid = torch.LongTensor([utils.resolve_id(sid, self.speakers_map, label="speaker")])
		lid = torch.LongTensor([utils.resolve_id(lid, self.languages_map, label="language")])
		return (text, spec, wav, sid, lid, pitch, energy, qwen_feat, qwen_len)

	def get_pitch(self, audiopath, spec_len):
		pitch = torch.zeros(spec_len, dtype=torch.float32)
		if not self.use_spp or not self.pitch_dir:
			return pitch

		pitch_path = utils.pitch_path_for_wav(audiopath, self.pitch_dir, self.base_dir)
		if not os.path.isfile(pitch_path):
			return pitch

		pitch_hz = np.load(pitch_path).astype(np.float32)
		pitch_norm = utils.normalize_pitch_hz(pitch_hz, self.f0_max)
		length = min(spec_len, pitch_norm.numel())
		pitch[:length] = pitch_norm[:length]
		return pitch

	def get_energy(self, audiopath, spec_len):
		energy = torch.zeros(spec_len, dtype=torch.float32)
		if not self.energy_dir:
			return energy

		energy_path = utils.energy_path_for_wav(audiopath, self.energy_dir, self.base_dir)
		if not os.path.isfile(energy_path):
			return energy

		energy_rms = np.load(energy_path).astype(np.float32)
		energy_log = utils.normalize_energy_rms(energy_rms)
		length = min(spec_len, energy_log.numel())
		energy[:length] = energy_log[:length]
		return energy

	def get_qwen(self, audiopath):
		qwen_dim = getattr(self.hparams.model, "qwen_teacher_dim", 2048)
		if not self.use_qwen_distill or not self.qwen_dir:
			return torch.zeros(0, qwen_dim, dtype=torch.float32), 0
		qwen_path = utils.qwen_path_for_wav(audiopath, self.qwen_dir, self.base_dir)
		if not os.path.isfile(qwen_path):
			return torch.zeros(0, qwen_dim, dtype=torch.float32), 0
		qwen_feat = np.load(qwen_path).astype(np.float32)
		if qwen_feat.ndim != 2:
			return torch.zeros(0, qwen_dim, dtype=torch.float32), 0
		return torch.from_numpy(qwen_feat), int(qwen_feat.shape[0])

	def get_audio(self, filename):
		audio, sampling_rate = utils.load_wav_to_torch(filename)
		if sampling_rate != self.sampling_rate: raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
		audio_norm = (audio / self.max_wav_value).unsqueeze(0)
		segment_size = self.hparams.train.segment_size
		if audio_norm.size(1) < segment_size: audio_norm = torch.nn.functional.pad(audio_norm, (0, segment_size - audio_norm.size(1)), 'constant')
		spec_filename = filename.replace(".wav", ".mel.pt")

		if os.path.exists(spec_filename):
			spec = torch.load(spec_filename)
			if spec.size(1) < segment_size // self.hop_length:
				spec = None
		else:
			spec = None

		if spec is None:
			spec = utils.mel_spectrogram_torch(audio_norm, self.filter_length, self.n_mel_channels, self.sampling_rate, self.hop_length, self.win_length, self.hparams.data.mel_fmin, self.hparams.data.mel_fmax, center=False)
			spec = torch.squeeze(spec, 0)
			torch.save(spec, spec_filename)
		return spec, audio_norm

	def __getitem__(self, index):
		return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

	def __len__(self):
		return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
	"""Zero-pads model inputs and targets"""
	def __init__(self, return_ids=False, qwen_teacher_dim=2048):
		self.return_ids = return_ids
		self.qwen_teacher_dim = qwen_teacher_dim

	def __call__(self, batch):
		_, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)
		max_text_len = max([len(x[0]) for x in batch])
		max_spec_len = max([x[1].size(1) for x in batch])
		max_wav_len = max([x[2].size(1) for x in batch])
		max_qwen_len = max([x[7].size(0) for x in batch], default=0)
		text_lengths = torch.LongTensor(len(batch))
		spec_lengths = torch.LongTensor(len(batch))
		wav_lengths = torch.LongTensor(len(batch))
		qwen_lengths = torch.LongTensor(len(batch))
		sid = torch.LongTensor(len(batch))
		lid = torch.LongTensor(len(batch))
		text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
		spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
		wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()
		pitch_padded = torch.FloatTensor(len(batch), max_spec_len).zero_()
		energy_padded = torch.FloatTensor(len(batch), max_spec_len).zero_()
		qwen_padded = torch.FloatTensor(len(batch), max_qwen_len, self.qwen_teacher_dim).zero_()

		for i, row_idx in enumerate(ids_sorted_decreasing):
			row = batch[row_idx]
			text, spec, wav, pitch, energy, qwen_feat = row[0], row[1], row[2], row[5], row[6], row[7]
			text_padded[i, :text.size(0)] = text
			text_lengths[i] = text.size(0)
			spec_padded[i, :, :spec.size(1)] = spec
			spec_lengths[i] = spec.size(1)
			wav_padded[i, :, :wav.size(1)] = wav
			wav_lengths[i] = wav.size(1)
			sid[i] = row[3]
			lid[i] = row[4]
			pitch_padded[i, :pitch.size(0)] = pitch
			energy_padded[i, :energy.size(0)] = energy
			if qwen_feat.size(0) > 0:
				qwen_padded[i, :qwen_feat.size(0)] = qwen_feat
				qwen_lengths[i] = qwen_feat.size(0)

		if self.return_ids:
			return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, lid, pitch_padded, energy_padded, qwen_padded, qwen_lengths, ids_sorted_decreasing
		return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, lid, pitch_padded, energy_padded, qwen_padded, qwen_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
	"""Maintains similar input lengths in a batch."""
	def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
		super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
		self.lengths = dataset.lengths
		self.batch_size = batch_size
		self.boundaries = boundaries
		self.buckets, self.num_samples_per_bucket = self._create_buckets()
		self.total_size = sum(self.num_samples_per_bucket)
		self.num_samples = self.total_size // self.num_replicas

	def _create_buckets(self):
		buckets = [[] for _ in range(len(self.boundaries) - 1)]

		for i, length in enumerate(self.lengths):
			idx_bucket = self._bisect(length)
			if idx_bucket != -1: buckets[idx_bucket].append(i)

		for i in range(len(buckets) - 1, -1, -1):
			if not buckets[i]:
				buckets.pop(i)
				self.boundaries.pop(i + 1)

		num_samples_per_bucket = []
		for bucket in buckets:
			len_bucket = len(bucket)
			total_batch_size = self.num_replicas * self.batch_size
			rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
			num_samples_per_bucket.append(len_bucket + rem)
		return buckets, num_samples_per_bucket

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch)

		if self.shuffle: indices = [torch.randperm(len(b), generator=g).tolist() for b in self.buckets]
		else: indices = [list(range(len(b))) for b in self.buckets]

		batches = []
		for i, bucket in enumerate(self.buckets):
			len_bucket = len(bucket)
			ids_bucket = indices[i]
			num_samples_bucket = self.num_samples_per_bucket[i]
			rem = num_samples_bucket - len_bucket
			ids_bucket += ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]  # Add extra samples
			ids_bucket = ids_bucket[self.rank::self.num_replicas]  # Subsample

			for j in range(len(ids_bucket) // self.batch_size):  # Batching
				batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
				batches.append(batch)

		if self.shuffle:
			batch_ids = torch.randperm(len(batches), generator=g).tolist()
			batches = [batches[i] for i in batch_ids]

		self.batches = batches
		assert len(self.batches) * self.batch_size == self.num_samples
		return iter(self.batches)

	def _bisect(self, x, lo=0, hi=None):
		if hi is None: hi = len(self.boundaries) - 1
		if hi > lo:
			mid = (hi + lo) // 2
			if self.boundaries[mid] < x <= self.boundaries[mid + 1]: return mid
			elif x <= self.boundaries[mid]: return self._bisect(x, lo, mid)
			else: return self._bisect(x, mid + 1, hi)
		return -1

	def __len__(self):
		return self.num_samples // self.batch_size


def feature_loss(fmap_r, fmap_g):
	loss = 0
	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):
			rl = rl.float().detach()
			gl = gl.float()
			loss += torch.mean(torch.abs(rl - gl))

	return loss * 2


def nonnegative_flow_nll(nll_per_item):
	"""Flow NLL can be < 0 when density > 1; train on |NLL| (distance from zero) like other losses."""
	return nll_per_item.float().abs().mean()


def unpack_synthesizer_forward(raw):
	"""Unpack net_g forward; supports legacy tuple layouts."""
	if len(raw) == 9:
		y_hat, l_length, l_pitch, attn, ids_slice, x_mask, y_mask, latents, prosody = raw
		l_pitch_l1, l_energy = None, None
	else:
		y_hat, l_length, l_pitch, l_pitch_l1, l_energy, attn, ids_slice, x_mask, y_mask, latents, prosody = raw
	z, z_p, m_p, logs_p, m_q, logs_q = latents
	hidden_x, logw, logw_ = prosody[0], prosody[1], prosody[2]
	logpitch = prosody[3] if len(prosody) > 3 else None
	logpitch_ = prosody[4] if len(prosody) > 4 else None
	vuv_logit = prosody[5] if len(prosody) > 5 else None
	energy_hat = prosody[6] if len(prosody) > 6 else None
	l_qwen = prosody[7] if len(prosody) > 7 else None
	return y_hat, l_length, l_pitch, l_pitch_l1, l_energy, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), hidden_x, logw, logw_, logpitch, logpitch_, vuv_logit, energy_hat, l_qwen


def _to_float(x):
	if torch.is_tensor(x):
		return float(x.detach().item())
	return float(x)


def compute_pitch_train_diagnostics(logpitch, pitch, ids_slice, segment_mel_frames, l_pitch_l1=None, f0_max=800.0):
	"""Pitch stats on the training segment (voiced MAE, unvoiced SPP leakage)."""
	out = {"pitch/l1_voiced_raw": 0.0, "pitch/spp_unvoiced_mean": 0.0, "pitch/spp_voiced_mae_hz": 0.0, "pitch/gt_voiced_frac": 0.0}
	if logpitch is None or pitch is None:
		return out
	if l_pitch_l1 is not None:
		out["pitch/l1_voiced_raw"] = _to_float(l_pitch_l1)
	pitch_mel = pitch.unsqueeze(1)
	pitch_seg = slice_segments(pitch_mel, ids_slice, segment_mel_frames)
	logpitch_seg = slice_segments(logpitch, ids_slice, segment_mel_frames)
	voiced = pitch_seg > 1e-4
	unvoiced = ~voiced
	out["pitch/gt_voiced_frac"] = _to_float(voiced.float().mean())
	if voiced.any():
		mae_norm = (logpitch_seg - pitch_seg).abs()[voiced].mean()
		out["pitch/spp_voiced_mae_hz"] = _to_float(mae_norm) * float(f0_max)
	if unvoiced.any():
		out["pitch/spp_unvoiced_mean"] = _to_float(logpitch_seg[unvoiced].mean())
	return out


def spectral_contrast_db(wav, hps):
	"""P90-P10 of in-band log-magnitude on loud frames: how deep the inter-harmonic valleys are. Mel pools 1025 linear bins into 128 triangular filters, so loss/g/mel cannot see this at all. When a decoder over-smooths, the valleys fill and this number falls away from the ground truth."""
	if wav is None:
		return None
	x = wav.squeeze(1) if wav.dim() == 3 else wav
	n_fft, hop, win, sr = hps.data.filter_length, hps.data.hop_length, hps.data.win_length, hps.data.sampling_rate
	window = torch.hann_window(win, device=x.device, dtype=torch.float32)
	spec = torch.stft(x.detach().float(), n_fft, hop, win, window=window, return_complex=True).abs()
	bin_hz = torch.arange(spec.size(1), device=x.device, dtype=torch.float32) * (sr / n_fft)
	spec = spec[:, (bin_hz >= hps.data.mel_fmin) & (bin_hz <= hps.data.mel_fmax)]
	if spec.size(1) < 8 or spec.size(2) < 1:
		return None
	lg = 20.0 * torch.log10(spec.clamp_min(1e-6)).permute(0, 2, 1).reshape(-1, spec.size(1))
	keep = lg.mean(1) > torch.quantile(lg.mean(1), 0.6)
	if int(keep.sum()) < 1:
		return None
	sel = lg[keep]
	q = torch.quantile(sel, torch.tensor([0.1, 0.9], device=sel.device), dim=1)
	return _to_float((q[1] - q[0]).mean())


def build_train_log_line(global_step, lr, scalars):
	"""Single grep-friendly training status line for log.txt."""
	def fmt(key, digits=3):
		val = scalars.get(key)
		if val is None:
			return None
		return f"{key.split('/')[-1]}={_to_float(val):.{digits}f}"

	parts = [f"step={global_step}", f"lr={lr:.6f}"]
	for key in ("loss/g/total", "loss/g/mel_unweighted", "loss/g/kl_unweighted", "loss/g/qwen_unweighted", "loss/g/stft_unweighted", "spectrum/contrast_gt", "pitch/l1_voiced_raw", "pitch/spp_unvoiced_mean", "pitch/spp_voiced_mae_hz", "audio/rms_posterior", ):
		text = fmt(key)
		if text is not None:
			parts.append(text)
	return " | ".join(parts)


def apply_tb_trend_labels_scalars(scalars):
	"""Rename TensorBoard scalar tags with trend hints (↓ lower, ↑ higher, →1 target)."""
	explicit = {"audio/rms_posterior": " ↔", "learning_rate": " ↘", "grad_norm_d": " ↔", "grad_norm_g": " ↔", "grad_norm_mrd": " ↔", "grad_norm_dur_disc": " ↔", "train/mas_noise_scale": " ↓", "pitch/gt_voiced_frac": " ↔", "eval/ltas_l1_db": " ↓", "eval/length_ratio": " ↔→1", "spectrum/contrast_gt": " ↔", "spectrum/contrast_post": " ↑→gt", "train/gst_active": "", "train/gst_same_clip": "", "train/freeze_decoder_only": "", "train/flow_prosody_teacher_forcing": "", "train/flow_prosody_predicted": "", "train/mas_flow_prosody_parity": "", }
	out = {}
	for key, value in scalars.items():
		if key in explicit:
			suffix = explicit[key]
		elif key.startswith("loss/"):
			suffix = " ↓"
		elif key.startswith("gap/"):
			suffix = " ↓"
		elif key.startswith("pitch/"):
			suffix = " ↓"
		else:
			suffix = ""
		out[key + suffix] = value
	return out


def build_diagnostic_images(hps, d):
	"""Progress plots: prosody overlays, duration, error maps, spectrum, waveform."""
	images = {}
	seg_frames = hps.train.segment_size // hps.data.hop_length
	f0_max = float(getattr(hps.data, "f0_max", 800.0))
	ids_slice = d["ids_slice"]

	if d["pitch"] is not None and d["logpitch"] is not None:
		gt = slice_segments(d["pitch"].unsqueeze(1), ids_slice, seg_frames)[0, 0].detach().float().cpu().numpy()
		hat = slice_segments(d["logpitch"], ids_slice, seg_frames)[0, 0].detach().float().cpu().numpy()
		images["prosody/pitch"] = utils.plot_curves_to_numpy([("gt (voiced)", np.where(gt > 1e-4, gt, np.nan) * f0_max), ("spp", hat * f0_max)], xlabel="Mel frame", ylabel="F0 (Hz)", title="Pitch contour: ground truth vs SPP")  # Unvoiced frames become NaN so the ground-truth line breaks instead of dropping to zero.

	if d["energy"] is not None and d["energy_hat"] is not None:
		gt = slice_segments(d["energy"].unsqueeze(1), ids_slice, seg_frames)[0, 0].detach().float().cpu().numpy()
		hat = slice_segments(d["energy_hat"], ids_slice, seg_frames)[0, 0].detach().float().cpu().numpy()
		images["prosody/energy"] = utils.plot_curves_to_numpy([("gt", gt), ("pred", hat)], xlabel="Mel frame", ylabel="log energy", title="Energy contour: ground truth vs predictor")

	if d["vuv_logit"] is not None and d["pitch"] is not None:
		t = min(d["vuv_logit"].size(2), d["pitch"].size(1))
		gt = (d["pitch"][0, :t] > 1e-4).float().detach().cpu().numpy()
		hat = torch.sigmoid(d["vuv_logit"][0, 0, :t]).detach().float().cpu().numpy()
		images["prosody/vuv"] = utils.plot_curves_to_numpy([("gt voiced", gt), ("p(voiced)", hat)], xlabel="Mel frame", ylabel="voicing", title="Voiced/unvoiced: target vs prediction")

	n_tok = int(d["x_mask"][0, 0].sum().item()) if d["x_mask"] is not None else 0
	if n_tok > 0 and d["logw"] is not None and d["logw_"] is not None:
		tgt = d["logw_"][0, 0, :n_tok].detach().float().exp().cpu().numpy()  # MAS never shows this: all/attn is teacher-forced, while inference rhythm comes from logw.
		pred = d["logw"][0, 0, :n_tok].detach().float().exp().cpu().numpy()
		images["duration/mas_vs_pred"] = utils.plot_curves_to_numpy([("MAS target", tgt), ("predictor", pred)], xlabel="Phoneme index", ylabel="Frames", title="Duration: MAS target vs duration predictor (governs inference rhythm)", step=True)

	if n_tok > 0 and d["attn"] is not None:
		widths = d["attn"][0, 0].sum(0)[:n_tok].detach().float().cpu().numpy()
		images["duration/mas_widths"] = utils.plot_curves_to_numpy([("frames per phoneme", widths)], xlabel="Phoneme index", ylabel="Frames", title="MAS duration profile", step=True)

	if d["y_mel"] is not None and d["y_hat_mel"] is not None:
		images["error/mel_posterior"] = utils.plot_heatmap_to_numpy((d["y_hat_mel"][0] - d["y_mel"][0]).abs().detach().float().cpu().numpy(), ylabel="Mel bin", title="|posterior - ground truth| mel")

	curves, freqs = [], None
	for label, wav in (("ground truth", d["y_gen"]), ("posterior", d["y_hat_gen"])):
		if wav is None:
			continue
		freqs, ltas = utils.long_term_average_spectrum(wav, hps.data.filter_length, hps.data.hop_length, hps.data.win_length, hps.data.sampling_rate)
		curves.append((label, ltas))
	if curves:
		images["spectrum/ltas"] = utils.plot_curves_to_numpy(curves, x=freqs, xlabel="Hz", ylabel="dB", title="Long-term average spectrum (air band / rolloff / noise floor)")

	if d["y_hat_gen"] is not None:
		mag = utils.stft_magnitude_torch(d["y_hat_gen"].squeeze(1).float(), hps.data.filter_length, hps.data.hop_length, hps.data.win_length, center=False)
		n_freq = mag.size(1)
		hi = int(np.ceil(hps.data.mel_fmax / (hps.data.sampling_rate / 2.0) * (n_freq - 1)))
		if hi < n_freq - 1:
			band = 20.0 * torch.log10(mag[0, hi:].clamp_min(1e-10))
			images["spectrum/out_of_band"] = utils.plot_heatmap_to_numpy(band.detach().float().cpu().numpy(), ylabel=f"bin (>= {int(hps.data.mel_fmax)} Hz)", title="Generated energy above mel_fmax (dB): hallucination check")

	if d["y_gen"] is not None and d["y_hat_gen"] is not None:
		n = min(4096, d["y_gen"].size(-1), d["y_hat_gen"].size(-1))
		wave = [("ground truth", d["y_gen"][0, 0, :n].detach().float().cpu().numpy()), ("posterior", d["y_hat_gen"][0, 0, :n].detach().float().cpu().numpy())]
		images["wave/overlay"] = utils.plot_curves_to_numpy(wave, xlabel=f"Sample (first {n} = {1000.0 * n / hps.data.sampling_rate:.0f} ms)", ylabel="Amplitude", title="Waveform overlay: phase / metallic artifact check")

	return images


def apply_tb_trend_labels_images(images):
	"""Rename TensorBoard image tags with visual convergence hints."""
	labels = {"slice/mel_org": "slice/mel_org (GT)", "slice/mel_gen": "slice/mel_gen (posterior ↓→ org)", "all/mel": "all/mel (GT full)", "all/attn": "all/attn (alignment)", "gen/mel": "gen/mel (infer ↓→ gt)", "gt/mel": "gt/mel (reference)", }
	return {labels.get(key, key): value for key, value in images.items()}


def apply_tb_trend_labels_audios(audios):
	labels = {"gen/audio": "gen/audio (infer)", "gt/audio": "gt/audio (reference)", }
	return {labels.get(key, key): value for key, value in audios.items()}


def sample_style_reference(spec, spec_lengths, speakers, gst_dropout, same_speaker_only=True, same_clip_prob=0.0):
	"""Pick GST style reference: none, same utterance, or a different batch item."""
	if random.random() < gst_dropout:
		return None, None
	if same_clip_prob > 0.0 and random.random() < same_clip_prob:
		return spec, spec_lengths

	if spec.size(0) <= 1:
		return spec, spec_lengths

	candidates = []
	for shift in range(1, spec.size(0)):
		perm = (torch.arange(spec.size(0), device=spec.device) + shift) % spec.size(0)
		if same_speaker_only and not torch.all(speakers[perm] == speakers):
			continue
		candidates.append(perm)

	if not candidates:
		shift = random.randint(1, spec.size(0) - 1)
		perm = (torch.arange(spec.size(0), device=spec.device) + shift) % spec.size(0)
	else:
		perm = candidates[random.randint(0, len(candidates) - 1)]

	return spec[perm], spec_lengths[perm]


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []

	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		dr = dr.float()
		dg = dg.float()
		r_loss = torch.mean((1 - dr)**2)
		g_loss = torch.mean(dg**2)
		loss += r_loss + g_loss
		r_losses.append(r_loss.item())
		g_losses.append(g_loss.item())
	return loss, r_losses, g_losses


def generator_loss(disc_outputs):
	loss = 0
	gen_losses = []

	for dg in disc_outputs:
		dg = dg.float()
		l = torch.mean((1 - dg)**2)
		gen_losses.append(l)
		loss += l
	return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
	"""z_p, logs_q: [b, h, t_t]
	m_p, logs_p: [b, h, t_t]
	"""
	z_p = z_p.float()
	logs_q = logs_q.float()
	m_p = m_p.float()
	logs_p = logs_p.float()
	z_mask = z_mask.float()

	kl = logs_p - logs_q - 0.5
	kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2.0 * logs_p)
	kl = torch.sum(kl * z_mask)
	l = kl / torch.sum(z_mask)
	return l


def main():
	"""Assume Single Node Multi GPUs Training Only"""
	assert torch.cuda.is_available(), "CPU training is not allowed."
	n_gpus = torch.cuda.device_count()
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "6060"
	hps = utils.get_hparams()
	mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
	net_dur_disc = None
	global global_step, run_start_step, resume_skip_step

	if rank == 0:
		logger = utils.get_logger(hps.model_dir)
		logger.info(hps)
		writer = SummaryWriter(log_dir=hps.model_dir)
		writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

	dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
	torch.manual_seed(hps.train.seed)
	torch.cuda.set_device(rank)

	train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
	train_sampler = DistributedBucketSampler(train_dataset, hps.train.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas=n_gpus, rank=rank, shuffle=True)
	collate_fn = TextAudioSpeakerCollate(qwen_teacher_dim=getattr(hps.model, "qwen_teacher_dim", 2048))
	train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)

	if rank == 0:
		eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
		eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn)

	if hps.data.n_speakers == 0:
		raise ValueError("n_speakers must be > 0")

	n_languages = getattr(hps.data, "n_languages", 0)

	print("Using noise scaled MAS for VITS2")
	noise_scale_delta = 2e-6
	freeze_decoder_only = getattr(hps.train, "freeze_decoder_only", False)
	net_dur_disc = DurationDiscriminatorV2(hps.model.hidden_channels, hps.model.hidden_channels, 3, 0.1, gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0).cuda(rank)
	net_g = SynthesizerTrn(len(symbols), 128, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, n_languages=n_languages, noise_scale_delta=noise_scale_delta, **utils.synthesizer_kwargs(hps)).cuda(rank)
	net_d = MultiPeriodDiscriminator(periods=tuple(getattr(hps.train, "mpd_periods", (2, 3, 5, 7, 11)))).cuda(rank)

	use_mrd = getattr(hps.train, "use_mrd", True)
	mrd_fft_sizes = tuple(getattr(hps.train, "mrd_fft_sizes", (2048, 1024, 512)))
	net_mrd = MultiResolutionDiscriminator(fft_sizes=mrd_fft_sizes).cuda(rank) if use_mrd else None

	if n_gpus == 1 and rank == 0:  # net_g: always find_unused_parameters=False (True double-marks qwen_proj). Aux losses must connect to o in forward. Single-GPU skips DDP entirely.
		print("Single GPU: DDP disabled")
	net_g = maybe_ddp(net_g, rank, n_gpus, find_unused_parameters=False)
	net_d = maybe_ddp(net_d, rank, n_gpus, find_unused_parameters=False)

	if net_mrd is not None:
		net_mrd = maybe_ddp(net_mrd, rank, n_gpus, find_unused_parameters=False)

	if net_dur_disc is not None:
		net_dur_disc = maybe_ddp(net_dur_disc, rank, n_gpus, find_unused_parameters=False)

	freeze_mode = utils.configure_generator_freeze(net_g, freeze_decoder_only)
	hps.train.freeze_decoder_only = freeze_mode == "decoder_only"
	if rank == 0 and freeze_mode == "decoder_only":
		print("Decoder-only training: frozen enc_p, enc_q, flow, dp, spp (dec.* trains)")

	weight_decay = float(getattr(hps.train, "weight_decay", 0.0))  # Optimizers and schedulers are built before the checkpoint load so that resume can restore Adam moments and the LR decay position into them.
	adam_kwargs = dict(betas=hps.train.betas, eps=hps.train.eps, weight_decay=weight_decay)
	optim_g = torch.optim.AdamW(utils.generator_parameters_for_optimizer(net_g, freeze_mode), hps.train.learning_rate, **adam_kwargs)
	optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, **adam_kwargs)
	optim_mrd = torch.optim.AdamW(net_mrd.parameters(), hps.train.learning_rate, **adam_kwargs) if net_mrd is not None else None
	optim_dur_disc = torch.optim.AdamW(net_dur_disc.parameters(), hps.train.learning_rate, **adam_kwargs) if net_dur_disc is not None else None

	make_scheduler = lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, gamma=hps.train.lr_decay, last_epoch=-1) if optim is not None else None
	scheduler_g = make_scheduler(optim_g)
	scheduler_d = make_scheduler(optim_d)
	scheduler_mrd = make_scheduler(optim_mrd)
	scheduler_dur_disc = make_scheduler(optim_dur_disc)

	resume = bool(getattr(hps.train, "resume", False))  # resume=false keeps the historical behaviour: load weights only, restart step/epoch/LR. resume=true continues the run, which also stops MAS alignment noise (annealed over the first 5000 steps) from being re-injected into an already converged model on every restart.
	epoch_str = 1
	global_step = 0
	resumed_from_step = None
	g_checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")

	if g_checkpoint_path is not None:
		mode = "resuming" if resume else "fine-tuning"
		try:
			_, _, _, _, (ckpt_step, ckpt_epoch) = utils.load_checkpoint(g_checkpoint_path, net_g, optim_g if resume else None, scheduler_g if resume else None)
			if resume:
				global_step, epoch_str = ckpt_step, ckpt_epoch
				resumed_from_step = global_step
			print(f"Loaded generator weights for {mode}: {g_checkpoint_path}")

			for name, net, optim, scheduler in (("D", net_d, optim_d, scheduler_d), ("MRD", net_mrd, optim_mrd, scheduler_mrd), ("DUR", net_dur_disc, optim_dur_disc, scheduler_dur_disc)):
				if net is None:
					continue
				path = utils.latest_checkpoint_path(hps.model_dir, f"{name}_*.pth")
				if path is None:
					print(f"No {name} checkpoint found, initializing from scratch")
					continue
				utils.load_checkpoint(path, net, optim if resume else None, scheduler if resume else None)
				print(f"Loaded {name} weights for {mode}: {path}")
		except Exception as e:
			print(f"Could not load checkpoint, starting from scratch. Error: {e}")
			epoch_str = 1
			global_step = 0
			resumed_from_step = None
	else:
		print("No generator checkpoint found, starting training from scratch")

	if resume and epoch_str > 1:
		for scheduler in (scheduler_g, scheduler_d, scheduler_mrd, scheduler_dur_disc):  # ExponentialLR cannot be constructed with last_epoch != -1 (KeyError: initial_lr), and older checkpoints carry no scheduler state, so step it forward instead.
			if scheduler is not None and scheduler.last_epoch <= 0:
				utils.advance_scheduler_to_epoch(scheduler, epoch_str)
		print(f"Resumed at epoch {epoch_str}, step {global_step}, lr {optim_g.param_groups[0]['lr']:.3e}")

	resume_skip_step = resumed_from_step
	run_start_step = global_step
	scaler = GradScaler(enabled=hps.train.fp16_run)
	for epoch in range(epoch_str, hps.train.epochs + 1):
		if rank == 0:
			train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_mrd, net_dur_disc], [optim_g, optim_d, optim_mrd, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_mrd, scheduler_dur_disc], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
		else:
			train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_mrd, net_dur_disc], [optim_g, optim_d, optim_mrd, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_mrd, scheduler_dur_disc], scaler, [train_loader, None], None, None)

		scheduler_g.step()
		scheduler_d.step()

		if scheduler_mrd is not None:
			scheduler_mrd.step()

		if net_dur_disc is not None:
			scheduler_dur_disc.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
	net_g, net_d, net_mrd, net_dur_disc = nets
	optim_g, optim_d, optim_mrd, optim_dur_disc = optims
	scheduler_g, scheduler_d, scheduler_mrd, scheduler_dur_disc = schedulers
	train_loader, eval_loader = loaders

	if writers is not None:
		writer, writer_eval = writers

	train_loader.batch_sampler.set_epoch(epoch)
	global global_step
	net_g.train()
	net_d.train()

	if net_mrd is not None:
		net_mrd.train()

	if net_dur_disc is not None:
		net_dur_disc.train()

	if rank == 0:
		loader = tqdm.tqdm(train_loader, desc="Loading train data")
	else:
		loader = train_loader

	use_spp = getattr(hps.model, "use_spp", False)
	use_gst = getattr(hps.model, "use_gst", False)
	c_pitch = getattr(hps.train, "c_pitch", 1.0)
	c_pitch_l1 = getattr(hps.train, "c_pitch_l1", 10.0)
	use_pitch_l1_loss = getattr(hps.train, "use_pitch_l1_loss", True)
	c_vuv = getattr(hps.train, "c_vuv", 1.0)
	c_energy = getattr(hps.train, "c_energy", 1.0)
	gst_dropout = getattr(hps.train, "gst_dropout", 0.5)
	gst_same_clip_prob = getattr(hps.train, "gst_same_clip_prob", 0.0)
	gst_same_speaker_only = getattr(hps.train, "gst_same_speaker_only", True)
	c_oob = getattr(hps.train, "c_oob", 0.0)
	use_oob_loss = getattr(hps.train, "use_oob_loss", c_oob > 0)
	c_mrd = getattr(hps.train, "c_mrd", 0.1)
	c_mrd_disc = getattr(hps.train, "c_mrd_disc", c_mrd)
	aux_warmup_steps = int(getattr(hps.train, "aux_warmup_steps", 0))  # Shared phase-in for loss terms that are newly enabled or re-scoped by a config change.
	c_fm = getattr(hps.train, "c_fm", 1.0)
	c_stft = getattr(hps.train, "c_stft", 0.0)
	use_stft_loss = getattr(hps.train, "use_stft_loss", c_stft > 0)
	stft_loss_scales = getattr(hps.train, "stft_loss_scales", utils.DEFAULT_STFT_LOSS_SCALES)
	stft_hf_fmin = getattr(hps.train, "stft_hf_fmin", 4000.0)
	stft_hf_fmax = getattr(hps.train, "stft_hf_fmax", 14000.0)
	stft_hf_weight = getattr(hps.train, "stft_hf_weight", 2.0)
	stft_log_weight = getattr(hps.train, "stft_log_weight", 0.0)
	stft_log_floor = getattr(hps.train, "stft_log_floor", 3e-3)
	freeze_decoder_only = getattr(hps.train, "freeze_decoder_only", False)
	use_qwen_distill = getattr(hps.train, "use_qwen_distill", False)
	c_qwen = getattr(hps.train, "c_qwen", 0.0)
	grad_clip = getattr(hps.train, "grad_clip", None)
	grad_clip_d = getattr(hps.train, "grad_clip_d", None)
	keep_last_checkpoints = getattr(hps.train, "keep_last_checkpoints", 0)

	for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, languages, pitch, energy, qwen_feat, qwen_lengths) in enumerate(loader):
		module_g = unwrap_ddp(net_g)
		current_mas_noise_scale = (0.01 - module_g.noise_scale_delta * global_step)
		module_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
		x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
		spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
		y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
		speakers = speakers.cuda(rank, non_blocking=True)
		languages = languages.cuda(rank, non_blocking=True)
		pitch = pitch.cuda(rank, non_blocking=True)
		energy = energy.cuda(rank, non_blocking=True)
		qwen_feat = qwen_feat.cuda(rank, non_blocking=True)
		qwen_lengths = qwen_lengths.cuda(rank, non_blocking=True)

		style_ref_mel, style_ref_lengths = None, None
		gst_active = 0.0
		gst_same_clip = 0.0
		if use_gst:
			style_ref_mel, style_ref_lengths = sample_style_reference(spec, spec_lengths, speakers, gst_dropout, gst_same_speaker_only, gst_same_clip_prob)
			if style_ref_mel is not None:
				gst_active = 1.0
				gst_same_clip = 1.0 if style_ref_mel is spec else 0.0

		with autocast(enabled=hps.train.fp16_run):
			raw = net_g(x, x_lengths, spec, spec_lengths, speakers, languages, pitch, energy=energy, style_ref_mel=style_ref_mel, style_ref_lengths=style_ref_lengths, global_step=global_step, qwen_feat=qwen_feat, qwen_lengths=qwen_lengths)
			(y_hat, l_length, l_pitch, l_pitch_l1, l_energy, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), hidden_x, logw, logw_, logpitch, logpitch_, vuv_logit, energy_hat, l_qwen_raw) = unpack_synthesizer_forward(raw)
			mel = spec
			y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
			y_hat_mel = utils.mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
			y = slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
			y_disc, y_hat_disc = utils.match_audio_lengths(y, y_hat.detach())

			if freeze_decoder_only:  # Discriminator
				loss_disc_all = torch.tensor(0.0, device=y.device)
				losses_disc_r, losses_disc_g = [], []
				loss_mrd_disc = torch.tensor(0.0, device=y.device)
				losses_mrd_disc_r, losses_mrd_disc_g = [], []
			else:
				y_d_hat_r, y_d_hat_g, _, _ = net_d(y_disc, y_hat_disc)
				with autocast(enabled=False):
					loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
					loss_disc_all = loss_disc
					loss_mrd_disc = torch.tensor(0.0, device=loss_disc.device)
					losses_mrd_disc_r = []
					losses_mrd_disc_g = []

					if net_mrd is not None:
						y_mrd, y_hat_mrd = utils.match_audio_lengths(y_disc.squeeze(1), y_hat_disc.squeeze(1))
						y_mrd_hat_r, y_mrd_hat_g, _, _ = net_mrd(y_mrd, y_hat_mrd)
						loss_mrd_disc, losses_mrd_disc_r, losses_mrd_disc_g = discriminator_loss(y_mrd_hat_r, y_mrd_hat_g)
						loss_mrd_disc = loss_mrd_disc / max(len(losses_mrd_disc_r), 1)
						loss_disc_all = loss_disc_all + c_mrd_disc * loss_mrd_disc

			loss_dur_disc_all = None  # Duration Discriminator (skip when adapting decoder only)
			if net_dur_disc is not None and not freeze_decoder_only:
				y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach())
				with autocast(enabled=False):
					(loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
					loss_dur_disc_all = loss_dur_disc

				optim_dur_disc.zero_grad()
				scaler.scale(loss_dur_disc_all).backward()
				scaler.unscale_(optim_dur_disc)
				grad_norm_dur_disc = grad_norm_and_clip(net_dur_disc.parameters(), grad_clip_d)
				scaler.step(optim_dur_disc)

		optim_d.zero_grad()
		if not freeze_decoder_only:
			scaler.scale(loss_disc).backward()
			scaler.unscale_(optim_d)
			grad_norm_d = grad_norm_and_clip(net_d.parameters(), grad_clip_d)
			scaler.step(optim_d)
		else:
			grad_norm_d = 0.0

		if net_mrd is not None and optim_mrd is not None and not freeze_decoder_only:
			optim_mrd.zero_grad()  # c_mrd_disc stays on the MRD's own update: it has a separate optimizer, so this factor is what keeps the band discriminator from outrunning the generator.
			scaler.scale(c_mrd_disc * loss_mrd_disc).backward()
			scaler.unscale_(optim_mrd)
			grad_norm_mrd = grad_norm_and_clip(net_mrd.parameters(), grad_clip_d)
			scaler.step(optim_mrd)
		else:
			grad_norm_mrd = 0.0

		with autocast(enabled=hps.train.fp16_run):
			y_gen, y_hat_gen = utils.match_audio_lengths(y, y_hat)  # Generator
			y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_gen, y_hat_gen)

			if net_dur_disc is not None and not freeze_decoder_only:
				y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)

			with autocast(enabled=False):
				loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
				loss_fm = feature_loss(fmap_r, fmap_g) * c_fm
				loss_gen, losses_gen = generator_loss(y_d_hat_g)
				loss_gen_all = loss_gen + loss_fm + loss_mel
				loss_oob = torch.tensor(0.0, device=loss_gen_all.device)
				loss_mrd_gen = torch.tensor(0.0, device=loss_gen_all.device)
				loss_fm_mrd = torch.tensor(0.0, device=loss_gen_all.device)
				loss_stft = torch.tensor(0.0, device=loss_gen_all.device)
				loss_qwen = torch.tensor(0.0, device=loss_gen_all.device)

				if net_mrd is not None:
					y_mrd, y_hat_mrd = utils.match_audio_lengths(y_gen.squeeze(1), y_hat_gen.squeeze(1))
					_, y_mrd_hat_g, fmap_mrd_r, fmap_mrd_g = net_mrd(y_mrd, y_hat_mrd)
					loss_mrd_gen, losses_mrd_gen = generator_loss(y_mrd_hat_g)
					loss_mrd_gen = loss_mrd_gen / max(len(losses_mrd_gen), 1)
					loss_fm_mrd = feature_loss(fmap_mrd_r, fmap_mrd_g) / max(len(fmap_mrd_r), 1)
					loss_gen_all = loss_gen_all + c_mrd * loss_ramp(aux_warmup_steps) * (loss_mrd_gen + loss_fm_mrd)  # Ramped: after a band-layout change the MRD needs to re-converge before its verdict is worth following, otherwise a near-random critic steers the decoder.

				if use_oob_loss and c_oob > 0:
					loss_oob = utils.out_of_band_stft_loss(y_hat_gen, y_gen, hps.data.filter_length, hps.data.hop_length, hps.data.win_length, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax) * c_oob
					loss_gen_all = loss_gen_all + loss_oob

				if use_stft_loss and c_stft > 0:
					loss_stft = utils.multi_resolution_stft_loss(y_hat_gen, y_gen, stft_loss_scales, hps.data.sampling_rate, hf_fmin=stft_hf_fmin, hf_fmax=stft_hf_fmax, hf_weight=stft_hf_weight, log_weight=stft_log_weight, log_floor=stft_log_floor) * c_stft
					loss_gen_all = loss_gen_all + loss_stft

				loss_pitch_nll = torch.tensor(0.0, device=loss_gen_all.device)
				loss_pitch_l1 = torch.tensor(0.0, device=loss_gen_all.device)
				loss_energy = torch.tensor(0.0, device=loss_gen_all.device)
				loss_vuv = torch.tensor(0.0, device=loss_gen_all.device)
				loss_dur_gen = torch.tensor(0.0, device=loss_gen_all.device)

				if freeze_decoder_only:
					loss_dur = torch.tensor(0.0, device=loss_gen_all.device)
					loss_kl = torch.tensor(0.0, device=loss_gen_all.device)
				else:
					loss_dur = nonnegative_flow_nll(l_length)
					loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask).abs() * hps.train.c_kl
					loss_gen_all = loss_gen_all + loss_dur + loss_kl

					if net_dur_disc is not None:
						loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
						loss_gen_all += loss_dur_gen

					if use_spp and l_pitch is not None:
						loss_pitch_nll = nonnegative_flow_nll(l_pitch) * c_pitch
						loss_gen_all += loss_pitch_nll

					if use_pitch_l1_loss and l_pitch_l1 is not None:
						loss_pitch_l1 = l_pitch_l1 * c_pitch_l1
						loss_gen_all += loss_pitch_l1

					if l_energy is not None:
						loss_energy = l_energy * c_energy
						loss_gen_all += loss_energy

					if use_spp and vuv_logit is not None:
						voiced_target = (pitch.unsqueeze(1) > 1e-4).float()
						voiced_target = voiced_target[:, :, :vuv_logit.size(2)]
						vuv_mask = z_mask[:, :, :vuv_logit.size(2)]
						loss_vuv = F.binary_cross_entropy_with_logits(vuv_logit, voiced_target, reduction="none")
						loss_vuv = torch.sum(loss_vuv * vuv_mask)
						denom = torch.clamp(torch.sum(vuv_mask), min=1.0)
						loss_vuv = (loss_vuv / denom) * c_vuv
						loss_gen_all += loss_vuv

					if use_qwen_distill and c_qwen > 0.0 and l_qwen_raw is not None:
						loss_qwen = l_qwen_raw * c_qwen
						loss_gen_all = loss_gen_all + loss_qwen

		optim_g.zero_grad()
		scaler.scale(loss_gen_all).backward()
		scaler.unscale_(optim_g)
		grad_norm_g = grad_norm_and_clip(net_g.parameters(), grad_clip)
		scaler.step(optim_g)
		scaler.update()

		if rank == 0:
			if global_step % hps.train.log_interval == 0:
				lr = optim_g.param_groups[0]["lr"]
				segment_mel_frames = hps.train.segment_size // hps.data.hop_length
				c_mel = hps.train.c_mel
				c_kl = hps.train.c_kl

				mel_unweighted = _to_float(loss_mel) / c_mel if c_mel > 0 else 0.0
				stft_unweighted = _to_float(loss_stft) / c_stft if c_stft > 0 else None
				kl_unweighted = _to_float(loss_kl) / c_kl if c_kl > 0 else None
				qwen_raw = _to_float(loss_qwen) / c_qwen if c_qwen > 0 and use_qwen_distill else None
				rms_posterior = _to_float(y_hat_gen.pow(2).mean().sqrt())
				contrast_gt = spectral_contrast_db(y_gen, hps)
				contrast_post = spectral_contrast_db(y_hat_gen, hps)
				pitch_diag = compute_pitch_train_diagnostics(logpitch, pitch, ids_slice, segment_mel_frames, l_pitch_l1=l_pitch_l1 if use_pitch_l1_loss else None, f0_max=getattr(hps.data, "f0_max", 800.0), )

				logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
				scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g, "train/freeze_decoder_only": float(freeze_decoder_only), "loss/g/mel_unweighted": mel_unweighted, "loss/g/kl_unweighted": kl_unweighted if kl_unweighted is not None else 0.0, "loss/g/qwen_unweighted": qwen_raw if qwen_raw is not None else 0.0, "audio/rms_posterior": rms_posterior, "train/gst_active": gst_active, "train/gst_same_clip": gst_same_clip, "train/mas_noise_scale": float(getattr(module_g, "current_mas_noise_scale", 0.0)), }
				if stft_unweighted is not None:
					scalar_dict["loss/g/stft_unweighted"] = stft_unweighted
				for tag, val in (("spectrum/contrast_gt", contrast_gt), ("spectrum/contrast_post", contrast_post)):
					if val is not None:
						scalar_dict[tag] = val
				scalar_dict.update(pitch_diag)
				scalar_dict["train/flow_prosody_teacher_forcing"] = float(getattr(module_g, "flow_prosody_teacher_forcing", False) and getattr(module_g, "use_flow_prosody", False))
				scalar_dict["train/flow_prosody_predicted"] = float(module_g._use_predicted_flow_prosody(global_step) if hasattr(module_g, "_use_predicted_flow_prosody") else 0.0)
				scalar_dict["train/mas_flow_prosody_parity"] = float(getattr(module_g, "mas_flow_prosody_parity", False))

				logger.info(build_train_log_line(global_step, lr, scalar_dict))

				if net_mrd is not None:
					scalar_dict.update({"loss/d/mrd": loss_mrd_disc, "loss/g/mrd": loss_mrd_gen, "loss/g/fm_mrd": loss_fm_mrd, "grad_norm_mrd": grad_norm_mrd, })

				if net_dur_disc is not None and loss_dur_disc_all is not None:
					scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all, "loss/g/dur_gen": loss_dur_gen, "grad_norm_dur_disc": grad_norm_dur_disc, })

				scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/oob": loss_oob, "loss/g/stft": loss_stft, "loss/g/qwen": loss_qwen, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/pitch_l1": loss_pitch_l1, "loss/g/spp_nll": loss_pitch_nll, "loss/g/energy": loss_energy, "loss/g/vuv": loss_vuv, })
				scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
				scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
				scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
				image_dict = {"slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()), "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()), "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()), }
				image_dict.update(build_diagnostic_images(hps, {"pitch": pitch, "logpitch": logpitch, "energy": energy, "energy_hat": energy_hat, "vuv_logit": vuv_logit, "logw": logw, "logw_": logw_, "x_mask": x_mask, "attn": attn, "ids_slice": ids_slice, "y_mel": y_mel, "y_hat_mel": y_hat_mel, "y_gen": y_gen, "y_hat_gen": y_hat_gen}))
				histogram_dict = {"latent/z_p": z_p[0].detach().float(), "latent/z": z[0].detach().float()}  # z_p should look standard normal if the flow is healthy; z_p drifting off N(0,1) is the earliest visible sign the prior path will not match the posterior at inference.
				utils.summarize(writer=writer, global_step=global_step, images=apply_tb_trend_labels_images(image_dict), scalars=apply_tb_trend_labels_scalars(scalar_dict), histograms=histogram_dict, )

			if global_step % hps.train.eval_interval == 0 and global_step != resume_skip_step:  # On resume the loop re-enters at the step that was just saved; re-saving would overwrite a good checkpoint with an identical one and waste an eval pass.
				evaluate(hps, net_g, eval_loader, writer_eval)
				lr_now = optim_g.param_groups[0]["lr"]
				utils.save_checkpoint(net_g, optim_g, lr_now, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)), global_step=global_step, scheduler=scheduler_g)
				utils.save_checkpoint(net_d, optim_d, lr_now, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)), global_step=global_step, scheduler=scheduler_d)

				if net_mrd is not None:
					utils.save_checkpoint(net_mrd, optim_mrd, lr_now, epoch, os.path.join(hps.model_dir, "MRD_{}.pth".format(global_step)), global_step=global_step, scheduler=scheduler_mrd)

				if net_dur_disc is not None:
					utils.save_checkpoint(net_dur_disc, optim_dur_disc, lr_now, epoch, os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)), global_step=global_step, scheduler=scheduler_dur_disc)

				utils.prune_old_checkpoints(hps.model_dir, keep=keep_last_checkpoints)
		global_step += 1

	if rank == 0:
		logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
	generator.eval()
	with torch.no_grad():
		x = None  # guards the empty-eval_loader check below
		for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, languages, pitch, energy, *_qwen) in enumerate(eval_loader):
			x, x_lengths = x.cuda(0), x_lengths.cuda(0)
			spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
			y, y_lengths = y.cuda(0), y_lengths.cuda(0)
			speakers = speakers.cuda(0)
			languages = languages.cuda(0)
			x = x[:1]
			x_lengths = x_lengths[:1]
			spec = spec[:1]
			spec_lengths = spec_lengths[:1]
			y = y[:1]
			y_lengths = y_lengths[:1]
			speakers = speakers[:1]
			languages = languages[:1]
			break

		if x is None:  # Check if we got any data from the eval_loader
			print("Warning: eval_loader is empty, skipping evaluation")
			generator.train()
			return

		y_hat, attn, mask, *_ = unwrap_ddp(generator).infer(x, x_lengths, sid=speakers, lid=languages)
		y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
		mel = spec
		y_hat_mel = utils.mel_spectrogram_torch(y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)

	image_dict = {"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
	audio_dict = {"gen/audio": y_hat[0, :, :y_hat_lengths[0]]}

	stft_args = (hps.data.filter_length, hps.data.hop_length, hps.data.win_length, hps.data.sampling_rate)  # Inference length never matches the reference, so a frame-wise eval loss would compare unrelated frames. LTAS averages time away, which keeps it valid across differing durations.
	freqs, ltas_gen = utils.long_term_average_spectrum(y_hat[0, :, :y_hat_lengths[0]], *stft_args)
	_, ltas_gt = utils.long_term_average_spectrum(y[0, :, :y_lengths[0]], *stft_args)
	scalar_dict = {"eval/ltas_l1_db": float(np.abs(ltas_gen - ltas_gt).mean()), "eval/length_ratio": float(int(y_hat_lengths[0]) / max(int(y_lengths[0]), 1))}
	image_dict["spectrum/ltas_eval"] = utils.plot_curves_to_numpy([("ground truth", ltas_gt), ("infer", ltas_gen)], x=freqs, xlabel="Hz", ylabel="dB", title="Eval LTAS: inference vs held-out reference")

	global logged_ground_truth  # Once per process, not once per training run: a resumed run writes a fresh event file and would otherwise have no ground-truth reference to compare its samples against.
	if not logged_ground_truth:
		logged_ground_truth = True
		image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
		audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

	utils.summarize(writer=writer_eval, global_step=global_step, images=apply_tb_trend_labels_images(image_dict), audios=apply_tb_trend_labels_audios(audio_dict), scalars=apply_tb_trend_labels_scalars(scalar_dict), audio_sampling_rate=hps.data.sampling_rate, )
	generator.train()


if __name__ == "__main__":
	main()
