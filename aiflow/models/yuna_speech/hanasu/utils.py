import json
import random
import wave
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from aiflow.models.yuna_speech.text import symbols, text_cleaners

VOCOS_GAIN_DB = -3.0


def load_config(path):
	with Path(path).open("r", encoding="utf-8") as f:
		return json.load(f)


def save_config(config, path):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(config, f, indent=2)


def save_json(data, path):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)


def load_json(path):
	with Path(path).open("r", encoding="utf-8") as f:
		return json.load(f)


def resolve_cache_dir(config, cache_dir=None):
	"""Resolve where prepared dataset tensors are stored. ``cache_dir`` overrides ``config['dataset']['cache_dir']``. Relative paths are interpreted from the current working directory."""
	dataset_cfg = config.get("dataset", {})
	path = Path(cache_dir or dataset_cfg.get("cache_dir", "data/cache"))
	if not path.is_absolute():
		path = Path.cwd() / path
	return path.expanduser().resolve()


def _extract_training_config(meta):
	if not meta:
		return None
	if "config" in meta and isinstance(meta["config"], dict):
		cfg = meta["config"]
		if "banana_config" in cfg:
			return cfg["banana_config"]
		if "project" in cfg and "model" in cfg:
			return cfg
	if "project" in meta and "model" in meta:
		return meta
	return None


def _load_safetensors_metadata(weights_path, metadata_path=None):
	weights_path = Path(weights_path)
	candidates = []
	if metadata_path:
		candidates.append(Path(metadata_path))
	candidates.extend([weights_path.with_suffix(".json"), weights_path.parent / "model_config.json", weights_path.with_name(weights_path.stem + "_config.json"), ])
	for candidate in candidates:
		if not candidate.exists():
			continue
		meta = load_json(candidate)
		config = _extract_training_config(meta)
		tokenizer = meta.get("tokenizer")
		if config or tokenizer:
			return {"config": config, "tokenizer": tokenizer}
	return {}


def load_checkpoint(checkpoint_path, map_location="cpu", metadata_path=None):
	path = Path(checkpoint_path)
	if path.suffix == ".safetensors":
		from safetensors.torch import load_file
		bundle = {"model": load_file(str(path))}
		bundle.update(_load_safetensors_metadata(path, metadata_path=metadata_path))
		return bundle
	ckpt = torch.load(path, map_location=map_location, weights_only=False)
	if isinstance(ckpt, dict) and "model" in ckpt:
		return ckpt
	return {"model": ckpt}


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def count_parameters(module, trainable_only=False):
	params = module.parameters()
	if trainable_only:
		params = (p for p in params if p.requires_grad)
	return sum(p.numel() for p in params)


def get_device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
	path = Path(path)
	path.mkdir(parents=True, exist_ok=True)
	return path


def format_param_count(n):
	if n >= 1_000_000:
		return f"{n / 1_000_000:.2f}M"
	if n >= 1_000:
		return f"{n / 1_000:.1f}K"
	return str(n)


def json_safe(value):
	if isinstance(value, dict):
		return {str(k): json_safe(v) for k, v in value.items()}
	if isinstance(value, list):
		return [json_safe(v) for v in value]
	if isinstance(value, tuple):
		return [json_safe(v) for v in value]
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def to_mono(audio):
	wav = torch.as_tensor(audio, dtype=torch.float32)
	if wav.ndim == 2:
		if wav.shape[0] <= 8:
			wav = wav.mean(dim=0)
		else:
			wav = wav.mean(dim=1)
	if wav.ndim != 1:
		raise ValueError(f"Expected mono or stereo audio, got shape {tuple(wav.shape)}")
	return wav.contiguous()


def resample_linear(wav, orig_sr, target_sr):
	if orig_sr == target_sr:
		return wav
	if wav.numel() == 0:
		return wav
	new_len = max(1, int(round(wav.numel() * target_sr / orig_sr)))
	wav_3d = wav.view(1, 1, -1)
	return F.interpolate(wav_3d, size=new_len, mode="linear", align_corners=False).view(-1)


def preemphasis_safe(wav):
	wav = torch.nan_to_num(wav.float())
	peak = wav.abs().max()
	if peak > 1.0:
		wav = wav / peak
	return wav.clamp(-1.0, 1.0)


class MelSpectrogram:
	"""Log-mel feature extractor that matches the Vocos vocoder exactly. Uses torchaudio.transforms.MelSpectrogram with the same parameters Vocos was trained on (htk mel scale, norm=None, power=1, center padding) and the same safe_log clip (1e-7). The waveform is peak-normalized then scaled by VOCOS_GAIN_DB so the feature distribution lines up with what Vocos saw in training. Output is (frames, n_mels) to match the rest of the pipeline."""
	def __init__(self, config):
		import torchaudio
		self.sample_rate = int(config["sample_rate"])
		self.n_fft = int(config["n_fft"])
		self.hop_length = int(config["hop_length"])
		self.win_length = int(config.get("win_length", config["n_fft"]))
		self.n_mels = int(config["n_mels"])
		self.f_min = float(config.get("f_min", 0.0))
		f_max = config.get("f_max", None)
		self.f_max = float(f_max) if f_max is not None else None
		self.power = float(config.get("power", 1.0))
		self.padding = config.get("padding", "center")
		self.clip_val = float(config.get("clip_val", 1e-7))
		self.gain = float(10.0**(float(config.get("gain_db", VOCOS_GAIN_DB)) / 20.0))
		self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, n_mels=self.n_mels, f_min=self.f_min, f_max=self.f_max, center=self.padding == "center", power=self.power, )

	def __call__(self, wav):
		wav = torch.nan_to_num(torch.as_tensor(wav, dtype=torch.float32))
		peak = wav.abs().max().clamp(min=1e-8)
		wav = wav / peak * self.gain
		if self.padding == "same":
			pad = self.win_length - self.hop_length
			wav = F.pad(wav, (pad // 2, pad // 2), mode="reflect")
		elif wav.numel() < self.win_length:
			wav = F.pad(wav, (0, self.win_length - wav.numel()))
		mel = self.mel_spec(wav)
		mel = torch.log(torch.clamp(mel, min=self.clip_val))
		return mel.transpose(0, 1).contiguous()


def save_wav(path, wav, sample_rate):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	wav_np = preemphasis_safe(wav).detach().cpu().numpy()
	wav_i16 = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
	with wave.open(str(path), "wb") as f:
		f.setnchannels(1)
		f.setsampwidth(2)
		f.setframerate(sample_rate)
		f.writeframes(wav_i16.tobytes())


def validate_audio(wav, sample_rate, min_seconds, max_seconds):
	if wav.numel() == 0:
		raise ValueError("empty audio")
	seconds = wav.numel() / sample_rate
	if seconds < min_seconds:
		raise ValueError(f"audio too short: {seconds:.3f}s")
	if seconds > max_seconds:
		raise ValueError(f"audio too long: {seconds:.3f}s")
	if not torch.isfinite(wav).all():
		raise ValueError("audio contains NaN or inf")


def load_wav(path):
	"""Load mono-ish audio from a path (soundfile, torchaudio, or stdlib wave)."""
	path = Path(path)
	try:
		import soundfile as sf
		data, sr = sf.read(str(path), dtype="float32", always_2d=False)
		return np.asarray(data, dtype=np.float32), int(sr)
	except ImportError:
		pass
	try:
		import torchaudio
		wav, sr = torchaudio.load(str(path))
		return wav.squeeze(0).numpy(), int(sr)
	except ImportError:
		pass
	with wave.open(str(path), "rb") as f:
		channels = f.getnchannels()
		sample_width = f.getsampwidth()
		sample_rate = f.getframerate()
		frames = f.readframes(f.getnframes())
	if sample_width == 1:
		data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
	elif sample_width == 2:
		data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
	elif sample_width == 4:
		data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
	else:
		raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
	if channels > 1:
		data = data.reshape(-1, channels)
	return data, sample_rate


def wav_to_model_mel(wav, sr, config):
	"""Extract a normalized log-mel for the acoustic model from a waveform."""
	audio_cfg = config["audio"]
	target_sr = int(audio_cfg["sample_rate"])
	wav = to_mono(wav)
	wav = resample_linear(wav, sr, target_sr)
	mel = MelSpectrogram(audio_cfg)(wav)
	mel_stats = config.get("audio", {}).get("mel_stats")
	if mel_stats and mel_stats.get("normalized_training", False):
		mean = float(mel_stats["mean"])
		std = max(float(mel_stats["std"]), 1e-5)
		mel = (mel - mean) / std
	return mel.contiguous()


def load_filepaths_and_text(filelist):
	"""Load a pipe-delimited filelist (one row per line)."""
	rows = []
	with Path(filelist).open("r", encoding="utf-8") as f:
		for line in f:
			line = line.rstrip("\n")
			if line.strip():
				rows.append(line.split("|"))
	return rows


def phonemize_text(text, language="en-us"):
	"""Convert normal text into IPA using the same cleaners/phonemizers as text.py."""
	return text_cleaners(text, language)


class PhonemeTokenizer:
	"""Character-level tokenizer using the fixed symbol table from text.py. Pad token is ``_`` (id 0). Unknown characters are skipped, matching ``text.cleaned_text_to_sequence``."""
	def __init__(self, symbol_table=None):
		self.symbols = list(symbol_table if symbol_table is not None else symbols)
		self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

	@property
	def pad_id(self):
		return self.symbol_to_id["_"]

	@property
	def vocab_size(self):
		return len(self.symbols)

	def normalize(self, text):
		return text.strip()

	def encode(self, text):
		text = self.normalize(text)
		ids = [self.symbol_to_id[s] for s in text if s in self.symbol_to_id]
		if not ids:
			raise ValueError("text has no symbols in the vocabulary")
		return ids

	def decode(self, ids):
		return "".join(self.symbols[i] for i in ids if 0 <= i < len(self.symbols))

	def to_dict(self):
		return {"type": "text_symbols", "symbols": self.symbols}

	@classmethod
	def from_dict(cls, data):
		stored = data.get("symbols")
		return cls(symbol_table=list(stored) if stored else None)


class VocosVocoder:
	"""Thin wrapper around a trained Vocos generator (backbone + ISTFT head). Accepts a log-mel of shape (frames, n_mels) in the Vocos feature space and returns a 1D waveform tensor on CPU."""
	def __init__(self, backbone, head, device):
		self.backbone = backbone
		self.head = head
		self.device = device

	@torch.inference_mode()
	def __call__(self, mel):
		features = mel.to(self.device).transpose(0, 1).unsqueeze(0)
		x = self.backbone(features)
		audio = self.head(x)
		return audio.squeeze(0).detach().cpu()


def denormalize_mel_tensor(mel, mel_stats):
	"""Differentiable inverse of the training mel z-score (no hard clamp, so gradients flow). Used to feed predicted mels back through Vocos for the waveform-domain training loss."""
	if not mel_stats or not mel_stats.get("normalized_training", False):
		return mel
	mean = float(mel_stats["mean"])
	std = max(float(mel_stats["std"]), 1e-5)
	return mel * std + mean


def _load_vocos_modules(vocos_dir, device, checkpoint_name="latest.ckpt"):
	"""Shared loader: build Vocos backbone+head from a weights folder and load the checkpoint."""
	import logging
	logging.getLogger("torio").setLevel(logging.WARNING)
	from aiflow.models.yuna_speech.vocos.models import ISTFTHead, VocosBackbone
	vocos_dir = Path(vocos_dir)
	config_path = vocos_dir / "config.json"
	checkpoint_path = vocos_dir / checkpoint_name
	if not config_path.exists() or not checkpoint_path.exists():
		raise FileNotFoundError(f"Vocos config or checkpoint missing in {vocos_dir}")
	cfg = load_json(config_path)
	backbone = VocosBackbone(**cfg["backbone"])
	head = ISTFTHead(**cfg["head"])
	ckpt = torch.load(checkpoint_path, map_location="cpu")
	state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
	for module, prefix in ((backbone, "backbone."), (head, "head.")):
		sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
		module.load_state_dict(sub, strict=False)
	backbone.eval().to(device)
	head.eval().to(device)
	return backbone, head


def load_vocos_vocoder(vocos_dir, device="cpu", checkpoint_name="latest.ckpt"):
	"""Load a trained Vocos vocoder from a weights folder (config.json + checkpoint). Model code comes from the bundled yuna_speech.vocos package."""
	backbone, head = _load_vocos_modules(vocos_dir, device, checkpoint_name)
	return VocosVocoder(backbone, head, device)


def load_vocos_generator(vocos_dir, device="cpu", checkpoint_name="latest.ckpt"):
	"""Load a frozen Vocos generator (backbone, head) for use as a training-time perceptual loss. Parameters are frozen and set to eval, but (unlike VocosVocoder) the modules are NOT wrapped in inference_mode, so gradients can flow back through the *input mel* to the acoustic model while the vocoder weights stay fixed."""
	backbone, head = _load_vocos_modules(vocos_dir, device, checkpoint_name)
	for module in (backbone, head):
		for p in module.parameters():
			p.requires_grad_(False)
	return backbone, head


def vocos_mel_to_wav(backbone, head, mel_features):
	"""Decode a waveform from log-mel features of shape (B, n_mels, T) using a Vocos generator."""
	return head(backbone(mel_features))
