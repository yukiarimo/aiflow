import os, glob, re, sys, argparse, logging, json, shutil
import numpy as np
import torch
from torch.nn import functional as F
import warnings
from packaging import version
import soundfile as sf

MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging
warnings.filterwarnings(action="ignore")
mel_basis = {}
hann_window = {}
_oob_bin_slices_cache = {}
DEFAULT_STFT_LOSS_SCALES = [{"n_fft": 2048, "hop": 256, "win": 2048, "weight": 1.0}, {"n_fft": 1024, "hop": 128, "win": 1024, "weight": 1.0}, {"n_fft": 512, "hop": 64, "win": 512, "weight": 1.0}, {"n_fft": 256, "hop": 32, "win": 256, "weight": 0.75}, ]
DEFAULT_PROSODY_CFG = {"soft_vuv": True, "vuv_temperature": 8.0, "boundary_ramp_frames": 6, "smooth_kernel_size": 5, "voiced_threshold": 1e-4, "energy_log_floor": -13.815510559647095, "smooth_pitch": True, "smooth_energy": True, "match_energy_loss_to_flow": True, "use_vuv_pred_for_smoothing": False, "gate_pitch_by_voicing": True, "gate_energy_by_voicing": False, "force_unvoiced_pitch": False, "pitch_value": None, "pitch_scale": 1.0, "pitch_bias": 0.0, }
DEFAULT_INFERENCE_CFG = {"noise_scale": 0.6, "noise_scale_w": 0.4, "noise_scale_p": 1.0, "length_scale": 1.0, "min_period_duration": 24.0, }


def spectral_normalize_torch(magnitudes):
	return torch.log(torch.clamp(magnitudes, min=1e-5) * 1)


def _get_window(y, win_size):
	dtype_device = str(y.dtype) + "_" + str(y.device)
	wnsize_dtype_device = str(win_size) + "_" + dtype_device
	if wnsize_dtype_device not in hann_window: hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
	return hann_window[wnsize_dtype_device], wnsize_dtype_device


def _get_mel_basis(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
	dtype_device = str(spec.dtype) + "_" + str(spec.device)
	fmax_dtype_device = str(fmax) + "_" + dtype_device
	if fmax_dtype_device not in mel_basis:
		from librosa.filters import mel as librosa_mel_fn  # lazy: only when mel features are computed
		mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
		mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
	return mel_basis[fmax_dtype_device], fmax_dtype_device


def _compute_stft(y, n_fft, hop_size, win_size, window, center):
	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect").squeeze(1)
	stft_args = {'input': y, 'n_fft': n_fft, 'hop_length': hop_size, 'win_length': win_size, 'window': window, 'center': center, 'pad_mode': 'reflect', 'normalized': False, 'onesided': True}
	if version.parse(torch.__version__) >= version.parse("2"): stft_args['return_complex'] = False
	return torch.stft(**stft_args)


def stft_magnitude_torch(y, n_fft, hop_size, win_size, center=False):
	"""Linear STFT magnitudes [B, n_fft // 2 + 1, T]."""
	window, _ = _get_window(y, win_size)
	spec = _compute_stft(y, n_fft, hop_size, win_size, window, center)
	return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def _oob_bin_slices(n_fft, sampling_rate, fmin, fmax):
	"""Return (low_end, high_start, n_freq) for STFT bins outside [fmin, fmax]."""
	key = (n_fft, sampling_rate, float(fmin), float(fmax))
	if key not in _oob_bin_slices_cache:
		n_freq = n_fft // 2 + 1
		low_end = min(int(np.ceil(fmin * n_fft / sampling_rate)), n_freq)
		high_start = min(int(np.floor(fmax * n_fft / sampling_rate)) + 1, n_freq)
		if high_start < low_end:
			high_start = low_end
		_oob_bin_slices_cache[key] = (low_end, high_start, n_freq)
	return _oob_bin_slices_cache[key]


def out_of_band_stft_loss(y_hat, y, n_fft, hop_size, win_size, sampling_rate, fmin, fmax, center=False):
	"""L1 on linear STFT magnitudes outside [fmin, fmax] vs detached target from y."""
	if y_hat.dim() == 3:
		y_hat = y_hat.squeeze(1)
	if y.dim() == 3:
		y = y.squeeze(1)

	mag_hat = stft_magnitude_torch(y_hat, n_fft, hop_size, win_size, center=center)
	with torch.no_grad():
		mag_y = stft_magnitude_torch(y, n_fft, hop_size, win_size, center=center)

	low_end, high_start, n_freq = _oob_bin_slices(n_fft, sampling_rate, fmin, fmax)
	parts_hat, parts_y = [], []
	if low_end > 0:
		parts_hat.append(mag_hat[:, :low_end, :])
		parts_y.append(mag_y[:, :low_end, :])
	if high_start < n_freq:
		parts_hat.append(mag_hat[:, high_start:, :])
		parts_y.append(mag_y[:, high_start:, :])
	if not parts_hat:
		return mag_hat.new_tensor(0.0)

	hat_oob = torch.cat(parts_hat, dim=1)
	y_oob = torch.cat(parts_y, dim=1)
	return F.l1_loss(hat_oob, y_oob)


def match_audio_lengths(y, y_hat):
	"""Trim waveforms to the same length (ISTFT output can be a few samples shorter)."""
	n = min(y.shape[-1], y_hat.shape[-1])
	return y[..., :n], y_hat[..., :n]


def _stft_bin_weights(n_fft, sampling_rate, hf_fmin, hf_fmax, hf_weight, device, dtype):
	n_freq = n_fft // 2 + 1
	weights = torch.ones(n_freq, device=device, dtype=dtype)
	if hf_fmin is not None and hf_fmax is not None and hf_weight != 1.0:
		hf_start = max(0, int(np.floor(hf_fmin * n_fft / sampling_rate)))
		hf_end = min(int(np.ceil(hf_fmax * n_fft / sampling_rate)) + 1, n_freq)
		if hf_end > hf_start:
			weights[hf_start:hf_end] = hf_weight
	return weights.view(1, n_freq, 1)


def multi_resolution_stft_loss(y_hat, y, scales, sampling_rate, hf_fmin=None, hf_fmax=None, hf_weight=2.0, center=False, log_weight=0.0, log_floor=3e-3):
	"""Multi-resolution STFT L1 vs detached target, with optional HF-band emphasis. log_weight adds a log-magnitude term alongside the linear one. Linear L1 is dominated by loud bins, so the low-level noise floor (breath, room, inter-word air) is nearly free to get wrong; the log term makes a quiet bin count as much as a loud one. log_floor is not cosmetic. d|log m|/dm is 1/m, so bins near the 1e-3 epsilon floor of stft_magnitude_torch carry ~1000x the gradient of a loud bin, and on 48 kHz audio brickwalled at 13.9 kHz roughly a quarter of all bins sit exactly on that floor. Clamping gives them zero gradient (clamp_min passes none through) while leaving every bin that carries real signal untouched."""
	if y_hat.dim() == 3:
		y_hat = y_hat.squeeze(1)
	if y.dim() == 3:
		y = y.squeeze(1)

	y_hat, y = match_audio_lengths(y_hat, y)
	total = y_hat.new_tensor(0.0)

	for scale in scales:
		n_fft = int(scale["n_fft"])
		hop = int(scale.get("hop", n_fft // 4))
		win = int(scale.get("win", n_fft))
		scale_weight = float(scale.get("weight", 1.0))

		mag_hat = stft_magnitude_torch(y_hat, n_fft, hop, win, center=center)
		with torch.no_grad():
			mag_y = stft_magnitude_torch(y, n_fft, hop, win, center=center)

		diff = (mag_hat - mag_y).abs()
		scale_log_weight = float(scale.get("log_weight", log_weight))
		if scale_log_weight > 0.0:
			floor = float(scale.get("log_floor", log_floor))
			diff = diff + scale_log_weight * (torch.log(mag_hat.clamp_min(floor)) - torch.log(mag_y.clamp_min(floor))).abs()
		scale_hf_fmin = scale.get("hf_fmin", hf_fmin)
		scale_hf_fmax = scale.get("hf_fmax", hf_fmax)
		scale_hf_weight = float(scale.get("hf_weight", hf_weight))

		if scale_hf_fmin is not None and scale_hf_weight != 1.0:
			w = _stft_bin_weights(n_fft, sampling_rate, scale_hf_fmin, scale_hf_fmax, scale_hf_weight, diff.device, diff.dtype)
			total = total + scale_weight * (diff * w).mean()
		else:
			total = total + scale_weight * diff.mean()

	return total


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
	if torch.min(y) < -1.0: print("min value is ", torch.min(y))
	if torch.max(y) > 1.0: print("max value is ", torch.max(y))
	window, _ = _get_window(y, win_size)
	mel_filter, _ = _get_mel_basis(y, n_fft, num_mels, sampling_rate, fmin, fmax)
	spec = _compute_stft(y, n_fft, hop_size, win_size, window, center)
	spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
	return spectral_normalize_torch(torch.matmul(mel_filter, spec))


def piecewise_rational_quadratic_transform(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails=None, tail_bound=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
	if tails is None: spline_fn, spline_kwargs = rational_quadratic_spline, {}
	else: spline_fn, spline_kwargs = unconstrained_rational_quadratic_spline, {"tails": tails, "tail_bound": tail_bound}
	return spline_fn(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=inverse, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative, **spline_kwargs)


def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails="linear", tail_bound=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
	inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
	if tails != "linear": raise RuntimeError(f"{tails} tails are not implemented.")
	constant = np.log(np.exp(1 - min_derivative) - 1)
	shape = list(unnormalized_derivatives.shape)
	shape[-1] = 1
	const_tensor = torch.full(shape, constant, dtype=unnormalized_derivatives.dtype, device=unnormalized_derivatives.device)
	unnormalized_derivatives = torch.cat([const_tensor, unnormalized_derivatives, const_tensor], dim=-1)  # Concatenate cleanly on the final dimension

	spline_inputs = torch.clamp(inputs, min=-tail_bound, max=tail_bound)  # Avoid boolean indexing (inputs[inside_interval_mask]) which breaks CoreML dynamic shapes. We clamp inputs to safely execute the math on ALL elements without OutOfBounds errors.
	spline_outputs, spline_logabsdet = rational_quadratic_spline(spline_inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=inverse, left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)

	outputs = torch.where(inside_interval_mask, spline_outputs, inputs)  # Use torch.where to cleanly merge the evaluated splines with the linear tails
	logabsdet = torch.where(inside_interval_mask, spline_logabsdet, torch.zeros_like(inputs))
	return outputs, logabsdet


def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
	num_bins = unnormalized_widths.shape[-1]

	widths = F.softmax(unnormalized_widths, dim=-1)  # Calculate widths
	widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
	cumwidths = torch.cumsum(widths, dim=-1)
	cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
	cumwidths = (right - left) * cumwidths + left

	left_t = torch.full_like(cumwidths[..., :1], left)  # Avoid inplace element assignment cumwidths[..., 0] = left
	right_t = torch.full_like(cumwidths[..., :1], right)
	cumwidths = torch.cat([left_t, cumwidths[..., 1:-1], right_t], dim=-1)

	widths = cumwidths[..., 1:] - cumwidths[..., :-1]

	derivatives = min_derivative + F.softplus(unnormalized_derivatives)  # Calculate derivatives

	heights = F.softmax(unnormalized_heights, dim=-1)  # Calculate heights
	heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
	cumheights = torch.cumsum(heights, dim=-1)
	cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
	cumheights = (top - bottom) * cumheights + bottom

	bottom_t = torch.full_like(cumheights[..., :1], bottom)  # Avoid inplace element assignment cumheights[..., 0] = bottom
	top_t = torch.full_like(cumheights[..., :1], top)
	cumheights = torch.cat([bottom_t, cumheights[..., 1:-1], top_t], dim=-1)
	heights = cumheights[..., 1:] - cumheights[..., :-1]

	bin_locations = cumheights if inverse else cumwidths  # Find bin indices and gather inputs

	last_elem = bin_locations[..., -1:] + 1e-6  # Avoid inplace addition bin_locations[..., -1] += 1e-6
	bin_locations = torch.cat([bin_locations[..., :-1], last_elem], dim=-1)
	bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1)[..., None] - 1

	bin_idx = bin_idx.long()  # Explicitly cast to Long/Integer so CoreML knows it's a valid index for .gather()!
	input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
	input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
	input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
	input_heights = heights.gather(-1, bin_idx)[..., 0]
	input_delta = (heights / widths).gather(-1, bin_idx)[..., 0]
	input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
	input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

	if inverse:
		a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives)
		b = input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
		c = -input_delta * (inputs - input_cumheights)
		discriminant = b.pow(2) - 4 * a * c

		discriminant = torch.clamp(discriminant, min=1e-6)  # Clamping discriminant strictly to prevent sqrt(negative) NaNs during tracing
		root = (2 * c) / (-b - torch.sqrt(discriminant))
		outputs = root * input_bin_widths + input_cumwidths
		theta_one_minus_theta = root * (1 - root)
		denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
		derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
		logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
		return outputs, -logabsdet
	else:
		theta = (inputs - input_cumwidths) / input_bin_widths
		theta_one_minus_theta = theta * (1 - theta)
		numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
		denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
		outputs = input_cumheights + numerator / denominator
		derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
		logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
		return outputs, logabsdet


def checkpoint_train_state(checkpoint_dict, checkpoint_path):
	"""(global_step, epoch) for resume. Legacy checkpoints stored the *epoch* in "iteration" and kept the step only in the filename, so read the explicit keys first and fall back to both older conventions."""
	epoch = int(checkpoint_dict.get("epoch") or checkpoint_dict.get("iteration") or 1)
	step = checkpoint_dict.get("global_step")
	if step is None:
		match = re.search(r"_(\d+)\.pth$", os.path.basename(checkpoint_path))
		step = int(match.group(1)) if match else 0
	return max(int(step), 0), max(epoch, 1)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
	iteration = checkpoint_dict["iteration"]
	learning_rate = checkpoint_dict["learning_rate"]
	name = os.path.basename(checkpoint_path)
	if optimizer is not None and checkpoint_dict.get("optimizer") is not None:  # Param-group layout changes (freeze mode, discriminator band count) make these incompatible; fresh moments are recoverable, a crash mid-resume is not.
		try:
			optimizer.load_state_dict(checkpoint_dict["optimizer"])
		except (ValueError, KeyError, RuntimeError) as e:
			print(f"NOTICE: optimizer state in {name} is incompatible ({e}); starting with fresh moments.")
	if scheduler is not None and checkpoint_dict.get("scheduler") is not None:
		try:
			scheduler.load_state_dict(checkpoint_dict["scheduler"])
		except (ValueError, KeyError, RuntimeError) as e:
			print(f"NOTICE: scheduler state in {name} is incompatible ({e}); rebuilding from epoch count.")
	saved_state_dict = checkpoint_dict["model"]
	state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
	new_state_dict = {}

	for k, v in state_dict.items():
		try:
			if k in saved_state_dict and saved_state_dict[k].shape == v.shape: new_state_dict[k] = saved_state_dict[k]
			else:
				if k in saved_state_dict: print(f"NOTICE: Size mismatch for {k}. Expected {v.shape}, got {saved_state_dict[k].shape}. Using model initialization.")  # Log message for mismatched shapes or missing keys
				else: print(f"{k} is not in the checkpoint")
				new_state_dict[k] = v
		except:
			print(f"{k} is not in the checkpoint")
			new_state_dict[k] = v

	target_model = model.module if hasattr(model, "module") else model
	target_model.load_state_dict(new_state_dict)
	resume_step, resume_epoch = checkpoint_train_state(checkpoint_dict, checkpoint_path)
	print(f"Loaded checkpoint '{checkpoint_path}' (epoch {resume_epoch}, step {resume_step})")
	return model, optimizer, learning_rate, iteration, (resume_step, resume_epoch)


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, global_step=None, scheduler=None):
	print(f"Saving model and optimizer state at epoch {epoch} / step {global_step} to {checkpoint_path}")
	state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
	payload = {"model": state_dict, "iteration": epoch, "epoch": epoch, "optimizer": optimizer.state_dict(), "learning_rate": learning_rate}  # "iteration" is kept as an alias of epoch so older loaders and exporters still work.
	if global_step is not None:
		payload["global_step"] = int(global_step)
	if scheduler is not None:
		payload["scheduler"] = scheduler.state_dict()
	torch.save(payload, checkpoint_path)


def prune_old_checkpoints(model_dir, prefixes=("G", "D", "MRD", "DUR"), keep=0):
	"""Keep only the newest `keep` checkpoints per prefix (keep <= 0 disables pruning)."""
	if not keep or int(keep) <= 0:
		return
	for prefix in prefixes:
		paths = scan_checkpoint(model_dir, f"{prefix}_*.pth") or []
		for path in paths[:-int(keep)]:
			try:
				os.remove(path)
			except OSError as e:
				print(f"NOTICE: could not prune {path}: {e}")


def advance_scheduler_to_epoch(scheduler, epoch):
	"""Fast-forward an epoch-stepped scheduler when resuming without saved scheduler state. Constructing ExponentialLR with last_epoch != -1 raises KeyError('initial_lr'), so step it forward through the public API instead."""
	for _ in range(max(int(epoch) - 1, 0)):
		scheduler.step()


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=48000):
	for k, v in scalars.items():
		writer.add_scalar(k, v, global_step)
	for k, v in histograms.items():
		writer.add_histogram(k, v, global_step)
	for k, v in images.items():
		writer.add_image(k, v, global_step, dataformats="HWC")
	for k, v in audios.items():
		writer.add_audio(k, v, global_step, audio_sampling_rate)


def scan_checkpoint(dir_path, regex):
	f_list = sorted(glob.glob(os.path.join(dir_path, regex)), key=lambda f: int("".join(filter(str.isdigit, f))))
	return f_list or None


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
	f_list = scan_checkpoint(dir_path, regex)
	if not f_list: return None
	x = f_list[-1]
	print(x)
	return x


def _figure_to_rgb_numpy(fig):
	"""Rasterize Agg figure to HxWx3 uint8 (matplotlib >=3.9 removed tostring_rgb)."""
	fig.canvas.draw()
	w, h = fig.canvas.get_width_height()
	if hasattr(fig.canvas, "buffer_rgba"):
		buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
		return buf[:, :, :3].copy()
	return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)


def plot_spectrogram_to_numpy(spectrogram):
	global MATPLOTLIB_FLAG

	if not MATPLOTLIB_FLAG:
		import matplotlib
		matplotlib.use("Agg")
		logging.getLogger("matplotlib").setLevel(logging.WARNING)
		MATPLOTLIB_FLAG = True
	import matplotlib.pylab as plt

	fig, ax = plt.subplots(figsize=(10, 2))
	im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
	plt.colorbar(im, ax=ax)
	plt.xlabel("Frames")
	plt.ylabel("Channels")
	plt.tight_layout()
	data = _figure_to_rgb_numpy(fig)
	plt.close()
	return data


def plot_alignment_to_numpy(alignment, info=None):
	global MATPLOTLIB_FLAG
	if not MATPLOTLIB_FLAG:
		import matplotlib
		matplotlib.use("Agg")
		logging.getLogger("matplotlib").setLevel(logging.WARNING)
		MATPLOTLIB_FLAG = True
	import matplotlib.pylab as plt

	fig, ax = plt.subplots(figsize=(6, 4))
	im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
	fig.colorbar(im, ax=ax)
	xlabel = "Decoder timestep"
	if info is not None: xlabel += f"\n\n{info}"
	plt.xlabel(xlabel)
	plt.ylabel("Encoder timestep")
	plt.tight_layout()
	data = _figure_to_rgb_numpy(fig)
	plt.close()
	return data


def _new_axes(figsize):
	"""Shared matplotlib bootstrap for the diagnostic plot helpers."""
	global MATPLOTLIB_FLAG
	if not MATPLOTLIB_FLAG:
		import matplotlib
		matplotlib.use("Agg")
		logging.getLogger("matplotlib").setLevel(logging.WARNING)
		MATPLOTLIB_FLAG = True
	import matplotlib.pylab as plt
	fig, ax = plt.subplots(figsize=figsize)
	return plt, fig, ax


def plot_curves_to_numpy(curves, x=None, xlabel="Frames", ylabel="Value", title=None, step=False, figsize=(10, 3)):
	"""Overlay named 1-D curves; curves = [(label, array), ...]. NaN values break a line."""
	plt, fig, ax = _new_axes(figsize)
	for label, series in curves:
		if series is None or len(series) == 0:
			continue
		axis = np.arange(len(series)) if x is None else np.asarray(x)[:len(series)]
		if step:
			ax.step(axis, series, where="mid", linewidth=1.0, label=label)
		else:
			ax.plot(axis, series, linewidth=1.0, label=label)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title: ax.set_title(title, fontsize=9)
	ax.legend(loc="upper right", fontsize=7)
	ax.grid(alpha=0.25, linewidth=0.4)
	plt.tight_layout()
	data = _figure_to_rgb_numpy(fig)
	plt.close()
	return data


def plot_heatmap_to_numpy(matrix, xlabel="Frames", ylabel="Channels", title=None, cmap="magma", figsize=(10, 2)):
	plt, fig, ax = _new_axes(figsize)
	im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="none", cmap=cmap)
	fig.colorbar(im, ax=ax)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title: ax.set_title(title, fontsize=9)
	plt.tight_layout()
	data = _figure_to_rgb_numpy(fig)
	plt.close()
	return data


def long_term_average_spectrum(wav, n_fft, hop_size, win_size, sampling_rate, eps=1e-10):
	"""Time-averaged magnitude in dB per STFT bin. Averaging out time makes this comparable between clips of different lengths, so it stays valid for inference vs reference."""
	if wav.dim() == 3:
		wav = wav.squeeze(1)
	if wav.dim() == 1:
		wav = wav.unsqueeze(0)
	mag = stft_magnitude_torch(wav.float(), n_fft, hop_size, win_size, center=False)
	ltas = 20.0 * torch.log10(mag.mean(dim=(0, 2)).clamp_min(eps))
	return np.linspace(0.0, sampling_rate / 2.0, ltas.numel()), ltas.detach().cpu().numpy()


def load_wav_to_torch(full_path):
	data, sampling_rate = sf.read(full_path)
	if len(data.shape) > 1:  # Ensure audio is 1D (mono)
		data = data[:, 0]

	if np.max(np.abs(data)) <= 1.0:  # If the wav is already naturally between -1 and 1, we multiply by 32768, so that the / max_wav_value math in train.py doesn't mute it completely.
		data = data * 32768.0

	return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
	with open(filename, encoding="utf-8") as f:
		return [line.strip().split(split) for line in f]


def hparams_to_dict(obj):
	"""Recursively convert HParams to plain dict (safe for iteration / .get())."""
	if isinstance(obj, HParams):
		return {k: hparams_to_dict(v) for k, v in obj.items()}
	if isinstance(obj, dict):
		return {k: hparams_to_dict(v) for k, v in obj.items()}
	return obj


def resolve_id(value, id_map, label="id"):
	"""Resolve a speaker/language given as a name (e.g. 'Yuki', 'en-us') or integer id string."""
	if value is None:
		raise ValueError(f"Missing {label}")
	if isinstance(id_map, HParams):
		id_map = hparams_to_dict(id_map)
	text = str(value).strip()
	if text.isdigit():
		return int(text)
	if id_map and text in id_map:
		return int(id_map[text])
	for name, idx in (id_map or {}).items():
		if str(idx) == text:
			return int(idx)
	raise ValueError(f"Unknown {label} '{value}' (known: {sorted((id_map or {}).keys())})")


def parse_phoneme_field(text):
	"""Accept pre-tokenized phoneme ids (from preprocess) or raw IPA/phoneme strings."""
	parts = text.strip().split()
	if parts and all(part.isdigit() for part in parts):
		return [int(part) for part in parts]
	from aiflow.models.yuna_speech.text import cleaned_text_to_sequence
	return cleaned_text_to_sequence(text)


def pitch_path_for_wav(wav_path, pitch_dir, base_dir=None):
	"""Mirror preprocess_voicedata.py pitch layout: pitch_dir + relpath.wav -> .npy"""
	if base_dir:
		rel_path = os.path.relpath(wav_path, base_dir)
	else:
		rel_path = os.path.basename(wav_path)
	return os.path.join(pitch_dir, rel_path).replace(".wav", ".npy")


def energy_path_for_wav(wav_path, energy_dir, base_dir=None):
	"""Mirror preprocess_voicedata.py energy layout: energy_dir + relpath.wav -> .npy"""
	if base_dir:
		rel_path = os.path.relpath(wav_path, base_dir)
	else:
		rel_path = os.path.basename(wav_path)
	return os.path.join(energy_dir, rel_path).replace(".wav", ".npy")


def qwen_path_for_wav(wav_path, qwen_dir, base_dir=None):
	"""Mirror preprocess_voicedata.py Qwen layout: qwen_dir + relpath.wav -> .npy"""
	if base_dir:
		rel_path = os.path.relpath(wav_path, base_dir)
	else:
		rel_path = os.path.basename(wav_path)
	return os.path.join(qwen_dir, rel_path).replace(".wav", ".npy")


def pool_latent_to_teacher_frames(z, teacher_lengths, source_lengths=None):
	"""Adaptive-average-pool mel-rate latent [B,C,T_mel] to per-item teacher frame counts. Each item is pooled from its own valid span to its own teacher length, then zero-padded to the batch maximum, so pooled frame k is the same instant as teacher frame k. Pooling the whole padded batch to one length is only approximately right: the mel:teacher frame ratio is near constant, so interior frames land close, but the trailing frames of every item shorter than the batch maximum average in zero padding, and the two lengths are rounded independently so a few percent of drift accumulates along the utterance."""
	lengths = [max(int(x), 1) for x in teacher_lengths]
	target_len = max(lengths)
	t_src = z.size(-1)
	pooled = []
	for i, teacher_len in enumerate(lengths):
		valid = t_src if source_lengths is None else max(min(int(source_lengths[i]), t_src), 1)
		frames = F.adaptive_avg_pool1d(z[i:i + 1, :, :valid], teacher_len)
		if teacher_len < target_len:
			frames = F.pad(frames, (0, target_len - teacher_len))
		pooled.append(frames)
	return torch.cat(pooled, dim=0)


def qwen_teacher_mask(teacher_lengths, max_len, device, dtype):
	"""Mask [B,1,T] for valid Qwen AuT frames, broadcast-compatible with y_mask."""
	lengths = torch.as_tensor(teacher_lengths, device=device, dtype=torch.long)
	valid = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
	return valid.unsqueeze(1).to(dtype=dtype)


def normalize_energy_rms(energy):
	"""Log-RMS per mel frame for NN targets / flow conditioning."""
	if not torch.is_tensor(energy):
		energy = torch.from_numpy(energy).float()
	return torch.log(energy.clamp(min=1e-6))


def resolve_prosody_cfg(cfg=None, overrides=None):
	"""Merge default, config, and runtime overrides for prosody smoothing."""
	out = dict(DEFAULT_PROSODY_CFG)
	if cfg is not None:
		if hasattr(cfg, "keys"):
			out.update({k: cfg[k] for k in cfg.keys()})
		else:
			for key in DEFAULT_PROSODY_CFG:
				if hasattr(cfg, key):
					out[key] = getattr(cfg, key)
	if overrides:
		out.update({k: v for k, v in overrides.items() if v is not None})
	return out


def resolve_inference_cfg(cfg=None, overrides=None):
	out = dict(DEFAULT_INFERENCE_CFG)
	if cfg is not None:
		if hasattr(cfg, "keys"):
			out.update({k: cfg[k] for k in cfg.keys()})
		else:
			for key in DEFAULT_INFERENCE_CFG:
				if hasattr(cfg, key):
					out[key] = getattr(cfg, key)
	if overrides:
		out.update({k: v for k, v in overrides.items() if v is not None})
	return out


def synthesizer_kwargs(hps):
	"""Central kwargs for SynthesizerTrn from full hparams."""
	legacy = {"ms_istft_vits", "subbands", "gen_istft_n_fft", "gen_istft_hop_size", "synthesis_kernel_size", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes", "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes", "use_transformer_flows", "use_duration_discriminator", "duration_discriminator_type", "use_mel_posterior_encoder", "n_layers_q", "use_spectral_norm", "use_sdp", }  # Drop legacy HiFi-GAN / MS-iSTFT / unused VITS1 config keys if present.
	kwargs = {k: v for k, v in dict(hps.model).items() if k not in legacy}
	kwargs["prosody_cfg"] = getattr(hps, "prosody", None)
	kwargs["hop_length"] = hps.data.hop_length
	kwargs["sampling_rate"] = hps.data.sampling_rate
	kwargs["filter_length"] = hps.data.filter_length
	kwargs["win_length"] = hps.data.win_length
	kwargs["mel_fmin"] = hps.data.mel_fmin
	kwargs["mel_fmax"] = hps.data.mel_fmax
	kwargs["use_pitch_l1_loss"] = getattr(hps.train, "use_pitch_l1_loss", True)
	kwargs["pitch_l1_noise_scale"] = getattr(hps.train, "pitch_l1_noise_scale", 0.0)
	kwargs["pitch_l1_smoothed_target"] = getattr(hps.train, "pitch_l1_smoothed_target", True)
	kwargs["mas_flow_prosody_parity"] = getattr(hps.train, "mas_flow_prosody_parity", False)
	kwargs["use_qwen_distill"] = getattr(hps.train, "use_qwen_distill", False)
	kwargs["qwen_teacher_dim"] = getattr(hps.model, "qwen_teacher_dim", 2048)
	kwargs["qwen_proj_hidden"] = getattr(hps.model, "qwen_proj_hidden", 512)
	kwargs["qwen_loss_type"] = getattr(hps.train, "qwen_loss_type", "mse")
	kwargs["qwen_norm_weight"] = getattr(hps.train, "qwen_norm_weight", 0.0)
	return kwargs


def _ensure_b1t(x):
	if x is None:
		return None
	if x.dim() == 2:
		return x.unsqueeze(1)
	return x


def soft_vuv_activity(vuv_logit, pitch, mask, soft_vuv=True, vuv_temperature=8.0, voiced_threshold=1e-4):
	"""Return [B,1,T] voicing activity in [0,1] for prosody gating."""
	if vuv_logit is not None:
		if soft_vuv:
			temp = max(float(vuv_temperature), 1e-3)
			return torch.sigmoid(vuv_logit / temp) * mask
		return (vuv_logit > 0.0).float() * mask
	if pitch is not None:
		return (pitch > float(voiced_threshold)).float() * mask
	return mask


def soften_voicing_boundaries(activity, ramp_frames, mask):
	"""Widen voicing on/off transitions over ramp_frames mel steps."""
	ramp_frames = int(ramp_frames)
	if ramp_frames <= 0:
		return activity * mask
	kernel = 2 * ramp_frames + 1
	x = (activity * mask).float()
	x = torch.nn.functional.pad(x, (ramp_frames, ramp_frames), mode="replicate")
	return torch.nn.functional.avg_pool1d(x, kernel_size=kernel, stride=1) * mask


def moving_average_masked(x, kernel_size, mask):
	kernel_size = int(kernel_size)
	if kernel_size <= 1:
		return x * mask
	if kernel_size % 2 == 0:
		kernel_size += 1
	pad = kernel_size // 2
	x = (x * mask).float()
	x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
	return torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=1) * mask


def smooth_prosody_channels(pitch, energy, mask, vuv_logit=None, pitch_raw=None, cfg=None, overrides=None):
	"""Apply soft V/UV gating, boundary ramps, and optional moving-average smoothing."""
	cfg = resolve_prosody_cfg(cfg, overrides)
	mask = _ensure_b1t(mask)
	pitch_in = _ensure_b1t(pitch_raw if pitch_raw is not None else pitch)
	energy_in = _ensure_b1t(energy)
	vuv_for_activity = vuv_logit if cfg.get("use_vuv_pred_for_smoothing", False) else None
	activity = soft_vuv_activity(vuv_for_activity, pitch_in if pitch_in is not None else pitch, mask, cfg["soft_vuv"], cfg["vuv_temperature"], cfg["voiced_threshold"])
	activity = soften_voicing_boundaries(activity, cfg["boundary_ramp_frames"], mask)

	pitch_out = pitch_in
	if pitch_out is not None and cfg["smooth_pitch"]:
		if cfg.get("gate_pitch_by_voicing", True):
			pitch_out = pitch_out * activity
		pitch_out = moving_average_masked(pitch_out, cfg["smooth_kernel_size"], mask)

	energy_out = energy_in
	if energy_out is not None and cfg["smooth_energy"]:
		if cfg.get("gate_energy_by_voicing", False):
			floor = float(cfg["energy_log_floor"])
			energy_out = energy_out * activity + floor * (1.0 - activity)
		energy_out = moving_average_masked(energy_out, cfg["smooth_kernel_size"], mask)

	if pitch_out is not None:  # Infer hacks: force / scale / bias flow pitch (can leave the trained [0, 1] range).
		if cfg.get("pitch_value", None) is not None:
			pitch_out = torch.full_like(pitch_out, float(cfg["pitch_value"]))
		elif cfg.get("force_unvoiced_pitch", False):
			pitch_out = torch.zeros_like(pitch_out)
		else:
			scale = float(cfg.get("pitch_scale", 1.0))
			bias = float(cfg.get("pitch_bias", 0.0))
			if scale != 1.0 or bias != 0.0:
				pitch_out = pitch_out * scale + bias

	return pitch_out, energy_out, activity


def normalize_pitch_hz(pitch_hz, f0_max=800.0):
	"""Scale raw RMVPE Hz F0 to [0, 1]. Unvoiced frames (0 Hz) remain 0."""
	if not torch.is_tensor(pitch_hz):
		pitch_hz = torch.from_numpy(pitch_hz).float()
	pitch_norm = torch.zeros_like(pitch_hz)
	voiced = pitch_hz > 0
	if voiced.any():
		pitch_norm[voiced] = (pitch_hz[voiced] / float(f0_max)).clamp(max=1.0)
	return pitch_norm


def denormalize_pitch_hz(pitch_norm, f0_max=800.0):
	"""Invert normalize_pitch_hz for inference/debug."""
	return pitch_norm * float(f0_max)


def align_pitch_to_phonemes(pitch_frames, attn, x_mask):
	"""Map frame-level normalized pitch [B, T_mel] to phoneme grid [B, 1, T_text] via MAS attention."""
	weights = attn.sum(2).clamp(min=1e-6)  # attn: [B, 1, T_mel, T_text]
	pitch_text = torch.matmul(attn.squeeze(1).transpose(1, 2), pitch_frames.unsqueeze(-1)).transpose(1, 2)
	pitch_text = (pitch_text / weights) * x_mask
	return pitch_text


def align_hidden_to_mel(hidden_text, attn):
	"""Expand phoneme-level hidden states to mel frames via alignment [B,1,T_mel,T_text]."""
	return torch.matmul(attn.squeeze(1), hidden_text.transpose(1, 2)).transpose(1, 2)


def set_module_requires_grad(module, requires_grad):
	for param in module.parameters():
		param.requires_grad = requires_grad


def configure_generator_freeze(net_g, freeze_except_decoder=False):
	"""Configure which generator submodules receive gradients."""
	module = net_g.module if hasattr(net_g, "module") else net_g
	if freeze_except_decoder:
		set_module_requires_grad(module, False)
		if hasattr(module, "dec"):
			set_module_requires_grad(module.dec, True)
		return "decoder_only"
	set_module_requires_grad(module, True)
	return "none"


def generator_parameters_for_optimizer(net_g, freeze_mode="none"):
	module = net_g.module if hasattr(net_g, "module") else net_g
	if freeze_mode == "decoder_only":
		return [p for p in module.dec.parameters() if p.requires_grad]
	return module.parameters()


def get_hparams(init=True):
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config", type=str, default="./config.json", help="JSON file for configuration")
	parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
	parser.add_argument("--freeze-decoder-only", action="store_true", help="Freeze all generator weights except dec.* (decoder adaptation)")
	args = parser.parse_args()
	model_dir = os.path.join("./logs", args.model)
	os.makedirs(model_dir, exist_ok=True)
	config_save_path = os.path.join(model_dir, "config.json")
	if init: shutil.copy(args.config, config_save_path)
	with open(config_save_path, "r") as f:
		config = json.load(f)
	hparams = HParams(**config)
	hparams.model_dir = model_dir
	if not hasattr(hparams, "train") or hparams.train is None:
		hparams.train = HParams()
	if args.freeze_decoder_only:
		hparams.train.freeze_decoder_only = True
	else:
		if not hasattr(hparams.train, "freeze_decoder_only"):
			hparams.train.freeze_decoder_only = False
	return hparams


def get_hparams_from_file(config_path):
	with open(config_path, "r") as f:
		config = json.load(f)
	return HParams(**config)


def get_logger(model_dir, filename="train.log"):
	global logger
	logger = logging.getLogger(os.path.basename(model_dir))
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
	os.makedirs(model_dir, exist_ok=True)
	h = logging.FileHandler(os.path.join(model_dir, filename))
	h.setLevel(logging.DEBUG)
	h.setFormatter(formatter)
	logger.addHandler(h)
	return logger


class HParams:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			if type(v) == dict: v = HParams(**v)
			self[k] = v

	def keys(self):
		return self.__dict__.keys()

	def items(self):
		return self.__dict__.items()

	def values(self):
		return self.__dict__.values()

	def __len__(self):
		return len(self.__dict__)

	def __getitem__(self, key):
		return getattr(self, key)

	def __setitem__(self, key, value):
		return setattr(self, key, value)

	def __contains__(self, key):
		return key in self.__dict__

	def __repr__(self):
		return self.__dict__.__repr__()
