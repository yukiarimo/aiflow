import os, glob, sys, argparse, logging, json, shutil
import numpy as np
import torch
from torch.nn import functional as F
import warnings
from librosa.filters import mel as librosa_mel_fn
from packaging import version
import soundfile as sf

MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging
warnings.filterwarnings(action="ignore")
mel_basis = {}
hann_window = {}


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
		mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
		mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

	return mel_basis[fmax_dtype_device], fmax_dtype_device


def _compute_stft(y, n_fft, hop_size, win_size, window, center):
	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect").squeeze(1)
	stft_args = {'input': y, 'n_fft': n_fft, 'hop_length': hop_size, 'win_length': win_size, 'window': window, 'center': center, 'pad_mode': 'reflect', 'normalized': False, 'onesided': True}
	if version.parse(torch.__version__) >= version.parse("2"): stft_args['return_complex'] = False

	return torch.stft(**stft_args)


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
	outside_interval_mask = ~inside_interval_mask

	outputs = torch.zeros_like(inputs)
	logabsdet = torch.zeros_like(inputs)

	if tails != "linear": raise RuntimeError(f"{tails} tails are not implemented.")

	unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
	constant = np.log(np.exp(1 - min_derivative) - 1)
	unnormalized_derivatives[..., 0] = constant
	unnormalized_derivatives[..., -1] = constant

	outputs[outside_interval_mask] = inputs[outside_interval_mask]
	logabsdet[outside_interval_mask] = 0
	outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(inputs[inside_interval_mask], unnormalized_widths[inside_interval_mask, :], unnormalized_heights[inside_interval_mask, :], unnormalized_derivatives[inside_interval_mask, :], inverse=inverse, left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)

	return outputs, logabsdet


def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
	if torch.min(inputs) < left or torch.max(inputs) > right: raise ValueError("Input to a transform is not within its domain")

	num_bins = unnormalized_widths.shape[-1]
	if min_bin_width * num_bins > 1.0: raise ValueError("Minimal bin width too large for the number of bins")
	if min_bin_height * num_bins > 1.0: raise ValueError("Minimal bin height too large for the number of bins")

	# Calculate widths
	widths = F.softmax(unnormalized_widths, dim=-1)
	widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
	cumwidths = torch.cumsum(widths, dim=-1)
	cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
	cumwidths = (right - left) * cumwidths + left
	cumwidths[..., 0], cumwidths[..., -1] = left, right
	widths = cumwidths[..., 1:] - cumwidths[..., :-1]

	# Calculate derivatives
	derivatives = min_derivative + F.softplus(unnormalized_derivatives)

	# Calculate heights
	heights = F.softmax(unnormalized_heights, dim=-1)
	heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
	cumheights = torch.cumsum(heights, dim=-1)
	cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
	cumheights = (top - bottom) * cumheights + bottom
	cumheights[..., 0], cumheights[..., -1] = bottom, top
	heights = cumheights[..., 1:] - cumheights[..., :-1]

	# Find bin indices and gather inputs
	bin_locations = cumheights if inverse else cumwidths
	bin_locations[..., -1] += 1e-6
	bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1)[..., None] - 1
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
		assert (discriminant >= 0).all()

		root = (2 * c) / (-b - torch.sqrt(discriminant))
		outputs = root * input_bin_widths + input_cumwidths

		theta_one_minus_theta = root * (1 - root)
		denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
		derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
		logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

		return outputs, -logabsdet
	# Forward pass
	else:
		theta = (inputs - input_cumwidths) / input_bin_widths
		theta_one_minus_theta = theta * (1 - theta)

		numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
		denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
		outputs = input_cumheights + numerator / denominator

		derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
		logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
		return outputs, logabsdet


def load_checkpoint(checkpoint_path, model, optimizer=None):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
	iteration = checkpoint_dict["iteration"]
	learning_rate = checkpoint_dict["learning_rate"]
	if optimizer is not None: optimizer.load_state_dict(checkpoint_dict["optimizer"])

	saved_state_dict = checkpoint_dict["model"]
	state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
	new_state_dict = {}

	for k, v in state_dict.items():
		try:
			if k in saved_state_dict and saved_state_dict[k].shape == v.shape: new_state_dict[k] = saved_state_dict[k]
			else:
				# Log message for mismatched shapes or missing keys
				if k in saved_state_dict: print(f"NOTICE: Size mismatch for {k}. Expected {v.shape}, got {saved_state_dict[k].shape}. Using model initialization.")
				else: print(f"{k} is not in the checkpoint")
				new_state_dict[k] = v
		except:
			print(f"{k} is not in the checkpoint")
			new_state_dict[k] = v

	target_model = model.module if hasattr(model, "module") else model
	target_model.load_state_dict(new_state_dict)
	print(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
	return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
	print(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
	state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
	torch.save({"model": state_dict, "iteration": iteration, "optimizer": optimizer.state_dict(), "learning_rate": learning_rate}, checkpoint_path)


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
	fig.canvas.draw()
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3, ))
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
	fig.canvas.draw()
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3, ))
	plt.close()
	return data


def load_wav_to_torch(full_path):
	data, sampling_rate = sf.read(full_path)
	# Ensure audio is 1D (mono)
	if len(data.shape) > 1:
		data = data[:, 0]

	# If the wav is already naturally between -1 and 1, we multiply by 32768, so that the / max_wav_value math in train.py doesn't mute it completely.
	if np.max(np.abs(data)) <= 1.0:
		data = data * 32768.0

	return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
	with open(filename, encoding="utf-8") as f:
		return [line.strip().split(split) for line in f]


def get_hparams(init=True):
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config", type=str, default="./config.json", help="JSON file for configuration")
	parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
	args = parser.parse_args()
	model_dir = os.path.join("./logs", args.model)
	os.makedirs(model_dir, exist_ok=True)
	config_save_path = os.path.join(model_dir, "config.json")

	if init: shutil.copy(args.config, config_save_path)
	with open(config_save_path, "r") as f:
		config = json.load(f)

	hparams = HParams(**config)
	hparams.model_dir = model_dir

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
