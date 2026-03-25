import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from utils import get_padding
import torchaudio
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

LRELU_SLOPE = 0.1


class HifiganGenerator(torch.nn.Module):
	def __init__(self, in_channels=128, resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)), resblock_kernel_sizes=(3, 7, 11), upsample_kernel_sizes=(20, 8, 4, 4), upsample_initial_channel=512, upsample_factors=(8, 4, 2, 2), inference_padding=5, sample_rate=48000):
		super().__init__()
		self.inference_padding = inference_padding
		self.num_kernels = len(resblock_kernel_sizes)
		self.num_upsamples = len(upsample_factors)
		self.sample_rate = sample_rate
		self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
		self.ups = nn.ModuleList()

		for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
			self.ups.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2**(i + 1)), k, u, padding=(k - u) // 2)))

		self.resblocks = nn.ModuleList()
		for i in range(len(self.ups)):
			ch = upsample_initial_channel // (2**(i + 1))
			for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
				self.resblocks.append(ResBlock(ch, k, d))

		self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

	def forward(self, x):
		o = self.conv_pre(x)

		for i in range(self.num_upsamples):
			o = F.leaky_relu(o, LRELU_SLOPE)
			o = self.ups[i](o)
			z_sum = None

			for j in range(self.num_kernels):
				if z_sum is None:
					z_sum = self.resblocks[i * self.num_kernels + j](o)
				else:
					z_sum += self.resblocks[i * self.num_kernels + j](o)
			o = z_sum / self.num_kernels

		o = F.leaky_relu(o)
		o = self.conv_post(o)
		o = torch.tanh(o)
		return o

	def remove_weight_norm(self):
		for layer in self.ups:
			remove_weight_norm(layer)
		for layer in self.resblocks:
			layer.remove_weight_norm()

		remove_weight_norm(self.conv_pre)
		remove_weight_norm(self.conv_post)


class ResBlock(torch.nn.Module):
	def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
		super().__init__()
		self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
		self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])

	def forward(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			xt = F.leaky_relu(x, LRELU_SLOPE)
			xt = c1(xt)
			xt = F.leaky_relu(xt, LRELU_SLOPE)
			xt = c2(xt)
			x = xt + x
		return x

	def remove_weight_norm(self):
		for l in self.convs1:
			remove_weight_norm(l)
		for l in self.convs2:
			remove_weight_norm(l)


class PeriodDiscriminator(torch.nn.Module):
	def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
		super().__init__()
		self.period = period
		norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
		self.convs = nn.ModuleList([norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
		self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

	def forward(self, x):
		feat = []
		b, c, t = x.shape

		if t % self.period != 0:
			n_pad = self.period - (t % self.period)
			x = F.pad(x, (0, n_pad), "reflect")
			t = t + n_pad

		x = x.view(b, c, t // self.period, self.period)
		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, LRELU_SLOPE)
			feat.append(x)

		x = self.conv_post(x)
		feat.append(x)
		x = torch.flatten(x, 1, -1)
		return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.discriminators = nn.ModuleList([PeriodDiscriminator(2), PeriodDiscriminator(3), PeriodDiscriminator(5), PeriodDiscriminator(7), PeriodDiscriminator(11)])

	def forward(self, x):
		scores = []
		feats = []

		for d in self.discriminators:
			score, feat = d(x)
			scores.append(score)
			feats.append(feat)
		return scores, feats


class ScaleDiscriminator(torch.nn.Module):
	def __init__(self, use_spectral_norm=False):
		super().__init__()
		norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
		self.convs = nn.ModuleList([norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)), norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2))])
		self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

	def forward(self, x):
		feat = []

		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, LRELU_SLOPE)
			feat.append(x)

		x = self.conv_post(x)
		feat.append(x)
		x = torch.flatten(x, 1, -1)
		return x, feat


class MultiScaleDiscriminator(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.discriminators = nn.ModuleList([ScaleDiscriminator(use_spectral_norm=True), ScaleDiscriminator(), ScaleDiscriminator()])
		self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

	def forward(self, x):
		scores = []
		feats = []

		for i, d in enumerate(self.discriminators):
			if i != 0:
				x = self.meanpools[i - 1](x)
			score, feat = d(x)
			scores.append(score)
			feats.append(feat)
		return scores, feats


class HifiganDiscriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.mpd = MultiPeriodDiscriminator()
		self.msd = MultiScaleDiscriminator()

	def forward(self, x):
		scores, feats = self.mpd(x)
		scores_, feats_ = self.msd(x)
		return scores + scores_, feats + feats_


def feature_loss(features_real, features_generate):
	loss = 0
	for r, g in zip(features_real, features_generate):
		for rl, gl in zip(r, g):
			loss += torch.mean(torch.abs(rl - gl))
	return loss * 2


def discriminator_loss(real, generated):
	loss = 0
	real_losses = []
	generated_losses = []

	for r, g in zip(real, generated):
		r_loss = torch.mean((1 - r)**2)
		g_loss = torch.mean(g**2)
		loss += r_loss + g_loss
		real_losses.append(r_loss.item())
		generated_losses.append(g_loss.item())
	return loss, real_losses, generated_losses


def generator_loss(discriminator_outputs):
	loss = 0
	generator_losses = []

	for x in discriminator_outputs:
		l = torch.mean((1 - x)**2)
		generator_losses.append(l)
		loss += l
	return loss, generator_losses


def load_model(checkpoint_path="model.pth"):
	"""Loads HiFi-GAN generator weights, strips out unnecessary modules, and removes weight norms for massive inference speedups. Supports MPS automatically."""
	device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Loading HiFi-GAN checkpoint from: {checkpoint_path} onto {device}")
	hifigan = HifiganGenerator().to(device)
	checkpoint = torch.load(checkpoint_path, map_location=device)

	# Determine the correct state dict key
	if "generator" in checkpoint and "model" in checkpoint["generator"]:
		generator_state = checkpoint["generator"]["model"]
	elif "generator" in checkpoint and isinstance(checkpoint["generator"], dict) and "state_dict" in checkpoint["generator"]:
		generator_state = checkpoint["generator"]["state_dict"]
	elif "generator" in checkpoint:
		generator_state = checkpoint["generator"]
	elif "model" in checkpoint:
		generator_state = checkpoint["model"]
	elif isinstance(checkpoint, dict) and any(k.startswith("conv_pre") for k in checkpoint.keys()):
		generator_state = checkpoint
	else:
		raise KeyError("Could not find generator state_dict in checkpoint.")

	consume_prefix_in_state_dict_if_present(generator_state, "module.")
	missing_keys, unexpected_keys = hifigan.load_state_dict(generator_state, strict=True)

	if missing_keys:
		print(f"Warning: Missing keys in HiFi-GAN state_dict: {missing_keys}")
	if unexpected_keys:
		print(f"Warning: Unexpected keys in HiFi-GAN state_dict: {unexpected_keys}")

	# Bakes normalization directly into weights to skip math overhead
	hifigan.eval()
	hifigan.remove_weight_norm()
	print("Successfully loaded HiFi-GAN model.")
	return hifigan


def infer_audio(model, mel_tensor, output_path="output.wav", sample_rate=48000):
	"""Run inference converting a Mel Spectrogram into raw audio waveform."""
	device = next(model.parameters()).device
	mel_tensor = mel_tensor.to(device)

	# Ensure shape is [batch, mel_channels, time]
	if len(mel_tensor.shape) == 2:
		mel_tensor = mel_tensor.unsqueeze(0)

	with torch.inference_mode():
		audio_waveform = model(mel_tensor).squeeze(1).cpu()

	torchaudio.save(output_path, audio_waveform, sample_rate)
	print(f"Saved generated audio to: {output_path}")
	return audio_waveform
