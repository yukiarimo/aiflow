import torch
from torch import nn
import torchaudio


class MelSpectrogramFeatures(torch.nn.Module):
	"""
    Log-mel-spectrogram feature extractor. Defaults are tuned for 48 kHz high-pitched vocals (n_fft=2048, hop_length=512, n_mels=128).

    Args:
        sample_rate: Audio sampling rate.
        n_fft: Size of the FFT.
        hop_length: Hop size between frames.
        n_mels: Number of mel bands.
        f_min: Lowest frequency (Hz) of the mel filterbank.
        f_max: Highest frequency (Hz) of the mel filterbank. None means Nyquist (sample_rate / 2).
        padding: Type of padding. Options are "center" or "same".
    """
	def __init__(self, sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128, f_min=0.0, f_max=None, padding="center", ):
		super().__init__()
		if padding not in ["center", "same"]:
			raise ValueError("Padding must be 'center' or 'same'.")
		self.padding = padding
		self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max, center=padding == "center", power=1, )

	def forward(self, audio):
		if self.padding == "same":
			pad = self.mel_spec.win_length - self.mel_spec.hop_length
			audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
		mel = self.mel_spec(audio)
		features = safe_log(mel)
		return features


class ConvNeXtBlock(nn.Module):
	"""
	ConvNeXt Block to 1D audio signal.

    Args:
        dim: Number of input channels.
        intermediate_dim: Dimensionality of the intermediate layer.
        layer_scale_init_value: Initial value for the layer scale. None means no scaling.
    """
	def __init__(self, dim, intermediate_dim, layer_scale_init_value):
		super().__init__()
		self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
		self.norm = nn.LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
		self.act = nn.GELU()
		self.pwconv2 = nn.Linear(intermediate_dim, dim)
		self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None)

	def forward(self, x):
		residual = x
		x = self.dwconv(x)
		x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
		x = residual + x
		return x


def safe_log(x, clip_val=1e-7):
	"""Element-wise logarithm with clipping to avoid near-zero values."""
	return torch.log(torch.clip(x, min=clip_val))


class VocosBackbone(nn.Module):
	"""
    Vocos backbone module built with ConvNeXt blocks. Preserves the same temporal resolution across all layers.

    Args:
        input_channels: Number of input feature channels (the number of mel bands).
        dim: Hidden dimension of the model.
        intermediate_dim: Intermediate dimension used in ConvNeXtBlock.
        num_layers: Number of ConvNeXtBlock layers.
        layer_scale_init_value: Initial value for layer scaling. Defaults to `1 / num_layers`.
    """
	def __init__(self, input_channels, dim, intermediate_dim, num_layers, layer_scale_init_value=None):
		super().__init__()
		self.input_channels = input_channels
		self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
		self.norm = nn.LayerNorm(dim, eps=1e-6)
		layer_scale_init_value = layer_scale_init_value or 1 / num_layers
		self.convnext = nn.ModuleList([ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_layers)])
		self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, (nn.Conv1d, nn.Linear)):
			nn.init.trunc_normal_(m.weight, std=0.02)
			nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.embed(x)
		x = self.norm(x.transpose(1, 2))
		x = x.transpose(1, 2)
		for conv_block in self.convnext:
			x = conv_block(x)
		x = self.final_layer_norm(x.transpose(1, 2))
		return x


class ISTFT(nn.Module):
	"""
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges. Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs. The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft: Size of Fourier transform.
        hop_length: The distance between neighboring sliding window frames.
        win_length: The size of window frame and STFT filter.
        padding: Type of padding. Options are "center" or "same".
    """
	def __init__(self, n_fft, hop_length, win_length, padding="same"):
		super().__init__()
		if padding not in ["center", "same"]:
			raise ValueError("Padding must be 'center' or 'same'.")
		self.padding = padding
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.win_length = win_length
		window = torch.hann_window(win_length)
		self.register_buffer("window", window)

	def forward(self, spec):
		"""
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec: Input complex spectrogram of shape (B, N, T), where B is the batch size,
                N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
		if self.padding == "center":
			# Fallback to pytorch native implementation
			return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
		elif self.padding == "same":
			pad = (self.win_length - self.hop_length) // 2
		else:
			raise ValueError("Padding must be 'center' or 'same'.")

		assert spec.dim() == 3, "Expected a 3D tensor as input"
		B, N, T = spec.shape

		# Inverse FFT
		ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
		ifft = ifft * self.window[None, :, None]

		# Overlap and Add
		output_size = (T - 1) * self.hop_length + self.win_length
		y = torch.nn.functional.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length), )[:, 0, 0, pad:-pad]

		# Window envelope
		window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
		window_envelope = torch.nn.functional.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length), ).squeeze()[pad:-pad]

		# Normalize
		assert (window_envelope > 1e-11).all()
		y = y / window_envelope

		return y


class ISTFTHead(torch.nn.Module):
	"""
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim: Hidden dimension of the model.
        n_fft: Size of Fourier transform.
        hop_length: The distance between neighboring sliding window frames, which should align with
            the resolution of the input features.
        padding: Type of padding. Options are "center" or "same".
    """
	def __init__(self, dim, n_fft, hop_length, padding="center"):
		super().__init__()
		out_dim = n_fft + 2
		self.out = torch.nn.Linear(dim, out_dim)
		self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

	def forward(self, x):
		"""
        Args:
            x: Input tensor of shape (B, L, H), where B is the batch size,
                L is the sequence length, and H denotes the model dimension.

        Returns:
            Reconstructed time-domain audio signal of shape (B, T).
        """
		x = self.out(x).transpose(1, 2)
		mag, p = x.chunk(2, dim=1)
		mag = torch.exp(mag)
		mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
		# wrapping happens here. These two lines produce the real and imaginary value
		real = torch.cos(p)
		imag = torch.sin(p)
		S = mag * (real + 1j * imag)
		audio = self.istft(S)
		return audio
