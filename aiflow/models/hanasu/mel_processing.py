import warnings
warnings.filterwarnings(action="ignore")
import torch
from librosa.filters import mel as librosa_mel_fn
from packaging import version
mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5): return torch.log(torch.clamp(x, min=clip_val) * C)
def dynamic_range_decompression_torch(x, C=1): return torch.exp(x) / C
def spectral_normalize_torch(magnitudes): return dynamic_range_compression_torch(magnitudes)
def spectral_de_normalize_torch(magnitudes): return dynamic_range_decompression_torch(magnitudes)

def _check_audio_range(y):
    if torch.min(y) < -1.0: print("min value is ", torch.min(y))
    if torch.max(y) > 1.0: print("max value is ", torch.max(y))

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

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    _check_audio_range(y)
    window, _ = _get_window(y, win_size)
    spec = _compute_stft(y, n_fft, hop_size, win_size, window, center)
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    mel_filter, _ = _get_mel_basis(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
    return spectral_normalize_torch(torch.matmul(mel_filter, spec))

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    _check_audio_range(y)
    window, _ = _get_window(y, win_size)
    mel_filter, _ = _get_mel_basis(y, n_fft, num_mels, sampling_rate, fmin, fmax)
    spec = _compute_stft(y, n_fft, hop_size, win_size, window, center)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spectral_normalize_torch(torch.matmul(mel_filter, spec))