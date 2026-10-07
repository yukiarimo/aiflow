import json
import os
import sys
from glob import glob
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 256
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048
CONST = 1997.3794084376191
N_MELS_RVC = 128
HOP_LENGTH_RVC = 160
WINDOW_LENGTH_RVC = 1024
MEL_FMAX_RVC = 8000
COREML_CHUNK_FRAMES = 512


def load_audio_16k(path, sr=SAMPLE_RATE):
	from aiflow.models.yuna_audio.audio_np import load_audio_np
	return load_audio_np(path, sample_rate=sr, mono=True)


class BiGRU(nn.Module):
	def __init__(self, input_features, hidden_features, num_layers):
		super(BiGRU, self).__init__()
		self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

	def forward(self, x):
		return self.gru(x)[0]


class BiLSTM(nn.Module):
	def __init__(self, input_features, hidden_features, num_layers):
		super(BiLSTM, self).__init__()
		self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

	def forward(self, x):
		return self.lstm(x)[0]


class STFT(nn.Module):
	def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
		super(STFT, self).__init__()
		if win_length is None:
			win_length = filter_length
		self.filter_length = filter_length
		self.hop_length = hop_length
		self.win_length = win_length
		self.window = window
		self.forward_transform = None
		fourier_basis = np.fft.fft(np.eye(self.filter_length))
		cutoff = int((self.filter_length / 2 + 1))
		fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
		forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
		if window is not None:
			assert (filter_length >= win_length)
			fft_window = get_window(window, win_length, fftbins=True)
			fft_window = pad_center(fft_window, filter_length)
			fft_window = torch.from_numpy(fft_window).float()
			forward_basis *= fft_window
		self.register_buffer('forward_basis', forward_basis.float())

	def forward(self, input_data):
		num_batches = input_data.size(0)
		num_samples = input_data.size(1)
		input_data = input_data.view(num_batches, 1, num_samples)
		forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)
		cutoff = int((self.filter_length / 2) + 1)
		real_part = forward_transform[:, :cutoff, :]
		imag_part = forward_transform[:, cutoff:, :]
		magnitude = torch.sqrt(real_part**2 + imag_part**2)
		phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
		return magnitude, phase


class MelSpectrogram(torch.nn.Module):
	def __init__(self, n_mels, sample_rate, filter_length, hop_length, win_length=None, mel_fmin=0.0, mel_fmax=None):
		super(MelSpectrogram, self).__init__()
		self.stft = STFT(filter_length, hop_length, win_length)
		mel_basis = mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)
		mel_basis = torch.from_numpy(mel_basis).float()
		self.register_buffer('mel_basis', mel_basis)

	def forward(self, y):
		assert (torch.min(y.data) >= -1)
		assert (torch.max(y.data) <= 1)
		magnitudes, phases = self.stft(y)
		magnitudes = magnitudes.data
		mel_output = torch.matmul(self.mel_basis, magnitudes)
		mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
		return mel_output


class ConvBlockRes(nn.Module):
	def __init__(self, in_channels, out_channels, momentum=0.01):
		super(ConvBlockRes, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), )
		if in_channels != out_channels:
			self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
			self.is_shortcut = True
		else:
			self.is_shortcut = False

	def forward(self, x):
		if self.is_shortcut:
			return self.conv(x) + self.shortcut(x)
		return self.conv(x) + x


class ResEncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
		super(ResEncoderBlock, self).__init__()
		self.n_blocks = n_blocks
		self.conv = nn.ModuleList()
		self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
		for i in range(n_blocks - 1):
			self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
		self.kernel_size = kernel_size
		if self.kernel_size is not None:
			self.pool = nn.AvgPool2d(kernel_size=kernel_size)

	def forward(self, x):
		for i in range(self.n_blocks):
			x = self.conv[i](x)
		if self.kernel_size is not None:
			return x, self.pool(x)
		return x


class ResDecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
		super(ResDecoderBlock, self).__init__()
		out_padding = (0, 1) if stride == (1, 2) else (1, 1)
		self.n_blocks = n_blocks
		self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), )
		self.conv2 = nn.ModuleList()
		self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
		for i in range(n_blocks - 1):
			self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

	def forward(self, x, concat_tensor):
		x = self.conv1(x)
		x = torch.cat((x, concat_tensor), dim=1)
		for i in range(self.n_blocks):
			x = self.conv2[i](x)
		return x


class Encoder(nn.Module):
	def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
		super(Encoder, self).__init__()
		self.n_encoders = n_encoders
		self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
		self.layers = nn.ModuleList()
		self.latent_channels = []
		for i in range(self.n_encoders):
			self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
			self.latent_channels.append([out_channels, in_size])
			in_channels = out_channels
			out_channels *= 2
			in_size //= 2
		self.out_size = in_size
		self.out_channel = out_channels

	def forward(self, x):
		concat_tensors = []
		x = self.bn(x)
		for i in range(self.n_encoders):
			_, x = self.layers[i](x)
			concat_tensors.append(_)
		return x, concat_tensors


class Intermediate(nn.Module):
	def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
		super(Intermediate, self).__init__()
		self.n_inters = n_inters
		self.layers = nn.ModuleList()
		self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
		for i in range(self.n_inters - 1):
			self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

	def forward(self, x):
		for i in range(self.n_inters):
			x = self.layers[i](x)
		return x


class Decoder(nn.Module):
	def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
		super(Decoder, self).__init__()
		self.layers = nn.ModuleList()
		self.n_decoders = n_decoders
		for i in range(self.n_decoders):
			out_channels = in_channels // 2
			self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
			in_channels = out_channels

	def forward(self, x, concat_tensors):
		for i in range(self.n_decoders):
			x = self.layers[i](x, concat_tensors[-1 - i])
		return x


class TimbreFilter(nn.Module):
	def __init__(self, latent_rep_channels):
		super(TimbreFilter, self).__init__()
		self.layers = nn.ModuleList()
		for latent_rep in latent_rep_channels:
			self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

	def forward(self, x_tensors):
		out_tensors = []
		for i, layer in enumerate(self.layers):
			out_tensors.append(layer(x_tensors[i]))
		return out_tensors


class DeepUnet(nn.Module):
	def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(DeepUnet, self).__init__()
		self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
		self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
		self.tf = TimbreFilter(self.encoder.latent_channels)
		self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

	def forward(self, x):
		x, concat_tensors = self.encoder(x)
		x = self.intermediate(x)
		concat_tensors = self.tf(concat_tensors)
		x = self.decoder(x, concat_tensors)
		return x


class DeepUnet0(nn.Module):
	def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(DeepUnet0, self).__init__()
		self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
		self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
		self.tf = TimbreFilter(self.encoder.latent_channels)
		self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

	def forward(self, x):
		x, concat_tensors = self.encoder(x)
		x = self.intermediate(x)
		x = self.decoder(x, concat_tensors)
		return x


class E2E(nn.Module):
	def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(E2E, self).__init__()
		self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
		self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
		self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
		if n_gru:
			self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
		else:
			self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

	def forward(self, x):
		mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
		x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
		hidden_vec = 0
		if len(self.fc) == 4:
			for i in range(len(self.fc)):
				x = self.fc[i](x)
				if i == 0:
					hidden_vec = x
		return hidden_vec, x


class E2E0(nn.Module):
	def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(E2E0, self).__init__()
		self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
		self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
		self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
		if n_gru:
			self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
		else:
			self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

	def forward(self, x):
		mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
		x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
		x = self.fc(x)
		return x


class DeepUnetRVC(nn.Module):
	def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(DeepUnetRVC, self).__init__()
		self.encoder = Encoder(in_channels, N_MELS_RVC, en_de_layers, kernel_size, n_blocks, en_out_channels)
		self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
		self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

	def forward(self, x):
		x, concat_tensors = self.encoder(x)
		x = self.intermediate(x)
		x = self.decoder(x, concat_tensors)
		return x


class E2E_RVC(nn.Module):
	"""Hub / RVC checkpoint: 128 mels, hop 160, mel passed in (no built-in spectrogram)."""
	def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
		super(E2E_RVC, self).__init__()
		self.unet = DeepUnetRVC(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
		self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
		if n_gru:
			self.fc = nn.Sequential(BiGRU(3 * N_MELS_RVC, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
		else:
			self.fc = nn.Sequential(nn.Linear(3 * N_MELS_RVC, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

	def forward(self, mel):
		mel = mel.transpose(-1, -2).unsqueeze(1)
		x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
		return self.fc(x)


def _read_config(model_path):
	config_path = os.path.join(os.path.dirname(model_path), 'config.json')
	if not os.path.isfile(config_path):
		return {}
	with open(config_path) as f:
		return json.load(f)


def _build_from_config(config, kind):
	n_blocks = config.get('n_blocks', 4)
	n_gru = config.get('n_gru', 1)
	kernel_size = tuple(config.get('kernel_size', [2, 2]))
	en_de_layers = config.get('en_de_layers', 5)
	inter_layers = config.get('inter_layers', 4)
	in_channels = config.get('in_channels', 1)
	en_out_channels = config.get('en_out_channels', 16)
	if kind == 'rvc':
		return E2E_RVC(n_blocks, n_gru, kernel_size, en_de_layers, inter_layers, in_channels, en_out_channels)
	hop_length = config.get('hop_length', int(20 / 1000 * SAMPLE_RATE))
	return E2E(hop_length, n_blocks, n_gru, kernel_size, en_de_layers, inter_layers, in_channels, en_out_channels)


def _checkpoint_kind(state_dict):
	keys = state_dict.keys() if hasattr(state_dict, 'keys') else []
	if any(k.startswith('mel.') for k in keys):
		return 'paper'
	if any(k.startswith('unet.tf.') for k in keys):
		return 'paper'
	gru_key = 'fc.0.gru.weight_ih_l0'
	if gru_key in state_dict and state_dict[gru_key].shape[1] == 3 * N_MELS_RVC:
		return 'rvc'
	return 'paper'


def load_model(model_path, device=None):
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = _read_config(model_path)

	if model_path.endswith('.safetensors'):
		from safetensors.torch import load_file
		state_dict = load_file(model_path, device=str(device))
		kind = _checkpoint_kind(state_dict)
		model = _build_from_config(config, kind)
		model.load_state_dict(state_dict, strict=True)
		model.eval()
		return model.to(device), kind

	prev_models = sys.modules.get('models')  # model.pt from convert_to_pt.py pickles full module as top-level `models.*`
	sys.modules['models'] = sys.modules[__name__]
	try:
		loaded = torch.load(model_path, map_location=device, weights_only=False)
	finally:
		if prev_models is None:
			sys.modules.pop('models', None)
		else:
			sys.modules['models'] = prev_models
	if isinstance(loaded, nn.Module):
		kind = 'rvc' if isinstance(loaded, E2E_RVC) else 'paper'
		return loaded.eval().to(device), kind

	state_dict = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else loaded
	kind = _checkpoint_kind(state_dict)
	model = _build_from_config(config, kind)
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	return model.to(device), kind


def audio_to_mel_rvc(audio):
	mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=WINDOW_LENGTH_RVC, hop_length=HOP_LENGTH_RVC, win_length=WINDOW_LENGTH_RVC, n_mels=N_MELS_RVC, fmin=MEL_FMIN, fmax=MEL_FMAX_RVC, htk=True, )
	mel = np.log(np.maximum(mel, 1e-5))
	return torch.from_numpy(mel).float()


def infer_rvc(model, audio, device):
	mel = audio_to_mel_rvc(audio.cpu().numpy()).unsqueeze(0).to(device)
	n_frames = mel.shape[-1]
	n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
	if n_pad:
		mel = F.pad(mel, (0, n_pad))
	with torch.no_grad():
		salience = model(mel)[:, :n_frames]
	return salience.squeeze(0)


def load_coreml_model(path):
	try:
		import coremltools as ct
	except ImportError as e:
		raise ImportError('pip install coremltools') from e
	return ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)


def infer_rvc_coreml(ml_model, audio, chunk_frames=COREML_CHUNK_FRAMES):
	if isinstance(audio, torch.Tensor):
		audio = audio.cpu().numpy()
	mel = audio_to_mel_rvc(audio).numpy()[np.newaxis, ...]
	n_frames = mel.shape[-1]
	n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
	if n_pad:
		mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)))
	chunks = []
	for start in range(0, mel.shape[-1], chunk_frames):
		end = min(start + chunk_frames, mel.shape[-1])
		valid = end - start
		chunk = mel[:, :, start:end]
		if chunk.shape[-1] < chunk_frames:
			chunk = np.pad(chunk, ((0, 0), (0, 0), (0, chunk_frames - chunk.shape[-1])))
		out = ml_model.predict({'mel': chunk.astype(np.float32)})
		salience = np.array(out['salience']).squeeze()
		if salience.ndim == 1:
			salience = salience[np.newaxis, :]
		chunks.append(salience[:valid])
	salience = np.concatenate(chunks, axis=0)[:n_frames]
	return torch.from_numpy(salience).float()


def smoothl1(inputs, targets, alpha=None):
	loss_f = nn.SmoothL1Loss(reduce=False)
	weight = torch.ones(inputs.shape, dtype=torch.float).to(inputs.device)
	if alpha is not None:
		weight[targets != 0] = float(alpha)
	loss = torch.mean(loss_f(inputs, targets) * weight)
	return loss


def bce(inputs, targets):
	loss_f = nn.BCELoss()
	return loss_f(inputs, targets)


def FL(inputs, targets, alpha, gamma):
	loss = F.binary_cross_entropy(inputs, targets, reduce=False)
	weight = torch.ones(inputs.shape, dtype=torch.float).to(inputs.device)
	weight[targets == 1] = float(alpha)
	loss_w = F.binary_cross_entropy(inputs, targets, weight=weight, reduce=False)
	pt = torch.exp(-loss)
	weight_gamma = (1 - pt)**gamma
	return torch.mean(weight_gamma * loss_w)


class MIR1K(Dataset):
	def __init__(self, path, hop_length, sequence_length=None, groups=None):
		self.path = path
		self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
		self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
		self.num_class = N_CLASS
		self.data = []

		print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
		      f"of {self.__class__.__name__} at {path}")
		for group in groups:
			for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
				self.data.extend(self.load(*input_files))

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	@staticmethod
	def availabe_groups():
		return ['test']

	def files(self, group):
		audio_files = glob(os.path.join(self.path, group, '*.wav'))
		label_files = [f.replace('.wav', '.pv') for f in audio_files]
		assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
		assert (all(os.path.isfile(label_file) for label_file in label_files))
		return sorted(zip(audio_files, label_files))

	def load(self, audio_path, label_path):
		data = []
		audio = load_audio_16k(audio_path, SAMPLE_RATE)
		audio_l = len(audio)
		audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
		audio = torch.from_numpy(audio).float()
		audio_steps = audio_l // self.HOP_LENGTH + 1
		pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
		voice_label = torch.zeros(audio_steps, dtype=torch.float)
		with open(label_path, 'r') as f:
			lines = f.readlines()
			i = 0
			for line in lines:
				i += 1
				if float(line) != 0:
					freq = 440 * (2.0**((float(line) - 69.0) / 12.0))
					cent = 1200 * np.log2(freq / 10)
					index = int(round((cent - CONST) / 20))
					pitch_label[i][index] = 1
					voice_label[i] = 1

		if self.seq_len is not None:
			n_steps = self.seq_len // self.HOP_LENGTH + 1
			for i in range(audio_l // self.seq_len):
				begin_t = i * self.seq_len
				end_t = begin_t + self.seq_len + WINDOW_LENGTH
				begin_step = begin_t // self.HOP_LENGTH
				end_step = begin_step + n_steps
				data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step], voice=voice_label[begin_step:end_step], file=audio_path))
			data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:], voice=voice_label[-n_steps:], file=audio_path))
		else:
			data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
		return data


class MIR_ST500(Dataset):
	def __init__(self, path, hop_length, sequence_length=None, groups=None):
		self.path = path
		self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
		self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
		self.num_class = N_CLASS
		self.data = []
		print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
		      f"of {self.__class__.__name__} at {path}")
		for group in groups:
			for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
				self.data.extend(self.load(*input_files))

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	@staticmethod
	def availabe_groups():
		return ['test']

	def files(self, group):
		audio_files = glob(os.path.join(self.path, group, '*.wav'))
		label_files = [f.replace('.wav', '.tsv') for f in audio_files]
		assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
		assert (all(os.path.isfile(label_file) for label_file in label_files))
		return sorted(zip(audio_files, label_files))

	def load(self, audio_path, label_path):
		data = []
		audio = load_audio_16k(audio_path, SAMPLE_RATE)
		audio_l = len(audio)
		audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
		audio = torch.from_numpy(audio).float()
		audio_steps = audio_l // self.HOP_LENGTH + 1
		pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
		voice_label = torch.zeros(audio_steps, dtype=torch.float)

		midi = np.loadtxt(label_path, delimiter='\t', skiprows=1)
		for onset, offset, note in midi:
			left = int(round(onset * SAMPLE_RATE / self.HOP_LENGTH))
			right = int(round(offset * SAMPLE_RATE / self.HOP_LENGTH)) + 1
			freq = 440 * (2.0**((float(note) - 69.0) / 12.0))
			cent = 1200 * np.log2(freq / 10)
			index = int(round((cent - CONST) / 20))
			pitch_label[left:right, index] = 1
			voice_label[left:right] = 1

		if self.seq_len is not None:
			n_steps = self.seq_len // self.HOP_LENGTH + 1
			for i in range(audio_l // self.seq_len):
				begin_t = i * self.seq_len
				end_t = begin_t + self.seq_len + WINDOW_LENGTH
				begin_step = begin_t // self.HOP_LENGTH
				end_step = begin_step + n_steps
				data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step], voice=voice_label[begin_step:end_step], file=audio_path))
			data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:], voice=voice_label[-n_steps:], file=audio_path))
		else:
			data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
		return data


class MDB(Dataset):
	def __init__(self, path, hop_length, sequence_length=None, groups=None):
		self.path = path
		self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
		self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
		self.num_class = N_CLASS
		self.data = []
		print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
		      f"of {self.__class__.__name__} at {path}")
		for group in groups:
			for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
				self.data.extend(self.load(*input_files))

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	@staticmethod
	def availabe_groups():
		return ['test']

	def files(self, group):
		audio_files = glob(os.path.join(self.path, group, '*.wav'))
		label_files = [f.replace('.wav', '.csv') for f in audio_files]
		assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
		assert (all(os.path.isfile(label_file) for label_file in label_files))
		return sorted(zip(audio_files, label_files))

	def load(self, audio_path, label_path):
		data = []
		audio = load_audio_16k(audio_path, SAMPLE_RATE)
		audio_l = len(audio)
		audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
		audio = torch.from_numpy(audio).float()
		audio_steps = audio_l // self.HOP_LENGTH + 1
		pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
		voice_label = torch.zeros(audio_steps, dtype=torch.float)
		df_label = pd.read_csv(label_path)
		for i in range(len(df_label)):
			if float(df_label['midi'][i]):
				freq = 440 * (2.0**((float(df_label['midi'][i]) - 69.0) / 12.0))
				cent = 1200 * np.log2(freq / 10)
				index = int(round((cent - CONST) / 20))
				pitch_label[i][index] = 1
				voice_label[i] = 1

		if self.seq_len is not None:
			n_steps = self.seq_len // self.HOP_LENGTH + 1
			for i in range(audio_l // self.seq_len):
				begin_t = i * self.seq_len
				end_t = begin_t + self.seq_len + WINDOW_LENGTH
				begin_step = begin_t // self.HOP_LENGTH
				end_step = begin_step + n_steps
				data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step], voice=voice_label[begin_step:end_step], file=audio_path))
			data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:], voice=voice_label[-n_steps:], file=audio_path))
		else:
			data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
		return data


class Inference:
	def __init__(self, model, seg_len, seg_frames, hop_length, batch_size, device):
		super(Inference, self).__init__()
		self.model = model.eval()
		self.seg_len = seg_len
		self.seg_frames = seg_frames
		self.batch_size = batch_size
		self.hop_length = hop_length
		self.device = device

	def inference(self, audio):
		with torch.no_grad():
			padded_audio = self.pad_audio(audio)
			segments = self.en_frame(padded_audio)
			hidden_vec_segments, out_segments = self.forward_in_mini_batch(self.model, segments)
			out_segments = self.de_frame(out_segments, type_seg='pitch')[:(len(audio) // self.hop_length + 1)]
			hidden_vec_segments = self.de_frame(hidden_vec_segments, type_seg='pitch')[:(len(audio) // self.hop_length + 1)]
			return hidden_vec_segments, out_segments

	def pad_audio(self, audio):
		audio_len = len(audio)
		seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
		pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
		padded_audio = torch.cat([torch.zeros(self.seg_len // 4).to(self.device), audio, torch.zeros(pad_len - self.seg_len // 4).to(self.device)])
		return padded_audio

	def en_frame(self, audio):
		audio_len = len(audio)
		assert audio_len % (self.seg_len // 2) == 0
		audio = torch.cat([torch.zeros(1024).to(self.device), audio, torch.zeros(1024).to(self.device)])
		segments = []
		start = 0
		while start + self.seg_len <= audio_len:
			segments.append(audio[start:start + self.seg_len + 2048])
			start += self.seg_len // 2
		segments = torch.stack(segments, dim=0)
		return segments

	def forward_in_mini_batch(self, model, segments):
		hidden_vec_segments = []
		out_segments = []
		segments_num = segments.shape[0]
		batch_start = 0
		while True:
			if batch_start + self.batch_size >= segments_num:
				batch_tmp = segments[batch_start:].shape[0]
				segment_in = torch.cat([segments[batch_start:], torch.zeros_like(segments)[:self.batch_size - batch_tmp].to(self.device)], dim=0)
				hidden_vec, out_tmp = model(segment_in)
				hidden_vec_segments.append(hidden_vec[:batch_tmp])
				out_segments.append(out_tmp[:batch_tmp])
				break
			segment_in = segments[batch_start:batch_start + self.batch_size]
			hidden_vec, out_tmp = model(segment_in)
			hidden_vec_segments.append(hidden_vec)
			out_segments.append(out_tmp)
			batch_start += self.batch_size
		hidden_vec_segments = torch.cat(hidden_vec_segments, dim=0)
		out_segments = torch.cat(out_segments, dim=0)
		return hidden_vec_segments, out_segments

	def de_frame(self, segments, type_seg='audio'):
		output = []
		if type_seg == 'audio':
			for segment in segments:
				output.append(segment[self.seg_len // 4:int(self.seg_len * 0.75)])
		else:
			for segment in segments:
				output.append(segment[self.seg_frames // 4:int(self.seg_frames * 0.75)])
		output = torch.cat(output, dim=0)
		return output
