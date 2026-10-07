import os
import glob
import json
import shutil
import subprocess
import io
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import random

from aiflow.models.yuna_audio.audio_np import SAMPLE_RATE, to_mono, resample_audio  # re-export soxr helpers (single definition)


def _ffmpeg_bin():
	ffmpeg_path = shutil.which("ffmpeg")
	if ffmpeg_path is None:
		raise RuntimeError("ffmpeg not found! Install with: brew install ffmpeg / sudo apt install ffmpeg")
	return ffmpeg_path


def _ffprobe_bin():
	ffprobe_path = shutil.which("ffprobe")
	if ffprobe_path is None:
		raise RuntimeError("ffprobe not found")
	return ffprobe_path


def _as_ffmpeg_input(file):
	"""Normalize path / BytesIO / bytes for ffmpeg."""
	if isinstance(file, io.BytesIO):
		file.seek(0)
		return file.read()
	if isinstance(file, (bytes, bytearray)):
		return bytes(file)
	return str(file)


def _probe_audio(input_data):
	ffprobe_path = _ffprobe_bin()
	if isinstance(input_data, bytes):
		probe_cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a:0", "-i", "pipe:0"]
		probe_result = subprocess.run(probe_cmd, input=input_data, capture_output=True)
	else:
		probe_cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a:0", str(input_data)]
		probe_result = subprocess.run(probe_cmd, capture_output=True)
	if probe_result.returncode != 0:
		raise RuntimeError(f"ffprobe failed: {probe_result.stderr.decode()}")
	probe_info = json.loads(probe_result.stdout.decode())
	if not probe_info.get("streams"):
		raise RuntimeError("No audio streams found in file")
	stream = probe_info["streams"][0]
	return int(stream.get("sample_rate", 44100)), int(stream.get("channels", 2))


def _decode_ffmpeg(input_data, sample_rate=None, nchannels=None):
	"""Decode to s16le PCM. If sample_rate/nchannels set, ffmpeg converts in one pass."""
	ffmpeg_path = _ffmpeg_bin()
	if sample_rate is None or nchannels is None:
		native_sr, native_ch = _probe_audio(input_data)
		if sample_rate is None:
			sample_rate = native_sr
		if nchannels is None:
			nchannels = native_ch

	base = [ffmpeg_path, "-threads", "0", "-nostdin", "-hide_banner", "-loglevel", "error"]
	if isinstance(input_data, bytes):
		decode_cmd = base + ["-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(nchannels), "pipe:1"]
		decode_result = subprocess.run(decode_cmd, input=input_data, capture_output=True)
	else:
		decode_cmd = base + ["-i", str(input_data), "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(nchannels), "pipe:1"]
		decode_result = subprocess.run(decode_cmd, capture_output=True)

	if decode_result.returncode != 0:
		raise RuntimeError(f"ffmpeg decoding failed: {decode_result.stderr.decode()}")

	samples = np.frombuffer(decode_result.stdout, dtype=np.int16)
	return samples, sample_rate, nchannels


def audio_read(file, always_2d=False, dtype="float32", sample_rate=None, mono=False):
	"""Decode audio via ffmpeg. Pass sample_rate/mono for one-pass convert (no soxr needed after)."""
	input_data = _as_ffmpeg_input(file)
	nchannels = 1 if mono else None
	samples, sample_rate, nchannels = _decode_ffmpeg(input_data, sample_rate=sample_rate, nchannels=nchannels)

	if nchannels > 1:
		samples = samples.reshape(-1, nchannels)

	if dtype in ("float32", "float64"):
		samples = samples.astype(dtype) / 32768.0
	elif dtype == "int16":
		pass
	else:
		samples = samples.astype(dtype)

	if always_2d and samples.ndim == 1:
		samples = samples[:, np.newaxis]

	return samples, sample_rate


def load_audio_np(file, sample_rate=SAMPLE_RATE, mono=True, dtype="float32"):
	"""Canonical path/bytes → mono float numpy @ sample_rate (ffmpeg one-pass)."""
	samples, sr = audio_read(file, always_2d=False, dtype=dtype, sample_rate=sample_rate, mono=mono)
	if mono:
		samples = to_mono(samples)
	elif samples.ndim > 1 and sample_rate != sr:
		samples = resample_audio(samples, sr, sample_rate)
	return np.asarray(samples, dtype=np.float32 if dtype == "float32" else dtype)


def get_model_path(path_or_hf_repo, **kwargs):
	model_path = Path(path_or_hf_repo)
	if model_path.exists():
		return model_path
	raise FileNotFoundError(f"Local path not found: {path_or_hf_repo}")


def load_config(model_path, **kwargs):
	if isinstance(model_path, str):
		model_path = get_model_path(model_path, **kwargs)

	config_file = model_path / "config.json"
	if config_file.exists():
		with open(config_file, encoding="utf-8") as f:
			return json.load(f)
	raise FileNotFoundError(f"Config not found at {model_path}")


def load_weights(model_path):
	weight_files = glob.glob(str(model_path / "*.safetensors"))
	if not weight_files:
		weight_files = glob.glob(str(model_path / "*.npz"))

	if not weight_files:
		raise FileNotFoundError(f"No weight files found in {model_path}")

	weights = {}
	for wf in weight_files:
		weights.update(mx.load(wf))
	return weights


def apply_quantization(model, config, weights, model_quant_predicate=None):
	quantization = config.get("quantization", None)
	if quantization is None:
		return
	group_size = quantization.get("group_size", 64)

	def get_class_predicate(p, m):
		if not hasattr(m, "to_quantized"):
			return False
		if hasattr(m, "weight") and m.weight.shape[-1] % group_size != 0:
			return False
		if model_quant_predicate is not None:
			pred_result = model_quant_predicate(p, m)
			if isinstance(pred_result, dict):
				return pred_result
			if not pred_result:
				return False
		if p in config["quantization"]:
			return config["quantization"][p]
		return f"{p}.scales" in weights

	nn.quantize(model, group_size=group_size, bits=quantization["bits"], mode=quantization.get("mode", "affine"), class_predicate=get_class_predicate)


def get_model_class():
	from . import qwen3_asr
	return qwen3_asr


def base_load_model(model_path, lazy=False, strict=False, **kwargs):
	if isinstance(model_path, str):
		model_path = get_model_path(model_path, **kwargs)
	elif not isinstance(model_path, Path):
		raise ValueError(f"Invalid model path type: {type(model_path)}")

	config = load_config(model_path)
	config["model_path"] = str(model_path)
	model_class = get_model_class()
	model_config = model_class.ModelConfig.from_dict(config)
	model = model_class.Model(model_config)
	weights = load_weights(model_path)

	if hasattr(model, "sanitize"):
		weights = model.sanitize(weights)

	model_quant_predicate = getattr(model, "model_quant_predicate", None)
	apply_quantization(model, config, weights, model_quant_predicate)
	model.load_weights(list(weights.items()), strict=strict)

	if not lazy:
		mx.eval(model.parameters())

	model.eval()

	if hasattr(model_class.Model, "post_load_hook"):
		model = model_class.Model.post_load_hook(model, model_path)

	return model


def audio_volume_normalize(audio, coeff=0.2):
	temp = np.sort(np.abs(audio))
	if temp[-1] < 0.1:
		scaling_factor = max(temp[-1], 1e-3)
		audio = audio / scaling_factor * 0.1

	temp = temp[temp > 0.01]
	L = temp.shape[0]

	if L <= 10:
		return audio

	volume = np.mean(temp[int(0.9 * L):int(0.99 * L)])
	audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

	max_value = np.max(np.abs(audio))
	if max_value > 1:
		audio = audio / max_value

	return audio


def random_select_audio_segment(audio, length):
	if audio.shape[0] < length:
		audio = np.pad(audio, (0, int(length - audio.shape[0])))
	start_index = random.randint(0, audio.shape[0] - length)
	end_index = int(start_index + length)
	return audio[start_index:end_index]


def load_audio(audio, sample_rate=SAMPLE_RATE, length=None, volume_normalize=False, segment_duration=None):
	if isinstance(audio, mx.array):
		return audio

	if not isinstance(audio, str):
		raise TypeError(f"audio must be str or mx.array, got {type(audio)}")

	if not os.path.exists(audio):
		raise FileNotFoundError(f"Audio file not found: {audio}")

	samples = load_audio_np(audio, sample_rate=sample_rate, mono=True)

	if segment_duration is not None:
		seg_length = int(sample_rate * segment_duration)
		samples = random_select_audio_segment(samples, seg_length)

	if volume_normalize:
		samples = audio_volume_normalize(samples)

	if length is not None:
		if samples.shape[0] > length:
			samples = samples[:length]
		else:
			samples = np.pad(samples, (0, int(length - samples.shape[0])))

	return mx.array(samples, dtype=mx.float32)


def load(model_path, lazy=False, strict=False, **kwargs):
	return base_load_model(model_path=model_path, lazy=lazy, strict=strict, **kwargs)


class _AudioTower:
	"""audio_tower + Whisper FE only. Same get_audio_features / _preprocess_audio as Qwen3ASRModel."""
	def __init__(self, tower, feature_extractor):
		self.audio_tower = tower
		self._feature_extractor = feature_extractor

	def get_audio_features(self, input_features, feature_attention_mask=None):
		return self.audio_tower(input_features, feature_attention_mask)

	def _preprocess_audio(self, audio):
		from .audio_encoder import get_feat_extract_output_lengths
		audio_input = audio[0] if isinstance(audio, list) else audio
		if isinstance(audio_input, str):
			audio_input = load_audio(audio_input)
		audio_np = np.array(audio_input) if isinstance(audio_input, mx.array) else np.asarray(audio_input, dtype=np.float32)
		if audio_np.ndim > 1:
			audio_np = audio_np.reshape(-1)
		audio_inputs = self._feature_extractor(audio_np, sampling_rate=SAMPLE_RATE, return_attention_mask=True, truncation=False, padding=True, return_tensors="np")
		input_features = mx.array(audio_inputs["input_features"])
		feature_attention_mask = mx.array(audio_inputs["attention_mask"])
		aftercnn_lens = get_feat_extract_output_lengths(feature_attention_mask.sum(axis=-1))
		return input_features, feature_attention_mask, int(aftercnn_lens[0].item())


def load_audio_encoder(model_path, lazy=False, **kwargs):
	"""Load Qwen3-ASR audio_tower + WhisperFeatureExtractor. No text LM."""
	from mlx.utils import tree_flatten
	import transformers
	from transformers import WhisperFeatureExtractor
	from .audio_encoder import AudioEncoder
	from .config import ModelConfig
	from .qwen3_asr import Qwen3ASRModel

	if isinstance(model_path, str):
		model_path = get_model_path(model_path, **kwargs)
	config = load_config(model_path)
	tower = AudioEncoder(ModelConfig.from_dict(config).audio_config)
	weights = Qwen3ASRModel.sanitize(load_weights(model_path))
	enc_weights = {k[len("audio_tower."):]: v for k, v in weights.items() if k.startswith("audio_tower.")}
	if not enc_weights:
		raise RuntimeError(f"No audio_tower.* weights in {model_path}")
	need = [k for k, _ in tree_flatten(tower.parameters()) if "positional_embedding" not in k]
	missing = [k for k in need if k not in enc_weights]
	if missing:
		raise RuntimeError(f"audio_tower missing {len(missing)} keys, first: {missing[:8]}")
	tower.load_weights([(k, enc_weights[k]) for k in need], strict=False)
	if not lazy:
		mx.eval(tower.parameters())
	tower.eval()
	prev = transformers.logging.get_verbosity()
	transformers.logging.set_verbosity_error()
	try:
		feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
	finally:
		transformers.logging.set_verbosity(prev)
	return _AudioTower(tower, feature_extractor)


def extract_aut_features(model, audio, sample_rate=SAMPLE_RATE):
	"""Wav / 1-D float → [T, output_dim] (2048). Same tower path as ASR encode."""
	if isinstance(audio, mx.array):
		audio = np.array(audio)
	if not isinstance(audio, str):
		audio = np.asarray(audio, dtype=np.float32)
		if audio.ndim > 1:
			audio = audio.reshape(-1)
		if int(sample_rate) != SAMPLE_RATE:
			audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
	input_features, feature_attention_mask, num_audio_tokens = model._preprocess_audio(audio)
	feats = model.get_audio_features(input_features, feature_attention_mask)
	mx.eval(feats)
	arr = np.array(feats, dtype=np.float32)
	if arr.ndim == 3:
		arr = arr.reshape(-1, arr.shape[-1])
	if arr.ndim != 2:
		raise RuntimeError(f"Qwen audio features must be [T, D], got {arr.shape}")
	if arr.shape[0] > num_audio_tokens:
		arr = arr[:num_audio_tokens]
	return arr
