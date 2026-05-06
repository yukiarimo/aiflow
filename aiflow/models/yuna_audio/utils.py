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
import soxr
import random

SAMPLE_RATE = 16000
MODEL_REMAPPING = {"qwen3_asr": "qwen3_asr"}


def _detect_format_from_bytes(data):
	if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
		return "wav"
	elif data[:3] == b"ID3" or (data[0:2] == b"\xff\xfb" or data[0:2] == b"\xff\xfa"):
		return "mp3"
	elif data[:4] == b"fLaC":
		return "flac"
	elif data[:4] == b"OggS":
		return "vorbis"
	elif data[4:8] == b"ftyp":
		return "m4a"
	else:
		raise ValueError("Unable to detect audio format from bytes")


def _decode_ffmpeg(input_data):
	ffmpeg_path = shutil.which("ffmpeg")
	if ffmpeg_path is None:
		raise RuntimeError("ffmpeg not found! Install with: brew install ffmpeg / sudo apt install ffmpeg")

	ffprobe_path = shutil.which("ffprobe")
	if ffprobe_path is None:
		raise RuntimeError("ffprobe not found")

	if isinstance(input_data, bytes):
		probe_cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a:0", "-i", "pipe:0", ]
		probe_result = subprocess.run(probe_cmd, input=input_data, capture_output=True)
	else:
		probe_cmd = [ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a:0", str(input_data), ]
		probe_result = subprocess.run(probe_cmd, capture_output=True)

	if probe_result.returncode != 0:
		raise RuntimeError(f"ffprobe failed: {probe_result.stderr.decode()}")

	probe_info = json.loads(probe_result.stdout.decode())
	if not probe_info.get("streams"):
		raise RuntimeError("No audio streams found in file")

	stream = probe_info["streams"][0]
	sample_rate = int(stream.get("sample_rate", 44100))
	nchannels = int(stream.get("channels", 2))

	if isinstance(input_data, bytes):
		decode_cmd = [ffmpeg_path, "-threads", "0", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(nchannels), "pipe:1"]
		decode_result = subprocess.run(decode_cmd, input=input_data, capture_output=True)
	else:
		decode_cmd = [ffmpeg_path, "-threads", "0", "-i", str(input_data), "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(nchannels), "pipe:1"]
		decode_result = subprocess.run(decode_cmd, capture_output=True)

	if decode_result.returncode != 0:
		raise RuntimeError(f"ffmpeg decoding failed: {decode_result.stderr.decode()}")

	samples = np.frombuffer(decode_result.stdout, dtype=np.int16)
	return samples, sample_rate, nchannels


def audio_read(file, always_2d=False, dtype="float64"):
	if isinstance(file, io.BytesIO):
		file.seek(0)
		input_data = file.read()
	else:
		input_data = file
	samples, sample_rate, nchannels = _decode_ffmpeg(input_data)

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
	else:
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


def get_model_class(model_type, model_name, category, model_remapping):
	model_type_mapped = model_remapping.get(model_type, None)

	models_dir = Path(__file__).parent / category / "models"
	available_models = []
	if models_dir.exists() and models_dir.is_dir():
		for item in models_dir.iterdir():
			if item.is_dir() and not item.name.startswith("__"):
				available_models.append(item.name)

	if model_name is not None and model_type_mapped != model_type:
		for part in model_name:
			if part in available_models:
				model_type = part
			if part in model_remapping:
				model_type = model_remapping[part]
				break
	elif model_type_mapped is not None:
		model_type = model_type_mapped

	from . import qwen3_asr
	arch = qwen3_asr
	return arch, model_type


def base_load_model(model_path, category, model_remapping, lazy=False, strict=False, **kwargs):
	model_name = None

	if isinstance(model_path, str):
		model_name = model_path.lower().split("/")[-1].split("-")
		model_path = get_model_path(model_path, **kwargs)
	elif isinstance(model_path, Path):
		try:
			index = model_path.parts.index("hub")
			model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
		except ValueError:
			model_name = model_path.name.lower().split("-")
	else:
		raise ValueError(f"Invalid model path type: {type(model_path)}")

	config = load_config(model_path)
	config["model_path"] = str(model_path)
	model_type = config.get("model_type", None)

	if model_type is None:
		model_type = config.get("architecture", None)

	if model_type is None:
		model_type = model_name[0].lower() if model_name is not None else None

	model_class, model_type = get_model_class(model_type=model_type, model_name=model_name, category=category, model_remapping=model_remapping)
	model_config = (model_class.ModelConfig.from_dict(config) if hasattr(model_class, "ModelConfig") else config)
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


def load_audio(audio, sample_rate=24000, length=None, volume_normalize=False, segment_duration=None):
	if isinstance(audio, mx.array):
		return audio

	if not isinstance(audio, str):
		raise TypeError(f"audio must be str or mx.array, got {type(audio)}")

	if not os.path.exists(audio):
		raise FileNotFoundError(f"Audio file not found: {audio}")

	samples, orig_sample_rate = audio_read(audio)
	shape = samples.shape

	if len(shape) > 1:
		samples = samples.sum(axis=1)
		samples = samples / shape[1]

	if sample_rate != orig_sample_rate:
		duration = samples.shape[0] / orig_sample_rate
		num_samples = int(duration * sample_rate)
		samples = resample_audio(samples, orig_sample_rate, sample_rate)

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


def resample_audio(audio, orig_sr, target_sr):
	return soxr.resample(audio, orig_sr, target_sr)


def load_audio_mono(file=None, sr=SAMPLE_RATE, from_stdin=False, dtype=mx.float32):
	audio, sample_rate = audio_read(file, always_2d=True)
	if sample_rate != sr:
		audio = resample_audio(audio, sample_rate, sr)
	return mx.array(audio, dtype=dtype).mean(axis=1)


def load_model(model_path, lazy=False, strict=False, **kwargs):
	return base_load_model(model_path=model_path, category="stt", model_remapping=MODEL_REMAPPING, lazy=lazy, strict=strict, **kwargs)


def load(model_path, lazy=False, strict=False, **kwargs):
	return load_model(model_path, lazy=lazy, strict=strict, **kwargs)
