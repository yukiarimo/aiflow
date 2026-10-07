import glob
import importlib
import inspect
import json
import logging
import re
from io import BytesIO
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from mlx.utils import tree_flatten
from PIL import Image, ImageOps
from transformers import AutoProcessor
from aiflow.models.yuna_audio.utils import load_audio_np, resample_audio as _resample_audio
from aiflow.models.yuna_audio.audio_encoder import get_feat_extract_output_lengths
from .base import BaseImageProcessor
from .yuna_tokenizer import load_tokenizer

MAX_FILE_SIZE_GB = 5
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def skip_multimodal_module(path):
	return ("vision_model" in path or "vision_tower" in path or "audio_model" in path or "audio_tower" in path)


def get_model_and_args(config):
	model_type = config["model_type"].lower()
	if model_type == "qwen3_avl":
		model_type = "qwen3_avlm"

	try:
		arch = importlib.import_module(f".{model_type}", package=__package__)
	except ImportError:
		msg = f"Model type {model_type} not supported."
		logging.error(msg)
		raise ValueError(msg)

	return arch, model_type


def get_model_path(path_or_hf_repo):
	model_path = Path(path_or_hf_repo)
	if not model_path.exists():
		raise FileNotFoundError(f"Local model path not found: {path_or_hf_repo}")
	return model_path


def load_model(model_path, lazy=False, **kwargs):
	model_path = Path(model_path)
	config = load_config(model_path, **kwargs)
	quantization = config.get("quantization", None)

	weight_files = [wf for wf in glob.glob(str(model_path / "*.safetensors")) if not wf.endswith("consolidated.safetensors")]

	if not weight_files:
		raise FileNotFoundError(f"No safetensors found in local path {model_path}")

	weights = {}
	for wf in weight_files:
		weights.update(mx.load(wf))

	import safetensors
	with safetensors.safe_open(weight_files[0], framework="np") as f:
		is_mlx_format = f.metadata() and f.metadata().get("format") == "mlx"

	model_class, _ = get_model_and_args(config=config)

	config.setdefault("text_config", {})
	config.setdefault("vision_config", {})
	config.setdefault("audio_config", {})

	model_config = model_class.ModelConfig.from_dict(config)
	model_config = update_module_configs(model_config, model_class, config, ["text", "vision", "audio"])

	model = model_class.Model(model_config)

	if not is_mlx_format:
		weights = sanitize_weights(model, weights)
		weights = sanitize_weights(model_class.VisionModel, weights, model_config.vision_config)
		weights = sanitize_weights(model_class.LanguageModel, weights, model_config.text_config)
		if hasattr(model_class, "AudioModel"):
			weights = sanitize_weights(model_class.AudioModel, weights, model_config.audio_config)
		elif hasattr(model_class, "AudioEncoder"):
			weights = sanitize_weights(model_class.AudioEncoder, weights, model_config.audio_config)

	if (quantization := config.get("quantization", None)) is not None:
		skip_vision = config.get("vision_config", {}).get("skip_vision", False)

		def get_class_predicate(p, m):
			if skip_multimodal_module(p) and skip_vision:
				return False
			if p in config["quantization"]:
				return config["quantization"][p]
			if not hasattr(m, "to_quantized"):
				return False
			if hasattr(m, "weight") and m.weight.size % 64 != 0:
				return False
			return f"{p}.scales" in weights

		nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"], mode=quantization.get("mode", "affine"), class_predicate=get_class_predicate)

	model.load_weights(list(weights.items()))
	if not lazy:
		mx.eval(model.parameters())

	model.eval()
	return model


def sanitize_weights(model_obj, weights, config=None):
	if hasattr(model_obj, "sanitize"):
		if config is not None:
			model_obj = model_obj(config)
		weights = model_obj.sanitize(weights)
	return weights


def update_module_configs(model_config, model_class, config, modules):
	for config_name in modules:
		config_attr = f"{config_name}_config"
		if hasattr(model_config, config_attr):
			config_class = getattr(model_class, f"{config_name.title()}Config")
			setattr(model_config, config_attr, config_class.from_dict(config[config_attr]))
	return model_config


def load(path_or_hf_repo, lazy=False, **kwargs):
	model_path = get_model_path(path_or_hf_repo)
	model = load_model(model_path, lazy, **kwargs)
	eos_token_id = getattr(model.config, "eos_token_id", None)
	processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
	return model, processor


def load_config(model_path, **kwargs):
	if isinstance(model_path, str):
		model_path = get_model_path(model_path)

	try:
		with open(model_path / "config.json", encoding="utf-8") as f:
			config = json.load(f)

		generation_config_file = model_path / "generation_config.json"
		if generation_config_file.exists():
			generation_config = {}
			try:
				with open(generation_config_file, "r") as f:
					generation_config = json.load(f)
			except json.JSONDecodeError:
				pass

			if eos_token_id := generation_config.get("eos_token_id", False):
				config["eos_token_id"] = eos_token_id

		return config
	except FileNotFoundError as exc:
		raise FileNotFoundError(f"Config not found at {model_path}") from exc


def load_processor(model_path, add_detokenizer=True, eos_token_ids=None, **kwargs):
	processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
	if getattr(processor, "feature_extractor", None) is None:
		pre = Path(model_path) / "preprocessor_config.json"
		if pre.exists():
			from transformers import WhisperFeatureExtractor
			processor.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
	if add_detokenizer:
		detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
		tokenizer_obj = (processor.tokenizer if hasattr(processor, "tokenizer") else processor)
		processor.detokenizer = detokenizer_class(tokenizer_obj)
		final_eos_token_ids = (eos_token_ids if eos_token_ids is not None else tokenizer_obj.eos_token_ids)
		criteria = StoppingCriteria(final_eos_token_ids, tokenizer_obj)

		if hasattr(processor, "tokenizer"):
			processor.tokenizer.stopping_criteria = criteria
		else:
			processor.stopping_criteria = criteria

	return processor


def make_shards(weights, max_file_size_gb=MAX_FILE_SIZE_GB):
	max_file_size_bytes = max_file_size_gb << 30
	shards = []
	shard, shard_size = {}, 0
	for k, v in weights.items():
		if shard_size + v.nbytes > max_file_size_bytes:
			shards.append(shard)
			shard, shard_size = {}, 0
		shard[k] = v
		shard_size += v.nbytes
	shards.append(shard)
	return shards


def apply_repetition_penalty(logits, generated_tokens, penalty):
	if len(generated_tokens) > 0:
		indices = mx.array([token for token in generated_tokens])
		selected_logits = logits[:, indices]
		selected_logits = mx.where(selected_logits < 0, selected_logits * penalty, selected_logits / penalty)
		logits[:, indices] = selected_logits
	return logits


def save_weights(save_path, model, donate_weights=False):
	if isinstance(save_path, str):
		save_path = Path(save_path)

	weights = dict(tree_flatten(model.parameters()))
	del model

	save_path.mkdir(parents=True, exist_ok=True)

	shards = make_shards(weights)
	shards_count = len(shards)
	shard_file_format = ("model-{:05d}-of-{:05d}.safetensors" if shards_count > 1 else "model.safetensors")

	total_size = sum(v.nbytes for v in weights.values())
	index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

	if donate_weights:
		weights.clear()
		del weights

	for i in range(len(shards)):
		shard = shards[i]
		shards[i] = None
		shard_name = shard_file_format.format(i + 1, shards_count)
		shard_path = save_path / shard_name

		mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

		for weight_name in shard.keys():
			index_data["weight_map"][weight_name] = shard_name
		del shard

	index_data["weight_map"] = {k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])}

	with open(save_path / "model.safetensors.index.json", "w") as f:
		json.dump(index_data, f, indent=4)


def save_config(config, config_path):
	config.pop("_name_or_path", None)
	config.pop("torch_dtype", None)
	config = dict(sorted(config.items()))
	with open(config_path, "w") as fid:
		json.dump(config, fid, indent=4)


def load_image(image_source, timeout=10):
	if (isinstance(image_source, BytesIO) or (isinstance(image_source, str) and image_source.startswith("data:image/")) or Path(image_source).is_file()):
		try:
			if image_source.startswith("data:image/"):
				import base64
				if "," not in image_source:
					raise ValueError("Invalid data URI format - missing comma separator")
				_, data = image_source.split(",", 1)
				image_source = BytesIO(base64.b64decode(data))
			image = Image.open(image_source)
		except IOError as e:
			raise ValueError(f"Failed to load image from {image_source} with error: {e}") from e
	elif image_source.startswith(("http://", "https://")):
		try:
			response = requests.get(image_source, stream=True, timeout=timeout)
			response.raise_for_status()
			image = Image.open(response.raw)
		except Exception as e:
			raise ValueError(f"Failed to load image from URL: {image_source} with error {e}") from e
	else:
		raise ValueError(f"The image {image_source} must be a valid URL or existing file.")

	image = ImageOps.exif_transpose(image)
	image = image.convert("RGB")
	return image


def resize_image(img, max_size):
	ratio = min(max_size[0] / img.width, max_size[1] / img.height)
	if ratio >= 1.0:
		return img  # No need to upscale
	new_size = (int(img.width * ratio), int(img.height * ratio))
	return img.resize(new_size, Image.Resampling.LANCZOS)


def process_image(img, resize_shape, image_processor):
	if isinstance(img, str):
		img = load_image(img)

	if resize_shape is None:  # hard limit so MLX doesn't VRAM-explode on huge files
		resize_shape = (448, 448)

	img = resize_image(img, resize_shape)
	return img


def resample_audio(audio, orig_sr, target_sr):
	return _resample_audio(audio, orig_sr, target_sr)


def load_audio(file, sr, timeout=10):
	"""Path or URL → mono float32 @ sr via ffmpeg (+ soxr when needed)."""
	if isinstance(file, str) and file.startswith(("http://", "https://")):
		response = requests.get(file, stream=True, timeout=timeout)
		response.raise_for_status()
		file = response.content
	return load_audio_np(file, sample_rate=sr, mono=True)


def expand_marker(text, marker, counts):
	"""Expand each marker run in order: counts[i] pads for site i."""
	pattern = re.escape(marker) + "+"
	sites = list(re.finditer(pattern, text))
	if len(sites) != len(counts):
		raise ValueError(f"prompt has {len(sites)} {marker!r} site(s), but got {len(counts)} count(s)")
	out = text
	for site, n in zip(reversed(sites), reversed(counts)):
		out = out[:site.start()] + (marker * int(n)) + out[site.end():]
	return out


def prepare_avlm_inputs(processor, prompts, images=None, audio=None, resize_shape=None, add_special_tokens=False, sampling_rate=16000):
	"""Build AVLM model inputs: expand <|audio_pad|>, Whisper mel, VL image pads."""
	text = prompts if isinstance(prompts, str) else prompts[0]
	out = {}
	if isinstance(images, list) and not images:
		images = None
	if isinstance(audio, list) and not audio:
		audio = None

	if audio is not None:
		if not isinstance(audio, list):
			audio = [audio]
		fe = getattr(processor, "feature_extractor", None)
		if fe is None:
			raise ValueError("AVLM processor missing feature_extractor (Whisper FE); re-merge or set preprocessor_config.json")
		sr = getattr(fe, "sampling_rate", sampling_rate) or sampling_rate
		wavs = []
		for a in audio:
			if isinstance(a, (str, bytes, bytearray)) or hasattr(a, "read"):
				wavs.append(load_audio(a, sr=sr))
			else:
				wavs.append(np.asarray(a, dtype=np.float32))
		feats = fe(wavs, sampling_rate=sr, return_attention_mask=True, truncation=False, padding=True, return_tensors="np")
		input_features = mx.array(feats["input_features"])
		feature_attention_mask = mx.array(feats["attention_mask"])
		feat_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
		pad_counts = [int(x) for x in np.array(get_feat_extract_output_lengths(feat_lens))]
		text = expand_marker(text, "<|audio_pad|>", pad_counts)
		out["input_features"] = input_features
		out["feature_attention_mask"] = feature_attention_mask
		out["audio_pad_counts"] = pad_counts

	if images is not None:
		if not isinstance(images, list):
			images = [images]
		image_processor = getattr(processor, "image_processor", None)
		pil = [process_image(img, resize_shape, image_processor) for img in images]
		n_sites = len(re.findall(re.escape("<|image_pad|>") + "+", text))
		if n_sites != len(pil):
			raise ValueError(f"prompt has {n_sites} <|image_pad|> site(s), but got {len(pil)} image(s)")
		proc_out = processor(text=[text], images=pil, return_tensors="np", add_special_tokens=add_special_tokens)
		out["input_ids"] = mx.array(proc_out["input_ids"])
		if "attention_mask" in proc_out:
			out["attention_mask"] = mx.array(proc_out["attention_mask"])
		out["pixel_values"] = mx.array(proc_out["pixel_values"])
		if "image_grid_thw" in proc_out:
			out["image_grid_thw"] = mx.array(proc_out["image_grid_thw"])
	else:
		tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
		out["input_ids"] = mx.array([tok.encode(text, add_special_tokens=add_special_tokens)])

	return out


def process_inputs(processor, prompts, images=None, audio=None, add_special_tokens=False, padding=True, padding_side="left", return_tensors="mlx", **kwargs):
	process_method = getattr(processor, "process", processor)
	parameters = inspect.signature(process_method).parameters

	args = {"text": prompts, "images": images, "padding": padding, "return_tensors": return_tensors}

	if "padding_side" in parameters:
		args["padding_side"] = padding_side

	if "add_special_tokens" in parameters:
		args["add_special_tokens"] = add_special_tokens

	for param in parameters.keys():
		if param in kwargs.keys():
			args[param] = kwargs.get(param, None)
			break

	if audio is not None:
		if "audio" in parameters:
			args["audio"] = audio
		else:
			raise ValueError(f"Processor {processor} does not support audio parameter")

	return process_method(**args)


def process_inputs_with_fallback(processor, prompts, images, audio, add_special_tokens=False, return_tensors="mlx", **kwargs):
	try:
		return process_inputs(processor, prompts=prompts, images=images, audio=audio, add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)
	except Exception as e:
		if return_tensors != "pt":
			try:
				return process_inputs(processor, prompts=prompts, images=images, audio=audio, add_special_tokens=add_special_tokens, return_tensors="pt", **kwargs)
			except Exception as fallback_error:
				raise ValueError(f"Failed to process inputs with error: {fallback_error}") from fallback_error
		raise ValueError(f"Failed to process inputs with error: {e}")


def prepare_inputs(processor, images=None, audio=None, prompts=None, image_token_index=None, resize_shape=None, add_special_tokens=False, padding=True, padding_side="left", pad_to_uniform_size=False, **kwargs):
	if not images and not audio:
		tokenizer = (processor.tokenizer if hasattr(processor, "tokenizer") else processor)
		if padding and tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token
		inputs = tokenizer(prompts, add_special_tokens=add_special_tokens, padding=padding, padding_side=padding_side)
		input_ids = mx.array([inputs.input_ids])
		mask = mx.array([inputs.attention_mask])
		return {"input_ids": input_ids, "attention_mask": mask}

	if images is not None:
		if not isinstance(images, list):
			images = [images]

		image_processor = (processor.image_processor if hasattr(processor, "image_processor") else None)
		images = [process_image(img, resize_shape, image_processor) for img in images]

		if len(images) > 1 and pad_to_uniform_size:
			target_size = None
			if image_processor is not None and hasattr(image_processor, "size"):
				size = image_processor.size
				if isinstance(size, tuple):
					target_size = size
				elif isinstance(size, dict):
					target_size = (size.get("height", 384), size.get("width", 384))
				elif isinstance(size, int):
					target_size = (size, size)

			if target_size is not None:
				resized_images = []
				for img in images:
					if img.size != (target_size[1], target_size[0]):
						img = img.resize((target_size[1], target_size[0]), Image.Resampling.BICUBIC)
					resized_images.append(img)
				images = resized_images
			else:
				max_width = max(img.width for img in images)
				max_height = max(img.height for img in images)

				padded_images = []
				for img in images:
					if img.width != max_width or img.height != max_height:
						padded_img = Image.new("RGB", (max_width, max_height), (255, 255, 255))
						x_offset = (max_width - img.width) // 2
						y_offset = (max_height - img.height) // 2
						padded_img.paste(img, (x_offset, y_offset))
						padded_images.append(padded_img)
					else:
						padded_images.append(img)
				images = padded_images

	if audio is not None:
		if not isinstance(audio, list):
			audio = [audio]

		if len(audio) > 1:
			print("\033[33mWarning\033[0m: Single prompt with multiple audio files is not supported yet. Using the first audio file.\n")
			audio = audio[:1]

		fe = getattr(processor, "feature_extractor", None)
		sr = getattr(fe, "sampling_rate", 16000) if fe is not None else 16000
		audio = [load_audio(audio_file, sr=sr) for audio_file in audio]

	model_inputs = {}

	if hasattr(processor, "image_processor") and isinstance(processor.image_processor, BaseImageProcessor):
		if not isinstance(prompts, list):
			prompts = [prompts]

		if processor.pad_token is None:
			processor.pad_token = processor.eos_token
		text_chunks = [[processor(chunk).input_ids for chunk in prompt.split("<image>")] for prompt in prompts]

		max_length = max(sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks)

		input_ids = []
		for chunks in text_chunks:
			ids = chunks[0] + [image_token_index] + chunks[1]
			padding_len = [processor.pad_token_id] * (max_length - len(ids))
			input_ids.append(mx.array(ids + padding_len))

		model_inputs["input_ids"] = mx.array(input_ids)
		pixel_values = processor.image_processor.preprocess(images=images)
		model_inputs["pixel_values"] = mx.array(np.stack(pixel_values))
		model_inputs["attention_mask"] = mx.array([(ids != processor.pad_token_id) for ids in input_ids]).astype(mx.int32)

	else:
		if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
			processor.tokenizer.pad_token = processor.tokenizer.eos_token

		inputs = process_inputs_with_fallback(processor, images=images, audio=audio, prompts=prompts, add_special_tokens=add_special_tokens, **kwargs)

		if "images" in inputs:
			inputs["pixel_values"] = inputs["images"]
			inputs.pop("images")

		model_inputs["attention_mask"] = (mx.array(inputs["attention_mask"]) if "attention_mask" in inputs else None)

		for key, value in inputs.items():
			if key not in model_inputs:
				if isinstance(value, (str, list, mx.array)):
					model_inputs[key] = value
				else:
					model_inputs[key] = mx.array(value)

	return model_inputs


class StoppingCriteria:
	def __init__(self, eos_token_ids, tokenizer=None):
		if isinstance(eos_token_ids, int):
			self.eos_token_ids = [eos_token_ids]
		else:
			self.eos_token_ids = eos_token_ids

		self.tokenizer = tokenizer

	def add_eos_token_ids(self, new_eos_token_ids=None):
		if new_eos_token_ids is None:
			return

		if self.tokenizer is None:
			raise ValueError("Processor is not provided")

		if new_eos_token_ids is not None:
			if isinstance(new_eos_token_ids, str):
				new_eos_token_ids = [new_eos_token_ids]
			new_eos_token_ids = [self.tokenizer.encode(" " + token, add_special_tokens=False)[-1] for token in new_eos_token_ids]
			self.eos_token_ids.extend(new_eos_token_ids)

	def reset(self, eos_token_ids=None):
		eos_token_ids = (eos_token_ids if eos_token_ids is not None else self.tokenizer.eos_token_ids)

		if isinstance(eos_token_ids, int):
			eos_token_ids = [eos_token_ids]

		if self.eos_token_ids != eos_token_ids:
			self.eos_token_ids = eos_token_ids

	def __call__(self, input_ids):
		return input_ids in self.eos_token_ids
