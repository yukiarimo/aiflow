import json
import os
from . import _hf5_mlx  # noqa: F401
import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache


def make_prompt_cache(model, max_kv_size=None):
	if hasattr(model, "make_cache"):
		return model.make_cache()

	num_layers = len(model.layers)

	if max_kv_size is not None:
		return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]
	else:
		return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(cache, path, input_ids):
	tensors = {}
	metadata = {"input_ids": input_ids, "layers": len(cache), "offsets": []}

	for i, layer_cache in enumerate(cache):
		if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):  # dynamic class types
			if layer_cache.keys is not None and layer_cache.values is not None:
				tensors[f"layer_{i}_keys"] = layer_cache.keys
				tensors[f"layer_{i}_values"] = layer_cache.values
				metadata["offsets"].append(layer_cache.offset)
			else:
				metadata["offsets"].append(0)
		else:
			metadata["offsets"].append(0)

	base, _ = os.path.splitext(path)
	if not path.endswith(".safetensors"):
		path = path + ".safetensors"

	safe_metadata = {str(k): json.dumps(v) for k, v in metadata.items()}  # SafeTensors metadata values must be strings
	try:
		mx.save_safetensors(path, tensors, metadata=safe_metadata)
	except TypeError:
		mx.save_safetensors(path, tensors)  # older MLX bindings without metadata kwargs
	with open(base + ".json", "w") as f:  # companion json for easy python loading
		json.dump(metadata, f)


def load_prompt_cache(path, model):
	base, _ = os.path.splitext(path)
	if not path.endswith(".safetensors"):
		path = path + ".safetensors"
	meta_path = base + ".json"

	if not os.path.exists(path) or not os.path.exists(meta_path):
		return None, []

	try:
		tensors = mx.load(path)
		with open(meta_path, "r") as f:
			metadata = json.load(f)
	except Exception as e:
		print(f"[WARNING] Failed to load cache: {e}")
		return None, []

	cache_obj = make_prompt_cache(model)

	if len(cache_obj) != metadata["layers"]:
		print(f"[WARNING] Cache layer count mismatch. Expected {len(cache_obj)}, got {metadata['layers']}")
		return None, []

	for i, layer_cache in enumerate(cache_obj):
		k_key = f"layer_{i}_keys"
		v_key = f"layer_{i}_values"

		if k_key in tensors and v_key in tensors:
			keys = tensors[k_key]
			values = tensors[v_key]
			offset = metadata["offsets"][i]

			if isinstance(layer_cache, (KVCache, RotatingKVCache)):
				layer_cache.keys = keys
				layer_cache.values = values
				layer_cache.offset = offset

	return cache_obj, metadata["input_ids"]


def trim_cache(cache, trim_len):
	for layer_cache in cache:
		if isinstance(layer_cache, KVCache):
			if layer_cache.keys is not None and layer_cache.keys.shape[2] > trim_len:
				layer_cache.keys = layer_cache.keys[:, :, :trim_len, :]
				layer_cache.values = layer_cache.values[:, :, :trim_len, :]
				layer_cache.offset = trim_len
		elif isinstance(layer_cache, RotatingKVCache):
			if layer_cache.offset > trim_len:
				layer_cache.offset = trim_len
