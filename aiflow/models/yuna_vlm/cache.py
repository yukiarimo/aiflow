import json
import os
import mlx.core as mx
from mlx_lm.models.cache import ChunkedKVCache, KVCache, RotatingKVCache, _BaseCache


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
		if isinstance(layer_cache, (KVCache, SimpleKVCache, StaticKVCache, RotatingKVCache)):
			if layer_cache.keys is not None and layer_cache.values is not None:
				tensors[f"layer_{i}_keys"] = layer_cache.keys
				tensors[f"layer_{i}_values"] = layer_cache.values
				metadata["offsets"].append(layer_cache.offset)
			else:
				metadata["offsets"].append(0)

	base, _ = os.path.splitext(path)
	if not path.endswith(".safetensors"):
		path = path + ".safetensors"

	mx.save_safetensors(path, tensors)

	with open(base + ".json", "w") as f:
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

			if isinstance(layer_cache, (KVCache, SimpleKVCache, StaticKVCache, RotatingKVCache)):
				layer_cache.keys = keys
				layer_cache.values = values
				layer_cache.offset = offset

	return cache_obj, metadata["input_ids"]


def trim_cache(cache, trim_len):
	for layer_cache in cache:
		if isinstance(layer_cache, (KVCache, SimpleKVCache, StaticKVCache)):
			if layer_cache.keys is not None and layer_cache.keys.shape[2] > trim_len:
				layer_cache.keys = layer_cache.keys[:, :, :trim_len, :]
				layer_cache.values = layer_cache.values[:, :, :trim_len, :]
				layer_cache.offset = trim_len
		elif isinstance(layer_cache, RotatingKVCache):
			if layer_cache.offset > trim_len:
				layer_cache.offset = trim_len


class SimpleKVCache:
	def __init__(self):
		self.keys = None
		self.values = None
		self.offset = 0

	def update_and_fetch(self, keys, values):
		if self.keys is None:
			self.keys = keys
			self.values = values
		else:
			self.keys = mx.concatenate([self.keys, keys], axis=2)
			self.values = mx.concatenate([self.values, values], axis=2)

		self.offset += keys.shape[2]
		return self.keys, self.values

	def fetch(self):
		return self.keys, self.values

	def update(self, keys, values):
		self.keys = keys
		self.values = values
		self.offset += keys.shape[2]


class SlidingWindowCache(_BaseCache):
	def __init__(self, max_size, step=256):
		self.max_size = max_size
		self.step = step
		self.keys = None
		self.values = None
		self.offset = 0

	def update_and_fetch(self, keys, values):
		B, n_kv_heads, seq_len, k_head_dim = keys.shape
		v_head_dim = values.shape[-1]

		if self.keys is None:
			k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
			v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
			self.keys = mx.zeros(k_shape, dtype=keys.dtype)
			self.values = mx.zeros(v_shape, dtype=values.dtype)

		if self.offset + seq_len <= self.max_size:
			start_idx = self.offset
			end_idx = self.offset + seq_len
			self.keys[:, :, start_idx:end_idx, :] = keys
			self.values[:, :, start_idx:end_idx, :] = values
			self.offset += seq_len
		else:
			if seq_len < self.max_size:
				shift_amount = min(seq_len, self.max_size - 1)
				self.keys[:, :, :-shift_amount, :] = self.keys[:, :, shift_amount:, :]
				self.values[:, :, :-shift_amount, :] = self.values[:, :, shift_amount:, :]
				self.keys[:, :, -shift_amount:, :] = keys[:, :, -shift_amount:, :]
				self.values[:, :, -shift_amount:, :] = values[:, :, -shift_amount:, :]
			else:
				self.keys = keys[:, :, -self.max_size:, :]
				self.values = values[:, :, -self.max_size:, :]
			self.offset = self.max_size

		return self.keys, self.values

	@property
	def state(self):
		if self.keys is None:
			return None, None
		return self.keys, self.values

	@state.setter
	def state(self, v):
		if v is not None and len(v) == 2:
			self.keys, self.values = v
			if self.keys is not None:
				self.offset = self.max_size

	def get_max_cache_shape(self):
		return self.max_size

	@property
	def meta_state(self):
		return tuple(map(str, (self.max_size, self.step, self.offset)))

	@meta_state.setter
	def meta_state(self, v):
		self.max_size, self.step, self.offset = map(int, v)

	def is_trimmable(self):
		return False

	def trim(self, n):
		return 0


class StaticKVCache(_BaseCache):
	def __init__(self, max_size, step=256):
		self.max_size = max_size
		self.step = step
		self.keys = None
		self.values = None
		self.offset = 0

	def update_and_fetch(self, keys, values):
		B, n_kv_heads, seq_len, k_head_dim = keys.shape
		v_head_dim = values.shape[-1]

		if self.keys is None:
			k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
			v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
			self.keys = mx.zeros(k_shape, dtype=keys.dtype)
			self.values = mx.zeros(v_shape, dtype=values.dtype)

		end_pos = min(self.offset + seq_len, self.max_size)
		actual_seq_len = end_pos - self.offset

		if actual_seq_len > 0:
			self.keys[:, :, self.offset:end_pos, :] = keys[:, :, :actual_seq_len, :]
			self.values[:, :, self.offset:end_pos, :] = values[:, :, :actual_seq_len, :]
			self.offset = end_pos

		return self.keys, self.values

	@property
	def state(self):
		if self.keys is None:
			return None, None
		return self.keys, self.values

	@state.setter
	def state(self, v):
		if v is not None and len(v) == 2:
			self.keys, self.values = v
			if self.keys is not None:
				self.offset = self.max_size

	@property
	def meta_state(self):
		return tuple(map(str, (self.max_size, self.step, self.offset)))

	@meta_state.setter
	def meta_state(self, v):
		self.max_size, self.step, self.offset = map(int, v)

	def is_trimmable(self):
		return True

	def trim(self, n):
		n = min(self.offset, n)
		self.offset -= n
		return n
