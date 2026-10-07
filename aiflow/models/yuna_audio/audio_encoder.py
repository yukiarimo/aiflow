import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _floor_div(a, b):
	return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def get_feat_extract_output_lengths(input_lengths):
	"""ASR conv stack length formula (3x stride-2 over time, in 100-frame blocks)."""
	input_lengths_leave = input_lengths % 100
	feat_lengths = _floor_div(input_lengths_leave - 1, 2) + 1
	output_lengths = (_floor_div(_floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 + (input_lengths // 100) * 13)
	return output_lengths


_get_feat_extract_output_lengths = get_feat_extract_output_lengths  # Back-compat alias for ASR call sites


class SinusoidalPositionEmbedding(nn.Module):
	def __init__(self, length, channels, max_timescale=10000.0):
		super().__init__()
		if channels % 2 != 0:
			raise ValueError("SinusoidalPositionEmbedding needs even channels input")
		log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
		inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32))
		positions = mx.arange(length, dtype=mx.float32)[:, None]
		scaled_time = positions * inv_timescales[None, :]
		self._positional_embedding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)

	def __call__(self, seqlen):
		return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.embed_dim = config.d_model
		self.num_heads = config.encoder_attention_heads
		self.head_dim = self.embed_dim // self.num_heads
		self.scaling = self.head_dim**-0.5
		if (self.head_dim * self.num_heads) != self.embed_dim:
			raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")
		self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
		self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
		self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
		self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

	def __call__(self, hidden_states, mask=None):
		bsz, seq_len, _ = hidden_states.shape
		query_states = self.q_proj(hidden_states) * self.scaling
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)
		query_states = query_states.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
		key_states = key_states.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
		value_states = value_states.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
		attn_output = mx.fast.scaled_dot_product_attention(query_states, key_states, value_states, scale=1.0, mask=mask)
		attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.embed_dim)
		return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.embed_dim = config.d_model
		self.self_attn = AudioAttention(config)
		self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
		self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
		self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
		self.final_layer_norm = nn.LayerNorm(self.embed_dim)

	def __call__(self, hidden_states, mask=None):
		residual = hidden_states
		hidden_states = self.self_attn_layer_norm(hidden_states)
		hidden_states = self.self_attn(hidden_states, mask=mask)
		hidden_states = residual + hidden_states
		residual = hidden_states
		hidden_states = self.final_layer_norm(hidden_states)
		hidden_states = nn.gelu(self.fc1(hidden_states))
		hidden_states = self.fc2(hidden_states)
		hidden_states = residual + hidden_states
		return hidden_states


class AudioEncoder(nn.Module):
	"""Qwen3-ASR audio encoder + multimodal projector (proj1/proj2). Shared by ASR and AVLM."""
	def __init__(self, config):
		super().__init__()
		self.config = config
		embed_dim = config.d_model
		self.num_mel_bins = config.num_mel_bins
		self.max_source_positions = config.max_source_positions
		self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
		self.n_window = config.n_window
		self.n_window_infer = config.n_window_infer
		self.conv_chunksize = config.conv_chunksize
		self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)
		self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)
		self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1)
		freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
		self.conv_out = nn.Linear(config.downsample_hidden_size * freq_after_conv, embed_dim, bias=False)
		self.positional_embedding = SinusoidalPositionEmbedding(self.max_source_positions, embed_dim)
		self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
		self.ln_post = nn.LayerNorm(embed_dim)
		self.proj1 = nn.Linear(embed_dim, embed_dim)
		self.proj2 = nn.Linear(embed_dim, config.output_dim)

	def _create_block_attention_mask(self, seq_len, cu_seqlens, dtype):
		mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
		for i in range(len(cu_seqlens) - 1):
			start = cu_seqlens[i]
			end = cu_seqlens[i + 1]
			mask[start:end, start:end] = 0.0
		return mask

	def __call__(self, input_features, feature_attention_mask=None):
		if feature_attention_mask is not None:
			feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
		else:
			feature_lens = mx.array([input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32)

		feature_lens_np = np.array(feature_lens)
		aftercnn_lens = get_feat_extract_output_lengths(feature_lens)
		chunk_size = self.n_window * 2
		chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)
		chunk_lengths = []

		for i in range(len(feature_lens_np)):
			num_chunks = int(chunk_num[i])
			feat_len = int(feature_lens_np[i])
			for j in range(num_chunks):
				if j == num_chunks - 1:
					remainder = feat_len % chunk_size
					chunk_lengths.append(chunk_size if remainder == 0 else remainder)
				else:
					chunk_lengths.append(chunk_size)

		chunk_lengths = np.array(chunk_lengths, dtype=np.int32)
		chunks = []

		for i in range(len(feature_lens_np)):
			feat = input_features[i]
			feat_len = int(feature_lens_np[i])
			num_chunks = int(chunk_num[i])
			pos = 0
			for j in range(num_chunks):
				if j == num_chunks - 1:
					remainder = feat_len % chunk_size
					clen = chunk_size if remainder == 0 else remainder
				else:
					clen = chunk_size
				chunk = feat[:, pos:pos + clen]
				chunks.append(chunk)
				pos += clen

		max_chunk_len = int(max(chunk_lengths))
		padded_chunks = []
		for i, chunk in enumerate(chunks):
			clen = int(chunk_lengths[i])
			if clen < max_chunk_len:
				chunk = mx.pad(chunk, [(0, 0), (0, max_chunk_len - clen)])
			padded_chunks.append(chunk)

		padded_feature = mx.stack(padded_chunks, axis=0)
		feature_lens_after_cnn = get_feat_extract_output_lengths(mx.array(chunk_lengths))
		feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
		max_len_after_cnn = int(feature_lens_after_cnn_np.max())

		x = padded_feature[:, :, :, None]
		x = nn.gelu(self.conv2d1(x))
		x = nn.gelu(self.conv2d2(x))
		x = nn.gelu(self.conv2d3(x))
		b, f, t, c = x.shape
		x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
		x = self.conv_out(x)
		pos_emb = self.positional_embedding(x.shape[1])
		x = x + pos_emb[None, :, :]

		hidden_list = []
		for i in range(x.shape[0]):
			valid_len = int(feature_lens_after_cnn_np[i])
			hidden_list.append(x[i, :valid_len])
		hidden_states = mx.concatenate(hidden_list, axis=0)

		aftercnn_lens_np = np.array(aftercnn_lens)
		window_aftercnn = max_len_after_cnn * (self.n_window_infer // (self.n_window * 2))
		cu_chunk_lens = [0]
		for cnn_len in aftercnn_lens_np:
			cnn_len = int(cnn_len)
			num_full_windows = cnn_len // window_aftercnn
			for _ in range(num_full_windows):
				cu_chunk_lens.append(window_aftercnn)
			remainder = cnn_len % window_aftercnn
			if remainder != 0:
				cu_chunk_lens.append(remainder)

		cu_seqlens = np.cumsum(cu_chunk_lens).tolist()
		seq_len = hidden_states.shape[0]
		attention_mask = self._create_block_attention_mask(seq_len, cu_seqlens, hidden_states.dtype)
		attention_mask = attention_mask[None, None, :, :]
		hidden_states = hidden_states[None, :, :]

		for layer in self.layers:
			hidden_states = layer(hidden_states, mask=attention_mask)

		hidden_states = hidden_states[0]
		hidden_states = self.ln_post(hidden_states)
		hidden_states = nn.gelu(self.proj1(hidden_states))
		hidden_states = self.proj2(hidden_states)
		return hidden_states

	@staticmethod
	def sanitize(weights):
		"""PyTorch Conv2d (O,I,K,K) → MLX (O,K,K,I) when still in HF layout."""
		sanitized = {}
		for k, v in weights.items():
			if "conv2d" in k and k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 4:
				if v.shape[2] == v.shape[3] and v.shape[1] != v.shape[2]:
					v = v.transpose(0, 2, 3, 1)
			sanitized[k] = v
		return sanitized


AudioModel = AudioEncoder  # AVLM / loaders alias
