import math
import queue
import threading
import os
import re
import subprocess
import contextlib
from contextlib import nullcontext
from pathlib import Path
import time
import concurrent.futures
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from . import utils
from .. import monotonic_align
from aiflow.models.yuna_speech.text import symbols, split_and_process_text, cleaned_text_to_sequence, _symbol_to_id
import numpy as np
from tqdm import tqdm
import soundfile as sf
from .utils import mel_spectrogram_torch, piecewise_rational_quadratic_transform, HParams
import gc
from aiflow.models.yuna_speech.vocos.models import VocosBackbone, ISTFT

_ab_tls = threading.local()
_ab_pretrained_iter = None
_ab_pretrained_lock = threading.Lock()
EXPORT_MAX_MEL_FRAMES = 16384
COREML_DYNAMIC_MAX_PHONEMES = 1000
COREML_BUCKET_WIDTHS = [128, 256, 384, 512]


def _deterministic_noise(shape, device, dtype, seed=8):
	"""Export-safe standard-normal-ish noise from index hash (seed mixes into phases)."""
	b, c, t = _coreml_trace_dim(shape[0]), _coreml_trace_dim(shape[1]), _coreml_trace_dim(shape[2])
	ti = torch.arange(t, device=device, dtype=dtype).view(1, 1, -1)
	ci = torch.arange(c, device=device, dtype=dtype).view(1, -1, 1)
	bi = torch.arange(b, device=device, dtype=dtype).view(-1, 1, 1)
	seed_f = float(seed)
	return (torch.sin((bi * 127.1 + ci * 311.7 + ti * 74.7 + seed_f) * 0.017) * torch.cos((bi * 269.5 + ci * 183.3 + ti * 419.2 + seed_f) * 0.013))


def _is_exporting():
	return torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling()


def _coreml_trace_dim(v):
	"""Core ML jit.trace: bake shape dims as Python ints (coremltools chokes on aten::Int)."""
	if torch.jit.is_tracing() and not torch.compiler.is_compiling():
		if torch.is_tensor(v):
			return int(v.item()) if v.numel() == 1 else int(v)
		return int(v)
	return v


def _shape_check(cond):
	if not _is_exporting():
		torch._check(cond)


def _slice_noise(noise, length):
	if not _is_exporting():
		torch._check(length <= noise.shape[-1])
	return noise[..., :length]


def _resolve_noise_like(ref, noise_scale=1.0, noise=None, seed=8):
	if noise is not None:
		n = _slice_noise(noise, ref.shape[-1])
		return n.to(device=ref.device, dtype=ref.dtype) * noise_scale
	if _is_exporting():
		return _deterministic_noise(ref.shape, ref.device, ref.dtype, seed=seed) * noise_scale
	return torch.randn_like(ref) * noise_scale


def _resolve_noise(shape, device, dtype, noise_scale=1.0, noise=None, seed=8):
	if noise is not None:
		n = _slice_noise(noise, shape[-1])
		return n.to(device=device, dtype=dtype) * noise_scale
	if torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling():
		return _deterministic_noise(shape, device, dtype, seed=seed) * noise_scale
	return torch.randn(shape, device=device, dtype=dtype) * noise_scale


def _infer_noise_like(ref, noise_scale=1.0, noise=None):
	return _resolve_noise_like(ref, noise_scale=noise_scale, noise=noise)


def _infer_noise(shape, device, dtype, noise_scale=1.0, noise=None):
	return _resolve_noise(shape, device, dtype, noise_scale=noise_scale, noise=noise)


def get_padding(kernel_size, dilation=1):
	return (kernel_size * dilation - dilation) // 2


def slice_segments(x, ids_str, segment_size=4):
	ret = torch.zeros_like(x[:, :, :segment_size])
	for i in range(x.size(0)):
		idx_str = ids_str[i]
		idx_end = idx_str + segment_size
		if x.size(2) == 0:
			print(f"Warning: tensor has zero time dimension at batch {i}")
			continue
		if idx_end > x.size(2):
			available_size = x.size(2) - idx_str
			if available_size > 0: ret[i, :, :available_size] = x[i, :, idx_str:x.size(2)]
		else: ret[i] = x[i, :, idx_str:idx_end]
	return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
	b, d, t = x.size()
	if x_lengths is None: x_lengths = t
	ids_str_max = x_lengths - segment_size + 1
	ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
	return slice_segments(x, ids_str, segment_size), ids_str


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
	in_act = input_a + input_b
	t_act = torch.tanh(in_act[:, :n_channels, :])
	s_act = torch.sigmoid(in_act[:, n_channels:, :])
	return t_act * s_act


def sequence_mask(length, max_length=None):
	length = length.reshape(-1)
	if max_length is None:
		end = length[0] if length.numel() == 1 else length.amax()
	else:
		end = max_length
	end = _coreml_trace_dim(end)
	positions = torch.arange(end, dtype=length.dtype, device=length.device)
	return positions.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
	"""Duration: [b, 1, t_x], mask: [b, 1, t_y, t_x]"""
	b, _, t_y, t_x = mask.shape
	b, t_y, t_x = _coreml_trace_dim(b), _coreml_trace_dim(t_y), _coreml_trace_dim(t_x)
	cum_duration = torch.cumsum(duration, -1)
	cum_duration_flat = cum_duration.view(b * t_x)
	path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
	path = path.view(b, t_x, t_y)
	path = path - torch.cat([torch.zeros(b, 1, t_y, dtype=path.dtype, device=path.device), path[:, :-1, :]], dim=1)
	return path.unsqueeze(1).transpose(2, 3) * mask


class Encoder(nn.Module):
	def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4, **kwargs):
		super().__init__()
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.window_size = window_size
		self.drop = nn.Dropout(p_dropout)
		self.attn_layers = nn.ModuleList()
		self.norm_layers_1 = nn.ModuleList()
		self.ffn_layers = nn.ModuleList()
		self.norm_layers_2 = nn.ModuleList()
		self.gin_channels = kwargs.get("gin_channels", 0)
		self.cond_layer_idx = kwargs.get("cond_layer_idx", min(2, n_layers - 1))
		assert self.cond_layer_idx < self.n_layers, "cond_layer_idx should be less than n_layers"
		self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels) if self.gin_channels > 0 else None
		for i in range(self.n_layers):
			self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
			self.norm_layers_1.append(LayerNorm(hidden_channels))
			self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
			self.norm_layers_2.append(LayerNorm(hidden_channels))

	def forward(self, x, x_mask, g=None):
		attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
		x = x * x_mask
		for i in range(self.n_layers):
			if i == self.cond_layer_idx and g is not None and self.spk_emb_linear is not None:
				g = self.spk_emb_linear(g.transpose(1, 2))
				g = g.transpose(1, 2)
				x = x + g
				x = x * x_mask
			y = self.attn_layers[i](x, x, attn_mask)
			y = self.drop(y)
			x = self.norm_layers_1[i](x + y)
			y = self.ffn_layers[i](x, x_mask)
			y = self.drop(y)
			x = self.norm_layers_2[i](x + y)
		x = x * x_mask
		return x


class MultiHeadAttention(nn.Module):
	def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
		super().__init__()
		assert channels % n_heads == 0
		self.channels = channels
		self.out_channels = out_channels
		self.n_heads = n_heads
		self.p_dropout = p_dropout
		self.window_size = window_size
		self.heads_share = heads_share
		self.block_length = block_length
		self.proximal_bias = proximal_bias
		self.proximal_init = proximal_init
		self.attn = None
		self.k_channels = channels // n_heads
		self.conv_q = nn.Conv1d(channels, channels, 1)
		self.conv_k = nn.Conv1d(channels, channels, 1)
		self.conv_v = nn.Conv1d(channels, channels, 1)
		self.conv_o = nn.Conv1d(channels, out_channels, 1)
		self.drop = nn.Dropout(p_dropout)
		if window_size is not None:
			n_heads_rel = 1 if heads_share else n_heads
			rel_stddev = self.k_channels**-0.5
			self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
			self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
		nn.init.xavier_uniform_(self.conv_q.weight)
		nn.init.xavier_uniform_(self.conv_k.weight)
		nn.init.xavier_uniform_(self.conv_v.weight)
		if proximal_init:
			with torch.inference_mode():
				self.conv_k.weight.copy_(self.conv_q.weight)
				self.conv_k.bias.copy_(self.conv_q.bias)

	def forward(self, x, c, attn_mask=None):
		q = self.conv_q(x)
		k = self.conv_k(c)
		v = self.conv_v(c)
		x, self.attn = self.attention(q, k, v, mask=attn_mask)
		x = self.conv_o(x)
		return x

	def attention(self, query, key, value, mask=None):
		b, d, t_s, t_t = (*key.size(), query.size(2))
		query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
		key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
		value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

		if torch.jit.is_tracing() and not torch.compiler.is_compiling():  # Core ML jit.trace: relative-attention ops don't convert; Core AI torch.export keeps full path.
			bool_mask = mask.bool() if mask is not None else None
			output = F.scaled_dot_product_attention(query, key, value, attn_mask=bool_mask, dropout_p=0.0)
			return output.transpose(2, 3).contiguous().view(b, d, t_t), None

		if self.window_size is None and not self.proximal_bias and self.block_length is None and not torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling():
			bool_mask = mask.bool() if mask is not None else None
			output = F.scaled_dot_product_attention(query, key, value, attn_mask=bool_mask, dropout_p=self.p_dropout if self.training else 0.0)
			return output.transpose(2, 3).contiguous().view(b, d, t_t), None

		scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
		if self.window_size is not None:
			assert t_s == t_t, ("Relative attention is only available for self-attention.")
			key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
			rel_logits = torch.matmul(query / math.sqrt(self.k_channels), key_relative_embeddings.unsqueeze(0).transpose(-2, -1))  # x: [b, h, l, d], y: [h or 1, m, d], ret: [b, h, l, m]
			scores_local = self._relative_position_to_absolute_position(rel_logits)
			scores = scores + scores_local

		if self.proximal_bias:
			assert t_s == t_t, "Proximal bias is only available for self-attention."
			scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e4)
			if self.block_length is not None:
				assert t_s == t_t, ("Local attention is only available for self-attention.")
				block_mask = (torch.ones_like(scores).triu(-self.block_length).tril(self.block_length))
				scores = scores.masked_fill(block_mask == 0, -1e4)

		p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
		p_attn = self.drop(p_attn)
		output = torch.matmul(p_attn, value)

		if self.window_size is not None:
			relative_weights = self._absolute_position_to_relative_position(p_attn)
			value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
			output = output + torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0))  # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]

		output = (output.transpose(2, 3).contiguous().view(b, d, t_t))  # [b, n_h, t_t, d_k] -> [b, d, t_t]
		return output, p_attn

	def _get_relative_embeddings(self, relative_embeddings, length):
		"""SymInt-safe relative position table lookup (torch.export + Core ML compatible)."""
		ws1 = self.window_size + 1
		emb_len = relative_embeddings.shape[1]
		pos = torch.arange(2 * length - 1, device=relative_embeddings.device, dtype=torch.long) + (ws1 - length)
		return relative_embeddings[:, pos.clamp(0, emb_len - 1), :]

	def _relative_position_to_absolute_position(self, x):
		"""x: [b, h, l, 2*l-1], ret: [b, h, l, l]"""
		batch, heads, length, _ = x.shape
		i = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(1)
		j = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(0)
		idx = (length - 1) - i + j
		idx = idx.unsqueeze(0).unsqueeze(0).expand(batch, heads, length, length)
		x_final = torch.gather(x, -1, idx)
		return x_final

	def _absolute_position_to_relative_position(self, x):
		"""x: [b, h, l, l], ret: [b, h, l, 2*l-1]. SymInt-safe for torch.export."""
		batch, heads, length, _ = x.shape
		left_pad = max(length - 1, 0)
		if left_pad > 0:
			x_pad = torch.cat([torch.zeros(batch, heads, length, left_pad, device=x.device, dtype=x.dtype), x], dim=-1)
		else:
			x_pad = x
		x_pad = torch.cat([x_pad, torch.zeros(batch, heads, length, length, device=x.device, dtype=x.dtype)], dim=-1)
		i = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(1)
		j = torch.arange(length * 2 - 1, device=x.device, dtype=torch.long).unsqueeze(0)
		idx = i + j
		idx = idx.unsqueeze(0).unsqueeze(0).expand(batch, heads, length, length * 2 - 1)
		return torch.gather(x_pad, -1, idx)

	def _attention_bias_proximal(self, length):
		"""Bias for self-attention to encourage attention to close positions."""
		r = torch.arange(length, dtype=torch.float32)
		diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
		return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
	def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
		super().__init__()
		self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
		self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
		self.drop = nn.Dropout(p_dropout)
		self.kernel_size = kernel_size

	def forward(self, x, x_mask):
		x = self.conv_1(self._same_padding(x * x_mask))
		x = torch.relu(x)
		x = self.drop(x)
		x = self.conv_2(self._same_padding(x * x_mask))
		return x * x_mask

	def _same_padding(self, x):
		if self.kernel_size == 1:
			return x
		pad_l = (self.kernel_size - 1) // 2
		pad_r = self.kernel_size // 2
		if pad_l > 0:
			x = torch.cat([torch.zeros(x.shape[0], x.shape[1], pad_l, device=x.device, dtype=x.dtype), x], dim=-1)
		if pad_r > 0:
			x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], pad_r, device=x.device, dtype=x.dtype)], dim=-1)
		return x


def _merge_cond_projections(g, g_lang, cond_layer, cond_lang_layer=None, g_style=None, cond_style_layer=None, prosody=None, cond_prosody_layer=None):
	"""Project speaker, language, GST style, and frame prosody into flow/WN cond."""
	proj = None
	for cond, layer in ((g, cond_layer), (g_lang, cond_lang_layer), (g_style, cond_style_layer), (prosody, cond_prosody_layer), ):
		if cond is not None and layer is not None:
			part = layer(torch.detach(cond))
			proj = part if proj is None else proj + part
	return proj


def _apply_additive_global_conds(x, g, g_lang, cond, cond_lang=None):
	if g is not None: x = x + cond(torch.detach(g))
	if g_lang is not None and cond_lang is not None: x = x + cond_lang(torch.detach(g_lang))
	return x


class LayerNorm(nn.Module):
	def __init__(self, channels, eps=1e-5):
		super().__init__()
		self.channels = channels
		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(channels))
		self.beta = nn.Parameter(torch.zeros(channels))

	def forward(self, x):
		x = x.transpose(1, -1)
		x = F.layer_norm(x, (self.channels, ), self.gamma, self.beta, self.eps)
		return x.transpose(1, -1)


class DDSConv(nn.Module):
	def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
		super().__init__()
		self.channels = channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
		self.drop = nn.Dropout(p_dropout)
		self.convs_sep = nn.ModuleList()
		self.convs_1x1 = nn.ModuleList()
		self.norms_1 = nn.ModuleList()
		self.norms_2 = nn.ModuleList()
		for i in range(n_layers):
			dilation = kernel_size**i
			padding = (kernel_size * dilation - dilation) // 2
			self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
			self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
			self.norms_1.append(LayerNorm(channels))
			self.norms_2.append(LayerNorm(channels))

	def forward(self, x, x_mask, g=None):
		if g is not None:
			x = x + g
		for i in range(self.n_layers):
			y = self.convs_sep[i](x * x_mask)
			y = self.norms_1[i](y)
			y = F.gelu(y)
			y = self.convs_1x1[i](y)
			y = self.norms_2[i](y)
			y = F.gelu(y)
			y = self.drop(y)
			x = x + y
		return x * x_mask


class WN(torch.nn.Module):
	def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, use_lang_cond=True, use_style_cond=True, use_prosody_cond=True):
		super(WN, self).__init__()
		assert kernel_size % 2 == 1
		self.hidden_channels = hidden_channels
		self.kernel_size = (kernel_size)
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.gin_channels = gin_channels
		self.p_dropout = p_dropout
		self.in_layers = torch.nn.ModuleList()
		self.res_skip_layers = torch.nn.ModuleList()
		self.drop = nn.Dropout(p_dropout)
		self.cond_layer = torch.nn.utils.weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight")
		self.cond_lang_layer = torch.nn.utils.weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight") if use_lang_cond else None
		self.cond_style_layer = torch.nn.utils.weight_norm(torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight") if use_style_cond else None
		self.cond_prosody_layer = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 2 * hidden_channels * n_layers, 1), name="weight") if use_prosody_cond else None
		for i in range(n_layers):
			dilation = dilation_rate**i
			padding = int((kernel_size * dilation - dilation) / 2)
			in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
			in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
			self.in_layers.append(in_layer)
			res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
			res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
			res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
			self.res_skip_layers.append(res_skip_layer)

	def forward(self, x, x_mask, g=None, g_lang=None, g_style=None, prosody=None, **kwargs):
		output = torch.zeros_like(x)
		g = _merge_cond_projections(g, g_lang, self.cond_layer, self.cond_lang_layer, g_style=g_style, cond_style_layer=self.cond_style_layer, prosody=prosody, cond_prosody_layer=self.cond_prosody_layer)
		for i in range(self.n_layers):
			x_in = self.in_layers[i](x)
			if g is not None:
				cond_offset = i * 2 * self.hidden_channels
				g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
			else:
				g_l = torch.zeros_like(x_in)
			acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
			acts = self.drop(acts)
			res_skip_acts = self.res_skip_layers[i](acts)
			if i < self.n_layers - 1:
				res_acts = res_skip_acts[:, :self.hidden_channels, :]
				x = (x + res_acts) * x_mask
				output = output + res_skip_acts[:, self.hidden_channels:, :]
			else:
				output = output + res_skip_acts
		return output * x_mask

	def remove_weight_norm(self):
		for layer in (*self.in_layers, *self.res_skip_layers, self.cond_layer):
			remove_weight_norm(layer)
		for layer in (self.cond_lang_layer, self.cond_style_layer, self.cond_prosody_layer):
			if layer is not None:
				remove_weight_norm(layer)


class Log(nn.Module):
	def forward(self, x, x_mask, reverse=False, **kwargs):
		if not reverse:
			min_val = torch.tensor(1e-5, dtype=x.dtype, device=x.device)
			y = torch.log(torch.maximum(x, min_val)) * x_mask
			logdet = torch.sum(-y, [1, 2])
			return y, logdet
		else:
			return torch.exp(x) * x_mask


class Flip(nn.Module):
	def forward(self, x, *args, reverse=False, **kwargs):
		x = torch.flip(x, [1])
		if not reverse:
			logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
			return x, logdet
		else:
			return x


class ElementwiseAffine(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.channels = channels
		self.m = nn.Parameter(torch.zeros(channels, 1))
		self.logs = nn.Parameter(torch.zeros(channels, 1))

	def forward(self, x, x_mask, reverse=False, **kwargs):
		if not reverse:
			y = self.m + torch.exp(self.logs) * x
			y = y * x_mask
			logdet = torch.sum(self.logs * x_mask, [1, 2])
			return y, logdet
		else:
			return (x - self.m) * torch.exp(-self.logs) * x_mask


class ConvFlow(nn.Module):
	def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
		super().__init__()
		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.num_bins = num_bins
		self.tail_bound = tail_bound
		self.half_channels = in_channels // 2
		self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
		self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
		self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
		self.proj.weight.data.zero_()
		self.proj.bias.data.zero_()

	def forward(self, x, x_mask, g=None, reverse=False):
		x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
		if not _is_exporting(): _shape_check(x0.size(-1) > 1)
		h = self.pre(x0)
		h = self.convs(h, x_mask, g=g)
		h = self.proj(h) * x_mask
		b, c, t = x0.shape
		num_bins_total = self.num_bins * 3 - 1
		h = h.view(b, c, num_bins_total, t).permute(0, 1, 3, 2)
		unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
		unnormalized_heights = h[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
		unnormalized_derivatives = h[..., 2 * self.num_bins:]
		x1, logabsdet = piecewise_rational_quadratic_transform(x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=reverse, tails="linear", tail_bound=self.tail_bound)
		x = torch.cat([x0, x1], 1) * x_mask
		logdet = torch.sum(logabsdet * x_mask, [1, 2])
		if not reverse:
			return x, logdet
		else:
			return x


class StochasticDurationPredictor(nn.Module):
	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
		super().__init__()
		filter_channels = in_channels
		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.n_flows = n_flows
		self.gin_channels = gin_channels
		self.log_flow = Log()
		self.flows = nn.ModuleList()
		self.flows.append(ElementwiseAffine(2))
		for _ in range(n_flows):
			self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.flows.append(Flip())
		self.post_pre = nn.Conv1d(1, filter_channels, 1)
		self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		self.post_flows = nn.ModuleList()
		self.post_flows.append(ElementwiseAffine(2))
		for _ in range(4):
			self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.post_flows.append(Flip())
		self.pre = nn.Conv1d(in_channels, filter_channels, 1)
		self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
		self.cond_lang = nn.Conv1d(gin_channels, filter_channels, 1)

	def forward(self, x, x_mask, w=None, g=None, g_lang=None, reverse=False, noise_scale=1.0, dur_noise=None):
		x = torch.detach(x)
		x = self.pre(x)
		x = _apply_additive_global_conds(x, g, g_lang, self.cond, getattr(self, "cond_lang", None))
		x = self.convs(x, x_mask)
		x = self.proj(x) * x_mask

		if not reverse:
			flows = self.flows
			assert w is not None
			logdet_tot_q = 0
			h_w = self.post_pre(w)
			h_w = self.post_convs(h_w, x_mask)
			h_w = self.post_proj(h_w) * x_mask
			e_q = (torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask)
			z_q = e_q

			for flow in self.post_flows:
				z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
				logdet_tot_q += logdet_q

			z_u, z1 = torch.split(z_q, [1, 1], 1)
			u = torch.sigmoid(z_u) * x_mask
			z0 = (w - u) * x_mask
			logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
			logq = (torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q)
			logdet_tot = 0
			z0, logdet = self.log_flow(z0, x_mask)
			logdet_tot += logdet
			z = torch.cat([z0, z1], 1)

			for flow in flows:
				z, logdet = flow(z, x_mask, g=x, reverse=reverse)
				logdet_tot = logdet_tot + logdet

			nll = (torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot)
			return nll + logq  # [b]
		else:
			flows = list(reversed(self.flows))
			flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
			z = _infer_noise_like(x[:, :2, :], noise_scale, noise=dur_noise)

			for flow in flows:
				z = flow(z, x_mask, g=x, reverse=reverse)

			z0, z1 = torch.split(z, [1, 1], 1)
			logw = z0
			return logw


class StochasticPitchPredictor(nn.Module):
	"""Flow-based pitch on mel frames — separate architecture from SDP (no post_flows / duration coupling)."""
	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
		super().__init__()
		filter_channels = in_channels
		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.n_flows = n_flows
		self.gin_channels = gin_channels
		self.flows = nn.ModuleList()
		self.flows.append(ElementwiseAffine(2))
		for _ in range(n_flows):
			self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.flows.append(Flip())
		self.pre = nn.Conv1d(in_channels, filter_channels, 1)
		self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
		self.cond_lang = nn.Conv1d(gin_channels, filter_channels, 1)

	def forward(self, x, x_mask, w=None, g=None, g_lang=None, reverse=False, noise_scale=1.0, loss_mask=None, pitch_noise=None):
		x = torch.detach(x)
		_shape_check(x.size(-1) > 0)
		h = self.pre(x)
		h = _apply_additive_global_conds(h, g, g_lang, self.cond, getattr(self, "cond_lang", None))
		h = self.convs(h, x_mask)
		h = self.proj(h) * x_mask

		if not reverse:
			assert w is not None
			eff_mask = loss_mask if loss_mask is not None else x_mask
			z_aux = torch.randn(w.size(0), 1, w.size(2), device=w.device, dtype=w.dtype)
			z = torch.cat([w, z_aux], dim=1) * eff_mask
			logdet_tot = 0
			for flow in self.flows:
				z, logdet = flow(z, eff_mask, g=h)
				logdet_tot += logdet
			z32 = z.float()
			mask32 = eff_mask.float()
			logdet32 = logdet_tot.float()
			nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z32**2)) * mask32, [1, 2]) - logdet32
			return nll
		else:
			flows = list(reversed(self.flows))
			flows = flows[:-2] + [flows[-1]]
			z = _infer_noise((x.size(0), 2, x.size(2)), device=x.device, dtype=x.dtype, noise_scale=noise_scale, noise=pitch_noise)
			z = z * x_mask
			for flow in flows:
				z = flow(z, x_mask, g=h, reverse=True)
			pitch, _ = torch.split(z, [1, 1], 1)
			return pitch.clamp(0.0, 1.0) * x_mask


class ConditionalVariancePredictor(nn.Module):
	"""Text-aligned variance head with speaker + language global conditioning."""
	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, out_channels=1, gin_channels=0):
		super().__init__()
		self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_1 = LayerNorm(filter_channels)
		self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_2 = LayerNorm(filter_channels)
		self.proj = nn.Conv1d(filter_channels, out_channels, 1)
		self.drop = nn.Dropout(p_dropout)
		self.gin_channels = gin_channels
		self.cond = nn.Conv1d(gin_channels, in_channels, 1)
		self.cond_lang = nn.Conv1d(gin_channels, in_channels, 1)

	def forward(self, x, x_mask, g=None, g_lang=None):
		x = torch.detach(x)
		x = _apply_additive_global_conds(x, g, g_lang, self.cond, getattr(self, "cond_lang", None))
		x = self.conv_1(x * x_mask)
		x = torch.relu(x)
		x = self.norm_1(x)
		x = self.drop(x)
		x = self.conv_2(x * x_mask)
		x = torch.relu(x)
		x = self.norm_2(x)
		x = self.drop(x)
		x = self.proj(x * x_mask)
		return x * x_mask


class MelStyleEncoder(nn.Module):
	"""Reference mel -> global style vector for GST-style flow conditioning."""
	def __init__(self, n_mels, style_hidden=256, gin_channels=256):
		super().__init__()
		self.convs = nn.Sequential(weight_norm(Conv1d(n_mels, style_hidden, 5, 1, padding=2)), nn.ReLU(inplace=True), weight_norm(Conv1d(style_hidden, style_hidden, 5, 1, padding=2)), nn.ReLU(inplace=True), weight_norm(Conv1d(style_hidden, style_hidden, 5, 1, padding=2)), nn.ReLU(inplace=True), )
		self.proj = nn.Linear(style_hidden, gin_channels)

	def forward(self, mel, mel_mask):
		x = self.convs(mel * mel_mask)
		x = (x * mel_mask).sum(dim=2) / mel_mask.sum(dim=2).clamp(min=1.0)
		return self.proj(x).unsqueeze(-1)


class DurationDiscriminatorV2(nn.Module):
	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
		super().__init__()
		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.gin_channels = gin_channels
		self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_1 = LayerNorm(filter_channels)
		self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_2 = LayerNorm(filter_channels)
		self.dur_proj = nn.Conv1d(1, filter_channels, 1)
		self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.pre_out_norm_1 = LayerNorm(filter_channels)
		self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.pre_out_norm_2 = LayerNorm(filter_channels)
		self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

	def forward_probability(self, x, x_mask, dur, g=None):
		dur = self.dur_proj(dur)
		x = torch.cat([x, dur], dim=1)
		x = self.pre_out_conv_1(x * x_mask)
		x = torch.relu(x)
		x = self.pre_out_norm_1(x)
		x = self.pre_out_conv_2(x * x_mask)
		x = torch.relu(x)
		x = self.pre_out_norm_2(x)
		x = x * x_mask
		x = x.transpose(1, 2)
		output_prob = self.output_layer(x)
		return output_prob

	def forward(self, x, x_mask, dur_r, dur_hat, g=None):
		x = torch.detach(x)
		x = self.conv_1(x * x_mask)
		x = torch.relu(x)
		x = self.norm_1(x)
		x = self.conv_2(x * x_mask)
		x = torch.relu(x)
		x = self.norm_2(x)
		output_probs = []
		for dur in [dur_r, dur_hat]:
			output_prob = self.forward_probability(x, x_mask, dur, g)
			output_probs.append([output_prob])
		return output_probs


class TextEncoder(nn.Module):
	def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=0, n_languages=0):
		super().__init__()
		self.n_vocab = n_vocab
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.gin_channels = gin_channels
		self.n_languages = n_languages
		self.emb = nn.Embedding(n_vocab, hidden_channels)
		nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
		if n_languages > 0:
			self.emb_l = nn.Embedding(n_languages, hidden_channels)
			nn.init.normal_(self.emb_l.weight, 0.0, hidden_channels**-0.5)
		else:
			self.emb_l = None
		self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=self.gin_channels)
		self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(self, x, x_lengths, g=None, lid=None):
		x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
		if self.emb_l is not None and lid is not None:
			x = x + self.emb_l(lid).unsqueeze(1)
		x = torch.transpose(x, 1, -1)  # [b, h, t]
		x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
		x = self.encoder(x * x_mask, x_mask, g=g)
		stats = self.proj(x) * x_mask
		m, logs = torch.split(stats, self.out_channels, dim=1)
		return x, m, logs, x_mask


class ResidualCouplingTransformersLayer(nn.Module):
	def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
		assert channels % 2 == 0, "channels should be divisible by 2"
		super().__init__()
		self.channels = channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.half_channels = channels // 2
		self.mean_only = mean_only
		self.pre_transformer = Encoder(self.half_channels, self.half_channels, n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, window_size=None)
		self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
		self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
		self.post_transformer = Encoder(self.hidden_channels, self.hidden_channels, n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, window_size=None)
		self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
		self.post.weight.data.zero_()
		self.post.bias.data.zero_()

	def forward(self, x, x_mask, g=None, g_lang=None, g_style=None, prosody=None, reverse=False):
		x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
		x0_ = self.pre_transformer(x0 * x_mask, x_mask)
		x0_ = x0_ + x0
		h = self.pre(x0_) * x_mask  # Changed from x0 to x0_ to retain x0 for the flow
		h = self.enc(h, x_mask, g=g, g_lang=g_lang, g_style=g_style, prosody=prosody)
		stats = self.post(h) * x_mask

		if not self.mean_only:
			m, logs = torch.split(stats, [self.half_channels] * 2, 1)
		else:
			m = stats
			logs = torch.zeros_like(m)

		if not reverse:
			x1 = m + x1 * torch.exp(logs) * x_mask
			x = torch.cat([x0, x1], 1)
			logdet = torch.sum(logs, [1, 2])
			return x, logdet
		else:
			x1 = (x1 - m) * torch.exp(-logs) * x_mask
			x = torch.cat([x0, x1], 1)
			return x


class ResidualCouplingTransformersBlock(nn.Module):
	def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
		super().__init__()
		self.channels = channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.n_flows = n_flows
		self.gin_channels = gin_channels
		self.flows = nn.ModuleList()
		for i in range(n_flows):
			self.flows.append(ResidualCouplingTransformersLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
			self.flows.append(Flip())

	def forward(self, x, x_mask, g=None, g_lang=None, g_style=None, prosody=None, reverse=False):
		if not reverse:
			for flow in self.flows:
				x, _ = flow(x, x_mask, g=g, g_lang=g_lang, g_style=g_style, prosody=prosody, reverse=reverse)
		else:
			for flow in reversed(self.flows):
				x = flow(x, x_mask, g=g, g_lang=g_lang, g_style=g_style, prosody=prosody, reverse=reverse)
		return x


class PosteriorEncoder(nn.Module):
	def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.gin_channels = gin_channels
		self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
		self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, use_lang_cond=False, use_style_cond=False, use_prosody_cond=False)
		self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(self, x, x_lengths, g=None):
		x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
		x = self.pre(x) * x_mask
		x = self.enc(x, x_mask, g=g)
		stats = self.proj(x) * x_mask
		m, logs = torch.split(stats, self.out_channels, dim=1)
		z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
		return z, m, logs, x_mask


def _latent_vocos_freq_mask(n_fft, sampling_rate, fmin_hz, fmax_hz, device, dtype):
	n_bins = n_fft // 2 + 1
	bin_hz = torch.arange(n_bins, device=device, dtype=dtype) * (sampling_rate / n_fft)
	return ((bin_hz >= fmin_hz) & (bin_hz <= fmax_hz)).to(dtype)


class LatentVocos48k(nn.Module):
	"""Pure latent Vocos decoder: z -> ConvNeXt backbone -> ISTFT at hop=256 (no mel, no band split)."""
	is_latent_vocos_decoder = True

	def __init__(self, latent_channels, gin_channels=0, sampling_rate=48000, n_fft=2048, hop_length=256, fmin_hz=120.0, fmax_hz=14000.0, dim=512, intermediate_dim=1536, num_layers=8, istft_padding="center", ):
		super().__init__()
		self.latent_channels = latent_channels
		self.gin_channels = gin_channels
		self.sampling_rate = sampling_rate
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.fmin_hz = fmin_hz
		self.fmax_hz = fmax_hz
		self.n_bins = n_fft // 2 + 1
		self.backbone = VocosBackbone(latent_channels, dim, intermediate_dim, num_layers)
		self.head = nn.Linear(dim, n_fft + 2)
		self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=istft_padding)
		self.latent_spk = Conv1d(gin_channels, latent_channels, 1) if gin_channels > 0 else None
		self.spk_backbone = Conv1d(gin_channels, dim, 1) if gin_channels > 0 else None
		self.register_buffer("freq_mask", _latent_vocos_freq_mask(n_fft, sampling_rate, fmin_hz, fmax_hz, torch.device("cpu"), torch.float32), persistent=False, )
		nn.init.trunc_normal_(self.head.weight, std=0.02)
		nn.init.constant_(self.head.bias, 0.0)

	def _freq_mask(self, device, dtype):
		if self.freq_mask.device != device or self.freq_mask.dtype != dtype:
			return _latent_vocos_freq_mask(self.n_fft, self.sampling_rate, self.fmin_hz, self.fmax_hz, device, dtype)
		return self.freq_mask

	def _synthesize(self, features, target_frames):
		"""features: (B, T, dim) -> wav (B, 1, T*hop)."""
		x = self.head(features).transpose(1, 2)
		mag, p = x.chunk(2, dim=1)
		mag = torch.exp(mag).clamp(max=1e2)
		mag = mag * self._freq_mask(mag.device, mag.dtype).view(1, -1, 1)
		spec = mag * (torch.cos(p) + 1j * torch.sin(p))

		target_length = target_frames * self.hop_length
		window = self.istft.window.to(device=spec.device, dtype=spec.real.dtype)
		if self.istft.padding == "center":
			wav = torch.istft(spec, self.n_fft, self.hop_length, self.istft.win_length, window=window, center=True, length=target_length)
		else:
			wav = self.istft(spec)
			if wav.shape[-1] > target_length:
				wav = wav[..., :target_length]
			elif wav.shape[-1] < target_length:
				wav = F.pad(wav, (0, target_length - wav.shape[-1]))

		if wav.dim() == 1:
			wav = wav.unsqueeze(0)
		return wav.unsqueeze(1)

	def forward(self, x, g=None):
		if g is not None and self.latent_spk is not None:
			x = x + self.latent_spk(g)
		features = self.backbone(x)
		if g is not None and self.spk_backbone is not None:
			features = features + self.spk_backbone(g).transpose(1, 2)
		return self._synthesize(features, x.shape[-1])


def decode_to_waveform(dec, z, g=None):
	"""Normalize decoder output to [B, 1, T] waveform."""
	out = dec(z, g=g)
	if out.dim() == 2:
		out = out.unsqueeze(1)
	return out


class DiscriminatorP(torch.nn.Module):
	def __init__(self, period, kernel_size=5, stride=3):
		super(DiscriminatorP, self).__init__()
		self.period = period
		norm_f = weight_norm
		self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))), ])
		self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

	def forward(self, x):
		fmap = []
		b, c, t = x.shape

		if t % self.period != 0:
			n_pad = self.period - (t % self.period)
			x = F.pad(x, (0, n_pad), "reflect")
			t = t + n_pad

		x = x.view(b, c, t // self.period, self.period)
		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, 0.1)
			fmap.append(x)

		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)
		return x, fmap


class DiscriminatorS(torch.nn.Module):
	def __init__(self):
		super(DiscriminatorS, self).__init__()
		norm_f = weight_norm
		self.convs = nn.ModuleList([norm_f(Conv1d(1, 16, 15, 1, padding=7)), norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)), norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)), norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)), norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2)), ])
		self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

	def forward(self, x):
		fmap = []

		for l in self.convs:
			x = l(x)
			x = F.leaky_relu(x, 0.1)
			fmap.append(x)

		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)
		return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
	"""HiFi-GAN MPD. Periods are in samples, so their effective frequency is sr/period: the stock [2,3,5,7,11] set was tuned at 22.05 kHz and lands ~2.2x higher at 48 kHz. Changing them rewrites what every sub-discriminator looks at without changing any tensor shape, so a checkpoint reloads silently into a critic that no longer means what it was trained to mean."""
	def __init__(self, periods=(2, 3, 5, 7, 11)):
		super(MultiPeriodDiscriminator, self).__init__()
		discs = [DiscriminatorS()]
		discs = discs + [DiscriminatorP(i) for i in periods]
		self.discriminators = nn.ModuleList(discs)

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []

		for i, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			y_d_gs.append(y_d_g)
			fmap_rs.append(fmap_r)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(torch.nn.Module):
	"""Band-split STFT discriminator (from Vocos). Expects mono audio [B, T]. Bands are fractions of Nyquist (24 kHz at 48 kHz sample rate), packed so that five of the six stacks fall inside the 120 Hz - 14 kHz synthesis band where the speech detail lives. The last stack watches 14 kHz - Nyquist purely as a hallucination guard."""
	def __init__(self, window_length, channels=32, hop_factor=0.25, bands=((0.0, 0.05), (0.05, 0.125), (0.125, 0.25), (0.25, 0.42), (0.42, 0.5834), (0.5834, 1.0))):
		super().__init__()
		from torchaudio.transforms import Spectrogram  # training-only discriminator
		self.window_length = window_length
		self.hop_factor = hop_factor
		self.spec_fn = Spectrogram(n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None)
		n_fft = window_length // 2 + 1
		bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
		self.bands = bands
		convs = lambda: nn.ModuleList([weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))), ])
		self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
		self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

	def spectrogram(self, x):
		if x.dim() == 3:
			x = x.squeeze(1)
		x = x - x.mean(dim=-1, keepdim=True)  # Match the known-good run: DC remove + per-sample peak norm. Absolute level is still policed by mel / MRSTFT / oob; this keeps the MRD focused on band shape like before.
		x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
		x = self.spec_fn(x)
		x = torch.view_as_real(x)
		x = x.permute(0, 3, 2, 1)
		return [x[..., b[0]:b[1]] for b in self.bands]

	def forward(self, x):
		x_bands = self.spectrogram(x)
		fmap = []
		bands_out = []
		for band, stack in zip(x_bands, self.band_convs):
			for i, layer in enumerate(stack):
				band = layer(band)
				band = F.leaky_relu(band, 0.1)
				if i > 0:
					fmap.append(band)
			bands_out.append(band)
		x = torch.cat(bands_out, dim=-1)
		x = self.conv_post(x)
		fmap.append(x)
		return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(torch.nn.Module):
	def __init__(self, fft_sizes=(2048, 1024, 512)):
		super().__init__()
		self.discriminators = nn.ModuleList([DiscriminatorR(window_length=w) for w in fft_sizes])

	def forward(self, y, y_hat):
		if y.dim() == 3:
			y = y.squeeze(1)
		if y_hat.dim() == 3:
			y_hat = y_hat.squeeze(1)
		y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
		for d in self.discriminators:
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			y_d_gs.append(y_d_g)
			fmap_rs.append(fmap_r)
			fmap_gs.append(fmap_g)
		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class QwenDistillProjector(nn.Module):
	"""Map posterior latent frames to frozen Qwen AuT feature space."""
	def __init__(self, in_channels, out_dim, hidden_dim=512):
		super().__init__()
		self.net = nn.Sequential(weight_norm(nn.Conv1d(in_channels, hidden_dim, 1)), nn.GELU(), weight_norm(nn.Conv1d(hidden_dim, out_dim, 1)), )

	def forward(self, z):
		return self.net(z)


class SynthesizerTrn(nn.Module):
	"""Synthesizer for Training"""
	def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, n_speakers=0, n_languages=0, gin_channels=0, **kwargs):
		super().__init__()
		self.n_vocab = n_vocab
		self.spec_channels = spec_channels
		self.inter_channels = inter_channels
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.segment_size = segment_size
		self.n_speakers = n_speakers
		self.n_languages = n_languages
		self.gin_channels = gin_channels
		self.use_spp = kwargs.get("use_spp", False)
		self.use_flow_prosody = kwargs.get("use_flow_prosody", True)
		self.flow_prosody_teacher_forcing = kwargs.get("flow_prosody_teacher_forcing", False)
		self.flow_prosody_warmup_steps = kwargs.get("flow_prosody_warmup_steps", 0)
		self.flow_prosody_train_noise_scale_p = kwargs.get("flow_prosody_train_noise_scale_p", 1.0)
		self.use_pitch_l1_loss = kwargs.get("use_pitch_l1_loss", True)
		self.pitch_l1_noise_scale = kwargs.get("pitch_l1_noise_scale", 0.0)
		self.pitch_l1_smoothed_target = kwargs.get("pitch_l1_smoothed_target", True)
		self.mas_flow_prosody_parity = kwargs.get("mas_flow_prosody_parity", False)
		self.use_gst = kwargs.get("use_gst", False)
		self.prosody_cfg = utils.resolve_prosody_cfg(kwargs.get("prosody_cfg"))
		self.use_qwen_distill = kwargs.get("use_qwen_distill", False)
		self.qwen_teacher_dim = kwargs.get("qwen_teacher_dim", 2048)
		self.qwen_loss_type = kwargs.get("qwen_loss_type", "mse")
		self.qwen_norm_weight = kwargs.get("qwen_norm_weight", 0.0)
		self.use_latent_vocos_decoder = True
		self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
		self.current_mas_noise_scale = 0.01
		self.enc_gin_channels = gin_channels if kwargs.get("use_spk_conditioned_encoder", True) else 0
		self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=self.enc_gin_channels, n_languages=n_languages)
		hop_length = kwargs.get("hop_length", 256)
		self.hop_length = hop_length
		sampling_rate = kwargs.get("sampling_rate", 48000)
		filter_length = kwargs.get("filter_length", 2048)
		mel_fmin = kwargs.get("mel_fmin", 120.0)
		mel_fmax = kwargs.get("mel_fmax", 14000.0)
		print(f"Latent Vocos decoder (sr={sampling_rate}, n_fft={filter_length}, hop={hop_length}, f=[{mel_fmin},{mel_fmax}] Hz)")
		self.dec = LatentVocos48k(latent_channels=inter_channels, gin_channels=gin_channels, sampling_rate=sampling_rate, n_fft=filter_length, hop_length=hop_length, fmin_hz=mel_fmin, fmax_hz=mel_fmax, dim=kwargs.get("latent_vocos_dim", 512), intermediate_dim=kwargs.get("latent_vocos_intermediate_dim", 1536), num_layers=kwargs.get("latent_vocos_num_layers", 8), istft_padding=kwargs.get("latent_vocos_istft_padding", "center"), )
		self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
		self.flow = ResidualCouplingTransformersBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
		self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
		if self.use_spp:
			self.spp = StochasticPitchPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
			self.vuv_pred = ConditionalVariancePredictor(hidden_channels, 256, 3, 0.5, out_channels=1, gin_channels=gin_channels)
			self.energy_pred = ConditionalVariancePredictor(hidden_channels, 256, 3, 0.15, out_channels=1, gin_channels=gin_channels)
		if self.use_gst:
			style_hidden = kwargs.get("gst_style_hidden", 256)
			self.style_encoder = MelStyleEncoder(spec_channels, style_hidden=style_hidden, gin_channels=gin_channels)
		else:
			self.style_encoder = None

		if kwargs.get("qwen_teacher_dim"):
			proj_hidden = kwargs.get("qwen_proj_hidden", 512)
			self.qwen_proj = QwenDistillProjector(inter_channels, self.qwen_teacher_dim, hidden_dim=proj_hidden)
		else:
			self.qwen_proj = None

		if n_speakers > 0:
			self.emb_g = nn.Embedding(n_speakers, gin_channels)
			if self.use_spp:
				self.emb_pitch_g = nn.Embedding(n_speakers, gin_channels)
				nn.init.normal_(self.emb_pitch_g.weight, 0.0, gin_channels**-0.5)
		else:
			self.emb_g = None

		if n_languages > 0:
			self.emb_l_g = nn.Embedding(n_languages, gin_channels)
			nn.init.normal_(self.emb_l_g.weight, 0.0, gin_channels**-0.5)
		else:
			self.emb_l_g = None

	def _global_speaker_cond(self, sid):
		if self.emb_g is None or sid is None:
			return None
		return self.emb_g(sid).unsqueeze(-1)

	def _global_language_cond(self, lid):
		if self.emb_l_g is None or lid is None:
			return None
		return self.emb_l_g(lid).unsqueeze(-1)

	def _speaker_pitch_cond(self, sid):
		if self.emb_g is None or sid is None:
			return None
		g = self.emb_g(sid).unsqueeze(-1)
		if hasattr(self, "emb_pitch_g"):
			g = g + self.emb_pitch_g(sid).unsqueeze(-1)
		return g

	def _prepare_flow_prosody(self, pitch, energy, y_mask, vuv_logit=None, pitch_raw=None, prosody_overrides=None):
		"""Smooth pitch/energy for flow conditioning (train + infer)."""
		if not self.use_flow_prosody:
			return None
		pitch_s, energy_s, _ = utils.smooth_prosody_channels(pitch, energy, y_mask, vuv_logit=vuv_logit, pitch_raw=pitch_raw, cfg=self.prosody_cfg, overrides=prosody_overrides)
		return self._build_prosody_cond(pitch_s, energy_s, y_mask)

	def qwen_distill_loss(self, z, qwen_feat, qwen_lengths, y_mask):
		"""Match pooled posterior latent frames to offline Qwen AuT teacher features."""
		if self.qwen_proj is None or qwen_feat is None or qwen_lengths is None:
			return None
		if not torch.is_tensor(qwen_feat):
			qwen_feat = torch.as_tensor(qwen_feat, device=z.device, dtype=z.dtype)
		if qwen_feat.dim() == 2:
			qwen_feat = qwen_feat.unsqueeze(0)
		b, t_q, d = qwen_feat.shape
		if t_q <= 0 or d != self.qwen_teacher_dim:
			return None

		mel_lengths = y_mask.squeeze(1).sum(-1).long()  # Pool each item from its own valid mel span to its own teacher length, so pooled frame k is the same instant as teacher frame k. Padding is excluded here, which is why the mask below is the teacher mask alone.
		z_pooled = utils.pool_latent_to_teacher_frames(z, qwen_lengths, source_lengths=mel_lengths)
		pred = self.qwen_proj(z_pooled)
		target = qwen_feat.transpose(1, 2).to(dtype=pred.dtype, device=pred.device)
		t_common = min(pred.size(-1), target.size(-1))
		if t_common <= 0:
			return None
		pred = pred[..., :t_common]
		target = target[..., :t_common].detach()

		lengths = torch.as_tensor(qwen_lengths, device=z.device, dtype=torch.long).clamp(max=t_common)
		mask = utils.qwen_teacher_mask(lengths, t_common, z.device, pred.dtype)
		denom = torch.clamp(mask.sum(), min=1.0)

		if self.qwen_loss_type == "cosine":
			pred_n = F.normalize(pred, dim=1, eps=1e-6)
			target_n = F.normalize(target, dim=1, eps=1e-6)
			cos = (pred_n * target_n).sum(dim=1, keepdim=True)
			loss = ((1.0 - cos) * mask).sum() / denom
			if self.qwen_norm_weight > 0.0:
				log_ratio = 0.5 * (torch.log(pred.pow(2).sum(dim=1, keepdim=True) + 1e-6) - torch.log(target.pow(2).sum(dim=1, keepdim=True) + 1e-6))  # Cosine gradient scales as 1 / ||pred||, so an unconstrained projector norm can drift upward and quench its own learning signal. Anchor it to the teacher norm. Squared-sum rather than .norm(): d||x||/dx is 0/0 at x=0, and padded frames pool to exactly zero, where a NaN would survive multiplication by mask=0.
				loss = loss + self.qwen_norm_weight * (log_ratio.abs() * mask).sum() / denom
			return loss

		diff = (pred - target).abs().mean(dim=1, keepdim=True)  # Reduce over channels before masking: mask is [B,1,T], so a raw [B,D,T] product would broadcast to [B,B,D,T].
		return (diff * mask).sum() / denom

	def _build_prosody_cond(self, pitch, energy, y_mask):
		"""Frame-wise [pitch, log_energy] for flow conditioning."""
		if not self.use_flow_prosody:
			return None
		b, _, t = y_mask.shape
		device, dtype = y_mask.device, y_mask.dtype
		if pitch is None:
			pitch = torch.zeros(b, 1, t, device=device, dtype=dtype)
		elif pitch.dim() == 2:
			pitch = pitch.unsqueeze(1)
		if energy is None:
			energy = torch.zeros(b, 1, t, device=device, dtype=dtype)
		elif energy.dim() == 2:
			energy = energy.unsqueeze(1)
		return torch.cat([pitch, energy], dim=1) * y_mask

	def _mas_attention(self, z_p_for_mas, m_p, logs_p, x_mask, y_mask):
		"""Monotonic alignment from prior negative centroids (no grad through MAS)."""
		with torch.no_grad():
			s_p_sq_r = torch.exp(-2 * logs_p)
			neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
			neg_cent2 = torch.matmul(-0.5 * (z_p_for_mas**2).transpose(1, 2), s_p_sq_r)
			neg_cent3 = torch.matmul(z_p_for_mas.transpose(1, 2), (m_p * s_p_sq_r))
			neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
			neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
			epsilon = (torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale)
			neg_cent = neg_cent + epsilon
			attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
			attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
		return attn.clone().detach()

	def _spp_prosody_block(self, x, attn, pitch, pitch_mel, energy_mel, y_mask, sid, g_lang, use_predicted_prosody):
		"""SPP + energy + flow prosody cond from an alignment."""
		out = {"l_pitch": None, "l_pitch_l1": None, "l_energy": None, "logpitch": None, "logpitch_": None, "vuv_logit": None, "energy_hat": None, "pitch_hat": None, "prosody": None}
		if not self.use_spp or pitch is None:
			return out
		x_aligned = utils.align_hidden_to_mel(x, attn)
		pitch_mel = pitch.unsqueeze(1) if pitch_mel is None else pitch_mel
		voiced_mask = (pitch > 1e-4).float().unsqueeze(1) * y_mask
		g_pitch = self._speaker_pitch_cond(sid)
		spp_noise_scale = self.flow_prosody_train_noise_scale_p if use_predicted_prosody else 1.0
		out["l_pitch"] = self.spp(x_aligned, y_mask, pitch_mel, g=g_pitch, g_lang=g_lang, loss_mask=voiced_mask)
		out["l_pitch"] = out["l_pitch"] / torch.clamp(torch.sum(voiced_mask, [1, 2]), min=1.0)
		out["logpitch"] = self.spp(x_aligned, y_mask, g=g_pitch, g_lang=g_lang, reverse=True, noise_scale=spp_noise_scale)
		out["pitch_hat"] = out["logpitch"]
		out["logpitch_"] = torch.log(pitch_mel + 1e-6) * y_mask
		out["vuv_logit"] = self.vuv_pred(torch.detach(x_aligned), y_mask, g=g_pitch, g_lang=g_lang)
		if self.use_pitch_l1_loss:
			pitch_l1_pred = self.spp(x_aligned, y_mask, g=g_pitch, g_lang=g_lang, reverse=True, noise_scale=self.pitch_l1_noise_scale)
			pitch_l1_target = pitch_mel
			if self.pitch_l1_smoothed_target:
				pitch_l1_target, _, _ = utils.smooth_prosody_channels(pitch_mel, energy_mel, y_mask, cfg=self.prosody_cfg)
			out["l_pitch_l1"] = torch.sum(torch.abs((pitch_l1_pred - pitch_l1_target) * voiced_mask)) / torch.clamp(torch.sum(voiced_mask), min=1.0)
		if hasattr(self, "energy_pred"):
			energy_target = energy_mel if energy_mel is not None else torch.zeros_like(pitch_mel)
			_, energy_target_s, _ = utils.smooth_prosody_channels(pitch_mel, energy_target, y_mask, cfg=self.prosody_cfg)
			if not self.prosody_cfg.get("match_energy_loss_to_flow", True):
				energy_target_s = energy_target
			out["energy_hat"] = self.energy_pred(x_aligned, y_mask, g=g_pitch, g_lang=g_lang)
			out["l_energy"] = torch.sum(torch.abs((out["energy_hat"] - energy_target_s) * y_mask)) / torch.clamp(torch.sum(y_mask), min=1.0)
		if use_predicted_prosody:
			out["prosody"] = self._prepare_flow_prosody(out["pitch_hat"], out["energy_hat"], y_mask, vuv_logit=out["vuv_logit"], pitch_raw=out["pitch_hat"])
		return out

	def _encode_style(self, style_ref_mel, style_ref_lengths):
		if self.style_encoder is None or style_ref_mel is None or style_ref_lengths is None:
			return None
		ref_mask = torch.unsqueeze(sequence_mask(style_ref_lengths, style_ref_mel.size(2)), 1).to(style_ref_mel.dtype)
		return self.style_encoder(style_ref_mel, ref_mask)

	def _use_predicted_flow_prosody(self, global_step):
		"""Professor forcing: flow cond uses SPP/energy_pred (train/infer parity)."""
		if not self.use_flow_prosody or not self.flow_prosody_teacher_forcing or not self.use_spp:
			return False
		if global_step is None:
			return True
		return global_step >= self.flow_prosody_warmup_steps

	def forward(self, x, x_lengths, y, y_lengths, sid=None, lid=None, pitch=None, energy=None, style_ref_mel=None, style_ref_lengths=None, global_step=None, qwen_feat=None, qwen_lengths=None):
		g_spk = self._global_speaker_cond(sid)
		g_lang = self._global_language_cond(lid)
		x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g_spk, lid=lid)
		z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_spk)

		pitch_mel = pitch.unsqueeze(1) if pitch is not None and pitch.dim() == 2 else pitch
		if energy is not None and energy.dim() == 2:
			energy_mel = energy.unsqueeze(1)
		else:
			energy_mel = energy
		g_style = self._encode_style(style_ref_mel, style_ref_lengths)
		use_predicted_prosody = self._use_predicted_flow_prosody(global_step)
		prosody_gt = None
		if self.use_flow_prosody and pitch_mel is not None:
			prosody_gt = self._prepare_flow_prosody(pitch_mel, energy_mel, y_mask)

		if use_predicted_prosody:
			z_p_for_mas = self.flow(z, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=None)
		elif prosody_gt is not None:
			z_p_for_mas = self.flow(z, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=prosody_gt)
		else:
			z_p_for_mas = self.flow(z, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=None)

		attn = self._mas_attention(z_p_for_mas, m_p, logs_p, x_mask, y_mask)
		attn.requires_grad = False

		w = attn.sum(2)
		l_length = self.dp(x, x_mask, w, g=g_spk, g_lang=g_lang)
		l_length = l_length / torch.sum(x_mask)
		logw = self.dp(x, x_mask, g=g_spk, g_lang=g_lang, reverse=True, noise_scale=1.0)
		logw_ = torch.log(w + 1e-6) * x_mask

		spp_out = self._spp_prosody_block(x, attn, pitch, pitch_mel, energy_mel, y_mask, sid, g_lang, use_predicted_prosody)
		l_pitch = spp_out["l_pitch"]
		l_pitch_l1 = spp_out["l_pitch_l1"]
		l_energy = spp_out["l_energy"]
		logpitch = spp_out["logpitch"]
		logpitch_ = spp_out["logpitch_"]
		vuv_logit = spp_out["vuv_logit"]
		energy_hat = spp_out["energy_hat"]
		pitch_hat = spp_out["pitch_hat"]
		prosody = spp_out["prosody"]

		if self.mas_flow_prosody_parity and use_predicted_prosody and prosody is not None:
			z_p_for_mas = self.flow(z, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=prosody)
			attn = self._mas_attention(z_p_for_mas, m_p, logs_p, x_mask, y_mask)
			attn.requires_grad = False
			w = attn.sum(2)
			logw_ = torch.log(w + 1e-6) * x_mask
			l_length = self.dp(x, x_mask, w, g=g_spk, g_lang=g_lang)
			l_length = l_length / torch.sum(x_mask)
			logw = self.dp(x, x_mask, g=g_spk, g_lang=g_lang, reverse=True, noise_scale=1.0)
			spp_out = self._spp_prosody_block(x, attn, pitch, pitch_mel, energy_mel, y_mask, sid, g_lang, use_predicted_prosody)
			l_pitch = spp_out["l_pitch"]
			l_pitch_l1 = spp_out["l_pitch_l1"]
			l_energy = spp_out["l_energy"]
			logpitch = spp_out["logpitch"]
			logpitch_ = spp_out["logpitch_"]
			vuv_logit = spp_out["vuv_logit"]
			energy_hat = spp_out["energy_hat"]
			pitch_hat = spp_out["pitch_hat"]
			prosody = spp_out["prosody"]

		if use_predicted_prosody and prosody is not None:
			z_p = self.flow(z, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=prosody)
		else:
			z_p = z_p_for_mas

		m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # Expand prior
		logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
		z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
		o = decode_to_waveform(self.dec, z_slice, g=g_spk)
		o = o.clamp(-1.0, 1.0)

		l_qwen = None
		if self.training and self.qwen_proj is not None:
			if self.use_qwen_distill and any(p.requires_grad for p in self.qwen_proj.parameters()):
				l_qwen = self.qwen_distill_loss(z, qwen_feat, qwen_lengths, y_mask)
			if l_qwen is None:
				l_qwen = self.qwen_proj(z[:, :, :1]).sum() * 0.0

		if self.training and self.style_encoder is not None and style_ref_mel is None:
			ref_mask = torch.ones(y.size(0), 1, 1, device=y.device, dtype=y.dtype)
			o = o + self.style_encoder(y[:, :, :1], ref_mask).sum() * 0.0

		if self.training and l_qwen is not None:  # DDP (find_unused_parameters=False): tie aux tensors into o so every submodule in the forward graph receives a grad hook through the main output.
			o = o + l_qwen * 0.0

		return (o, l_length, l_pitch, l_pitch_l1, l_energy, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (x, logw, logw_, logpitch, logpitch_, vuv_logit, energy_hat, l_qwen))

	def infer(self, x, x_lengths, sid=None, lid=None, noise_scale=0.5, length_scale=1, noise_scale_w=0.5, noise_scale_p=1.0, min_period_duration=24.0, style_ref_mel=None, style_ref_lengths=None, prosody_overrides=None, duration_noise=None, pitch_noise=None, latent_noise=None):
		"""Inference method relying on the model's native duration predictions, with an added rule to enforce a minimum duration strictly for periods (.)."""
		skip_autocast = torch.jit.is_tracing() or torch.jit.is_scripting() or torch.compiler.is_compiling()
		autocast_ctx = nullcontext() if skip_autocast else torch.autocast(device_type=x.device.type, enabled=True)
		with autocast_ctx:
			if sid is not None and not torch.is_tensor(sid):
				sid = torch.LongTensor([sid]).to(x.device)
			if lid is not None and not torch.is_tensor(lid):
				lid = torch.LongTensor([lid]).to(x.device)
			g_spk = self._global_speaker_cond(sid)
			g_lang = self._global_language_cond(lid)
			x_tokens = x  # Store original token IDs before encoding to identify periods
			x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g_spk, lid=lid)  # Text Encoder

			logw = self.dp(x, x_mask, g=g_spk, g_lang=g_lang, reverse=True, noise_scale=noise_scale_w, dur_noise=duration_noise)  # Duration Prediction. The noise_scale_w controls the stochasticity of the duration (rhythm)

			w = torch.exp(logw) * x_mask * length_scale  # Convert log duration to linear time and apply global speed (length_scale)

			if '.' in _symbol_to_id:
				period_id = _symbol_to_id['.']
				period_mask = (x_tokens == period_id).unsqueeze(1)
				min_pd = torch.tensor(min_period_duration, dtype=w.dtype, device=w.device)
				w = torch.where(period_mask, torch.maximum(w, min_pd), w)

			w_ceil = torch.ceil(w)  # Ceiling ensures every token has at least 1 frame of duration if w > 0

			w_sum = torch.sum(w_ceil, [1, 2])  # Generate Alignment Path
			min_len = torch.tensor(1.0, dtype=w_sum.dtype, device=w_sum.device)
			y_lengths = torch.maximum(w_sum, min_len).long()
			mel_export_cap = None
			if torch.compiler.is_compiling():
				if pitch_noise is not None:
					mel_export_cap = pitch_noise.shape[-1]
				elif latent_noise is not None:
					mel_export_cap = latent_noise.shape[-1]
			y_mask = torch.unsqueeze(sequence_mask(y_lengths, mel_export_cap), 1).to(x_mask.dtype)
			attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
			attn = generate_path(w_ceil, attn_mask)

			m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # Expand Prior to aligned timeline
			logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

			x_aligned = utils.align_hidden_to_mel(x, attn)
			g_pitch = self._speaker_pitch_cond(sid)
			pitch_hat = None
			energy_hat = None
			pitch_raw = None
			vuv_logit = None
			if self.use_spp:
				pitch_raw = self.spp(x_aligned, y_mask, g=g_pitch, g_lang=g_lang, reverse=True, noise_scale=noise_scale_p, pitch_noise=pitch_noise)
				vuv_logit = self.vuv_pred(torch.detach(x_aligned), y_mask, g=g_pitch, g_lang=g_lang)
				if hasattr(self, "energy_pred"):
					energy_hat = self.energy_pred(x_aligned, y_mask, g=g_pitch, g_lang=g_lang)
				pitch_hat, energy_hat, _ = utils.smooth_prosody_channels(pitch_raw, energy_hat, y_mask, vuv_logit=vuv_logit, pitch_raw=pitch_raw, cfg=self.prosody_cfg, overrides=prosody_overrides)

			prosody = self._build_prosody_cond(pitch_hat, energy_hat, y_mask)
			g_style = self._encode_style(style_ref_mel, style_ref_lengths)

			noise = _infer_noise_like(m_p, noise=latent_noise)  # Latent Variable Inference (Flow Reverse)
			z_p = m_p + noise * torch.exp(logs_p) * noise_scale
			z = self.flow(z_p, y_mask, g=g_spk, g_lang=g_lang, g_style=g_style, prosody=prosody, reverse=True)

			o = decode_to_waveform(self.dec, z * y_mask, g=g_spk)  # Waveform Decoding

		return o, attn, y_mask, (z, z_p, m_p, logs_p), w_ceil, pitch_hat

	def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, noise_scale=0.0):
		"""noise_scale: Defaults to 0.0. Standard VITS uses random sampling (noise). For VC, we want the exact content, so we use the mean (0 noise)."""
		assert self.n_speakers > 0, "n_speakers have to be larger than 0."
		g_src = self.emb_g(sid_src).unsqueeze(-1)
		g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
		_, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)  # Extract Posterior (Audio -> Latent). We ignore the 'z' returned by enc_q because it has random noise baked in. We grab 'm_q' (mean) and 'logs_q' (variance) instead
		z_clean = (m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale) * y_mask  # Create Clean Latent. If noise_scale is 0, z = m_q (The most accurate representation of the audio)
		z_p = self.flow(z_clean, y_mask, g=g_src)  # Source -> Prior (Remove Source Speaker traits)
		z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)  # Prior -> Target (Add Target Speaker traits)
		o_hat = decode_to_waveform(self.dec, z_hat * y_mask, g=g_tgt)
		return o_hat, y_mask, (z_clean, z_p, z_hat)


def export_max_mel_frames(max_phonemes=500, frames_per_phoneme=32):
	"""Upper bound on mel frames baked into Core AI export (drives peak RAM at inference)."""
	return max_phonemes * frames_per_phoneme


def _coreml_compute_units(compute_unit="auto"):
	import coremltools as ct
	u = (compute_unit or "auto").lower()
	if u == "cpu":
		return ct.ComputeUnit.CPU_ONLY
	if u in {"ane", "neural_engine", "neural-engine", "all"}:  # ANE/ALL is opt-in only: on many Macs ANECCompile() fails and ComputeUnit.ALL wastes minutes attempting (and failing) the ANE compile before falling back to GPU.
		return ct.ComputeUnit.ALL
	return ct.ComputeUnit.CPU_AND_GPU  # auto/default/gpu/mps -> GPU (fast Metal compile, reliable).


def _coreml_uses_all_units(compute_unit="auto"):
	u = (compute_unit or "auto").lower()
	return u in {"ane", "neural_engine", "neural-engine", "all"}


def _looks_like_coreml_buckets(path):
	"""A bucketed Core ML model is a directory (not an .mlpackage) holding manifest.json."""
	try:
		p = str(path).rstrip("/")
		return os.path.isdir(p) and not p.lower().endswith(".mlpackage") and os.path.exists(os.path.join(p, "manifest.json"))
	except Exception:
		return False


def _is_coreml_bucket_handle(model):
	return isinstance(model, dict) and model.get("backend") == "coreml_buckets"


def _load_coreml_buckets(dir_path, compute_unit="auto", instances=1):
	"""Load a multi-width Core ML bundle. Models are loaded lazily per bucket on first use.
	instances: copies of each bucket's MLModel to keep in the pool. CoreML serializes predict() per MLModel instance, so >1 instance lets the GPU overlap concurrent chunks (round-robined).
	"""
	import json
	d = Path(dir_path)
	manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
	widths = sorted(int(w) for w in manifest["widths"])
	files = {int(k): str(d / v) for k, v in manifest["files"].items()}
	print(f"Loading Core ML buckets {widths} (lazy per-width; compute_unit={compute_unit}; instances={instances})...")
	return {"backend": "coreml_buckets", "dir": str(d), "widths": widths, "files": files, "pool": {}, "instances": max(1, int(instances)), "rr": {}, "rr_lock": threading.Lock(), "compute_unit": compute_unit, "lock": threading.Lock(), "gst_baked": bool(manifest.get("gst_baked", True))}


def _coreml_pick_width(widths, n):
	"""Smallest bucket width that fits n tokens (else the largest available)."""
	for w in widths:
		if n <= w:
			return w
	return widths[-1]


def _coreml_ensure_instances(handle, width, n, show_progress=False):
	"""Lazily load up to n MLModel instances for one bucket width. Thread-safe."""
	with handle["lock"]:
		pool = handle["pool"].setdefault(width, [])
		while len(pool) < n:
			t0 = time.time()
			if show_progress:
				units = _coreml_compute_units(handle["compute_unit"]).name
				print(f"Loading Core ML bucket b{width} instance {len(pool) + 1}/{n} ({units})...")
			m = load_model(None, handle["files"][width], backend="coreml", compute_unit=handle["compute_unit"])
			if show_progress:
				print(f"  bucket b{width} instance {len(pool) + 1}/{n} ready ({time.time() - t0:.1f}s)")
			pool.append(m)
	return handle["pool"][width]


def _coreml_next_instance(handle, width, show_progress=False):
	"""Round-robin an MLModel instance for a width (loads at least one on demand)."""
	pool = handle["pool"].get(width)
	if not pool:
		pool = _coreml_ensure_instances(handle, width, 1, show_progress=show_progress)
	if len(pool) == 1:
		return pool[0]
	with handle["rr_lock"]:
		c = handle["rr"].get(width, 0)
		handle["rr"][width] = c + 1
	return pool[c % len(pool)]


def _run_coreml_buckets(handle, seq, sid, lid, use_gst=False, style_mel_np=None, style_mel_lengths_np=None, trim=True, show_progress=False):
	"""Route one token sequence to the smallest fitting bucket, retrying a larger one if durations saturate the baked audio buffer."""
	widths = handle["widths"]
	max_w = widths[-1]
	if len(seq) > max_w:
		parts = [_run_coreml_buckets(handle, s, sid, lid, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, trim=trim, show_progress=show_progress) for s in split_phoneme_sequences(seq, max_w)]
		return _finalize_audio(np.concatenate(parts) if parts else np.array([], dtype=np.float32))
	gst_for_inputs = bool(use_gst and not handle.get("gst_baked", True))
	start = _coreml_pick_width(widths, len(seq))
	last_chunk = np.array([], dtype=np.float32)
	for w in [x for x in widths if x >= start]:
		mdl = _coreml_next_instance(handle, w, show_progress=show_progress)
		outputs = mdl.predict(build_predict_inputs(seq, sid, lid, backend="coreml", use_gst=gst_for_inputs, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=w))
		chunk = np.squeeze(outputs["audio_output"])
		buf = chunk.size
		last_chunk = chunk
		if not trim:
			return _finalize_audio(chunk)
		n = int(np.asarray(outputs["audio_length"]).reshape(-1)[0]) if "audio_length" in outputs else 0
		if n <= 0:
			return _finalize_audio(_trim_coreml_audio(chunk, phoneme_count=len(seq)))
		if n >= buf and w != max_w:
			continue  # baked buffer saturated -> true audio was truncated; retry a larger bucket
		return _finalize_audio(chunk[:min(n, buf)])
	return _finalize_audio(last_chunk)


def load_model(config_path, model_path, device="mps", backend="pytorch", compute_unit="auto"):
	"""Load a checkpoint for inference: PyTorch .pth, Core ML .mlpackage, Core AI .aimodel, or ONNX."""
	if (backend := _normalize_backend(backend)) == "pytorch":
		hps = utils.get_hparams_from_file(config_path)
		net_g = SynthesizerTrn(len(symbols), hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, n_languages=getattr(hps.data, "n_languages", 0), **utils.synthesizer_kwargs(hps)).to(device)
		utils.load_checkpoint(model_path, net_g, None)
		net_g.eval()
		net_g.hps = hps
		return net_g
	if backend == "coreml":
		if _looks_like_coreml_buckets(model_path):
			return _load_coreml_buckets(model_path, compute_unit)
		import coremltools as ct
		units = _coreml_compute_units(compute_unit)
		t0 = time.time()
		if _coreml_uses_all_units(compute_unit):
			print(f"Loading Core ML ({units.name}) — ANE compile on first load may take several minutes...")
		else:
			print(f"Loading Core ML ({units.name})...")
		m = ct.models.MLModel(model_path, compute_units=units)
		print(f"Core ML ready ({time.time() - t0:.1f}s)")
		return m
	if backend == "coreai":
		fn, loop, cu = _load_coreai_runtime(model_path, compute_unit)
		return {"backend": "coreai", "fn": fn, "loop": loop, "compute_unit": cu}
	if backend == "onnx":
		import onnxruntime as ort
		return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
	raise ValueError(f"Unsupported backend: {backend}")


def _normalize_backend(backend):
	b = (backend or "pytorch").lower()
	return "pytorch" if b in {"torch", "pytorch"} else "coreai" if b in {"ane", "neural_engine", "neural-engine"} else b


def _resolve_inference_backend(backend=None, coreml=False):
	return "coreml" if coreml else _normalize_backend(backend or "pytorch")


def sample_export_infer_noise(max_phonemes=500, max_mel=EXPORT_MAX_MEL_FRAMES, latent_channels=192, rng=None, deterministic=False, seed=8):
	"""Sample external infer noise buffers for exported Core AI models."""
	r = np.random.default_rng(seed) if deterministic else (rng or np.random.default_rng())
	return {"duration_noise": r.standard_normal((1, 2, max_phonemes), dtype=np.float32), "pitch_noise": r.standard_normal((1, 2, max_mel), dtype=np.float32), "latent_noise": r.standard_normal((1, latent_channels, max_mel), dtype=np.float32)}


def _ensure_coreai_session(model, max_phonemes, max_mel_frames, use_gst, style_mel_np, style_mel_lengths_np, n_mel_channels=128, style_mel_max_frames=1000):
	"""Reuse Core AI noise/style buffers across chunks (avoids ~13MB alloc per forward)."""
	if not isinstance(model, dict) or model.get("backend") != "coreai":
		return None
	key = (max_phonemes, max_mel_frames, use_gst, id(style_mel_np) if style_mel_np is not None else None)
	if model.get("_coreai_sess_key") == key:
		return model["_coreai_sess"]
	from coreai.runtime import NDArray
	n = sample_export_infer_noise(max_phonemes=max_phonemes, max_mel=max_mel_frames)
	sess = {"duration_noise": n["duration_noise"], "pitch_noise": n["pitch_noise"], "latent_noise": n["latent_noise"], "rng": np.random.default_rng()}
	if use_gst and style_mel_np is not None:
		sess["style_mel_nd"] = NDArray(style_mel_np.astype(np.float32, copy=False))
		sess["style_len_nd"] = NDArray(np.asarray(style_mel_lengths_np, dtype=np.int32).reshape(1))
	else:
		sess["style_mel_nd"] = NDArray(np.zeros((1, n_mel_channels, style_mel_max_frames), dtype=np.float32))
		sess["style_len_nd"] = NDArray(np.array([0], dtype=np.int32))
	model["_coreai_sess_key"] = key
	model["_coreai_sess"] = sess
	return sess


def _refresh_coreai_noise(sess, deterministic_noise=False, seed=8):
	if deterministic_noise:
		n = sample_export_infer_noise(max_phonemes=sess["duration_noise"].shape[-1], max_mel=sess["pitch_noise"].shape[-1], latent_channels=sess["latent_noise"].shape[1], deterministic=True, seed=seed)
		np.copyto(sess["duration_noise"], n["duration_noise"])
		np.copyto(sess["pitch_noise"], n["pitch_noise"])
		np.copyto(sess["latent_noise"], n["latent_noise"])
	else:
		r = sess["rng"]
		np.copyto(sess["duration_noise"], r.standard_normal(sess["duration_noise"].shape, dtype=np.float32))
		np.copyto(sess["pitch_noise"], r.standard_normal(sess["pitch_noise"].shape, dtype=np.float32))
		np.copyto(sess["latent_noise"], r.standard_normal(sess["latent_noise"].shape, dtype=np.float32))


def make_export_noise_tensors(max_phonemes=500, max_mel=EXPORT_MAX_MEL_FRAMES, latent_channels=192, seed=8):
	"""Fixed noise tensors for torch.export trace."""
	n = sample_export_infer_noise(max_phonemes, max_mel, latent_channels, deterministic=True, seed=seed)
	return (torch.from_numpy(n["duration_noise"]), torch.from_numpy(n["pitch_noise"]), torch.from_numpy(n["latent_noise"]))


def pad_phoneme_sequence(seq, max_phonemes=500):
	"""Pad or truncate a phoneme id sequence to a fixed width for static Core AI export."""
	return list(seq[:(ln := min(len(seq), max_phonemes))]) + [0] * (max_phonemes - ln), ln


def split_phoneme_sequences(seq, max_phonemes=500):
	"""Split a phoneme sequence into chunks that fit the exported static width."""
	return [seq[i:i + max_phonemes] for i in range(0, len(seq), max_phonemes)] if seq else []


def _coreai_specialization_options(compute_unit="auto"):
	from coreai.runtime import ComputeUnitKind, SpecializationOptions
	if (u := compute_unit.lower()) in {"auto", "default"}: return None
	if u == "cpu": return SpecializationOptions.cpu_only()
	if u == "gpu": return SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
	if u in {"ane", "neural_engine", "neural-engine"}: return SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
	raise ValueError(f"Unsupported compute unit: {compute_unit}")


def pad_style_mel(style_mel, style_lengths, max_frames=1000):
	"""Pad mel spectrogram to a fixed width for exported runtimes (returns float32 numpy)."""
	mel = style_mel.detach().cpu().float() if torch.is_tensor(style_mel) else torch.from_numpy(np.asarray(style_mel, dtype=np.float32))
	ln = max(0, min(int((style_lengths.reshape(-1)[0].item() if torch.is_tensor(style_lengths) else np.asarray(style_lengths).reshape(-1)[0])), max_frames))
	padded = torch.zeros(1, mel.size(1), max_frames, dtype=torch.float32)
	if ln > 0: padded[:, :, :ln] = mel[:, :, :ln]
	return padded.numpy(), np.array([ln], dtype=np.int32)


def load_style_ref_mel(style_ref_wav, hps, device):
	"""Load reference wav as mel for optional GST style conditioning at inference."""
	audio, sr = utils.load_wav_to_torch(style_ref_wav)
	if sr != hps.data.sampling_rate: raise ValueError(f"{sr} SR doesn't match target {hps.data.sampling_rate} SR")
	spec = utils.mel_spectrogram_torch((audio / hps.data.max_wav_value).unsqueeze(0), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax, center=False).to(device)
	return spec, torch.LongTensor([spec.size(2)]).to(device)


def load_style_ref_for_export(style_ref_wav, config_path, max_frames=1000):
	"""Load a reference wav and return padded style mel inputs for exported inference."""
	mel, ln = load_style_ref_mel(style_ref_wav, utils.get_hparams_from_file(config_path), device="cpu")
	return pad_style_mel(mel, ln, max_frames=max_frames)


def build_predict_inputs(seq, sid, lid, backend="coreml", use_gst=False, style_mel_np=None, style_mel_lengths_np=None, max_phonemes=None, max_mel_frames=EXPORT_MAX_MEL_FRAMES, infer_noise=None, deterministic_noise=False, n_mel_channels=128, style_mel_max_frames=1000, coreai_session=None):
	"""Build predict inputs for Core ML, Core AI, or ONNX runtimes."""
	actual_len = len(seq)
	if (backend := _normalize_backend(backend)) in {"coreai", "coreml"} and max_phonemes is not None:
		if actual_len > max_phonemes: raise ValueError(f"phoneme sequence length {actual_len} exceeds max {max_phonemes}")
		seq, _ = pad_phoneme_sequence(seq, max_phonemes=max_phonemes)
	inputs = {"x": np.array([seq], dtype=np.int32), "x_lengths": np.array([actual_len], dtype=np.int32), "sid": np.array([int(sid)], dtype=np.int32), "lid": np.array([int(lid)], dtype=np.int32)}
	if backend == "coreai":
		if coreai_session:
			_refresh_coreai_noise(coreai_session, deterministic_noise=deterministic_noise)
			from coreai.runtime import NDArray
			return {"x": NDArray(inputs["x"]), "x_lengths": NDArray(inputs["x_lengths"]), "sid": NDArray(inputs["sid"]), "lid": NDArray(inputs["lid"]), "duration_noise": NDArray(coreai_session["duration_noise"]), "pitch_noise": NDArray(coreai_session["pitch_noise"]), "latent_noise": NDArray(coreai_session["latent_noise"]), "style_mel": coreai_session["style_mel_nd"], "style_mel_lengths": coreai_session["style_len_nd"]}
		n = infer_noise or sample_export_infer_noise(max_phonemes=max_phonemes or 500, max_mel=max_mel_frames, deterministic=deterministic_noise)
		inputs.update({"duration_noise": n["duration_noise"], "pitch_noise": n["pitch_noise"], "latent_noise": n["latent_noise"]})
	if use_gst:
		if style_mel_np is None or style_mel_lengths_np is None: raise ValueError("GST exported models require style_mel_np and style_mel_lengths_np")
		inputs.update({"style_mel": style_mel_np.astype(np.float32, copy=False), "style_mel_lengths": np.asarray(style_mel_lengths_np, dtype=np.int32).reshape(1)})
	elif backend == "coreai" and not coreai_session:
		inputs.update({"style_mel": np.zeros((1, n_mel_channels, style_mel_max_frames), dtype=np.float32), "style_mel_lengths": np.array([0], dtype=np.int32)})
	if backend == "coreai" and not coreai_session:
		from coreai.runtime import NDArray
		return {k: NDArray(v) for k, v in inputs.items()}
	return inputs


def _trim_exported_audio(audio, phoneme_count=None, hop_length=256, frames_per_phoneme=32, pad_samples=1024):
	"""Safety trim for legacy .aimodel bundles that still emit padded tail audio."""
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	if phoneme_count and phoneme_count > 0 and audio.size > (cap := int(phoneme_count * frames_per_phoneme * hop_length) + pad_samples): audio = audio[:cap]
	if (idx := np.where(np.abs(audio) > 1e-3)[0]).size == 0: return audio
	return audio[max(0, int(idx[0]) - pad_samples):min(len(audio), int(idx[-1]) + 1 + pad_samples)]


def _trim_coreml_audio(audio, phoneme_count, hop_length=256, frames_per_phoneme=32, pad_samples=2048):
	"""Trim Core ML output to estimated length from phoneme count (no energy gate)."""
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	if not phoneme_count or phoneme_count <= 0:
		return audio
	cap = int(phoneme_count * frames_per_phoneme * hop_length) + pad_samples
	return audio[:min(len(audio), cap)]


def _finalize_audio(audio, ceiling=0.99):
	"""Native model output — no boost. Scale down only if peak exceeds ceiling (avoids hard clip on write/play)."""
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	if audio.size == 0:
		return audio
	peak = float(np.max(np.abs(audio)))
	if peak > ceiling:
		audio = audio * (ceiling / peak)
	return audio


def _warn_coreai_model_path(model_path):
	p = os.path.abspath(str(model_path))
	if "Mobile Documents/com~apple~CloudDocs" in p or "/CloudDocs/" in p:
		raise ValueError(f"Core AI cannot load models from iCloud Drive ({p}). "
		                 "Copy the .aimodel bundle to a local path (e.g. ~/Models/YunaVITS3.aimodel) and point model_path there.")


async def _load_coreai_fn(model_path, compute_unit="auto"):
	from coreai.runtime import AIModel
	_warn_coreai_model_path(model_path)
	u = (compute_unit or "auto").lower()
	if u in {"auto", "default"}:
		candidates = ["ane", None]
	elif u == "cpu":
		candidates = ["cpu"]
	else:
		candidates = [u, "ane", None]
	last_err = None
	for unit in candidates:
		label = unit or "auto"
		spec = _coreai_specialization_options(unit) if unit else None
		try:
			model = await AIModel.load(str(model_path), specialization_options=spec) if spec else await AIModel.load(str(model_path))
			fn = model.load_function("main")
			print(f"Core AI loaded (compute_unit={label})")
			return fn, label
		except RuntimeError as e:
			last_err = e
			if unit != candidates[-1]:
				print(f"Core AI {label} failed ({e}); trying next backend.")
	raise RuntimeError(f"Core AI failed to load. Try clearing ANE cache: rm -rf ~/Library/Caches/com.apple.coreml ~/Library/Caches/com.apple.aned\nLast error: {last_err}")


def _load_coreai_runtime(model_path, compute_unit="auto"):
	import asyncio
	loop = asyncio.new_event_loop()
	fn, label = loop.run_until_complete(_load_coreai_fn(model_path, compute_unit))
	return fn, loop, label


def _coreai_await(loop, coro):
	import asyncio
	if loop.is_running():
		return asyncio.run_coroutine_threadsafe(coro, loop).result()
	return loop.run_until_complete(coro)


def _run_exported_forward(model, backend, seq, sid, lid, use_gst=False, style_mel_np=None, style_mel_lengths_np=None, max_phonemes=500, max_mel_frames=EXPORT_MAX_MEL_FRAMES, deterministic_noise=False, trim=True, dynamic_phoneme_cap=COREML_DYNAMIC_MAX_PHONEMES):
	"""Run one phoneme sequence through a Core ML / Core AI / ONNX runtime."""
	if _is_coreml_bucket_handle(model):
		return _run_coreml_buckets(model, seq, sid, lid, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, trim=trim)
	backend = _normalize_backend(backend)
	if backend == "coreml" and max_phonemes is None and len(seq) > dynamic_phoneme_cap:
		parts = [_run_exported_forward(model, backend, s_part, sid, lid, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=None, max_mel_frames=max_mel_frames, deterministic_noise=deterministic_noise, trim=trim, dynamic_phoneme_cap=dynamic_phoneme_cap) for s_part in split_phoneme_sequences(seq, dynamic_phoneme_cap)]
		return _finalize_audio(np.concatenate(parts) if parts else np.array([], dtype=np.float32))
	if backend == "coreml":
		outputs = model.predict(build_predict_inputs(seq, sid, lid, backend=backend, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=max_phonemes))
		chunk = np.squeeze(outputs["audio_output"])
		if trim and "audio_length" in outputs:
			n = int(np.asarray(outputs["audio_length"]).reshape(-1)[0])
			if 0 < n < chunk.size:
				chunk = chunk[:n]
		elif trim:
			chunk = _trim_coreml_audio(chunk, phoneme_count=len(seq))
		return _finalize_audio(chunk)
	if backend == "onnx":
		return _finalize_audio(np.squeeze(model.run(None, build_predict_inputs(seq, sid, lid, backend=backend, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np))[0]))
	if backend != "coreai": raise ValueError(f"Unsupported exported backend: {backend}")
	fn, loop, parts = model["fn"], model["loop"], []
	coreai_sess = _ensure_coreai_session(model, max_phonemes, max_mel_frames, use_gst, style_mel_np, style_mel_lengths_np)
	for s_part in split_phoneme_sequences(seq, max_phonemes):
		inputs = build_predict_inputs(s_part, sid, lid, backend=backend, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=max_phonemes, max_mel_frames=max_mel_frames, deterministic_noise=deterministic_noise, coreai_session=coreai_sess)
		outputs = _coreai_await(loop, fn(inputs=inputs))
		chunk = np.squeeze(outputs["audio"].numpy())
		if trim and "audio_length" in outputs:
			n = int(np.asarray(outputs["audio_length"].numpy()).reshape(-1)[0])
			if 0 < n < chunk.size: chunk = chunk[:n]
		elif trim: chunk = _trim_exported_audio(chunk, phoneme_count=len(s_part))
		parts.append(chunk)
		del outputs, inputs
	out = parts[0] if len(parts) == 1 else np.concatenate(parts) if parts else np.array([], dtype=np.float32)
	return _finalize_audio(out)


class _ExportInferWrapper(nn.Module):
	def __init__(self, model, infer_cfg, use_gst=False, bake_gst=False, style_mel_np=None, style_mel_lengths_np=None, include_noise=False, return_audio_length=False, style_mel_max_frames=1000):
		super().__init__()
		self.model, self.use_gst, self.bake_gst, self.include_noise, self.return_audio_length, self.style_mel_max_frames = model, use_gst or bake_gst, bake_gst, include_noise, return_audio_length, style_mel_max_frames
		for k, v in infer_cfg.items():
			self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
		if return_audio_length: self.register_buffer("hop_length", torch.tensor(model.hop_length, dtype=torch.int64))
		if bake_gst:
			if style_mel_np is None or style_mel_lengths_np is None: raise ValueError("bake_gst requires style_mel_np and style_mel_lengths_np")
			self.register_buffer("_style_mel", torch.from_numpy(np.asarray(style_mel_np, dtype=np.float32)))
			self.register_buffer("_style_mel_lengths", torch.from_numpy(np.asarray(style_mel_lengths_np, dtype=np.int64).reshape(-1)))

	def forward(self, x, x_lengths, sid, lid, style_mel=None, style_mel_lengths=None, duration_noise=None, pitch_noise=None, latent_noise=None):
		if self.bake_gst:
			style_ref_mel, style_ref_lengths = self._style_mel, self._style_mel_lengths
		elif self.use_gst:
			style_ref_mel, style_ref_lengths = style_mel, style_mel_lengths
		else:
			style_ref_mel = style_ref_lengths = None
		infer_kwargs = {"noise_scale": self.noise_scale, "length_scale": self.length_scale, "noise_scale_w": self.noise_scale_w, "noise_scale_p": self.noise_scale_p, "min_period_duration": self.min_period_duration, "style_ref_mel": style_ref_mel, "style_ref_lengths": style_ref_lengths}
		if self.include_noise: infer_kwargs.update(duration_noise=duration_noise, pitch_noise=pitch_noise, latent_noise=latent_noise)
		audio, _, _, _, w_ceil, _ = self.model.infer(x, x_lengths, sid=sid, lid=lid, **infer_kwargs)
		return (audio, torch.sum(w_ceil, dim=[1, 2]).long().clamp(min=1) * self.hop_length) if self.return_audio_length else audio


@contextlib.contextmanager
def _export_trace_mode():
	s_auto, s_trace = torch.autocast, torch.jit.is_tracing
	torch.autocast, torch.jit.is_tracing = lambda *a, **kw: nullcontext(), lambda: True
	try:
		yield
	finally:
		torch.autocast, torch.jit.is_tracing = s_auto, s_trace


def _load_export_checkpoint(config_path, model_path, device="cpu"):
	m = load_model(config_path, model_path, device=device, backend="pytorch")
	return utils.get_hparams_from_file(config_path), m.float()


def _resolve_export_style_inputs(hps, model_g, style_ref_wav, style_mel_max_frames):
	if style_ref_wav:
		print(f"Using GST style reference: {style_ref_wav}")
		mel, ln = load_style_ref_mel(style_ref_wav, hps, device="cpu")
	else:
		print("No style_ref_wav provided; tracing with silent GST reference.")
		mel, ln = torch.zeros(1, hps.data.n_mel_channels, style_mel_max_frames), torch.LongTensor([1])
	return pad_style_mel(mel, ln, max_frames=style_mel_max_frames)


def _infer_export_format(output_path):
	p = str(output_path).rstrip("/").lower()
	if p.endswith(".mlpackage"):
		return "coreml"
	if p.endswith(".aimodel"):
		return "coreai"
	if p.endswith(".onnx"):
		return "onnx"
	return "coreai"


def _default_use_gst(backend, style_ref_wav):
	if backend == "coreai":
		return True
	return False


def _resolve_max_phonemes(backend, max_phonemes):
	"""Core ML MS-iSTFT needs fixed phoneme width; Core AI uses static export width."""
	if max_phonemes is not None:
		return max_phonemes
	if backend in {"coreai", "coreml"}:
		return 500
	return None


def export_model(config_path, model_path, output_path, format=None, device="cpu", style_ref_wav=None, max_phonemes=500, max_mel_frames=None, style_mel_max_frames=1000, dynamic_phonemes=False, bake_gst=True, compress=False, buckets=None):
	"""Export a VITS checkpoint to onnx, coreml, or coreai.
	buckets: for coreml, export a ladder of length-bucketed graphs into output_path (a directory). Pass True for the default ladder, or a list of token widths (e.g. [128, 256, 384, 512]).
	"""
	f = (format or _infer_export_format(output_path)).lower()
	if format and format.lower() != _infer_export_format(output_path):
		print(f"Note: format={f} (output suffix is {_infer_export_format(output_path)})")
	if f not in ("onnx", "coreml", "coreai"):
		raise ValueError(f"Unsupported export format: {f}. Choose from onnx, coreml, coreai")
	if f == "onnx":
		return _export_onnx(config_path, model_path, output_path, device=device)
	if f == "coreml":
		if buckets:
			return _export_coreml_buckets(config_path, model_path, output_path, style_ref_wav=style_ref_wav, style_mel_max_frames=style_mel_max_frames, compress=compress, widths=(None if buckets is True else buckets), bake_gst=bake_gst)
		return _export_coreml(config_path, model_path, output_path, style_ref_wav=style_ref_wav, style_mel_max_frames=style_mel_max_frames, compress=compress, max_phonemes=max_phonemes, dynamic_phonemes=dynamic_phonemes, bake_gst=bake_gst)
	return _export_coreai(config_path, model_path, output_path, style_ref_wav=style_ref_wav, max_phonemes=max_phonemes, max_mel_frames=max_mel_frames, style_mel_max_frames=style_mel_max_frames, dynamic_phonemes=dynamic_phonemes)


def _export_onnx(config_path, model_path, output_path, device="cpu"):
	print("Loading model...")
	hps, model_g = _load_export_checkpoint(config_path, model_path, device=device)
	model_g.forward = model_g.infer
	seq = torch.randint(low=0, high=len(symbols), size=(1, 50), dtype=torch.long).to(device)
	out_f = Path(output_path)
	out_f.parent.mkdir(parents=True, exist_ok=True)
	temp_path = out_f.with_suffix(".onnx")
	torch.onnx.export(model=model_g, args=(seq, torch.LongTensor([seq.size(1)]).to(device)), f=str(temp_path), verbose=False, opset_version=17, input_names=["x", "x_lengths"], output_names=["output", "attn", "y_mask", "internals"], dynamic_axes={"x": {0: "batch_size", 1: "phonemes"}, "x_lengths": {0: "batch_size"}, "output": {0: "batch_size", 2: "time"}}, kwargs={"sid": torch.LongTensor([0]).to(device) if hps.data.n_speakers > 0 else None, "noise_scale": 0.4, "length_scale": 1.0, "noise_scale_w": 0.4})
	print(f"ONNX export complete: {temp_path}")


def _export_coreai(config_path, model_path, output_path, style_ref_wav=None, max_phonemes=500, max_mel_frames=None, style_mel_max_frames=1000, dynamic_phonemes=False):
	max_mel_frames = max_mel_frames or export_max_mel_frames(max_phonemes)
	print(f"Core AI mel cap: {max_mel_frames} frames (pass the same max_mel_frames to inference(backend='coreai'))")
	from coreai_torch import TorchConverter, get_decomp_table
	hps, model_g = _load_export_checkpoint(config_path, model_path, device="cpu")
	wrapped = _ExportInferWrapper(model_g, utils.resolve_inference_cfg(getattr(hps, "inference", None)), use_gst=True, include_noise=True, return_audio_length=True, style_mel_max_frames=style_mel_max_frames).eval()
	mel_np, len_np = _resolve_export_style_inputs(hps, model_g, style_ref_wav, style_mel_max_frames)
	p_seq, act_len = pad_phoneme_sequence([0, 28, 0, 45, 0, 42, 0, 7, 0, 15, 0], max_phonemes=max_phonemes)
	args = (torch.tensor([p_seq], dtype=torch.long), torch.tensor([act_len], dtype=torch.long), torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long), torch.from_numpy(mel_np), torch.from_numpy(len_np).long(), *make_export_noise_tensors(max_phonemes=max_phonemes, max_mel=max_mel_frames))
	print("Sanity-check PyTorch forward...")
	with torch.no_grad(), _export_trace_mode():
		audio, a_len = wrapped(*args)
	print(f"  audio shape: {tuple(audio.shape)}  audio_length: {int(a_len.reshape(-1)[0])}  phoneme width: {max_phonemes}")
	kwargs = {"dynamic_shapes": {"x": {1: torch.export.Dim("phonemes", min=1, max=max_phonemes)}, "x_lengths": {}, "sid": {}, "lid": {}, "style_mel": {}, "style_mel_lengths": {}, "duration_noise": {}, "pitch_noise": {}, "latent_noise": {}}} if dynamic_phonemes else {}
	print("Exporting with torch.export...")
	with _export_trace_mode():
		exp_prog = torch.export.export(wrapped, args=args, **kwargs)
	exp_prog = exp_prog.run_decompositions(get_decomp_table())
	print("Converting to Core AI IR...")
	ai_prog = TorchConverter().add_exported_program(exp_prog, input_names=["x", "x_lengths", "sid", "lid", "style_mel", "style_mel_lengths", "duration_noise", "pitch_noise", "latent_noise"], output_names=["audio", "audio_length"]).to_coreai()
	ai_prog.optimize()
	out_f = Path(output_path)
	out_f.parent.mkdir(parents=True, exist_ok=True)
	if out_f.exists():
		import shutil
		shutil.rmtree(out_f) if out_f.is_dir() else out_f.unlink()
	ai_prog.save_asset(out_f)
	print(f"Success! Saved: {out_f}")


def _coreml_fill_sequence(width):
	"""Realistic dense phoneme fill of a given token width (sets the baked mel/audio buffer)."""
	from aiflow.models.yuna_speech.text import text_to_sequence
	seed = text_to_sequence("ABOUT 13.5 BILLION YEARS AGO, MATTER, energy, time and space came into being in what is known as the Big Bang.", language="en-us")
	fill = list(seed)
	while len(fill) < width:
		fill = fill + seed
	p_seq, _ = pad_phoneme_sequence(fill[:width], max_phonemes=width)
	return p_seq


def _prepare_coreml_export(config_path, model_path, style_ref_wav, style_mel_max_frames, bake_gst):
	"""Load checkpoint and build the export wrapper once (shared across bucket widths)."""
	hps, model_g = _load_export_checkpoint(config_path, model_path, device="cpu")
	use_gst = getattr(model_g, "use_gst", False)
	bake = bool(bake_gst and use_gst)
	if use_gst and bake and not style_ref_wav:
		raise ValueError("Core ML export with GST requires style_ref_wav (style is baked into the model at export time)")
	style_mel_np = style_mel_lengths_np = None
	if bake:
		style_mel_np, style_mel_lengths_np = _resolve_export_style_inputs(hps, model_g, style_ref_wav, style_mel_max_frames)
		print("Baking GST style reference into Core ML graph (no style_mel inputs at inference)")
	wrapped = _ExportInferWrapper(model_g, utils.resolve_inference_cfg(getattr(hps, "inference", None)), use_gst=use_gst, bake_gst=bake, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, return_audio_length=True, style_mel_max_frames=style_mel_max_frames).eval()
	return hps, model_g, wrapped, use_gst, bake


def _coreml_convert_fixed(hps, model_g, wrapped, use_gst, bake, style_ref_wav, style_mel_max_frames, width, compress, save_path):
	"""Trace + convert + save one fixed-width Core ML graph (ALL compute units, FP16)."""
	import coremltools as ct
	p_seq = _coreml_fill_sequence(width)
	t_args = (torch.LongTensor([p_seq]), torch.LongTensor([width]), torch.LongTensor([0]), torch.LongTensor([0]))
	if use_gst and not bake:
		m_np, l_np = _resolve_export_style_inputs(hps, model_g, style_ref_wav, style_mel_max_frames)
		t_args += (torch.from_numpy(m_np), torch.from_numpy(l_np))
	with torch.no_grad(), _export_trace_mode():
		sanity_audio, sanity_len = wrapped(*t_args)
		print(f"  width {width}: {sanity_audio.shape[-1] / 48000:.1f}s baked buffer, fill audio_length={int(sanity_len.reshape(-1)[0]) / 48000:.1f}s")
		traced = torch.jit.trace(wrapped, t_args, check_trace=False)
	in_types = [ct.TensorType(name="x", shape=(1, width)), ct.TensorType(name="x_lengths", shape=(1, )), ct.TensorType(name="sid", shape=(1, )), ct.TensorType(name="lid", shape=(1, ))]
	if use_gst and not bake:
		in_types.extend([ct.TensorType(name="style_mel", shape=(1, hps.data.n_mel_channels, style_mel_max_frames)), ct.TensorType(name="style_mel_lengths", shape=(1, ))])
	out_types = [ct.TensorType(name="audio_output"), ct.TensorType(name="audio_length")]
	try:
		mlm = ct.convert(traced, inputs=in_types, outputs=out_types, convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL, compute_precision=ct.precision.FLOAT16)
	except Exception as e:
		print(f"FP16 conversion failed ({e}); retrying in FP32...")
		mlm = ct.convert(traced, inputs=in_types, outputs=out_types, convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL, compute_precision=ct.precision.FLOAT32)
	if compress:
		import coremltools.optimize.coreml as cto
		print("  compressing weights (int8)...")
		mlm = cto.linear_quantize_weights(mlm, cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")))
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)
	mlm.save(str(save_path))
	return mlm


def _export_coreml(config_path, model_path, output_path, style_ref_wav=None, style_mel_max_frames=1000, compress=False, max_phonemes=500, dynamic_phonemes=False, bake_gst=True):
	if dynamic_phonemes:
		print("Core ML dynamic phonemes are unsupported for MS-iSTFT (conv_transpose output_shape is trace-fixed); using fixed width.")
	hps, model_g, wrapped, use_gst, bake = _prepare_coreml_export(config_path, model_path, style_ref_wav, style_mel_max_frames, bake_gst)
	print(f"Core ML fixed phoneme width: {max_phonemes} (pass max_phonemes={max_phonemes} to inference)")
	print("Converting to CoreML (fixed shape, ALL compute units, FP16)...")
	_coreml_convert_fixed(hps, model_g, wrapped, use_gst, bake, style_ref_wav, style_mel_max_frames, max_phonemes, compress, output_path)
	print(f"Success! Saved: {output_path} (fixed width {max_phonemes}, GST {'baked' if bake else 'runtime' if use_gst else 'off'})")


def _export_coreml_buckets(config_path, model_path, output_dir, style_ref_wav=None, style_mel_max_frames=1000, compress=False, widths=None, bake_gst=True):
	"""Export a ladder of fixed-width Core ML graphs into a directory + manifest.json for length-bucketed inference."""
	import json
	widths = sorted(set(int(w) for w in (widths or COREML_BUCKET_WIDTHS)))
	out_dir = Path(output_dir)
	if out_dir.suffix.lower() == ".mlpackage":
		out_dir = out_dir.with_suffix("")
	out_dir.mkdir(parents=True, exist_ok=True)
	hps, model_g, wrapped, use_gst, bake = _prepare_coreml_export(config_path, model_path, style_ref_wav, style_mel_max_frames, bake_gst)
	files = {}
	for i, w in enumerate(widths):
		print(f"=== Core ML bucket b{w} ({i + 1}/{len(widths)}) ===")
		fname = f"b{w}.mlpackage"
		_coreml_convert_fixed(hps, model_g, wrapped, use_gst, bake, style_ref_wav, style_mel_max_frames, w, compress, out_dir / fname)
		files[str(w)] = fname
	manifest = {"type": "coreml_buckets", "widths": widths, "files": files, "gst_baked": bake, "style_mel_max_frames": style_mel_max_frames}
	(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	print(f"Success! Saved {len(widths)} Core ML buckets to {out_dir} (widths={widths}, GST {'baked' if bake else 'runtime' if use_gst else 'off'})")
	return str(out_dir)


def compress_coreml_model(modelpath="vits.mlpackage", output_path="vits-int8.mlpackage"):
	"""Optional int8 weight compression for an existing Core ML model."""
	import coremltools as ct
	import coremltools.optimize.coreml as cto
	model = ct.models.MLModel(modelpath)
	print("Model loaded. Starting compression...")
	op_config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"))
	compressed_model = cto.linear_quantize_weights(model, op_config)
	compressed_model.save(output_path)
	print(f"Success! Compressed model saved as: {output_path}")
	return output_path


def inference(model=None, text=None, sid=0, noise_scale=None, noise_scale_w=None, noise_scale_p=None, length_scale=None, min_period_duration=None, device="mps", stream=False, output_file=None, language="en-us", language_map=None, languages=None, language_speakers=None, backend=None, coreml=False, use_gst=None, coreml_use_gst=None, style_ref_wav=None, style_ref_mel=None, style_ref_lengths=None, style_mel_np=None, style_mel_lengths_np=None, coreml_style_mel_np=None, coreml_style_mel_lengths_np=None, config_path=None, prosody_overrides=None, max_phonemes=None, max_mel_frames=EXPORT_MAX_MEL_FRAMES, deterministic_noise=False, chunk_pause_sec=0, max_text_chunk=300, show_progress=True):
	"""Synthesize speech with pytorch, coreml, coreai, or onnx backends."""
	backend = _resolve_inference_backend(backend, coreml=coreml)
	exported = backend in {"coreml", "coreai", "onnx"}
	max_phonemes = _resolve_max_phonemes(backend, max_phonemes)
	use_gst = coreml_use_gst if use_gst is None and coreml_use_gst is not None else _default_use_gst(backend, style_ref_wav) if use_gst is None else use_gst
	style_mel_np, style_mel_lengths_np = style_mel_np or coreml_style_mel_np, style_mel_lengths_np or coreml_style_mel_lengths_np
	language_map, language_speakers = language_map or {"en-us": "en-us", "ru": "ru", "ja": "ja"}, language_speakers or {"en-us": sid, "ru": sid, "ja": sid}
	hps = getattr(model, "hps", None) or (utils.get_hparams_from_file(config_path) if config_path else None)
	if languages is None and hps: languages = getattr(hps.data, "languages", None)
	infer_cfg = utils.resolve_inference_cfg(getattr(hps, "inference", None) if hps else None)
	noise_scale, noise_scale_w, noise_scale_p, length_scale, min_period_duration = [v if v is not None else infer_cfg[k] for k, v in zip(["noise_scale", "noise_scale_w", "noise_scale_p", "length_scale", "min_period_duration"], [noise_scale, noise_scale_w, noise_scale_p, length_scale, min_period_duration])]
	processed_chunks = split_and_process_text(text, language=language, max_length=max_text_chunk, combine=True, language_map=language_map)
	c_style_mel, c_style_lengths = style_ref_mel, style_ref_lengths
	if not exported and c_style_mel is None and style_ref_wav is not None and getattr(model, "use_gst", False): c_style_mel, c_style_lengths = load_style_ref_mel(style_ref_wav, model.hps, device)
	if exported and use_gst and style_mel_np is None:
		if not style_ref_wav or not config_path: raise ValueError("use_gst=True requires style_ref_wav AND config_path")
		style_mel_np, style_mel_lengths_np = load_style_ref_for_export(style_ref_wav, config_path)

	def run_synth(stn, c_sid, c_lid):
		if exported:
			return _run_exported_forward(model, backend, stn, c_sid, c_lid, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=max_phonemes, max_mel_frames=max_mel_frames, deterministic_noise=deterministic_noise)
		with torch.inference_mode():
			x = torch.LongTensor(stn).to(device).unsqueeze(0)
			res = model.infer(x=x, x_lengths=torch.LongTensor([len(stn)]).to(device), sid=c_sid, lid=c_lid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, noise_scale_p=noise_scale_p, length_scale=length_scale, min_period_duration=min_period_duration, style_ref_mel=c_style_mel, style_ref_lengths=c_style_lengths, prosody_overrides=prosody_overrides)
			chunk = res[0][0, 0].detach().cpu().float().numpy()
			del res, x
			return _finalize_audio(chunk)

	def process_chunk_obj(obj):
		c_lid = utils.resolve_id(obj["language"], languages, label="language") if languages else {"en-us": 0, "ja": 1, "ru": 2}.get(obj["language"], 0)
		return run_synth(cleaned_text_to_sequence(obj["text"]), language_speakers.get(obj["language"], sid), c_lid)

	def clean_cache():
		gc.collect()
		if not exported and device in ["mps", "cuda"]: getattr(torch, device).empty_cache()

	if not stream:
		all_audio, sf_file = [], sf.SoundFile(output_file, mode="w", samplerate=48000, channels=1) if output_file else None
		pause = np.zeros(int(48000 * chunk_pause_sec), dtype=np.float32) if chunk_pause_sec > 0 else None
		iterator = tqdm(processed_chunks, desc="Generating audio", unit="chunk") if show_progress else processed_chunks
		for i, obj in enumerate(iterator):
			chunk = process_chunk_obj(obj)
			if sf_file:
				sf_file.write(chunk)
				if pause is not None and i + 1 < len(processed_chunks):
					sf_file.write(pause)
			else:
				all_audio.append(chunk)
				if pause is not None and i + 1 < len(processed_chunks):
					all_audio.append(pause)
			if not exported: del chunk
		if sf_file: sf_file.close()
		clean_cache()
		if sf_file: return
		return all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

	def audio_generator():
		q, pbar = queue.Queue(maxsize=4), tqdm(total=len(processed_chunks), desc="Generating audio", unit="chunk")

		def worker():
			try:
				for obj in processed_chunks:
					q.put(process_chunk_obj(obj))
					pbar.update(1)
			except Exception as e:
				print(f"Error in generation worker: {e}")
			finally:
				clean_cache()
				q.put(None)

		threading.Thread(target=worker, daemon=True).start()
		while True:
			try:
				if (chunk := q.get(timeout=30)) is None: break
				yield chunk
			except queue.Empty:
				break

	return audio_generator()


def _ab_default_workers(backend, compute_unit="auto"):
	if backend in {"pytorch", "coreai"}:
		return 1
	if backend == "coreml" and _coreml_uses_all_units(compute_unit):
		return 1
	cpu = os.cpu_count() or 1
	return max(1, min(4, cpu - 1))


def _ab_worker_init(model_path, config_path, backend, compute_unit, use_gst, style_ref_wav, max_phonemes, max_mel_frames):
	global _ab_pretrained_iter
	if _ab_pretrained_iter is not None:
		with _ab_pretrained_lock:
			_ab_tls.model = next(_ab_pretrained_iter)
	else:
		_ab_tls.model = load_model(config_path, model_path, backend=backend, compute_unit=compute_unit)
	_ab_tls.backend = backend
	_ab_tls.use_gst = use_gst
	_ab_tls.max_phonemes = max_phonemes
	_ab_tls.max_mel_frames = max_mel_frames
	_ab_tls.style_mel_np = _ab_tls.style_mel_lengths_np = None
	if use_gst:
		_ab_tls.style_mel_np, _ab_tls.style_mel_lengths_np = load_style_ref_for_export(style_ref_wav, config_path)


def _ab_preload_coreml_models(model_path, config_path, compute_unit, workers, show_progress):
	"""Load one MLModel per worker sequentially — parallel ANE compile deadlocks."""
	import coremltools as ct  # once in main thread (avoids per-thread re-registration spam)
	_ = ct
	preloaded = []
	for i in range(workers):
		if show_progress:
			print(f"Preloading Core ML model {i + 1}/{workers}...")
		preloaded.append(load_model(config_path, model_path, backend="coreml", compute_unit=compute_unit))
	return preloaded


def _ab_worker_ping():
	return True


def _ab_worker_chunk(chunk_obj, deterministic_noise, language_speakers, speaker_id):
	stn = cleaned_text_to_sequence(chunk_obj["text"])
	if not stn:
		return np.array([], dtype=np.float32)
	lang = chunk_obj.get("language", "en-us")
	sid = language_speakers.get(lang, speaker_id)
	lid = {"en-us": 0, "ja": 1, "ru": 2}.get(lang, 0)
	return _run_exported_forward(_ab_tls.model, _ab_tls.backend, stn, sid, lid, use_gst=_ab_tls.use_gst, style_mel_np=_ab_tls.style_mel_np, style_mel_lengths_np=_ab_tls.style_mel_lengths_np, max_phonemes=_ab_tls.max_phonemes, max_mel_frames=_ab_tls.max_mel_frames, deterministic_noise=deterministic_noise)


def _ab_collect_pool(pool, submit_fn, chunks, show_progress):
	results = [None] * len(chunks)
	futures = {submit_fn(c, i): i for i, c in enumerate(chunks)}
	pbar = tqdm(total=len(chunks), desc="Generating audiobook", unit="chunk") if show_progress else None
	for fut in concurrent.futures.as_completed(futures):
		results[futures[fut]] = fut.result()
		if pbar:
			pbar.update(1)
	if pbar:
		pbar.close()
	return [r if r is not None else np.array([], dtype=np.float32) for r in results]


def _ab_thread_chunk(model, backend, chunk_obj, use_gst, style_mel_np, style_mel_lengths_np, max_phonemes, max_mel_frames, deterministic_noise, language_speakers, speaker_id):
	stn = cleaned_text_to_sequence(chunk_obj["text"])
	if not stn:
		return np.array([], dtype=np.float32)
	lang = chunk_obj.get("language", "en-us")
	sid = language_speakers.get(lang, speaker_id)
	lid = {"en-us": 0, "ja": 1, "ru": 2}.get(lang, 0)
	return _run_exported_forward(model, backend, stn, sid, lid, use_gst=use_gst, style_mel_np=style_mel_np, style_mel_lengths_np=style_mel_lengths_np, max_phonemes=max_phonemes, max_mel_frames=max_mel_frames, deterministic_noise=deterministic_noise)


def _ab_bucket_parallel(chunks, handle, use_gst, language_speakers, speaker_id, workers, show_progress):
	"""Length-bucketed parallel inference over a shared (thread-safe) Core ML bucket handle. Buckets that the book actually needs are pre-loaded sequentially (to avoid a parallel ANE compile storm), then chunks run concurrently against the shared MLModels."""
	tokens = []
	for c in chunks:
		stn = cleaned_text_to_sequence(c["text"])
		lang = c.get("language", "en-us")
		c_sid = language_speakers.get(lang, speaker_id)
		c_lid = {"en-us": 0, "ja": 1, "ru": 2}.get(lang, 0)
		tokens.append((stn, c_sid, c_lid))
	widths = handle["widths"]
	counts = {}
	for t in tokens:
		if t[0]:
			w = _coreml_pick_width(widths, len(t[0]))
			counts[w] = counts.get(w, 0) + 1
	needed = sorted(counts)
	per_width = {w: max(1, min(handle["instances"], workers, counts[w])) for w in needed}  # Per width, don't load more instances than there are chunks for it (or workers).
	if show_progress and needed:
		print(f"Bucket widths needed: {counts} (instances {per_width}); preloading sequentially...")
	for w in needed:
		_coreml_ensure_instances(handle, w, per_width[w], show_progress=show_progress)

	def run(i):
		stn, c_sid, c_lid = tokens[i]
		if not stn:
			return np.array([], dtype=np.float32)
		return _run_coreml_buckets(handle, stn, c_sid, c_lid, use_gst=use_gst, trim=True)

	results = [None] * len(chunks)
	with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
		futures = {pool.submit(run, i): i for i in range(len(chunks))}
		pbar = tqdm(total=len(chunks), desc="Generating audiobook", unit="chunk") if show_progress else None
		for fut in concurrent.futures.as_completed(futures):
			results[futures[fut]] = fut.result()
			if pbar:
				pbar.update(1)
		if pbar:
			pbar.close()
	return [r if r is not None else np.array([], dtype=np.float32) for r in results]


def _ab_parallel_exported(chunks, model_path, config_path, backend, model, use_gst, style_ref_wav, compute_unit, max_phonemes, max_mel_frames, deterministic_noise, language_speakers, speaker_id, workers, show_progress):
	global _ab_pretrained_iter
	if _is_coreml_bucket_handle(model):
		return _ab_bucket_parallel(chunks, model, use_gst, language_speakers, speaker_id, workers, show_progress)
	if backend == "coreai":
		raise ValueError("Core AI does not support parallel workers — use workers=1 (matches infervitsnew_test / ANE path)")
	if backend in {"coreml", "onnx"}:
		workers = min(workers, len(chunks))
		if show_progress:
			print(f"Parallel {backend}: {workers} workers (one model per thread)")
		initargs = (model_path, config_path, backend, compute_unit, use_gst, style_ref_wav, max_phonemes, max_mel_frames)
		try:
			if backend == "coreml" and workers > 1:
				if show_progress and _coreml_uses_all_units(compute_unit):
					print("Preloading models sequentially (parallel ANE compile hangs)...")
				_ab_pretrained_iter = iter(_ab_preload_coreml_models(model_path, config_path, compute_unit, workers, show_progress))
			with concurrent.futures.ThreadPoolExecutor(max_workers=workers, initializer=_ab_worker_init, initargs=initargs) as pool:
				if show_progress and backend == "coreml" and workers == 1:
					print("Loading Core ML model...")
				for f in [pool.submit(_ab_worker_ping) for _ in range(workers)]:
					f.result()
				return _ab_collect_pool(pool, lambda c, i: pool.submit(_ab_worker_chunk, c, deterministic_noise, language_speakers, speaker_id), chunks, show_progress)
		finally:
			_ab_pretrained_iter = None
	owns_model = model is None
	if owns_model:
		model = load_model(config_path, model_path, backend=backend, compute_unit=compute_unit)
	style_mel_np = style_mel_lengths_np = None
	if use_gst:
		style_mel_np, style_mel_lengths_np = load_style_ref_for_export(style_ref_wav, config_path)
	try:
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			return _ab_collect_pool(pool, lambda c, i: pool.submit(_ab_thread_chunk, model, backend, c, use_gst, style_mel_np, style_mel_lengths_np, max_phonemes, max_mel_frames, deterministic_noise, language_speakers, speaker_id), chunks, show_progress)
	finally:
		if owns_model and backend == "coreai" and isinstance(model, dict):
			model["loop"].close()


def _ab_write_chunks(output_wav, parts, chunk_pause_sec):
	pause = np.zeros(int(48000 * chunk_pause_sec), dtype=np.float32) if chunk_pause_sec > 0 else None
	merged = []
	for i, part in enumerate(parts):
		if len(part) == 0:
			continue
		merged.append(part)
		if pause is not None and i + 1 < len(parts):
			merged.append(pause)
	if not merged:
		raise RuntimeError("No audio generated")
	sf.write(str(output_wav), np.concatenate(merged), 48000)


def audiobook_generate(text, output_wav, model_path, config_path, backend="coreai", model=None, style_ref_wav=None, sid=0, language="en-us", language_map=None, language_speakers=None, device="mps", compute_unit="ane", workers=None, bucket_instances=2, chunk_pause_sec=0, max_text_chunk=300, max_phonemes=None, max_mel_frames=None, use_gst=None, deterministic_noise=False, noise_scale=None, noise_scale_w=None, noise_scale_p=None, length_scale=None, min_period_duration=None, prosody_overrides=None, show_progress=True):
	"""Synthesize long text into one WAV. Core AI uses sequential ANE inference (workers=1) like infervitsnew_test."""
	if isinstance(text, Path):
		text = text.read_text(encoding="utf-8")
	text = text.strip()
	if not text:
		raise ValueError("Empty text")

	backend = _normalize_backend(backend or "coreai")
	output_wav = Path(output_wav)
	output_wav.parent.mkdir(parents=True, exist_ok=True)
	max_mel_frames = max_mel_frames or EXPORT_MAX_MEL_FRAMES
	max_phonemes = _resolve_max_phonemes(backend, max_phonemes)
	exported = backend in {"coreml", "coreai", "onnx"}
	use_gst = use_gst if use_gst is not None else _default_use_gst(backend, style_ref_wav)
	language_map = language_map or {"en-us": "en-us", "ru": "ru", "ja": "ja"}
	language_speakers = language_speakers or {"en-us": sid, "ru": sid, "ja": sid}
	bucketed = backend == "coreml" and _looks_like_coreml_buckets(model_path)
	if bucketed and model is None:
		model = load_model(config_path, model_path, device=device, backend=backend, compute_unit=compute_unit)
	if bucketed and _is_coreml_bucket_handle(model):
		model["instances"] = max(1, int(bucket_instances))
	if bucketed:
		worker_count = workers if workers is not None else max(1, min(4, (os.cpu_count() or 1) - 1))
	else:
		worker_count = workers if workers is not None else _ab_default_workers(backend, compute_unit)
	if backend == "coreai" and worker_count > 1:
		print("Core AI: forcing workers=1 (parallel breaks ANE)")
		worker_count = 1
	if backend == "coreml" and not bucketed and worker_count > 1 and _coreml_uses_all_units(compute_unit) and workers is not None:
		print("Note: workers>1 with ComputeUnit.ALL preloads sequentially; workers=1 is faster to start. Use compute_unit='gpu' for parallel GPU.")
	t0 = time.time()

	if (worker_count <= 1 or not exported) and not bucketed:
		owns_model = model is None
		if owns_model:
			model = load_model(config_path, model_path, device=device, backend=backend, compute_unit=compute_unit)
		try:
			inference(model=model, text=text, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, noise_scale_p=noise_scale_p, length_scale=length_scale, min_period_duration=min_period_duration, device=device, output_file=str(output_wav), language=language, language_map=language_map, language_speakers=language_speakers, backend=backend, use_gst=use_gst, style_ref_wav=style_ref_wav, config_path=config_path, prosody_overrides=prosody_overrides, max_phonemes=max_phonemes, max_mel_frames=max_mel_frames, deterministic_noise=deterministic_noise, chunk_pause_sec=chunk_pause_sec, max_text_chunk=max_text_chunk, show_progress=show_progress)
		finally:
			if owns_model and backend == "coreai" and isinstance(model, dict):
				model["loop"].close()
	else:
		if use_gst and not style_ref_wav:
			raise ValueError(f"{backend} with use_gst=True requires style_ref_wav")
		chunks = split_and_process_text(text, language=language, max_length=max_text_chunk, combine=True, language_map=language_map)
		if not chunks:
			raise RuntimeError("No speakable chunks after text processing")
		if show_progress:
			print(f"Chunks: {len(chunks)}  workers: {worker_count}  backend: {backend}")
		parts = _ab_parallel_exported(chunks, model_path, config_path, backend, model, use_gst, style_ref_wav, compute_unit, max_phonemes, max_mel_frames, deterministic_noise, language_speakers, sid, worker_count, show_progress)
		_ab_write_chunks(output_wav, parts, chunk_pause_sec)

	if show_progress:
		duration = sf.info(str(output_wav)).duration
		elapsed = time.time() - t0
		rtf = duration / elapsed if elapsed > 0 else 0.0
		cu = model.get("compute_unit") if isinstance(model, dict) else None
		extra = f"  compute_unit={cu}" if backend == "coreai" and cu else ""
		print(f"Wrote {output_wav} — {duration:.1f}s audio in {elapsed:.1f}s ({rtf:.2f}x realtime){extra}")
	return str(output_wav)


def audiobook_creation(series_name="Title", volume_num="01", author=None, narrator="Yuna Ai", genre="Action", year="2026", language="eng", publisher="Yuna Audio", cover_art_file=None, files_before_chapters=None, files_after_chapters=None):
	"""Combines WAV files into a chapterized M4B audiobook using FFmpeg."""
	book_name = f"{series_name} Vol {volume_num}" if volume_num else series_name
	source_path, output_filename = Path(book_name), f"{book_name}.m4b"
	files_before, files_after = files_before_chapters or [], files_after_chapters or []
	if not (wav_files := list(source_path.glob("*.wav"))): return print("No .wav files found in directory!")
	pre_list = sorted([w for w in wav_files if w.stem in files_before], key=lambda x: files_before.index(x.stem))
	post_list = sorted([w for w in wav_files if w.stem in files_after], key=lambda x: files_after.index(x.stem))
	main_list = sorted([w for w in wav_files if w.stem not in files_before and w.stem not in files_after], key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s.name)])
	meta_content = [";FFMETADATA1", f"title={book_name}", f"album={book_name}", f"artist={author}", f"album_artist={author}", f"composer={narrator}", f"genre={genre}", f"date={year}", f"language={language}", f"copyright=© {year} {author}", f"publisher={publisher}", f"sort_name={f'{series_name} {volume_num}' if volume_num else series_name}", f"grouping={series_name}"]
	current_time_sec, concat_entries = 0.0, []
	for wav in (pre_list + main_list + post_list):
		print(f"Analyzing: {wav.name}")
		duration_sec = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(wav)], stdout=subprocess.PIPE, text=True).stdout.strip())
		end_time_sec = current_time_sec + duration_sec
		stem = wav.stem
		ch_title = stem if (stem in files_before or stem in files_after) else stem.replace("-", " ").replace("_", " ").title() if any(k in stem.lower() for k in ["prologue", "epilogue", "interlude", "introduction", "foreword", "afterword", "credits", "appendix"]) else f"Chapter {stem.split('-')[0]} - End" if ("-" in stem and stem.split("-")[1].lower() == "end") else f"Chapters {stem.split('-')[0]}-{stem.split('-')[1]}" if "-" in stem else f"Chapter {stem}"
		meta_content.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(current_time_sec * 1000)}", f"END={int(end_time_sec * 1000)}", f"title={ch_title}"])
		apath = str(wav.absolute()).replace("'", "'\\''")
		concat_entries.append(f"file '{apath}'")
		current_time_sec = end_time_sec
	with open((c_txt := source_path / "concat_list.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(concat_entries))
	with open((m_txt := source_path / "ffmetadata.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(meta_content))
	print("Starting conversion and merging...")
	cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(c_txt), "-i", str(m_txt)]
	has_cover = cover_art_file and Path(cover_art_file).exists()
	if has_cover: cmd.extend(["-i", cover_art_file])
	else: print("Warning: Cover art not found or skipped.")
	cmd.extend(["-map_metadata", "1", "-map", "0:a"])
	if has_cover: cmd.extend(["-map", "2", "-disposition:v", "attached_pic"])
	cmd.extend(["-c:a", "alac", "-ar", "48000", "-c:v", "copy", output_filename])
	try:
		subprocess.run(cmd, check=True)
		print(f"\nSUCCESS! Created: {output_filename}")
	except subprocess.CalledProcessError:
		print("\nError: FFmpeg failed to process the file.")
	finally:
		if c_txt.exists(): os.remove(c_txt)
		if m_txt.exists(): os.remove(m_txt)


def voice_conversion_inference(model=None, source_wav_path=None, source_speaker_id=0, target_speaker_id=1, device="mps", hps=None, pitch_shift=0):
	"""Voice conversion between existing speakers only."""
	import soxr
	cfg = getattr(model, "hps", hps)
	if cfg is None: raise ValueError("Model must have hps attribute or hps argument must be provided")
	cfg = HParams(**cfg) if isinstance(cfg, dict) else cfg
	sr = cfg.data.sampling_rate
	audio, orig_sr = sf.read(source_wav_path, always_2d=False, dtype="float32")
	if getattr(audio, "ndim", 1) > 1: audio = audio.mean(axis=-1)
	if orig_sr != sr: audio = soxr.resample(audio, orig_sr, sr)
	if pitch_shift != 0:
		import librosa  # pitch_shift quality needs librosa; load/resample already fast-pathed
		audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift, bins_per_octave=12, res_type="kaiser_best")
	mel = mel_spectrogram_torch(torch.FloatTensor(audio.astype(np.float32)).unsqueeze(0), cfg.data.filter_length, cfg.data.n_mel_channels, sr, cfg.data.hop_length, cfg.data.win_length, cfg.data.mel_fmin, cfg.data.mel_fmax, center=False)
	with torch.inference_mode():
		y = mel.to(device)
		s_s, s_t = torch.LongTensor([source_speaker_id]).to(device), torch.LongTensor([target_speaker_id]).to(device)
		out = model.voice_conversion(y, torch.LongTensor([y.shape[2]]).to(device), sid_src=s_s, sid_tgt=s_t, noise_scale=0.0)[0][0, 0].data.cpu().float().numpy()
		del y, s_s, s_t
		if device in ["mps", "cuda"]: getattr(torch, device).empty_cache()
	return out
