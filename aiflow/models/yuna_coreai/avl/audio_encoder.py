from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder

_ASR_AUDIO_KEYS = ("num_mel_bins", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "d_model", "dropout", "attention_dropout", "activation_function", "activation_dropout", "scale_embedding", "initializer_range", "n_window", "output_dim", "n_window_infer", "downsample_hidden_size", "max_position_embeddings", )


class AVLProjector(nn.Module):
	def __init__(self, audio_config):
		super().__init__()
		self.linear_1 = nn.Linear(audio_config.d_model, audio_config.d_model)
		self.act = ACT2FN[audio_config.activation_function]
		self.linear_2 = nn.Linear(audio_config.d_model, audio_config.output_dim)

	def forward(self, audio_features):
		return self.linear_2(self.act(self.linear_1(audio_features)))


def build_attn_bias(n_tokens_max: int, n_valid: int, window: int = 104, dtype: torch.dtype = torch.float32, ) -> torch.Tensor:
	"""Static additive mask ``[1, S, S]``: in-window + valid attend; else -inf."""
	s = n_tokens_max
	neg = torch.finfo(dtype).min
	idx = torch.arange(s)
	valid = idx < n_valid
	same_window = (idx[:, None] // window) == (idx[None, :] // window)
	allow = same_window & valid[None, :] & valid[:, None]
	bias = torch.where(allow, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
	return bias.unsqueeze(0)


def encoder_config_from_avl(model_path: str | Path) -> Qwen3ASREncoderConfig:
	cfg = json.loads((Path(model_path) / "config.json").read_text())
	ac = dict(cfg.get("audio_config") or {})
	if "max_position_embeddings" not in ac and "max_source_positions" in ac:
		ac["max_position_embeddings"] = ac["max_source_positions"]
	return Qwen3ASREncoderConfig(**{k: ac[k] for k in _ASR_AUDIO_KEYS if k in ac})


def _iter_weight_files(src: Path):
	single = src / "model.safetensors"
	if single.is_file():
		yield single
		return
	index = src / "model.safetensors.index.json"
	if index.is_file():
		weight_map = json.loads(index.read_text())["weight_map"]
		for name in sorted(set(weight_map.values())):
			path = src / name
			if path.is_file():
				yield path
		return
	raise FileNotFoundError(f"No safetensors in {src}")


def load_avl_audio(model_path: str | Path, *, dtype: torch.dtype = torch.float16):
	"""Load only ``audio_tower`` + ``multi_modal_projector`` from YunaAVL."""
	src = Path(model_path).expanduser().resolve()
	enc_cfg = encoder_config_from_avl(src)
	tower = Qwen3ASREncoder(enc_cfg)
	projector = AVLProjector(enc_cfg)
	tower_sd: dict[str, torch.Tensor] = {}
	proj_sd: dict[str, torch.Tensor] = {}
	from safetensors import safe_open

	for wf in _iter_weight_files(src):
		with safe_open(str(wf), framework="pt") as f:
			for key in f.keys():
				if key.startswith("model.audio_tower."):
					tower_sd[key[len("model.audio_tower."):]] = f.get_tensor(key)
				elif key.startswith("model.multi_modal_projector."):
					proj_sd[key[len("model.multi_modal_projector."):]] = f.get_tensor(key)
	if not tower_sd:
		raise FileNotFoundError(f"No model.audio_tower.* weights in {src}")
	if not proj_sd:
		raise FileNotFoundError(f"No model.multi_modal_projector.* weights in {src}")
	missing, unexpected = tower.load_state_dict(tower_sd, strict=False)
	if missing:
		raise RuntimeError(f"audio_tower missing {missing[:8]}…")
	projector.load_state_dict(proj_sd, strict=True)
	return tower.to(dtype=dtype).eval(), projector.to(dtype=dtype).eval(), enc_cfg


class AVLAudioEncoderStatic(nn.Module):
	"""One baked-K window. Host loops for longer audio."""
	def __init__(self, tower: nn.Module, projector: nn.Module, n_chunks: int) -> None:
		super().__init__()
		self.t = tower
		self.proj = projector
		self.K = int(n_chunks)
		self.chunk_mel = int(tower.n_window) * 2
		self.tok_per_chunk = ((((self.chunk_mel - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1)
		self.H = tower.config.encoder_attention_heads
		self.hd = tower.config.d_model // self.H
		self.scaling = self.hd**-0.5
		infer = int(getattr(tower, "n_window_infer", tower.config.n_window_infer))
		self.window = self.tok_per_chunk * (infer // self.chunk_mel)
		self.num_mel_bins = int(tower.config.num_mel_bins)
		self.output_dim = int(tower.config.output_dim)

	@property
	def n_tokens(self) -> int:
		return self.K * self.tok_per_chunk

	def _layer(self, layer, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
		residual = x
		h = layer.self_attn_layer_norm(x)
		s, d = h.shape
		sa = layer.self_attn
		q = sa.q_proj(h).view(s, self.H, self.hd).transpose(0, 1)
		k = sa.k_proj(h).view(s, self.H, self.hd).transpose(0, 1)
		v = sa.v_proj(h).view(s, self.H, self.hd).transpose(0, 1)
		scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
		scores = scores + bias
		attn = torch.softmax(scores, dim=-1)
		o = torch.matmul(attn, v).transpose(0, 1).reshape(s, d)
		x = residual + sa.out_proj(o)
		residual = x
		h = layer.final_layer_norm(x)
		h = layer.fc2(layer.activation_fn(layer.fc1(h)))
		x = residual + h
		if x.dtype == torch.float16:
			clamp = torch.finfo(x.dtype).max - 1000
			x = torch.clamp(x, min=-clamp, max=clamp)
		return x

	def forward(self, input_features: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
		"""input_features ``[1, 128, 100*K]`` → audio_embeds ``[K*13, 2048]``."""
		t = self.t
		nmel = self.num_mel_bins
		x = input_features.reshape(nmel, self.K, self.chunk_mel).permute(1, 0, 2).contiguous().unsqueeze(1)
		x = F.gelu(t.conv2d1(x))
		x = F.gelu(t.conv2d2(x))
		x = F.gelu(t.conv2d3(x))
		b, c, f, tt = x.shape
		x = t.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, tt, c * f))
		pe = t.positional_embedding.positional_embedding[:tt, :].unsqueeze(0).to(x.dtype)
		x = (x + pe).reshape(b * tt, -1)
		for layer in t.layers:
			x = self._layer(layer, x, attn_bias)
		x = t.ln_post(x)
		return self.proj(x)
