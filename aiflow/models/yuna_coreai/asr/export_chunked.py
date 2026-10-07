from __future__ import annotations
import argparse
import gc
import json
import shutil
import types
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
import transformers.utils.generic as _tf_generic  # noqa: E402  # --- compat shim: qwen-asr 0.0.6 was written when transformers exposed `check_model_inputs` as a decorator *factory* (used as `@check_model_inputs()`). transformers >= 5.12 made it a bare decorator (`@check_model_inputs`), so the factory call style raises "missing 1 required positional argument: 'func'". Wrap it so both call styles work, before importing qwen_asr's modeling code.

_orig_check_model_inputs = _tf_generic.check_model_inputs


def _compat_check_model_inputs(*args, **kwargs):
	if len(args) == 1 and callable(args[0]) and not kwargs:
		return _orig_check_model_inputs(args[0])

	def _decorator(func):
		return _orig_check_model_inputs(func)

	return _decorator


_tf_generic.check_model_inputs = _compat_check_model_inputs

import sys as _sys  # noqa: E402  # qwen_asr/__init__ eagerly imports the forced aligner, which imports `nagisa` (a heavy Japanese tokenizer pulling DyNet) only used at call time. We never run the aligner, so stub the module to satisfy the top-level import.
import types as _types  # noqa: E402

_sys.modules.setdefault("nagisa", _types.ModuleType("nagisa"))

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (  # noqa: E402
    Qwen3ASRForConditionalGeneration, _get_feat_extract_output_lengths,
)
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (  # noqa: E402
    Qwen3ASRConfig as _Qwen3ASRConfig, )


def _safe_get_text_config(self, decoder: bool = False):  # compat: transformers >=5.12 runs `validate_token_ids` inside PretrainedConfig __init__ (via the strict dataclass), which calls get_text_config() before qwen_asr's Qwen3ASRConfig.__init__ has assigned self.thinker_config. Make it tolerate the not-yet-set attribute during that early validation pass.
	thinker = getattr(self, "thinker_config", None)
	if thinker is None:
		return self
	return thinker.get_text_config()


_Qwen3ASRConfig.get_text_config = _safe_get_text_config

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS  # noqa: E402  # compat: transformers >=5.12 removed the "default" key from ROPE_INIT_FUNCTIONS (standard RoPE is handled elsewhere now). qwen-asr's vendored rotary embedding still does `ROPE_INIT_FUNCTIONS[self.rope_type]` with rope_type="default", so re-register a standard inverse-frequency initializer under that key.


def _default_rope_init(config, device=None, seq_len=None, **kwargs):
	base = getattr(config, "rope_theta", None)
	if base is None:
		rp = getattr(config, "rope_parameters", None) or {}
		base = rp.get("rope_theta", 10000.0)
	head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
	partial = getattr(config, "partial_rotary_factor", 1.0)
	dim = int(head_dim * partial)
	inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim))
	return inv_freq, 1.0


if "default" not in _ROPE_FNS:
	_ROPE_FNS["default"] = _default_rope_init

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (  # noqa: E402  # compat: transformers >=5.12 `_init_weights` re-inits "default" RoPE modules via `module.compute_default_rope_parameters(config)`, which qwen-asr's rotary class doesn't define. Attach it.
    Qwen3ASRThinkerTextRotaryEmbedding as _Qwen3ASRRotary, )


def _compute_default_rope_parameters(self, config=None, device=None, **kwargs):
	return _default_rope_init(config if config is not None else self.config, device)


_Qwen3ASRRotary.compute_default_rope_parameters = _compute_default_rope_parameters

from transformers.masking_utils import create_causal_mask  # noqa: E402

import transformers.configuration_utils as _cfgu  # noqa: E402  # compat: transformers >=5.12 no longer materializes bos/eos/pad_token_id (and friends) as default attributes on PretrainedConfig (they live in the generation config now). qwen-asr reads e.g. `config.pad_token_id` directly, so add a fallback returning None for any missing *_token_id attribute.

_PC = _cfgu.PretrainedConfig
if not getattr(_PC, "_yuna_token_id_fallback", False):
	_prev_getattr = _PC.__dict__.get("__getattr__", None)

	def _pc_getattr(self, name):
		if name.endswith("_token_id"):
			return None
		if _prev_getattr is not None:
			return _prev_getattr(self, name)
		raise AttributeError(name)

	_PC.__getattr__ = _pc_getattr
	_PC._yuna_token_id_fallback = True
from safetensors import safe_open
from torch.nn.utils import parametrize

MODEL_PATH: str | None = None
OUTPUT_DIR = Path(".")
EMBED_WEIGHT_KEY = "thinker.model.embed_tokens.weight"
EMBED_SIDECAR_NAME = "YunaASREmbed_quantized.pt"
EMBED_SWIFT_META_NAME = "YunaASREmbed_meta.json"
EMBED_SWIFT_QUANT_NAME = "YunaASREmbed_quantized.bin"
EMBED_SWIFT_SCALE_NAME = "YunaASREmbed_scales.bin"
MAX_CACHE_LEN = 1024
BATCH_SIZE = 1
DECODER_CHUNK_LAYERS = 6
CHUNK_FRAMES = 100  # n_window * 2 (one CNN chunk)
N_CHUNKS_PER_WINDOW = 8  # n_window_infer // (n_window * 2) = 800 // 100
TOKENS_PER_SUBCHUNK = 13  # _get_feat_extract_output_lengths(100)
MAX_MEL_FRAMES = CHUNK_FRAMES * N_CHUNKS_PER_WINDOW  # 800 mel frames = one encoder window
AUDIO_TOKENS_PER_CHUNK = TOKENS_PER_SUBCHUNK * N_CHUNKS_PER_WINDOW  # 104 tokens per window


def _free_memory() -> None:
	gc.collect()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()


def _save_aimodel(program, output_path: Path) -> None:
	if output_path.exists():
		shutil.rmtree(output_path)
	program.save_asset(output_path)


def freeze_module(module: nn.Module) -> None:
	module.eval()
	for param in module.parameters():
		param.requires_grad_(False)


def _quant_storage_dtype(nbits: int) -> torch.dtype:
	if nbits == 4:
		return torch.int4
	if nbits == 8:
		return torch.int8
	raise ValueError(f"Unsupported quant bits: {nbits}. Use 4 or 8.")


def _symmetric_quantize_matrix(weight: torch.Tensor, nbits: int, channel_axis: int = 0):
	qmin = -(2**(nbits - 1))
	qmax = 2**(nbits - 1) - 1
	w = weight.detach().float()
	if channel_axis != 0:
		w = w.movedim(channel_axis, 0)
	reduce_dims = tuple(range(1, w.ndim))
	wmax = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
	scale = (wmax / qmax).to(torch.bfloat16)
	q = torch.round(w / scale.float()).clamp(qmin, qmax).to(torch.int8)
	if channel_axis != 0:
		q = q.movedim(0, channel_axis)
		scale = scale.movedim(0, channel_axis)
	return q, scale, _quant_storage_dtype(nbits)


def _register_quantized_weight(module: nn.Module, param_name: str, nbits: int, channel_axis: int = 0) -> None:
	if parametrize.is_parametrized(module, param_name):
		return
	weight = getattr(module, param_name)
	q, scale, storage_dtype = _symmetric_quantize_matrix(weight, nbits, channel_axis=channel_axis)
	param_cls = wrap_for_parametrization(WeightDequantizeModule)
	parametrize.register_parametrization(module, param_name, param_cls(quantized_data=q, scale=scale, zero_point=None, input_dtype=storage_dtype, output_dtype=torch.bfloat16, ), unsafe=True, )


def quantize_module_weights(module: nn.Module, nbits: int = 8, *, skip_audio: bool = True) -> int:
	quantized = 0
	for name, submodule in module.named_modules():
		if skip_audio and "audio_tower" in name:
			continue
		if isinstance(submodule, nn.Linear):
			_register_quantized_weight(submodule, "weight", nbits, channel_axis=0)
			quantized += 1
		elif isinstance(submodule, nn.Embedding):
			_register_quantized_weight(submodule, "weight", nbits, channel_axis=0)
			quantized += 1
	return quantized


def export_quantized_embedding(model_path: str, output_dir: Path, nbits: int = 8) -> Path:
	weights_path = Path(model_path) / "model.safetensors"
	if not weights_path.exists():
		raise FileNotFoundError(f"Missing {weights_path}")
	print(f"Quantizing embed_tokens to int{nbits} sidecar...")
	with safe_open(weights_path, framework="pt") as handle:
		weight = handle.get_tensor(EMBED_WEIGHT_KEY).to(torch.bfloat16)
	q, scale, storage_dtype = _symmetric_quantize_matrix(weight, nbits, channel_axis=0)
	sidecar_path = output_dir / EMBED_SIDECAR_NAME
	torch.save({"quantized_data": q.cpu(), "scale": scale.cpu(), "weight_bits": nbits, "storage_dtype": str(storage_dtype), "vocab_size": weight.shape[0], "hidden_size": weight.shape[1], }, sidecar_path, )
	q.numpy().tofile(output_dir / EMBED_SWIFT_QUANT_NAME)
	scale.float().numpy().tofile(output_dir / EMBED_SWIFT_SCALE_NAME)
	swift_meta = {"vocab_size": int(weight.shape[0]), "hidden_size": int(weight.shape[1]), "weight_bits": nbits, "quantized_file": EMBED_SWIFT_QUANT_NAME, "scale_file": EMBED_SWIFT_SCALE_NAME, }
	(output_dir / EMBED_SWIFT_META_NAME).write_text(json.dumps(swift_meta, indent=2))
	print(f"✨ Quantized embedding → {sidecar_path}")
	del weight, q, scale
	_free_memory()
	return sidecar_path


def bundle_tokenizer_assets(model_path: str, output_dir: Path) -> None:
	for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json", "merges.txt", "chat_template.json", ):
		src = Path(model_path) / name
		if src.exists():
			shutil.copy2(src, output_dir / name)
			print(f"📎 Asset → {output_dir / name}")


class ASRAudioEncoderExportWrapper(nn.Module):
	"""One inference window: N_CHUNKS_PER_WINDOW sub-chunks of CHUNK_FRAMES mel frames each -> AUDIO_TOKENS_PER_CHUNK tokens, attended bidirectionally within the window (matching the real encoder's n_window_infer blocks). `attn_mask` is a runtime input (additive, [1,1,T,T]) so the caller can mask the padding tokens of a partial final window."""
	def __init__(self, encoder: nn.Module, cu_seqlens: torch.Tensor):
		super().__init__()
		self.encoder = encoder
		self.register_buffer("cu_seqlens", cu_seqlens)

	def forward(self, input_features: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
		enc = self.encoder
		chunks = input_features.T.reshape(N_CHUNKS_PER_WINDOW, CHUNK_FRAMES, 128).transpose(1, 2).unsqueeze(1)  # [128, N_CHUNKS*CHUNK_FRAMES] -> [N_CHUNKS, 1, 128, CHUNK_FRAMES]
		padded_embed = F.gelu(enc.conv2d1(chunks))
		padded_embed = F.gelu(enc.conv2d2(padded_embed))
		padded_embed = F.gelu(enc.conv2d3(padded_embed))
		b, c, f, t = padded_embed.size()
		padded_embed = enc.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
		pos = enc.positional_embedding.positional_embedding[:t, :].unsqueeze(0).to(padded_embed.dtype)
		padded_embed = padded_embed + pos
		hidden_states = padded_embed.reshape(b * t, -1)  # [N_CHUNKS * t, d]
		for layer in enc.layers:
			hidden_states = layer(hidden_states, self.cu_seqlens, attention_mask=attn_mask)[0]
		hidden_states = enc.ln_post(hidden_states)
		hidden_states = enc.act(enc.proj1(hidden_states))
		return enc.proj2(hidden_states)


def _bake_audio_encoder(encoder: nn.Module) -> torch.Tensor:
	encoder.config._attn_implementation = "eager"
	return torch.tensor([0, AUDIO_TOKENS_PER_CHUNK], dtype=torch.int32)  # Single attention block spanning the whole window; partial windows are handled by the runtime `attn_mask` input, not cu_seqlens.


def export_audio_encoder(model) -> Path:
	output_path = OUTPUT_DIR / "YunaASRAudioEncoder.aimodel"
	encoder = model.thinker.audio_tower
	cu_seqlens = _bake_audio_encoder(encoder)
	wrapper = ASRAudioEncoderExportWrapper(encoder, cu_seqlens).eval()
	freeze_module(wrapper)
	dummy = torch.zeros(128, MAX_MEL_FRAMES, dtype=torch.bfloat16)
	dummy_mask = torch.zeros(1, 1, AUDIO_TOKENS_PER_CHUNK, AUDIO_TOKENS_PER_CHUNK, dtype=torch.bfloat16)
	print("Capturing audio encoder graph...")
	with torch.no_grad():
		exported = torch.export.export(wrapper, args=(dummy, dummy_mask))
	print("Compiling audio encoder to Core AI...")
	exported = exported.run_decompositions(get_decomp_table())
	program = (TorchConverter().add_exported_program(exported, input_names=["input_features", "attn_mask"], output_names=["audio_features"], entrypoint_name="audio_encoder", ).to_coreai())
	program.optimize()
	_save_aimodel(program, output_path)
	print(f"✨ Audio encoder → {output_path}")
	del exported, program, wrapper
	_free_memory()
	return output_path


class StatefulKVCache:
	"""HF-compatible cache backed by in-place mutated buffers → Core AI states. `key_buf` / `value_buf` are registered buffers of shape [chunk_layers, 1, kv_heads, MAX_CACHE_LEN, head_dim]. We write the new token's K/V at `cache_position` with an in-place `index_copy_`, which torch.export captures as a buffer mutation and Core AI converts into a state (no per-step cache I/O across the Swift boundary)."""

	is_compileable = True

	def __init__(self, key_buf, value_buf, cache_position: torch.Tensor, layer_offset: int):
		self.key_buf = key_buf
		self.value_buf = value_buf
		self.cache_position = cache_position
		self.layer_offset = layer_offset

	def update(self, key_states, value_states, layer_idx, *args, **kwargs):
		local = layer_idx - self.layer_offset
		pos = torch.arange(key_states.shape[-2], device=key_states.device, dtype=torch.long) + self.cache_position[0]
		self.key_buf[local].index_copy_(2, pos, key_states)
		self.value_buf[local].index_copy_(2, pos, value_states)
		return self.key_buf[local], self.value_buf[local]

	def get_seq_length(self):
		return self.cache_position[0]

	def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
		return MAX_CACHE_LEN, 0


def patch_text_for_export(text_model):
	text_model.config._attn_implementation = "sdpa"  # SDPA -> Core AI fused attention, bf16 throughout (eager upcasts softmax to fp32, which blocks the ANE).
	for layer in text_model.layers:
		layer.self_attn.config._attn_implementation = "sdpa"


class ASRDecoderChunk(nn.Module):
	"""Decoder slice whose KV cache lives in in-place Core AI *state* buffers."""
	def __init__(self, text_model, lm_head: nn.Linear | None, layer_start: int, layer_end: int, kv_heads: int, head_dim: int, ):
		super().__init__()
		self.text_model = text_model
		self.lm_head = lm_head
		self.layer_start = layer_start
		self.layer_end = layer_end
		self.num_chunk_layers = layer_end - layer_start
		cache_shape = (self.num_chunk_layers, BATCH_SIZE, kv_heads, MAX_CACHE_LEN, head_dim)
		self.register_buffer("key_cache", torch.zeros(cache_shape, dtype=torch.bfloat16))
		self.register_buffer("value_cache", torch.zeros(cache_shape, dtype=torch.bfloat16))

	def forward(self, hidden_states, position_ids, cache_position):
		lm = self.text_model
		text_position_ids = position_ids[0]
		position_embeddings = lm.rotary_emb(hidden_states, position_ids)
		kv_cache = StatefulKVCache(self.key_cache, self.value_cache, cache_position, self.layer_start)
		attention_mask = create_causal_mask(config=lm.config, inputs_embeds=hidden_states, attention_mask=None, past_key_values=kv_cache, position_ids=text_position_ids, )
		for layer_idx in range(self.layer_start, self.layer_end):
			hidden_states = lm.layers[layer_idx](hidden_states, attention_mask=attention_mask, position_ids=text_position_ids, past_key_values=kv_cache, use_cache=True, cache_position=cache_position, position_embeddings=position_embeddings, )
		hidden_states = lm.norm(hidden_states) if self.lm_head is not None else hidden_states
		outputs = [hidden_states]
		if self.lm_head is not None:
			outputs.append(self.lm_head(hidden_states))
		return tuple(outputs)


def _chunk_io_names(has_lm_head: bool):
	inputs = ["hidden_states", "position_ids", "cache_position"]
	outputs = ["hidden_states"]
	if has_lm_head:
		outputs.append("logits")
	states = ["key_cache", "value_cache"]
	return inputs, outputs, states


def export_text_decoder(model, *, quant_bits: int | None = None) -> list[Path]:
	thinker = model.thinker
	text_config = thinker.model.config
	num_layers = text_config.num_hidden_layers
	hidden_size = text_config.hidden_size
	num_chunks = (num_layers + DECODER_CHUNK_LAYERS - 1) // DECODER_CHUNK_LAYERS
	text_model = thinker.model
	lm_head = thinker.lm_head
	kv_heads = text_config.num_key_value_heads
	head_dim = text_config.head_dim
	patch_text_for_export(text_model)
	freeze_module(text_model)
	freeze_module(lm_head)

	chunk_paths = []
	metadata = {"model_type": "qwen3_asr", "max_cache_len": MAX_CACHE_LEN, "batch_size": BATCH_SIZE, "hidden_size": hidden_size, "num_layers": num_layers, "chunk_layers": DECODER_CHUNK_LAYERS, "num_key_value_heads": kv_heads, "head_dim": head_dim, "kv_cache_layout": "stacked_states", "max_mel_frames": MAX_MEL_FRAMES, "audio_tokens_per_chunk": AUDIO_TOKENS_PER_CHUNK, "audio_token_id": thinker.config.audio_token_id, "audio_start_token_id": thinker.config.audio_start_token_id, "audio_end_token_id": thinker.config.audio_end_token_id, "eos_token_ids": [151645, 151643], "chunks": [], }
	if quant_bits is not None:
		metadata["quantization"] = {"enabled": True, "weight_bits": quant_bits, "embed_path": EMBED_SIDECAR_NAME, "embed_swift_meta": EMBED_SWIFT_META_NAME, }

	hidden_states = torch.zeros(BATCH_SIZE, 1, hidden_size, dtype=torch.bfloat16)
	position_ids = torch.zeros(3, BATCH_SIZE, 1, dtype=torch.long)
	cache_position = torch.tensor([0], dtype=torch.long)

	for chunk_idx in range(num_chunks):
		layer_start = chunk_idx * DECODER_CHUNK_LAYERS
		layer_end = min(layer_start + DECODER_CHUNK_LAYERS, num_layers)
		is_last = chunk_idx == num_chunks - 1
		chunk = ASRDecoderChunk(text_model, lm_head if is_last else None, layer_start, layer_end, kv_heads, head_dim, ).eval()
		freeze_module(chunk)
		args = (hidden_states, position_ids, cache_position)
		print(f"Capturing text decoder chunk {chunk_idx + 1}/{num_chunks} (layers {layer_start}-{layer_end - 1})...")
		with torch.no_grad():
			exported = torch.export.export(chunk, args=args, strict=False)
		input_names, output_names, state_names = _chunk_io_names(has_lm_head=is_last)
		output_path = OUTPUT_DIR / f"YunaASRTextDecoder_chunk{chunk_idx}.aimodel"
		print(f"Compiling chunk {chunk_idx} to Core AI...")
		exported = exported.run_decompositions(get_decomp_table())
		program = (TorchConverter().add_exported_program(exported, input_names=input_names, output_names=output_names, state_names=state_names, entrypoint_name=f"decoder_chunk_{chunk_idx}", ).to_coreai())
		program.optimize()
		_save_aimodel(program, output_path)
		chunk_paths.append(output_path)
		print(f"✨ Text decoder chunk {chunk_idx} → {output_path}")
		metadata["chunks"].append({"index": chunk_idx, "path": output_path.name, "layer_start": layer_start, "layer_end": layer_end, "inputs": input_names, "outputs": output_names, "states": state_names, })
		del chunk, exported, program
		_free_memory()

	meta_path = OUTPUT_DIR / "YunaASRTextDecoder_manifest.json"
	meta_path.write_text(json.dumps(metadata, indent=2))
	print(f"📋 Decoder manifest → {meta_path}")
	return chunk_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export Qwen3-ASR 0.6B to Core AI")
	parser.add_argument("--model-path", type=str, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True, help="Export int-quantized decoder weights (~4x smaller than bf16). Default: on.", )
	parser.add_argument("--quant-bits", type=int, choices=(4, 8), default=8)
	parser.add_argument("--audio-only", action="store_true", help="Skip text decoder export.")
	parser.add_argument("--text-only", action="store_true", help="Skip audio encoder export.")
	parser.add_argument("--embed-only", action="store_true", help="Only write quantized embedding sidecar.")
	return parser.parse_args()


def run(*, model_path: str, output_dir: Path, quant_bits: int | None = 8, audio_only: bool = False, text_only: bool = False, embed_only: bool = False, ) -> None:
	global OUTPUT_DIR, MODEL_PATH
	OUTPUT_DIR = output_dir
	MODEL_PATH = model_path
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	if quant_bits is not None:
		export_quantized_embedding(MODEL_PATH, OUTPUT_DIR, nbits=quant_bits)
		bundle_tokenizer_assets(MODEL_PATH, OUTPUT_DIR)
		if embed_only:
			print(f"\nDone. Quantized embedding only → {OUTPUT_DIR.resolve()}")
			return

	print("Loading Qwen3-ASR (transformers backend)...")
	model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True, ).eval()
	freeze_module(model)

	if quant_bits is not None:
		count = quantize_module_weights(model.thinker, nbits=quant_bits, skip_audio=True)
		print(f"Quantized {count} Linear/Embedding tensors in text decoder to int{quant_bits}")

	if not text_only:
		export_audio_encoder(model)
	else:
		print("Skipping audio encoder export (--text-only)")

	if not audio_only:
		export_text_decoder(model, quant_bits=quant_bits)
	else:
		print("Skipping text decoder export (--audio-only)")

	if quant_bits is None:
		bundle_tokenizer_assets(MODEL_PATH, OUTPUT_DIR)

	del model
	_free_memory()
	print(f"\nFull export complete → {OUTPUT_DIR.resolve()}")


def main():
	args = parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, quant_bits=None if not args.quantize else args.quant_bits, audio_only=args.audio_only, text_only=args.text_only, embed_only=args.embed_only, )


if __name__ == "__main__":
	main()
