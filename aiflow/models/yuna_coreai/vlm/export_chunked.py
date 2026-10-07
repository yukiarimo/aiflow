import argparse
import gc
import json
import shutil
import types
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextModel, apply_rotary_pos_emb_vision, eager_attention_forward, )
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from safetensors import safe_open

# ── paths & constants ──────────────────────────────────────────────────────────
MODEL_PATH: str | None = None
OUTPUT_DIR = Path(".")
EMBED_WEIGHT_KEY = "model.language_model.embed_tokens.weight"
EMBED_SIDECAR_NAME = "YunaEmbed_quantized.pt"
EMBED_SWIFT_META_NAME = "YunaEmbed_meta.json"
EMBED_SWIFT_QUANT_NAME = "YunaEmbed_quantized.bin"
EMBED_SWIFT_SCALE_NAME = "YunaEmbed_scales.bin"
MAX_CACHE_LEN = 1024
BATCH_SIZE = 1
DECODER_CHUNK_LAYERS = 6
GRID_T, GRID_H, GRID_W = 1, 28, 28  # Fixed vision grid for export: 1 frame, 28×28 patches (448×448 → 784 patches → 196 LLM tokens)
IMAGE_PATCHES = GRID_H * GRID_W
PATCH_DIM = 3 * 2 * 16 * 16


def _free_memory() -> None:
	gc.collect()
	if torch.backends.mps.is_available():
		torch.mps.empty_cache()


def _save_aimodel(program, output_path: Path):
	if output_path.exists():
		shutil.rmtree(output_path)
	program.save_asset(output_path)


def freeze_module(module: nn.Module):
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
	"""Per-channel symmetric quantization along channel_axis."""
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


def quantize_module_weights(module: nn.Module, nbits: int = 8) -> int:
	"""Replace Linear/Embedding weights with Core-AI-exportable int4/int8 weights."""
	quantized = 0
	for submodule in module.modules():
		if isinstance(submodule, nn.Linear):
			_register_quantized_weight(submodule, "weight", nbits, channel_axis=0)
			quantized += 1
		elif isinstance(submodule, nn.Embedding):
			_register_quantized_weight(submodule, "weight", nbits, channel_axis=0)
			quantized += 1
	return quantized


def export_quantized_embedding(model_path: str, output_dir: Path, nbits: int = 8) -> Path:
	"""Write a small quantized embed sidecar without loading the full model."""
	weights_path = Path(model_path) / "model.safetensors"
	if not weights_path.exists():
		raise FileNotFoundError(f"Missing {weights_path}")

	print(f"Quantizing embed_tokens to int{nbits} sidecar...")
	with safe_open(weights_path, framework="pt") as handle:
		weight = handle.get_tensor(EMBED_WEIGHT_KEY).to(torch.bfloat16)

	q, scale, storage_dtype = _symmetric_quantize_matrix(weight, nbits, channel_axis=0)
	sidecar_path = output_dir / EMBED_SIDECAR_NAME
	torch.save({"quantized_data": q.cpu(), "scale": scale.cpu(), "weight_bits": nbits, "storage_dtype": str(storage_dtype), "vocab_size": weight.shape[0], "hidden_size": weight.shape[1], }, sidecar_path, )
	sidecar_mb = sidecar_path.stat().st_size / 1e6
	print(f"✨ Quantized embedding → {sidecar_path} ({sidecar_mb:.0f} MB)")

	q.numpy().tofile(output_dir / EMBED_SWIFT_QUANT_NAME)  # Swift-friendly flat binaries (no PyTorch dependency in the app).
	scale.float().numpy().tofile(output_dir / EMBED_SWIFT_SCALE_NAME)
	swift_meta = {"vocab_size": int(weight.shape[0]), "hidden_size": int(weight.shape[1]), "weight_bits": nbits, "quantized_file": EMBED_SWIFT_QUANT_NAME, "scale_file": EMBED_SWIFT_SCALE_NAME, }
	(output_dir / EMBED_SWIFT_META_NAME).write_text(json.dumps(swift_meta, indent=2))
	print(f"✨ Swift embed assets → {EMBED_SWIFT_META_NAME}")

	del weight, q, scale
	_free_memory()
	return sidecar_path


def bundle_tokenizer_assets(model_path: str, output_dir: Path) -> None:
	"""Copy tokenizer files next to Core AI assets for the Swift demo app."""
	for name in ("tokenizer.json", "tokenizer_config.json"):
		src = Path(model_path) / name
		if src.exists():
			shutil.copy2(src, output_dir / name)
			print(f"📎 Tokenizer asset → {output_dir / name}")


# ── vision export helpers ──────────────────────────────────────────────────────
def exportable_vision_attention_forward(attn_module, hidden_states, position_embeddings):
	seq_length = hidden_states.shape[0]
	query_states, key_states, value_states = (attn_module.qkv(hidden_states).reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
	cos, sin = position_embeddings
	query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
	query_states = query_states.transpose(0, 1).unsqueeze(0)
	key_states = key_states.transpose(0, 1).unsqueeze(0)
	value_states = value_states.transpose(0, 1).unsqueeze(0)
	attn_output, _ = eager_attention_forward(attn_module, query_states, key_states, value_states, attention_mask=None, scaling=attn_module.scaling, dropout=0.0, is_causal=False, )
	return attn_module.proj(attn_output.reshape(seq_length, -1).contiguous())


def patch_vision_for_export(visual_model, grid_thw: torch.Tensor):
	with torch.no_grad():
		pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
		rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
		seq_len = pos_embeds.shape[0]
		rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
		emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
		cos, sin = emb.cos(), emb.sin()
		cu_seqlens = F.pad(torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32), (1, 0), value=0, )

	visual_model.register_buffer("_export_pos_embeds", pos_embeds, persistent=False)
	visual_model.register_buffer("_export_cos", cos, persistent=False)
	visual_model.register_buffer("_export_sin", sin, persistent=False)
	visual_model.register_buffer("_export_cu_seqlens", cu_seqlens, persistent=False)

	position_embeddings = (visual_model._export_cos, visual_model._export_sin)
	for block in visual_model.blocks:

		def make_attn_forward(pos_emb):
			def forward(self, hidden_states, *_args, **_kwargs):
				return exportable_vision_attention_forward(self, hidden_states, pos_emb)

			return forward

		block.attn.forward = types.MethodType(make_attn_forward(position_embeddings), block.attn)

	def exportable_visual_forward(self, hidden_states, grid_thw=None, **kwargs):
		hidden_states = self.patch_embed(hidden_states) + self._export_pos_embeds
		seq_len, _ = hidden_states.size()
		hidden_states = hidden_states.reshape(seq_len, -1)
		pos_emb = (self._export_cos, self._export_sin)
		deepstack_feature_lists = []
		for layer_num, blk in enumerate(self.blocks):
			hidden_states = blk(hidden_states, cu_seqlens=self._export_cu_seqlens, position_embeddings=pos_emb, **kwargs)
			if layer_num in self.deepstack_visual_indexes:
				deepstack_feature_lists.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states))
		return self.merger(hidden_states), deepstack_feature_lists

	visual_model.forward = types.MethodType(exportable_visual_forward, visual_model)


class YunaVisionWrapper(nn.Module):
	def __init__(self, visual_model):
		super().__init__()
		self.visual = visual_model

	def forward(self, pixel_values):
		merged, deepstack = self.visual(pixel_values)
		return merged, *deepstack


def export_vision(model) -> Path:
	output_path = OUTPUT_DIR / "YunaVisionEncoder.aimodel"
	visual = model.model.visual
	patch_vision_for_export(visual, torch.tensor([[GRID_T, GRID_H, GRID_W]], dtype=torch.long))
	wrapper = YunaVisionWrapper(visual)
	dummy_pixels = torch.zeros((IMAGE_PATCHES, PATCH_DIM), dtype=torch.bfloat16)

	print("Capturing vision graph...")
	with torch.no_grad():
		exported = torch.export.export(wrapper, args=(dummy_pixels, ))

	num_deepstack = len(visual.deepstack_visual_indexes)
	vision_outputs = ["merged_hidden_states"] + [f"deepstack_{i}" for i in range(num_deepstack)]

	print("Compiling vision encoder to Core AI...")
	exported = exported.run_decompositions(get_decomp_table())
	program = (TorchConverter().add_exported_program(exported, input_names=["pixel_values"], output_names=vision_outputs, entrypoint_name="vision").to_coreai())
	program.optimize()
	_save_aimodel(program, output_path)
	print(f"✨ Vision encoder → {output_path}")
	del exported, program, wrapper
	_free_memory()
	return output_path


# ── text decoder: in-place KV-cache as Core AI states ─────────────────────────
class StatefulKVCache:
	"""HF-compatible cache backed by in-place mutated buffers → Core AI states. `key_buf` / `value_buf` are registered buffers of shape [chunk_layers, 1, kv_heads, MAX_CACHE_LEN, head_dim]. The new token's K/V is written at `cache_position` with an in-place `index_copy_`, which torch.export captures as a buffer mutation and Core AI lowers to an in-place state. This removes the per-step KV tensor I/O of the old functional cache."""

	is_compileable = True

	def __init__(self, key_buf, value_buf, cache_position: torch.Tensor, layer_offset: int):
		self.key_buf = key_buf
		self.value_buf = value_buf
		self.cache_position = cache_position
		self.layer_offset = layer_offset

	def update(self, key_states, value_states, layer_idx, **kwargs):
		local = layer_idx - self.layer_offset
		pos = torch.arange(key_states.shape[-2], device=key_states.device, dtype=torch.long) + self.cache_position[0]
		self.key_buf[local].index_copy_(2, pos, key_states)
		self.value_buf[local].index_copy_(2, pos, value_states)
		return self.key_buf[local], self.value_buf[local]

	def get_seq_length(self):
		return self.cache_position[0]

	def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
		return MAX_CACHE_LEN, 0


def patch_text_for_export(language_model: Qwen3VLTextModel):
	language_model.config._attn_implementation = "sdpa"  # SDPA lowers to Core AI's fused attention kernel and stays in bf16, unlike eager (which upcasts softmax to fp32 -> blocks the ANE).

	def exportable_deepstack(self, hidden_states, visual_pos_masks, visual_embeds):
		mask = visual_pos_masks.to(hidden_states.dtype).unsqueeze(-1)
		return hidden_states + mask * visual_embeds

	language_model._deepstack_process = types.MethodType(exportable_deepstack, language_model)


class YunaDecoderChunk(nn.Module):
	"""Decoder slice whose KV cache lives in in-place Core AI *state* buffers."""
	def __init__(self, language_model: Qwen3VLTextModel, lm_head: nn.Linear | None, layer_start: int, layer_end: int, kv_heads: int, head_dim: int, ):
		super().__init__()
		self.language_model = language_model
		self.lm_head = lm_head
		self.layer_start = layer_start
		self.layer_end = layer_end
		self.num_chunk_layers = layer_end - layer_start
		cache_shape = (self.num_chunk_layers, BATCH_SIZE, kv_heads, MAX_CACHE_LEN, head_dim)
		self.register_buffer("key_cache", torch.zeros(cache_shape, dtype=torch.bfloat16))
		self.register_buffer("value_cache", torch.zeros(cache_shape, dtype=torch.bfloat16))

	def _split_position_ids(self, position_ids):
		if position_ids.ndim == 3 and position_ids.shape[0] == 4:
			return position_ids[0], position_ids[1:]
		return position_ids[0] if position_ids.ndim == 3 else None, position_ids

	def forward(self, hidden_states, position_ids, cache_position, deepstack_active, deepstack_0, deepstack_1, deepstack_2, ):
		lm = self.language_model
		text_position_ids, mrope_position_ids = self._split_position_ids(position_ids)
		position_embeddings = lm.rotary_emb(hidden_states, mrope_position_ids)

		kv_cache = StatefulKVCache(self.key_cache, self.value_cache, cache_position, self.layer_start)
		attention_mask = create_causal_mask(config=lm.config, inputs_embeds=hidden_states, attention_mask=None, past_key_values=kv_cache, position_ids=text_position_ids, )

		for layer_idx in range(self.layer_start, self.layer_end):
			hidden_states = lm.layers[layer_idx](hidden_states, attention_mask=attention_mask, position_ids=text_position_ids, past_key_values=kv_cache, use_cache=True, position_embeddings=position_embeddings, )
			if layer_idx < 3 and deepstack_active is not None:
				ds = [deepstack_0, deepstack_1, deepstack_2][layer_idx]
				hidden_states = lm._deepstack_process(hidden_states, deepstack_active.to(torch.bool), ds)

		hidden_states = lm.norm(hidden_states) if self.lm_head is not None else hidden_states
		outputs = [hidden_states]
		if self.lm_head is not None:
			outputs.append(self.lm_head(hidden_states))
		return tuple(outputs)


def _chunk_io_names(has_lm_head: bool):
	inputs = ["hidden_states", "position_ids", "cache_position", "deepstack_active", "deepstack_0", "deepstack_1", "deepstack_2", ]
	outputs = ["hidden_states"]
	if has_lm_head:
		outputs.append("logits")
	states = ["key_cache", "value_cache"]
	return inputs, outputs, states


def export_text_decoder(model, *, quant_bits: int | None = None) -> list[Path]:
	text_config = model.config.text_config
	num_layers = text_config.num_hidden_layers
	hidden_size = text_config.hidden_size
	num_chunks = (num_layers + DECODER_CHUNK_LAYERS - 1) // DECODER_CHUNK_LAYERS

	language_model = model.model.language_model
	lm_head = model.lm_head
	kv_heads = text_config.num_key_value_heads
	head_dim = text_config.head_dim
	patch_text_for_export(language_model)
	freeze_module(language_model)
	freeze_module(lm_head)

	chunk_paths = []
	metadata = {"max_cache_len": MAX_CACHE_LEN, "batch_size": BATCH_SIZE, "hidden_size": hidden_size, "num_layers": num_layers, "chunk_layers": DECODER_CHUNK_LAYERS, "num_key_value_heads": kv_heads, "head_dim": head_dim, "kv_cache_layout": "stacked_states", "chunks": [], }
	if quant_bits is not None:
		metadata["quantization"] = {"enabled": True, "weight_bits": quant_bits, "embed_path": EMBED_SIDECAR_NAME, "embed_swift_meta": EMBED_SWIFT_META_NAME, }

	hidden_states = torch.zeros(BATCH_SIZE, 1, hidden_size, dtype=torch.bfloat16)
	position_ids = torch.zeros(4, BATCH_SIZE, 1, dtype=torch.long)
	cache_position = torch.tensor([0], dtype=torch.long)
	deepstack_active = torch.zeros(BATCH_SIZE, 1, dtype=torch.bfloat16)
	deepstack = torch.zeros(BATCH_SIZE, 1, hidden_size, dtype=torch.bfloat16)

	for chunk_idx in range(num_chunks):
		layer_start = chunk_idx * DECODER_CHUNK_LAYERS
		layer_end = min(layer_start + DECODER_CHUNK_LAYERS, num_layers)
		is_last = chunk_idx == num_chunks - 1

		chunk = YunaDecoderChunk(language_model, lm_head if is_last else None, layer_start, layer_end, kv_heads, head_dim, ).eval()
		freeze_module(chunk)

		args = (hidden_states, position_ids, cache_position, deepstack_active, deepstack, deepstack, deepstack, )

		print(f"Capturing text decoder chunk {chunk_idx + 1}/{num_chunks} (layers {layer_start}-{layer_end - 1})...")
		with torch.no_grad():
			exported = torch.export.export(chunk, args=args)

		input_names, output_names, state_names = _chunk_io_names(has_lm_head=is_last)
		output_path = OUTPUT_DIR / f"YunaTextDecoder_chunk{chunk_idx}.aimodel"

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

	meta_path = OUTPUT_DIR / "YunaTextDecoder_manifest.json"
	meta_path.write_text(json.dumps(metadata, indent=2))
	print(f"📋 Decoder manifest → {meta_path}")
	return chunk_paths


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export Qwen3-VL to Core AI")
	parser.add_argument("--model-path", type=str, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True, help="Export int-quantized weights (~4x smaller than bf16). Default: on.", )
	parser.add_argument("--quant-bits", type=int, choices=(4, 8), default=8, help="Weight bit-width when --quantize is enabled (default: 8).", )
	parser.add_argument("--text-only", action="store_true", help="Skip vision encoder export to reduce peak RAM.", )
	parser.add_argument("--embed-only", action="store_true", help="Only write the quantized embedding sidecar (very low memory).", )
	parser.add_argument("--vision-only", action="store_true", help="Only export YunaVisionEncoder.aimodel (reuse existing decoder chunks).", )
	return parser.parse_args()


def run(*, model_path: str, output_dir: Path, quant_bits: int | None = 8, text_only: bool = False, embed_only: bool = False, vision_only: bool = False, ) -> None:
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

	print("Loading Qwen3-VL (low_cpu_mem_usage)...")
	model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ).eval()
	freeze_module(model)

	if quant_bits is not None:
		count = quantize_module_weights(model, nbits=quant_bits)
		print(f"Quantized {count} Linear/Embedding weight tensors to int{quant_bits}")

	if not text_only:
		export_vision(model)
	else:
		print("Skipping vision export (--text-only)")

	if vision_only:
		del model
		_free_memory()
		print(f"\nVision export complete → {OUTPUT_DIR.resolve()}")
		return

	export_text_decoder(model, quant_bits=quant_bits)
	del model
	_free_memory()

	print(f"\nFull export complete → {OUTPUT_DIR.resolve()}")
	if not text_only:
		print("   YunaVisionEncoder.aimodel")
	print("   YunaTextDecoder_chunk0.. chunks")
	if quant_bits is not None:
		print(f"   int{quant_bits} quantized weights")
		print(f"   {EMBED_SIDECAR_NAME}")
	print(f"   Functional KV-cache (max {MAX_CACHE_LEN} tokens)")


def main():
	args = parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, quant_bits=None if not args.quantize else args.quant_bits, text_only=args.text_only, embed_only=args.embed_only, vision_only=args.vision_only, )


if __name__ == "__main__":
	main()
