from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from .._vendor import ensure_python_path

ensure_python_path()
import torch
import coreai.runtime as rt
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.compression import quantize_pytorch_model
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.qwen3_vl import (PIPELINED_STATE_NAMES, Qwen3VLPipelinedForCausalLM, Qwen3VLVisionEncoder, )
from transformers import AutoTokenizer

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
	return {
	    "execution_mode": "eager",
	    "global_config": {
	        "op_state_spec": {
	            "weight": {
	                "dtype": dtype,  # Embeddings take the same INT8 block spec as the linears. An INT4 embedding was measured as the fuzzy one.
	                "qscheme": "symmetric_with_clipping",
	                "granularity": {
	                    "type": "per_block",
	                    "block_size": 32,
	                    "axis": 1
	                },
	            }
	        },
	        "op_input_spec": None,
	        "op_output_spec": None,
	    },
	    "module_type_configs": {
	        "coreai_models.primitives.macos.sdpa.SDPA": None,
	        "coreai_models.primitives.macos.rope.RoPE": None,
	        "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
	    },
	    "module_name_configs": {
	        r".*lm_head$": None
	    },
	}


def head_quant_spec() -> dict:
	return {"op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric", "granularity": {"type": "per_block", "block_size": 32, "axis": 1}, }}, "op_input_spec": None, "op_output_spec": None, }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, vocab: int, max_ctx: int, *, grid: int) -> None:
	meta = {"metadata_version": "0.2", "kind": "llm", "name": name, "assets": {"main": f"{name}.aimodel"}, "language": {"tokenizer": hf_id, "vocab_size": vocab, "max_context_length": max_ctx, "embedded_tokenizer": True, "function_map": {"main": ["main"]}, }, "vlm": {"image_tokens": grid * grid, "grid": grid, "vision_asset": "YunaVisionEncoder.aimodel", }, "source": {"model_definition": "torch", "hf_model_id": hf_id}, "compression": "int8hu", "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}, }
	(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def export_vision_encoder(*, vision_model_path: str, bundle_dir: Path, grid: int, ) -> Path:
	"""Export zoo-format vision encoder: patches → (image_embeds, deepstack_embeds)."""
	out_path = bundle_dir / "YunaVisionEncoder.aimodel"
	print(f"loading vision tower from {vision_model_path} (grid {grid}x{grid}) ...")
	vis = Qwen3VLVisionEncoder.from_hf(vision_model_path, target_dtype=DTYPE, grid_h=grid, grid_w=grid)
	vcfg = vis.vcfg
	patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
	patches = torch.zeros(vis.n_patches, patch_dim, dtype=DTYPE)

	print("exporting vision graph ...")
	prog = export_to_coreai(vis, {"patches": patches}, dynamic_shapes={"patches": None}, input_names=("patches", ), output_names=("image_embeds", "deepstack_embeds"), state_names=(), )
	prog.optimize()
	if out_path.exists():
		shutil.rmtree(out_path)
	prog.save_asset(out_path, rt.AIModelAssetMetadata())
	sz = subprocess.run(["du", "-sh", str(out_path)], capture_output=True, text=True).stdout.split()[0]
	print(f"SAVED {out_path} ({sz})")
	return out_path


def run(*, model_path: str, output_dir: Path, name: str = "YunaVLMLM", max_ctx: int = 4096, grid: int = 14, mode: str = "int8hu", vision_model_path: str | None = None, vision_src: Path | None = None, skip_vision: bool = False, skip_dynamic: bool = True, skip_s1: bool = False, ) -> Path:
	model_root = Path(model_path).expanduser().resolve()
	vision_root = Path(vision_model_path or model_path).expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	bundle_dir = output_dir if output_dir.name == name else output_dir / name

	if bundle_dir.exists():
		shutil.rmtree(bundle_dir)
	bundle_dir.mkdir(parents=True)

	print(f"loading pipelined VLM decoder from {model_root} (grid {grid}x{grid}) ...")
	model = Qwen3VLPipelinedForCausalLM.from_hf(str(model_root), grid_h=grid, grid_w=grid, target_dtype=DTYPE, max_context_length=max_ctx, ).eval()
	cfg = model.config

	spec = model.build_export_spec(DTYPE, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

	if mode in ("int8lin", "int8hu"):
		cfg_q = linear_quant_config("int8")
		if mode == "int8hu":
			cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
			model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
		print(f"quantizing decoder ({mode}) ...")
		model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q, )

	variants: list[tuple[str, dict, str | None]] = []
	if not skip_dynamic:
		variants.append((name, spec, None))
	if not skip_s1:
		s1_name = f"{name}_s1"
		variants.append((s1_name, model.build_export_spec(DTYPE, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1), s1_name, ))
	if not variants:
		raise SystemExit("Nothing to export — enable dynamic and/or S=1.")

	ship_name = f"{name}_s1" if not skip_s1 and skip_dynamic else name
	for vname, vspec, subdir in variants:
		print(f"exporting decoder graph ({vname}) ...")
		prog = export_to_coreai(model, vspec["reference_inputs"], dynamic_shapes=vspec["dynamic_shapes"], input_names=vspec["input_names"], output_names=vspec["output_names"], state_names=PIPELINED_STATE_NAMES, )
		prog.optimize()

		flatten_s1 = skip_dynamic and not skip_s1 and subdir is not None  # Ship layout: when only the S=1 bundle is exported, place the .aimodel at the bundle root.
		out_parent = bundle_dir if flatten_s1 else (bundle_dir / subdir if subdir else bundle_dir)
		out_parent.mkdir(parents=True, exist_ok=True)
		aimodel = out_parent / f"{name}.aimodel"
		if aimodel.exists():
			shutil.rmtree(aimodel)
		prog.save_asset(aimodel, rt.AIModelAssetMetadata())
		sz = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
		print(f"SAVED {aimodel} ({sz})")

	write_bundle_metadata(bundle_dir, name, str(vision_root), cfg.vocab_size, max_ctx, grid=grid)
	AutoTokenizer.from_pretrained(str(model_root)).save_pretrained(bundle_dir / "tokenizer")

	if not skip_vision:
		export_vision_encoder(vision_model_path=str(vision_root), bundle_dir=bundle_dir, grid=grid, )

	print(f"bundle ready → {bundle_dir}")
	print("Ship config: S=1 static-query decoder (int8hu) + fp16 vision encoder. "
	      "Set COREAI_CHUNK_THRESHOLD=1 at runtime; do not use the dynamic-query twin on macOS 27 beta.")
	return bundle_dir


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model-path", type=str, required=True)
	ap.add_argument("--output-dir", type=Path, required=True)
	ap.add_argument("--name", default="YunaVLMLM")
	ap.add_argument("--max-ctx", type=int, default=4096)
	ap.add_argument("--grid", type=int, default=14)
	ap.add_argument("--mode", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
	ap.add_argument("--vision-model-path", type=str, default=None, help="Full Qwen3-VL HF folder for vision export (default: --model-path).", )
	ap.add_argument("--vision-src", type=Path, default=None)
	ap.add_argument("--skip-vision", action="store_true")
	ap.add_argument("--skip-dynamic", action="store_true", default=True)
	ap.add_argument("--skip-s1", action="store_true")
	args = ap.parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, name=args.name, max_ctx=args.max_ctx, grid=args.grid, mode=args.mode, vision_model_path=args.vision_model_path, vision_src=args.vision_src, skip_vision=args.skip_vision, skip_dynamic=args.skip_dynamic, skip_s1=args.skip_s1, )


if __name__ == "__main__":
	main()
