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
from coreai_models.models.macos.qwen3_avl import Qwen3AVLPipelinedForCausalLM
from coreai_models.models.macos.qwen3_vl import PIPELINED_STATE_NAMES
from transformers import AutoTokenizer
from ..vlm.export_pipelined import (DTYPE, export_vision_encoder, head_quant_spec, linear_quant_config, )


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, vocab: int, max_ctx: int, *, grid: int, n_audio: int, ) -> None:
	meta = {"metadata_version": "0.2", "kind": "llm", "name": name, "assets": {"main": f"{name}.aimodel"}, "language": {"tokenizer": hf_id, "vocab_size": vocab, "max_context_length": max_ctx, "embedded_tokenizer": True, "function_map": {"main": ["main"]}, }, "vlm": {"image_tokens": grid * grid, "grid": grid, "vision_asset": "YunaVisionEncoder.aimodel", "image_slot_base": 0, }, "avl": {"audio_tokens": n_audio, "audio_slot_base": grid * grid, "hidden": 2048, "audio_asset": "YunaAudioEncoder.aimodel", "window_tokens": 104, "window_mel": 800, }, "source": {"model_definition": "torch", "hf_model_id": hf_id}, "compression": "int8hu", "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}, }
	(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def run(*, model_path: str, output_dir: Path, name: str = "YunaAVL", max_ctx: int = 4096, grid: int = 14, n_audio: int = 832, mode: str = "int8hu", vision_model_path: str | None = None, skip_vision: bool = False, skip_audio: bool = False, skip_dynamic: bool = False, skip_s1: bool = True, tokenizer_src: str | None = None, ) -> Path:
	model_root = Path(model_path).expanduser().resolve()
	vision_root = Path(vision_model_path or model_path).expanduser().resolve()
	tok_root = Path(tokenizer_src or vision_model_path or model_path).expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	bundle_dir = output_dir if output_dir.name == name else output_dir / name

	if bundle_dir.exists() and not (skip_vision or skip_audio):
		shutil.rmtree(bundle_dir)
	bundle_dir.mkdir(parents=True, exist_ok=True)
	stale = bundle_dir / f"{name}.aimodel"
	if stale.exists():
		shutil.rmtree(stale)

	print(f"loading pipelined AVL decoder from {model_root} "
	      f"(grid {grid}x{grid}, n_audio={n_audio}) ...")
	model = Qwen3AVLPipelinedForCausalLM.from_hf(str(model_root), grid_h=grid, grid_w=grid, n_audio_tokens=n_audio, target_dtype=DTYPE, max_context_length=max_ctx, ).eval()
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

	for vname, vspec, subdir in variants:
		print(f"exporting decoder graph ({vname}) ...")
		prog = export_to_coreai(model, vspec["reference_inputs"], dynamic_shapes=vspec["dynamic_shapes"], input_names=vspec["input_names"], output_names=vspec["output_names"], state_names=PIPELINED_STATE_NAMES, )
		prog.optimize()
		flatten_s1 = skip_dynamic and not skip_s1 and subdir is not None
		out_parent = bundle_dir if flatten_s1 else (bundle_dir / subdir if subdir else bundle_dir)
		out_parent.mkdir(parents=True, exist_ok=True)
		aimodel = out_parent / f"{name}.aimodel"
		if aimodel.exists():
			shutil.rmtree(aimodel)
		prog.save_asset(aimodel, rt.AIModelAssetMetadata())
		sz = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
		print(f"SAVED {aimodel} ({sz})")

	write_bundle_metadata(bundle_dir, name, str(tok_root), cfg.vocab_size, max_ctx, grid=grid, n_audio=n_audio)
	AutoTokenizer.from_pretrained(str(tok_root), trust_remote_code=True).save_pretrained(bundle_dir / "tokenizer")

	if not skip_vision:
		export_vision_encoder(vision_model_path=str(vision_root), bundle_dir=bundle_dir, grid=grid, )

	if not skip_audio:
		from .export_audio import run as export_audio

		export_audio(model_path=str(vision_root), output_dir=bundle_dir)

	print(f"bundle ready → {bundle_dir}")
	print("Ship config: dynamic-query AVL decoder (int8hu, one graph for prefill and decode) + fp16 ViT + fp16 AuT→2048. "
	      "The app prefills 128 tokens a pass (COREAI_CHUNK_THRESHOLD=128); an S=1 decoder crashes it. Image slots vocab+0..195; audio slots vocab+196...")
	return bundle_dir


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model-path", type=str, required=True)
	ap.add_argument("--output-dir", type=Path, required=True)
	ap.add_argument("--name", default="YunaAVL")
	ap.add_argument("--max-ctx", type=int, default=4096)
	ap.add_argument("--grid", type=int, default=14)
	ap.add_argument("--n-audio", type=int, default=832)
	ap.add_argument("--mode", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
	ap.add_argument("--vision-model-path", type=str, default=None)
	ap.add_argument("--skip-vision", action="store_true")
	ap.add_argument("--skip-audio", action="store_true")
	ap.add_argument("--skip-dynamic", action="store_true")
	ap.add_argument("--skip-s1", action="store_true", default=True)
	args = ap.parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, name=args.name, max_ctx=args.max_ctx, grid=args.grid, n_audio=args.n_audio, mode=args.mode, vision_model_path=args.vision_model_path, skip_vision=args.skip_vision, skip_audio=args.skip_audio, skip_dynamic=args.skip_dynamic, skip_s1=args.skip_s1, )


if __name__ == "__main__":
	main()
