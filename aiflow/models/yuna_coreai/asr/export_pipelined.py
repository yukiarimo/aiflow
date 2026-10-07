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
from coreai_models.models.macos.qwen3_asr_pipelined import (PIPELINED_STATE_NAMES, Qwen3ASRPipelinedForCausalLM, )
from transformers import AutoTokenizer

DTYPE = torch.float16
DEFAULT_N_AUDIO = 832


def linear_quant_config() -> dict:
	return {"execution_mode": "eager", "global_config": {"op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric_with_clipping", "granularity": {"type": "per_block", "block_size": 32, "axis": 1}, }}, "op_input_spec": None, "op_output_spec": None, }, "module_type_configs": {"coreai_models.primitives.macos.sdpa.SDPA": None, "coreai_models.primitives.macos.rope.RoPE": None, "coreai_models.primitives.macos.rms_norm.RMSNorm": None, "torch.nn.modules.sparse.Embedding": None, }, "module_name_configs": {r".*lm_head$": None}, }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, vocab: int, max_ctx: int) -> None:
	meta = {"metadata_version": "0.2", "kind": "llm", "name": name, "assets": {"main": f"{name}.aimodel"}, "language": {"tokenizer": hf_id, "vocab_size": vocab, "max_context_length": max_ctx, "embedded_tokenizer": True, "function_map": {"main": ["main"]}, }, "source": {"model_definition": "torch", "hf_model_id": hf_id}, "compression": "int8", "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}, }
	(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _resolve_encoder_src(encoder_src: Path | None, bundle_dir: Path) -> Path | None:
	if encoder_src is not None:
		return encoder_src if encoder_src.exists() else None
	for candidate in (bundle_dir / "YunaASRAudioEncoder.aimodel", bundle_dir.parent / "YunaASRAudioEncoder.aimodel", ):
		if candidate.exists():
			return candidate
	return None


def run(*, ckpt: Path, out_dir: Path, name: str = "YunaASRLM", n_audio: int = DEFAULT_N_AUDIO, max_ctx: int = 4096, encoder_src: Path | None = None, skip_dynamic: bool = False, skip_s1: bool = True, ) -> Path:
	if not ckpt.exists():
		raise FileNotFoundError(f"Missing checkpoint {ckpt} — run `convert asr make-ckpt` first")

	print(f"loading ASR text decoder (fp16, n_audio={n_audio}) from {ckpt} ...")
	model = Qwen3ASRPipelinedForCausalLM.from_hf(str(ckpt), n_audio_tokens=n_audio, target_dtype=DTYPE, max_context_length=max_ctx, ).eval()

	spec = model.build_export_spec(DTYPE, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
	print("quantizing (int8lin: symmetric per-block-32) ...")
	model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], linear_quant_config(), )

	variants: list[tuple[str, dict, str | None]] = []
	if not skip_dynamic:
		variants.append((name, spec, None))
	if not skip_s1:
		variants.append((f"{name}_s1", model.build_export_spec(DTYPE, max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1), f"{name}_s1", ))
	if not variants:
		raise SystemExit("Nothing to export — enable dynamic and/or S=1.")

	bundle_dir = out_dir / name
	if bundle_dir.exists():
		shutil.rmtree(bundle_dir)
	bundle_dir.mkdir(parents=True)

	for vname, vspec, subdir in variants:
		print(f"exporting decoder graph ({vname}) ...")
		prog = export_to_coreai(model, vspec["reference_inputs"], dynamic_shapes=vspec["dynamic_shapes"], input_names=vspec["input_names"], output_names=vspec["output_names"], state_names=PIPELINED_STATE_NAMES, )
		prog.optimize()
		out_parent = bundle_dir / subdir if subdir else bundle_dir
		out_parent.mkdir(parents=True, exist_ok=True)
		aimodel = out_parent / f"{name}.aimodel"
		shutil.rmtree(aimodel, ignore_errors=True)
		prog.save_asset(aimodel, rt.AIModelAssetMetadata())
		sz = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
		print(f"SAVED {aimodel} ({sz})")

	write_bundle_metadata(bundle_dir, name, str(ckpt), model.config.vocab_size, max_ctx)
	AutoTokenizer.from_pretrained(str(ckpt)).save_pretrained(bundle_dir / "tokenizer")

	src_encoder = _resolve_encoder_src(encoder_src, bundle_dir)
	dst_encoder = bundle_dir / "YunaASRAudioEncoder.aimodel"
	if src_encoder is not None:
		if dst_encoder.exists():
			shutil.rmtree(dst_encoder)
		shutil.copytree(src_encoder, dst_encoder)
		print(f"copied audio encoder -> {dst_encoder}")
	else:
		print("WARNING: YunaASRAudioEncoder.aimodel not found — run:\n"
		      "  python -m aiflow.models.yuna_coreai.convert asr --model-path ... --output-dir ... --encoder-only")

	print(f"bundle ready: {bundle_dir}")
	return bundle_dir


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--out-dir", type=Path, required=True)
	ap.add_argument("--name", default="YunaASRLM")
	ap.add_argument("--n-audio", type=int, default=DEFAULT_N_AUDIO)
	ap.add_argument("--max-ctx", type=int, default=4096)
	ap.add_argument("--encoder-src", type=Path, default=None)
	ap.add_argument("--skip-dynamic", action="store_true")
	ap.add_argument("--skip-s1", action="store_true", default=True)
	args = ap.parse_args()
	run(ckpt=args.ckpt, out_dir=args.out_dir, name=args.name, n_audio=args.n_audio, max_ctx=args.max_ctx, encoder_src=args.encoder_src, skip_dynamic=args.skip_dynamic, skip_s1=args.skip_s1, )


if __name__ == "__main__":
	main()
