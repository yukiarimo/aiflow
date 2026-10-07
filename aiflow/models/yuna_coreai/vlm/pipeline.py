from __future__ import annotations
import gc
import shutil
from pathlib import Path


def _free_memory() -> None:
	gc.collect()
	try:
		import torch

		if torch.backends.mps.is_available():
			torch.mps.empty_cache()
	except ImportError:
		pass


def _staging_dir(output_dir: Path) -> Path:
	return output_dir.parent / f".{output_dir.name}_staging"


def run(*, model_path: str, output_dir: Path, name: str = "YunaVLMLM", max_ctx: int = 4096, grid: int = 14, mode: str = "int8hu", vision_src: Path | None = None, decoder_only: bool = False, vision_only: bool = False, keep_staging: bool = False, ) -> Path:
	model_root = Path(model_path).expanduser().resolve()
	if not model_root.is_dir():
		raise FileNotFoundError(f"Missing Qwen3-VL folder: {model_root}")

	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if vision_only:
		print(f"\n=== VLM pipelined step 1/1: vision encoder → {output_dir} ===")
		from . import export_chunked

		export_chunked.run(model_path=str(model_root), output_dir=output_dir, quant_bits=None, vision_only=True, )
		_free_memory()
		return output_dir

	staging = _staging_dir(output_dir)
	text_ckpt = staging / "text_ckpt"
	ckpt_for_decoder = model_root

	if not decoder_only:
		if staging.exists():
			shutil.rmtree(staging)
		staging.mkdir(parents=True)
		print(f"\n=== VLM pipelined step 1/2: extract text decoder → {text_ckpt} ===")
		from ..text import make_ckpt

		make_ckpt.run(src=model_root, dst=text_ckpt)
		_free_memory()
		ckpt_for_decoder = text_ckpt

	step = "1/1" if decoder_only else "2/2"
	print(f"\n=== VLM pipelined step {step}: decoder + vision → {output_dir} ===")
	from . import export_pipelined

	bundle = export_pipelined.run(model_path=str(ckpt_for_decoder), output_dir=output_dir, name=name, max_ctx=max_ctx, grid=grid, mode=mode, vision_model_path=str(model_root), vision_src=vision_src or (output_dir.parent / "YunaNewVLM" / "YunaVisionEncoder.aimodel"), skip_vision=False, skip_dynamic=True, skip_s1=False, )
	_free_memory()

	if not keep_staging and staging.exists():
		shutil.rmtree(staging)
		print(f"Removed staging → {staging}")

	print(f"\nPipelined VLM bundle ready → {bundle}")
	return bundle
