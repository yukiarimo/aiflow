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


def run(*, model_path: str, output_dir: Path, name: str = "YunaAVL", max_ctx: int = 4096, grid: int = 14, n_audio: int = 832, mode: str = "int8hu", decoder_only: bool = False, encoder_only: bool = False, vision_only: bool = False, keep_staging: bool = False, ) -> Path:
	if sum(bool(x) for x in (decoder_only, encoder_only, vision_only)) > 1:
		raise SystemExit("Use at most one of --decoder-only / --encoder-only / --vision-only.")

	model_root = Path(model_path).expanduser().resolve()
	if not model_root.is_dir():
		raise FileNotFoundError(f"Missing YunaAVL folder: {model_root}")

	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if encoder_only:
		print(f"\n=== AVL step 1/1: audio encoder → {output_dir} ===")
		from .export_audio import run as export_audio

		export_audio(model_path=str(model_root), output_dir=output_dir)
		_free_memory()
		return output_dir

	if vision_only:
		print(f"\n=== AVL step 1/1: vision encoder → {output_dir} ===")
		from ..vlm.export_pipelined import export_vision_encoder

		export_vision_encoder(vision_model_path=str(model_root), bundle_dir=output_dir, grid=grid)
		_free_memory()
		return output_dir

	staging = _staging_dir(output_dir)
	text_ckpt = staging / "text_ckpt"
	ckpt_for_decoder = model_root

	if not decoder_only:
		if staging.exists():
			shutil.rmtree(staging)
		staging.mkdir(parents=True)
		print(f"\n=== AVL step 1/2: extract text decoder → {text_ckpt} ===")
		from ..text import make_ckpt

		make_ckpt.run(src=model_root, dst=text_ckpt)
		_free_memory()
		ckpt_for_decoder = text_ckpt

	step = "1/1" if decoder_only else "2/2"
	print(f"\n=== AVL step {step}: decoder + vision + audio → {output_dir} ===")
	from .export_pipelined import run as export_pipelined

	bundle = export_pipelined(model_path=str(ckpt_for_decoder), output_dir=output_dir, name=name, max_ctx=max_ctx, grid=grid, n_audio=n_audio, mode=mode, vision_model_path=str(model_root), tokenizer_src=str(model_root), skip_vision=decoder_only, skip_audio=decoder_only, skip_dynamic=False, skip_s1=True, )
	_free_memory()

	if not keep_staging and staging.exists():
		shutil.rmtree(staging)
		print(f"Removed staging → {staging}")

	print(f"\nPipelined AVL bundle ready → {bundle}")
	return bundle
