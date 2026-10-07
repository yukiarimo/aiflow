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


def run(*, model_path: str, output_dir: Path, n_audio: int = 832, max_ctx: int = 4096, encoder_src: Path | None = None, encoder_only: bool = False, decoder_only: bool = False, keep_staging: bool = False, ) -> Path:
	"""Export Qwen3-ASR like ``convert vlm``: one CLI, separate heavy steps inside."""
	if encoder_only and decoder_only:
		raise SystemExit("Use at most one of --encoder-only / --decoder-only.")

	model_root = Path(model_path).expanduser().resolve()
	if not model_root.is_dir():
		hint = ""
		if "path/to" in str(model_path):
			hint = " (did you copy the README placeholder? Use your real qwen3-asr HF folder.)"
		raise FileNotFoundError(f"Missing qwen3-asr folder: {model_root}{hint}")

	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	staging = _staging_dir(output_dir)
	text_ckpt = staging / "text_ckpt"
	encoder_stage = staging / "encoder"
	staged_encoder = encoder_stage / "YunaASRAudioEncoder.aimodel"

	if encoder_only:
		print(f"\n=== ASR step 1/1: audio encoder → {output_dir} ===")
		from .._torch_util import ensure_torch

		ensure_torch()
		from . import export_chunked

		export_chunked.run(model_path=str(model_root), output_dir=output_dir, quant_bits=None, audio_only=True, )
		_free_memory()
		print(f"\nAudio encoder ready → {output_dir / 'YunaASRAudioEncoder.aimodel'}")
		return output_dir

	if not decoder_only:
		if staging.exists():
			shutil.rmtree(staging)
		staging.mkdir(parents=True)
		encoder_stage.mkdir(parents=True)

		print(f"\n=== ASR step 1/3: extract text decoder → {text_ckpt} ===")
		from . import make_ckpt

		make_ckpt.run(src=model_root, dst=text_ckpt)
		_free_memory()

		print(f"\n=== ASR step 2/3: audio encoder → {encoder_stage} ===")
		from .._torch_util import ensure_torch

		ensure_torch()
		from . import export_chunked

		export_chunked.run(model_path=str(model_root), output_dir=encoder_stage, quant_bits=None, audio_only=True, )
		_free_memory()
		resolved_encoder = encoder_src or staged_encoder
	else:
		if text_ckpt.is_dir():
			print(f"Reusing staged text checkpoint → {text_ckpt}")
		else:
			staging.mkdir(parents=True, exist_ok=True)
			print(f"\n=== ASR step 1/1: extract text decoder → {text_ckpt} ===")
			from . import make_ckpt

			make_ckpt.run(src=model_root, dst=text_ckpt)
			_free_memory()

		resolved_encoder = encoder_src or (output_dir / "YunaASRAudioEncoder.aimodel")
		if not resolved_encoder.exists():
			raise FileNotFoundError(f"Missing {resolved_encoder} — run full export or pass --encoder-src")

	step_label = "1/1" if decoder_only else "3/3"
	print(f"\n=== ASR step {step_label}: pipelined LanguageBundle → {output_dir} ===")
	from .._torch_util import ensure_torch

	ensure_torch()
	from . import export_pipelined

	bundle_dir = export_pipelined.run(ckpt=text_ckpt, out_dir=output_dir.parent, name=output_dir.name, n_audio=n_audio, max_ctx=max_ctx, encoder_src=resolved_encoder, )
	_free_memory()

	if not keep_staging and staging.exists():
		shutil.rmtree(staging)
		print(f"Removed staging → {staging}")

	print(f"\nASR bundle ready → {bundle_dir}")
	return bundle_dir
