from __future__ import annotations
import argparse
import asyncio
import shutil
from pathlib import Path
from .._vendor import ensure_python_path

ensure_python_path()

from coreai_models.export.pipeline import ExportConfig, export_model


async def _export_async(*, ckpt: Path, output_dir: Path, name: str, max_ctx: int) -> Path:
	bundle_parent = output_dir.parent if output_dir.name == name else output_dir
	bundle_parent.mkdir(parents=True, exist_ok=True)

	if (bundle_parent / name).exists():
		shutil.rmtree(bundle_parent / name)

	path = await export_model(ExportConfig(hf_model_id=str(ckpt), variant="macOS", max_context_length=max_ctx, compute_precision="float16", compression="8bit", output_dir=str(bundle_parent), output_name=name, overwrite=True, ))
	return Path(path)


def run(*, model_path: str, output_dir: Path, name: str = "YunaTextLM", max_ctx: int = 4096) -> Path:
	from . import make_ckpt

	model_root = Path(model_path).expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	staging = output_dir.parent / f".{output_dir.name}_staging"
	text_ckpt = staging / "text_ckpt"

	if staging.exists():
		shutil.rmtree(staging)
	staging.mkdir(parents=True)

	print(f"\n=== Text step 1/2: extract text decoder → {text_ckpt} ===")
	make_ckpt.run(src=model_root, dst=text_ckpt)

	print(f"\n=== Text step 2/2: pipelined LanguageBundle → {output_dir} ===")
	bundle = asyncio.run(_export_async(ckpt=text_ckpt, output_dir=output_dir, name=name, max_ctx=max_ctx))

	shutil.rmtree(staging, ignore_errors=True)
	print(f"\nText bundle ready → {bundle}")
	return bundle


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model-path", type=str, required=True)
	ap.add_argument("--output-dir", type=Path, required=True)
	ap.add_argument("--name", default="YunaTextLM")
	ap.add_argument("--max-ctx", type=int, default=4096)
	args = ap.parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, name=args.name, max_ctx=args.max_ctx, )


if __name__ == "__main__":
	main()
