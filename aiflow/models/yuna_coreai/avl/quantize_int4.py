from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from coreai.authoring import AIModelAsset
from coreai_models.export.compiler import apply_mlir_quantization


async def run(src: Path, dst: Path) -> None:
	src = src.expanduser().resolve()
	dst = dst.expanduser().resolve()
	if dst.exists():
		shutil.rmtree(dst)
	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copytree(src, dst)

	asset = AIModelAsset.load(dst)
	before = asset.summary()
	print("before:", before)

	prog = await apply_mlir_quantization(asset.program, {"type": "int8", "symmetric": True, "granularity": "per_block", "block_size": 32, }, )
	if dst.exists():
		shutil.rmtree(dst)
	prog.save_asset(dst)
	after = AIModelAsset.load(dst).summary()
	print("after:", after)


def main() -> None:
	import asyncio

	ap = argparse.ArgumentParser()
	ap.add_argument("--src", type=Path, required=True)
	ap.add_argument("--dst", type=Path, required=True)
	args = ap.parse_args()
	asyncio.run(run(args.src, args.dst))


if __name__ == "__main__":
	main()
