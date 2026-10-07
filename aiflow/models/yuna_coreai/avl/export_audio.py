from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from .._vendor import ensure_python_path

ensure_python_path()

import torch
from .audio_encoder import AVLAudioEncoderStatic, load_avl_audio

DTYPE = torch.float16
DEFAULT_CHUNKS = 8  # 800 mel frames = one host window, 104 tokens


def run(*, model_path: str, output_dir: Path, chunks: int = DEFAULT_CHUNKS, name: str = "YunaAudioEncoder.aimodel", ) -> Path:
	from .._torch_util import ensure_torch

	ensure_torch()
	import coreai.runtime as rt
	from coreai_models.export.macos import export_to_coreai

	src = Path(model_path).expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / name

	print(f"loading AVL audio tower from {src} (K={chunks}) ...")
	tower, projector, _cfg = load_avl_audio(src, dtype=DTYPE)
	enc = AVLAudioEncoderStatic(tower, projector, chunks).eval()
	s = enc.n_tokens
	example = {"input_features": torch.zeros(1, enc.num_mel_bins, chunks * enc.chunk_mel, dtype=DTYPE), "attn_bias": torch.zeros(1, s, s, dtype=DTYPE), }
	print(f"exporting audio graph ({s} tokens × {enc.output_dim}) ...")
	prog = export_to_coreai(enc, example, dynamic_shapes={"input_features": None, "attn_bias": None}, input_names=("input_features", "attn_bias"), output_names=("audio_embeds", ), state_names=(), )
	prog.optimize()
	if out_path.exists():
		shutil.rmtree(out_path)
	prog.save_asset(out_path, rt.AIModelAssetMetadata())
	sz = subprocess.run(["du", "-sh", str(out_path)], capture_output=True, text=True).stdout.split()[0]
	print(f"SAVED {out_path} ({sz})")
	return out_path


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model-path", type=str, required=True)
	ap.add_argument("--output-dir", type=Path, required=True)
	ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
	args = ap.parse_args()
	run(model_path=args.model_path, output_dir=args.output_dir, chunks=args.chunks)


if __name__ == "__main__":
	main()
