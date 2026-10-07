from __future__ import annotations
import argparse
from pathlib import Path
from ._torch_util import ensure_torch


def _add_asr_flags(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--model-path", type=str, help="Full qwen3-asr HF folder")
	parser.add_argument("--output-dir", type=Path, help="Bundle output directory")
	parser.add_argument("--n-audio", type=int, default=832, help="Max audio token slots (default: 832)")
	parser.add_argument("--max-ctx", type=int, default=4096)
	parser.add_argument("--encoder-src", type=Path, default=None, help="Existing YunaASRAudioEncoder.aimodel (decoder-only or override copy source).", )
	parser.add_argument("--encoder-only", action="store_true", help="Export audio encoder only.")
	parser.add_argument("--decoder-only", action="store_true", help="Export pipelined decoder only (reuse encoder in output-dir or --encoder-src).", )
	parser.add_argument("--keep-staging", action="store_true", help="Keep .YunaASRLM_staging/ after a full export (text ckpt for re-runs).", )


def _build_parser() -> argparse.ArgumentParser:
	root = argparse.ArgumentParser(prog="python -m aiflow.models.yuna_coreai.convert", description="Export Qwen3-VL / Qwen3-ASR / YunaAVL Hugging Face checkpoints to Core AI bundles.", )
	sub = root.add_subparsers(dest="task", required=True)

	vlm = sub.add_parser("vlm", help="Qwen3-VL: chunked int8 decoder + YunaVisionEncoder")
	vlm.add_argument("--model-path", type=str, required=True, help="HF Qwen3-VL folder")
	vlm.add_argument("--output-dir", type=Path, required=True, help="Bundle output directory")
	vlm.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True, help="Export int-quantized decoder weights (default: on).", )
	vlm.add_argument("--quant-bits", type=int, choices=(4, 8), default=8)
	vlm.add_argument("--text-only", action="store_true", help="Skip vision encoder export.")
	vlm.add_argument("--embed-only", action="store_true", help="Only write quantized embed sidecar.")
	vlm.add_argument("--vision-only", action="store_true", help="Only export YunaVisionEncoder.aimodel (reuse existing decoder chunks).", )
	vlm.add_argument("--pipelined", action="store_true", help="Export pipelined VLM LanguageBundle (YunaVLMLM) instead of chunked decoder.", )
	vlm.add_argument("--grid", type=int, default=14, help="Merged vision grid side (default: 14 → 196 tokens).")
	vlm.add_argument("--max-ctx", type=int, default=4096, help="Max context for pipelined export (default: 4096).", )
	vlm.add_argument("--vlm-name", default="YunaVLMLM", help="Pipelined bundle folder name (default: YunaVLMLM).", )
	vlm.add_argument("--vision-src", type=Path, default=None, help="Existing YunaVisionEncoder.aimodel to copy into pipelined bundle.", )
	vlm.add_argument("--decoder-only", action="store_true", help="Pipelined: export decoder only (reuse vision in output-dir).", )

	asr = sub.add_parser("asr", help="Qwen3-ASR: pipelined LanguageBundle (runs text-ckpt, encoder, decoder separately)", )
	_add_asr_flags(asr)

	asr_sub = asr.add_subparsers(dest="asr_mode", required=False)

	mk = asr_sub.add_parser("make-ckpt", help="[advanced] Extract text decoder HF checkpoint")
	mk.add_argument("--src", type=Path, required=True, help="Full qwen3-asr HF folder")
	mk.add_argument("--dst", type=Path, required=True, help="Output text-only Qwen3 checkpoint")

	pip = asr_sub.add_parser("pipelined", help="[advanced] Pipelined LanguageBundle only")
	pip.add_argument("--ckpt", type=Path, required=True, help="Text decoder HF folder (make-ckpt output)")
	pip.add_argument("--out-dir", type=Path, required=True, help="Parent dir; writes <name>/ bundle")
	pip.add_argument("--name", default="YunaASRLM", help="Bundle folder name (default: YunaASRLM)")
	pip.add_argument("--n-audio", type=int, default=832, help="Max audio token slots (default: 832)")
	pip.add_argument("--max-ctx", type=int, default=4096)
	pip.add_argument("--encoder-src", type=Path, default=None, help="YunaASRAudioEncoder.aimodel to copy into the bundle.", )
	pip.add_argument("--skip-dynamic", action="store_true", help="Skip dynamic-query ship bundle.")
	pip.add_argument("--skip-s1", action="store_true", default=True, help="Skip static S=1 oracle bundle.")

	ch = asr_sub.add_parser("chunked", help="[advanced] Legacy chunked decoder + audio encoder export")
	ch.add_argument("--model-path", type=str, required=True, help="Full qwen3-asr HF folder")
	ch.add_argument("--output-dir", type=Path, required=True)
	ch.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
	ch.add_argument("--quant-bits", type=int, choices=(4, 8), default=8)
	ch.add_argument("--audio-only", action="store_true")
	ch.add_argument("--text-only", action="store_true")
	ch.add_argument("--embed-only", action="store_true")

	text = sub.add_parser("text", help="Qwen3-VL text → pipelined YunaTextLM (fast text-only path)")
	text.add_argument("--model-path", type=str, required=True, help="HF Qwen3-VL folder (e.g. YunaNew)")
	text.add_argument("--output-dir", type=Path, required=True, help="Bundle output directory (YunaTextLM/)")
	text.add_argument("--name", default="YunaTextLM", help="Bundle folder name (default: YunaTextLM)")
	text.add_argument("--max-ctx", type=int, default=4096)

	avl = sub.add_parser("avl", help="YunaAVL: pipelined 2B LM + ViT + 2048-d audio encoder")
	avl.add_argument("--model-path", type=str, required=True, help="HF YunaAVL folder")
	avl.add_argument("--output-dir", type=Path, required=True, help="Bundle output directory (YunaAVL/)")
	avl.add_argument("--name", default="YunaAVL", help="Bundle folder name (default: YunaAVL)")
	avl.add_argument("--grid", type=int, default=14)
	avl.add_argument("--max-ctx", type=int, default=4096)
	avl.add_argument("--n-audio", type=int, default=832, help="Max audio token slots (default: 832)")
	avl.add_argument("--encoder-only", action="store_true", help="Export YunaAudioEncoder.aimodel only.")
	avl.add_argument("--vision-only", action="store_true", help="Export YunaVisionEncoder.aimodel only.")
	avl.add_argument("--decoder-only", action="store_true", help="Export pipelined decoder only.")
	avl.add_argument("--keep-staging", action="store_true")

	return root


def _run_asr_unified(args: argparse.Namespace) -> None:
	if not args.model_path or not args.output_dir:
		raise SystemExit("asr requires --model-path and --output-dir (or use an asr subcommand).")

	from .asr import pipeline

	pipeline.run(model_path=args.model_path, output_dir=args.output_dir, n_audio=args.n_audio, max_ctx=args.max_ctx, encoder_src=args.encoder_src, encoder_only=args.encoder_only, decoder_only=args.decoder_only, keep_staging=args.keep_staging, )


def main(argv: list[str] | None = None) -> None:
	parser = _build_parser()
	args = parser.parse_args(argv)

	if args.task == "vlm":
		ensure_torch()
		if args.pipelined:
			from .vlm import pipeline

			pipeline.run(model_path=args.model_path, output_dir=args.output_dir, name=args.vlm_name, max_ctx=args.max_ctx, grid=args.grid, vision_src=args.vision_src, decoder_only=args.decoder_only, vision_only=args.vision_only, )
			return

		from .vlm import export_chunked

		export_chunked.run(model_path=args.model_path, output_dir=args.output_dir, quant_bits=None if not args.quantize else args.quant_bits, text_only=args.text_only, embed_only=args.embed_only, vision_only=args.vision_only, )
		return

	if args.task == "asr":
		if args.asr_mode is None:
			_run_asr_unified(args)
			return

		if args.asr_mode == "make-ckpt":
			from .asr import make_ckpt

			make_ckpt.run(src=args.src, dst=args.dst)
			return

		if args.asr_mode == "pipelined":
			ensure_torch()
			from .asr import export_pipelined

			export_pipelined.run(ckpt=args.ckpt, out_dir=args.out_dir, name=args.name, n_audio=args.n_audio, max_ctx=args.max_ctx, encoder_src=args.encoder_src, skip_dynamic=args.skip_dynamic, skip_s1=args.skip_s1, )
			return

		if args.asr_mode == "chunked":
			ensure_torch()
			from .asr import export_chunked

			export_chunked.run(model_path=args.model_path, output_dir=args.output_dir, quant_bits=None if not args.quantize else args.quant_bits, audio_only=args.audio_only, text_only=args.text_only, embed_only=args.embed_only, )
			return

	if args.task == "text":
		ensure_torch()
		from .text import export_pipelined

		export_pipelined.run(model_path=args.model_path, output_dir=args.output_dir, name=args.name, max_ctx=args.max_ctx, )
		return

	if args.task == "avl":
		from .avl import pipeline

		pipeline.run(model_path=args.model_path, output_dir=args.output_dir, name=args.name, max_ctx=args.max_ctx, grid=args.grid, n_audio=args.n_audio, decoder_only=args.decoder_only, encoder_only=args.encoder_only, vision_only=args.vision_only, keep_staging=args.keep_staging, )
		return

	parser.error(f"Unknown task {args.task!r}")


if __name__ == "__main__":
	main()
