from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import mlx.core as mx

AUDIO_SPECIAL_TOKENS = ["<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<non_speech>", "<|audio_pad|>", ]


def _load_json(path: Path):
	with open(path, encoding="utf-8") as f:
		return json.load(f)


def _save_json(path: Path, data):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)
		f.write("\n")


def build_avlm_config(vl_cfg: dict, asr_cfg: dict) -> dict:
	text_config = dict(vl_cfg["text_config"])
	rope = text_config.get("rope_parameters") or {}  # normalize rope for MLX TextConfig (also keeps HF rope_parameters)
	if "rope_theta" in rope:
		text_config["rope_theta"] = rope["rope_theta"]
	if "rope_scaling" not in text_config and rope:
		scaling = {k: v for k, v in rope.items() if k != "rope_theta"}
		if "rope_type" in scaling and "type" not in scaling:
			scaling["type"] = scaling["rope_type"]
		text_config["rope_scaling"] = scaling

	audio_config = dict(asr_cfg["audio_config"])
	if "max_source_positions" not in audio_config and "max_position_embeddings" in audio_config:
		audio_config["max_source_positions"] = audio_config["max_position_embeddings"]

	return {"architectures": ["Qwen3AVLMForConditionalGeneration"], "model_type": "qwen3_avlm", "dtype": vl_cfg.get("dtype", "bfloat16"), "bos_token_id": vl_cfg.get("bos_token_id", 151643), "eos_token_id": vl_cfg.get("eos_token_id", 151643), "pad_token_id": vl_cfg.get("pad_token_id", 151654), "tie_word_embeddings": True, "image_token_id": vl_cfg.get("image_token_id", 151655), "video_token_id": vl_cfg.get("video_token_id", 151656), "vision_start_token_id": vl_cfg.get("vision_start_token_id", 151652), "vision_end_token_id": vl_cfg.get("vision_end_token_id", 151653), "audio_token_id": asr_cfg.get("audio_token_id", 151676), "audio_start_token_id": 151669, "audio_end_token_id": 151670, "text_config": text_config, "vision_config": vl_cfg["vision_config"], "audio_config": audio_config, "vocab_size": text_config.get("vocab_size", 151936), }


def merge_weights(vl_path: Path, asr_path: Path) -> dict:
	from .qwen3_avlm import Model
	from ..qwen3_vl.vision import VisionModel

	vl_w = mx.load(str(vl_path / "model.safetensors"))
	asr_w = mx.load(str(asr_path / "model.safetensors"))

	raw = {}  # keep HF-style keys first, then same sanitize path as load_model
	for k, v in vl_w.items():
		if k.startswith("model.visual.") or k.startswith("model.language_model."):
			raw[k] = v
	for k, v in asr_w.items():
		if k.startswith("model.audio_tower.") or k.startswith("model.multi_modal_projector."):
			raw[k] = v

	merged = Model.sanitize(raw)
	merged = VisionModel.sanitize(VisionModel, merged)  # VisionModel.sanitize is an instance method that ignores self
	return merged


def _asr_token_id(asr_tok_json, token):
	"""Resolve token → id from ASR tokenizer.json (added_tokens + model.vocab)."""
	for entry in asr_tok_json.get("added_tokens") or []:
		if entry.get("content") == token:
			return int(entry["id"])
	vocab = (asr_tok_json.get("model") or {}).get("vocab") or {}
	if token in vocab:
		return int(vocab[token])
	return None


def copy_tokenizer_with_audio_tokens(vl_path: Path, asr_path: Path, out_path: Path):
	"""VL tokenizer as base; inject ASR audio specials at their canonical IDs."""
	tok_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
	for name in tok_names:
		src = vl_path / name
		if src.exists():
			shutil.copy2(src, out_path / name)
	if not (out_path / "tokenizer.json").exists():
		raise FileNotFoundError(f"VL tokenizer.json missing at {vl_path}")

	vl_tok = _load_json(out_path / "tokenizer.json")
	asr_tok = _load_json(asr_path / "tokenizer.json")
	asr_added = {e["content"]: e for e in (asr_tok.get("added_tokens") or []) if "content" in e}
	vl_added = {e["content"]: e for e in (vl_tok.get("added_tokens") or []) if "content" in e}
	vocab = (vl_tok.setdefault("model", {})).setdefault("vocab", {})

	expected = {}
	for token in AUDIO_SPECIAL_TOKENS:
		tid = _asr_token_id(asr_tok, token)
		if tid is None:
			raise ValueError(f"ASR tokenizer missing audio special {token!r}")
		expected[token] = tid
		if token in vl_added and int(vl_added[token]["id"]) != tid:
			raise ValueError(f"VL already has {token!r} at id {vl_added[token]['id']}, want {tid}")
		if token in vocab and int(vocab[token]) != tid:
			raise ValueError(f"VL vocab has {token!r} at id {vocab[token]}, want {tid}")
		entry = dict(asr_added.get(token) or {"id": tid, "content": token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True})
		entry["id"] = tid
		entry["content"] = token
		vl_added[token] = entry
		vocab[token] = tid

	vl_tok["added_tokens"] = sorted(vl_added.values(), key=lambda e: int(e["id"]))  # keep added_tokens sorted by id (HF convention)
	_save_json(out_path / "tokenizer.json", vl_tok)

	cfg_path = out_path / "tokenizer_config.json"
	if cfg_path.exists():
		cfg = _load_json(cfg_path)
		extra = list(cfg.get("additional_special_tokens") or [])
		for token in AUDIO_SPECIAL_TOKENS:
			if token not in extra:
				extra.append(token)
		cfg["additional_special_tokens"] = extra
		_save_json(cfg_path, cfg)

	from transformers import AutoTokenizer
	tok = AutoTokenizer.from_pretrained(str(out_path), trust_remote_code=True, fix_mistral_regex=True)
	ids = {t: tok.convert_tokens_to_ids(t) for t in AUDIO_SPECIAL_TOKENS}
	for t, want in expected.items():
		got = ids[t]
		if got != want:
			raise ValueError(f"Tokenizer id mismatch for {t!r}: got {got}, want {want}")
	print(f"[INFO] Tokenizer audio specials (VL-base): {ids}")


def merge(vl_path: str | Path, asr_path: str | Path, out_path: str | Path):
	vl_path = Path(vl_path)
	asr_path = Path(asr_path)
	out_path = Path(out_path)
	out_path.mkdir(parents=True, exist_ok=True)

	vl_cfg = _load_json(vl_path / "config.json")
	asr_cfg = _load_json(asr_path / "config.json")
	avlm_cfg = build_avlm_config(vl_cfg, asr_cfg)
	_save_json(out_path / "config.json", avlm_cfg)

	print("[INFO] Merging weights...")
	merged = merge_weights(vl_path, asr_path)
	print(f"[INFO] Writing {len(merged)} tensors -> {out_path / 'model.safetensors'}")
	mx.save_safetensors(str(out_path / "model.safetensors"), merged, metadata={"format": "mlx"})

	vl_proc = _load_json(vl_path / "processor_config.json") if (vl_path / "processor_config.json").exists() else {}  # VL image/video + ASR feature extractor
	asr_proc = _load_json(asr_path / "processor_config.json") if (asr_path / "processor_config.json").exists() else {}
	proc = dict(vl_proc)
	if "feature_extractor" in asr_proc:
		proc["feature_extractor"] = asr_proc["feature_extractor"]
	proc["processor_class"] = "Qwen3VLProcessor"
	_save_json(out_path / "processor_config.json", proc)

	if "feature_extractor" in asr_proc:  # ASR preprocessor_config naming some loaders expect
		_save_json(out_path / "preprocessor_config.json", asr_proc["feature_extractor"])

	for name in ("generation_config.json", "chat_template.jinja"):
		for src_root in (vl_path, asr_path):
			src = src_root / name
			if src.exists() and not (out_path / name).exists():
				shutil.copy2(src, out_path / name)

	copy_tokenizer_with_audio_tokens(vl_path, asr_path, out_path)

	index = {"metadata": {"total_size": sum(int(v.nbytes) for v in merged.values())}, "weight_map": {k: "model.safetensors" for k in sorted(merged)}, }  # weight map index (single shard)
	_save_json(out_path / "model.safetensors.index.json", index)

	prefixes = {}  # quick architecture report
	for k in merged:
		p = ".".join(k.split(".")[:2])
		prefixes[p] = prefixes.get(p, 0) + 1
	print("[INFO] Merged prefix counts:")
	for p, n in sorted(prefixes.items(), key=lambda x: -x[1]):
		print(f"  {n:4d}  {p}")
	print(f"[INFO] Done -> {out_path}")
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Merge Qwen3-VL + Qwen3-ASR audio into Qwen3-AVLM MLX weights")
	parser.add_argument("--vl", required=True, help="Path to Yuna/Qwen3-VL checkpoint")
	parser.add_argument("--asr", required=True, help="Path to Qwen3-ASR checkpoint")
	parser.add_argument("--out", required=True, help="Output directory for merged MLX AVLM")
	args = parser.parse_args()
	merge(args.vl, args.asr, args.out)


if __name__ == "__main__":
	main()
