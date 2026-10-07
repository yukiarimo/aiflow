from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _remap_key(key: str) -> str | None:
	if key.startswith("model.language_model.model."):
		return "model." + key[len("model.language_model.model."):]
	if key.startswith("model.language_model."):
		rest = key[len("model.language_model."):]
		if rest.startswith("layers.") or rest.startswith("norm."):
			return "model." + rest
		if rest.startswith("embed_tokens."):
			return "model." + rest
	if key == "lm_head.weight":
		return key
	return None


def _weight_files(src: Path) -> list[Path]:
	single = src / "model.safetensors"
	if single.is_file():
		return [single]
	index = src / "model.safetensors.index.json"
	if index.is_file():
		weight_map = json.loads(index.read_text())["weight_map"]
		files = [src / name for name in sorted(set(weight_map.values()))]
		missing = [p for p in files if not p.is_file()]
		if missing:
			raise FileNotFoundError(f"Missing shards: {missing[:3]}")
		return files
	raise FileNotFoundError(f"Missing model.safetensors in {src}")


def run(*, src: Path, dst: Path) -> Path:
	if not src.is_dir():
		raise FileNotFoundError(f"Missing Qwen3-VL folder: {src}")
	dst.mkdir(parents=True, exist_ok=True)

	cfg = json.loads((src / "config.json").read_text())
	tc = cfg.get("text_config") or cfg
	rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
	theta = rope.get("rope_theta") or tc.get("rope_theta") or 1_000_000

	qcfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": tc["hidden_size"], "num_hidden_layers": tc["num_hidden_layers"], "num_attention_heads": tc["num_attention_heads"], "num_key_value_heads": tc["num_key_value_heads"], "head_dim": tc["head_dim"], "intermediate_size": tc["intermediate_size"], "vocab_size": tc["vocab_size"], "rms_norm_eps": tc.get("rms_norm_eps", 1e-6), "max_position_embeddings": tc.get("max_position_embeddings", 65536), "tie_word_embeddings": tc.get("tie_word_embeddings", True), "hidden_act": tc.get("hidden_act", "silu"), "attention_bias": tc.get("attention_bias", False), "rope_theta": theta, "rope_interleaved": False, "torch_dtype": "bfloat16", "eos_token_id": cfg.get("eos_token_id", tc.get("eos_token_id", 151643)), }
	(dst / "config.json").write_text(json.dumps(qcfg, indent=2))
	print("rope_interleaved =", qcfg["rope_interleaved"], "| rope_theta =", theta)

	weight_files = _weight_files(src)
	tensors = {}
	for weights in weight_files:
		with safe_open(str(weights), framework="pt") as f:
			for k in f.keys():
				if k.startswith("model.visual.") or k.startswith("model.audio_tower.") or k.startswith("model.multi_modal_projector."):
					continue
				nk = _remap_key(k)
				if nk is None:
					continue
				tensors[nk] = f.get_tensor(k).to(torch.bfloat16).contiguous()

	print(f"remapped {len(tensors)} tensors; tied head =", "lm_head.weight" not in tensors)
	save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

	import shutil

	for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab.json", "merges.txt", "chat_template.json", ):
		src_file = src / name
		if src_file.exists():
			shutil.copy2(src_file, dst / name)
			print("copied", name)

	print("done ->", dst)
	return dst
