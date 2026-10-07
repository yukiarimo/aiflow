from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import ml_dtypes
from safetensors import safe_open
from safetensors.numpy import save_file

_BF16 = ml_dtypes.bfloat16


def _to_bfloat16(arr: np.ndarray) -> np.ndarray:
	return np.asarray(arr, dtype=np.float32).astype(_BF16)


def run(*, src: Path, dst: Path) -> Path:
	if not src.is_dir():
		raise FileNotFoundError(f"Missing qwen3-asr folder: {src}")
	dst.mkdir(parents=True, exist_ok=True)

	cfg = json.loads((src / "config.json").read_text())
	tc = cfg["thinker_config"]["text_config"]
	rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
	theta = rope.get("rope_theta") or tc.get("rope_theta") or 1_000_000

	qcfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": tc["hidden_size"], "num_hidden_layers": tc["num_hidden_layers"], "num_attention_heads": tc["num_attention_heads"], "num_key_value_heads": tc["num_key_value_heads"], "head_dim": tc["head_dim"], "intermediate_size": tc["intermediate_size"], "vocab_size": tc["vocab_size"], "rms_norm_eps": tc.get("rms_norm_eps", 1e-6), "max_position_embeddings": tc.get("max_position_embeddings", 65536), "tie_word_embeddings": tc.get("tie_word_embeddings", True), "hidden_act": tc.get("hidden_act", "silu"), "attention_bias": tc.get("attention_bias", False), "rope_theta": theta, "rope_interleaved": False, "torch_dtype": "bfloat16", "eos_token_id": cfg.get("eos_token_id", 151645), }
	(dst / "config.json").write_text(json.dumps(qcfg, indent=2))
	print("rope_interleaved =", qcfg["rope_interleaved"], "| rope_theta =", theta)

	pref = "thinker.model."
	tensors = {}
	with safe_open(src / "model.safetensors", framework="numpy") as f:
		for k in f.keys():
			if k.startswith("thinker.audio_tower."):
				continue
			if k.startswith(pref):
				nk = "model." + k[len(pref):]
			elif k == "thinker.lm_head.weight":
				nk = "lm_head.weight"
			else:
				continue
			tensors[nk] = np.ascontiguousarray(_to_bfloat16(f.get_tensor(k)))

	print(f"remapped {len(tensors)} tensors; tied head =", "lm_head.weight" not in tensors)
	save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

	for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "chat_template.json", ):
		src_file = src / name
		if src_file.exists():
			shutil.copy2(src_file, dst / name)
			print("copied", name)

	print("done ->", dst)
	return dst


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--src", type=Path, required=True)
	ap.add_argument("--dst", type=Path, required=True)
	args = ap.parse_args()
	run(src=args.src, dst=args.dst)


if __name__ == "__main__":
	main()
