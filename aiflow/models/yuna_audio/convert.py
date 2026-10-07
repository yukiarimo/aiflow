import argparse
import glob
import shutil
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from aiflow.models.yuna_audio import qwen3_asr
from aiflow.models.yuna_audio.utils import get_model_path, load_config

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]
QUANT_MODES = ["affine", "mxfp4", "nvfp4", "mxfp8"]


def load_weights(model_path):
	weight_files = glob.glob(str(model_path / "*.safetensors"))
	if not weight_files:
		raise FileNotFoundError(f"No safetensors found in {model_path}")
	weights = {}
	for wf in weight_files:
		if "tokenizer" in wf:
			continue
		weights.update(mx.load(wf))
	return weights


def build_quant_predicate(model, quant_predicate_name=None):
	model_quant_predicate = getattr(model, "model_quant_predicate", lambda p, m: True)

	def base_requirements(path, module):
		return (hasattr(module, "weight") and module.weight.shape[-1] % 64 == 0 and hasattr(module, "to_quantized") and model_quant_predicate(path, module))

	if not quant_predicate_name:
		return base_requirements

	from mlx_lm.convert import mixed_quant_predicate_builder
	mixed_predicate = mixed_quant_predicate_builder(quant_predicate_name, model)
	return lambda p, m: base_requirements(p, m) and mixed_predicate(p, m)


def copy_model_files(source, dest):
	patterns = ["*.py", "*.json", "*.yaml", "*.tiktoken", "*.model", "*.txt", "*.wav", "*.pt", "*.safetensors"]
	for pattern in patterns:
		for file in glob.glob(str(source / pattern)):
			name = Path(file).name
			if name == "model.safetensors.index.json" or (name.startswith("model") and name.endswith(".safetensors")):
				continue
			shutil.copy(file, dest)
		for file in glob.glob(str(source / "**" / pattern), recursive=True):
			rel_path = Path(file).relative_to(source)
			if len(rel_path.parts) <= 1:
				continue
			name = Path(file).name
			if name == "model.safetensors.index.json":
				continue
			dest_dir = dest / rel_path.parent
			dest_dir.mkdir(parents=True, exist_ok=True)
			shutil.copy(file, dest_dir)


def convert(local_path, mlx_path="mlx_model", quantize=False, q_group_size=None, q_bits=None, dtype=None, dequantize=False, quant_predicate=None, q_mode="affine"):
	from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

	if quantize and dequantize:
		raise ValueError("Choose either quantize or dequantize, not both.")

	print(f"[INFO] Loading local ASR model from {local_path}")
	model_path = get_model_path(local_path)
	config = load_config(model_path)
	model_config = qwen3_asr.ModelConfig.from_dict(config)
	weights = load_weights(model_path)
	model = qwen3_asr.Model(model_config)

	if hasattr(model, "sanitize"):
		weights = model.sanitize(weights)

	model.load_weights(list(weights.items()))
	weights = dict(tree_flatten(model.parameters()))

	target_dtype = dtype or config.get("torch_dtype")
	if target_dtype and target_dtype in MODEL_CONVERSION_DTYPES:
		print(f"[INFO] Converting to {target_dtype}")
		mx_dtype = getattr(mx, target_dtype)
		weights = {k: v.astype(mx_dtype) for k, v in weights.items()}

	if quantize:
		final_predicate = build_quant_predicate(model, quant_predicate)
		model.load_weights(list(weights.items()))
		weights, config = quantize_model(model, config, q_group_size, q_bits, mode=q_mode, quant_predicate=final_predicate)

	if dequantize:
		print("[INFO] Dequantizing")
		model = dequantize_model(model)
		weights = dict(tree_flatten(model.parameters()))

	mlx_path = Path(mlx_path)
	mlx_path.mkdir(parents=True, exist_ok=True)
	copy_model_files(model_path, mlx_path)

	save_model(mlx_path, model, donate_model=True)
	config["model_type"] = "qwen3_asr"
	save_config(config, config_path=mlx_path / "config.json")
	print(f"[INFO] Conversion complete! Model saved to {mlx_path}")


def configure_parser():
	parser = argparse.ArgumentParser(description="Convert local Qwen3 ASR model to MLX format")
	parser.add_argument("--local-path", type=str, required=True, help="Path to the local model.")
	parser.add_argument("--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model.")
	parser.add_argument("-q", "--quantize", action="store_true", help="Generate a quantized model.")
	parser.add_argument("--q-group-size", type=int, default=None, help="Group size for quantization.")
	parser.add_argument("--q-bits", type=int, default=None, help="Bits per weight for quantization.")
	parser.add_argument("--q-mode", choices=QUANT_MODES, type=str, default="affine", help="Quantization mode.")
	parser.add_argument("--quant-predicate", choices=QUANT_RECIPES, type=str, help="Mixed-bit quantization recipe.")
	parser.add_argument("--dtype", type=str, choices=MODEL_CONVERSION_DTYPES, default=None, help="Data type for weights.")
	parser.add_argument("-d", "--dequantize", action="store_true", help="Dequantize a quantized model.")
	return parser


def main():
	parser = configure_parser()
	args = parser.parse_args()
	convert(**vars(args))


if __name__ == "__main__":
	main()
