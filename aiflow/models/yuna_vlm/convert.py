import argparse
import glob
import shutil
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_map_with_path
from mlx_lm.utils import dequantize_model, quantize_model
from .utils import (MODEL_CONVERSION_DTYPES, get_model_path, save_config, save_weights, skip_multimodal_module)
from .utils import load

QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_5", "mixed_3_6", "mixed_3_8", "mixed_4_6", "mixed_4_8"]


def mixed_quant_predicate_builder(recipe, model):
	group_size = 64

	recipe_config = {"mixed_2_6": (2, 6), "mixed_3_4": (3, 4), "mixed_3_5": (3, 5), "mixed_3_6": (3, 6), "mixed_3_8": (3, 8), "mixed_4_6": (4, 6), "mixed_4_8": (4, 8)}

	if recipe not in recipe_config:
		raise ValueError(f"Invalid quant recipe {recipe}")

	low_bits, high_bits = recipe_config[recipe]

	down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
	if len(down_keys) == 0:
		raise ValueError("Model does not have expected keys for mixed quant.")

	layer_location = 0
	for i, k in enumerate(down_keys[0].split(".")):
		if k.isdigit():
			layer_location = i
			break

	num_layers = len(model.layers)

	def mixed_quant_predicate(path, module):
		if skip_multimodal_module(path):
			return False
		if not hasattr(module, "to_quantized"):
			return False
		if module.weight.shape[1] % group_size != 0:
			return False

		path_parts = path.split(".")
		index = 0

		if len(path_parts) > layer_location:
			element = path_parts[layer_location]
			if element.isdigit():
				index = int(element)

		use_more_bits = (index < num_layers // 8 or index >= 7 * num_layers // 8 or (index - num_layers // 8) % 3 == 2)

		if use_more_bits and ("v_proj" in path or "down_proj" in path):
			return {"group_size": group_size, "bits": high_bits}

		if "lm_head" in path or "embed_tokens" in path:
			return {"group_size": group_size, "bits": high_bits}

		return {"group_size": group_size, "bits": low_bits}

	return mixed_quant_predicate


def convert(local_path, mlx_path="mlx_model", quantize=False, q_group_size=64, q_bits=4, dtype=None, dequantize=False, quant_predicate=None):
	print("[INFO] Loading local model...")
	model_path = get_model_path(local_path)
	model, processor = load(model_path, lazy=True)
	config = model.config.__dict__

	def base_quant_predicate(path, module):
		if skip_multimodal_module(path):
			return False
		if not hasattr(module, "to_quantized"):
			return False
		if module.weight.shape[1] % q_group_size != 0:
			return False
		return True

	if isinstance(quant_predicate, str):
		quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

	quant_predicate = quant_predicate or base_quant_predicate

	if dtype is None:
		dtype = config.get("torch_dtype", None)
	if dtype in MODEL_CONVERSION_DTYPES:
		print("[INFO] Using dtype:", dtype)
		dtype = getattr(mx, dtype)
		cast_predicate = getattr(model, "cast_predicate", lambda _: True)

		def set_dtype(k, v):
			if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
				return v.astype(dtype)
			else:
				return v

		model.update(tree_map_with_path(set_dtype, model.parameters()))

	if quantize and dequantize:
		raise ValueError("Choose either quantize or dequantize, not both.")

	if quantize:
		print("[INFO] Quantizing")
		config.setdefault("vision_config", {})
		model, config = quantize_model(model, config, q_group_size, q_bits, quant_predicate=quant_predicate)

	if dequantize:
		print("[INFO] Dequantizing")
		model = dequantize_model(model)

	if isinstance(mlx_path, str):
		mlx_path = Path(mlx_path)

	save_weights(mlx_path, model, donate_weights=True)

	for pattern in ["*.py", "*.json"]:
		files = glob.glob(str(model_path / pattern))
		for file in files:
			shutil.copy(file, mlx_path)

	processor.save_pretrained(mlx_path)
	save_config(config, config_path=mlx_path / "config.json")
	print(f"[INFO] Conversion complete! Model saved to {mlx_path}")


def configure_parser():
	parser = argparse.ArgumentParser(description="Convert local model to MLX format")
	parser.add_argument("--local-path", type=str, required=True, help="Path to the local model.")
	parser.add_argument("--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model.")
	parser.add_argument("-q", "--quantize", help="Generate a quantized model.", action="store_true")
	parser.add_argument("--q-group-size", help="Group size for quantization.", type=int, default=64)
	parser.add_argument("--q-bits", help="Bits per weight for quantization.", type=int, default=4)
	parser.add_argument("--dtype", help="Type to save the parameter.", type=str, choices=MODEL_CONVERSION_DTYPES, default=None)
	parser.add_argument("--quant-predicate", help="Mixed-bit quantization recipe.", choices=QUANT_RECIPES, type=str, required=False)
	parser.add_argument("-d", "--dequantize", help="Dequantize a quantized model.", action="store_true", default=False)
	return parser


def main():
	parser = configure_parser()
	args = parser.parse_args()
	convert(**vars(args))


if __name__ == "__main__":
	main()
