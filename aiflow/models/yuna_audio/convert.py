import argparse
import glob
import importlib
import json
import logging
import shutil
from enum import Enum
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]
QUANT_MODES = ["affine", "mxfp4", "nvfp4", "mxfp8"]


class Domain(str, Enum):
	STT = "stt"


class DomainConfig:
	def __init__(self, name, tags, cli_example, python_example):
		self.name = name
		self.tags = tags
		self.cli_example = cli_example
		self.python_example = python_example


_model_types_cache = {}
_detection_hints_cache = {}


def _discover_model_types(domain):
	models_dir = Path(__file__).parent / domain / "models"
	if not models_dir.exists():
		return set()
	return {d.name for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()}


def get_model_types(domain):
	domain_str = domain.value
	if domain_str not in _model_types_cache:
		_model_types_cache[domain_str] = _discover_model_types(domain_str)
	return _model_types_cache[domain_str]


def _get_config_keys(config_class):
	return set(vars(config_class).keys()) if hasattr(config_class, "__dict__") else set()


def _discover_detection_hints(domain):
	hints = {"config_keys": {}, "architectures": {}, "path_patterns": {}}

	for model_type in get_model_types(Domain(domain)):
		module_path = f"yuna_audio.{domain}.models.{model_type}"

		try:
			module = importlib.import_module(module_path)
			if hasattr(module, "DETECTION_HINTS"):
				model_hints = module.DETECTION_HINTS
				if "config_keys" in model_hints:
					hints["config_keys"][model_type] = set(model_hints["config_keys"])
				if "architectures" in model_hints:
					hints["architectures"][model_type] = set(model_hints["architectures"])
				if "path_patterns" in model_hints:
					hints["path_patterns"][model_type] = set(model_hints["path_patterns"])
			else:
				if hasattr(module, "ModelConfig"):
					config_keys = _get_config_keys(module.ModelConfig)
					hints["config_keys"][model_type] = config_keys
				hints["path_patterns"][model_type] = {model_type, model_type.replace("_", "")}
		except ImportError:
			continue
	return hints


def get_detection_hints(domain):
	domain_str = domain.value
	if domain_str not in _detection_hints_cache:
		_detection_hints_cache[domain_str] = _discover_detection_hints(domain_str)
	return _detection_hints_cache[domain_str]


def get_model_path(local_path):
	model_path = Path(local_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Local model path not found: {local_path}")
	return model_path


def load_config(model_path):
	config_path = model_path / "config.json"
	if config_path.exists():
		with open(config_path, "r", encoding="utf-8") as f:
			return json.load(f)
	raise FileNotFoundError(f"Config not found at {model_path}")


def _match_by_model_type(model_type):
	if not model_type:
		return None
	for domain in Domain:
		if model_type in get_model_types(domain):
			return domain
	return None


def _get_model_identifier(config):
	return config.get("model_type", "").lower() or config.get("name", "").lower()


def _match_by_config_keys(config):
	config_keys = set(config.keys())
	best_match = None
	best_score = 0

	for domain in Domain:
		hints = get_detection_hints(domain)
		for model_type, model_keys in hints.get("config_keys", {}).items():
			intersection = config_keys & model_keys
			if model_keys:
				score = len(intersection) / len(model_keys)
				if score > best_score and score > 0.3:
					best_score = score
					best_match = (domain, model_type)
	return best_match


def _match_by_path(model_path):
	path_str = str(model_path).lower()
	for domain in Domain:
		hints = get_detection_hints(domain)
		for model_type, patterns in hints.get("path_patterns", {}).items():
			if any(pattern in path_str for pattern in patterns):
				return (domain, model_type)
	return None


def detect_model_domain(config, model_path):
	model_identifier = _get_model_identifier(config)
	match = _match_by_path(model_path)
	if match: return match[0]
	domain = _match_by_model_type(model_identifier)
	if domain: return domain
	match = _match_by_config_keys(config)
	if match: return match[0]
	return Domain.STT


def get_model_type(config, model_path, domain):
	model_type = config.get("model_type", "").lower()
	model_name = config.get("name", "").lower()

	for candidate in [model_type, model_name]:
		if candidate and candidate in get_model_types(domain):
			return candidate

	hints = get_detection_hints(domain)
	config_keys = set(config.keys())
	best_match = None
	best_score = 0

	for mt, model_keys in hints.get("config_keys", {}).items():
		if model_keys:
			intersection = config_keys & model_keys
			score = len(intersection) / len(model_keys)
			if score > best_score:
				best_score = score
				best_match = mt

	if best_match and best_score > 0.3:
		return best_match

	path_str = str(model_path).lower()
	for mt, patterns in hints.get("path_patterns", {}).items():
		if any(pattern in path_str for pattern in patterns):
			return mt

	model_types = get_model_types(domain)
	return next(iter(model_types), "unknown") if model_types else "unknown"


def get_model_class(model_type, domain):
	module_path = f"yuna_audio.{domain.value}.models.{model_type}"
	try:
		return importlib.import_module(module_path)
	except ImportError as e:
		msg = f"Model type '{model_type}' not supported for {domain.name}. Error: {e}"
		logging.error(msg)
		raise ValueError(msg)


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


def convert(local_path, mlx_path="mlx_model", quantize=False, q_group_size=None, q_bits=None, dtype=None, dequantize=False, quant_predicate=None, q_mode="affine", model_domain=None):
	from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

	if quantize and dequantize:
		raise ValueError("Choose either quantize or dequantize, not both.")

	print(f"[INFO] Loading local model from {local_path}")
	model_path = get_model_path(local_path)
	config = load_config(model_path)

	if model_domain is None:
		domain = detect_model_domain(config, model_path)
	else:
		domain = Domain(model_domain)

	model_type = get_model_type(config, model_path, domain)
	print(f"\n[INFO] Model domain: {domain.name}, type: {model_type}")

	model_class = get_model_class(model_type, domain)
	model_config = (model_class.ModelConfig.from_dict(config) if hasattr(model_class, "ModelConfig") else config)

	if hasattr(model_config, "model_path"):
		model_config.model_path = model_path

	weights = load_weights(model_path)
	model = model_class.Model(model_config)

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
	config["model_type"] = model_type
	save_config(config, config_path=mlx_path / "config.json")
	print(f"[INFO] Conversion complete! Model saved to {mlx_path}")


def configure_parser():
	parser = argparse.ArgumentParser(description="Convert Local model to MLX format")
	parser.add_argument("--local-path", type=str, required=True, help="Path to the local model.")
	parser.add_argument("--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model.")
	parser.add_argument("-q", "--quantize", action="store_true", help="Generate a quantized model.")
	parser.add_argument("--q-group-size", type=int, default=None, help="Group size for quantization.")
	parser.add_argument("--q-bits", type=int, default=None, help="Bits per weight for quantization.")
	parser.add_argument("--q-mode", choices=QUANT_MODES, type=str, default="affine", help="Quantization mode.")
	parser.add_argument("--quant-predicate", choices=QUANT_RECIPES, type=str, help="Mixed-bit quantization recipe.")
	parser.add_argument("--dtype", type=str, choices=MODEL_CONVERSION_DTYPES, default=None, help="Data type for weights.")
	parser.add_argument("-d", "--dequantize", action="store_true", help="Dequantize a quantized model.")
	parser.add_argument("--model-domain", type=str, choices=["stt"], default=None, help="Force model domain.")
	return parser


def main():
	parser = configure_parser()
	args = parser.parse_args()
	convert(**vars(args))


if __name__ == "__main__":
	main()
