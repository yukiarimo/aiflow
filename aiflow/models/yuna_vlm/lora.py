import math
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class LoRaLayer(nn.Module):
	def __init__(self, linear, rank, alpha=0.1, dropout=0.0):
		super().__init__()

		self.original_layer = linear
		self.dropout = nn.Dropout(p=dropout)

		output_dims, input_dims = linear.weight.shape
		if isinstance(linear, nn.QuantizedLinear):
			input_dims *= 32 // linear.bits

		std_dev = 1 / math.sqrt(rank)

		self.A = mx.random.uniform(low=-std_dev, high=std_dev, shape=(input_dims, rank))
		self.B = mx.zeros((rank, output_dims))
		self.alpha = alpha

	def __call__(self, x):
		y = self.original_layer(x)
		lora_update = (self.dropout(x) @ self.A) @ self.B
		return y + (self.alpha * lora_update).astype(x.dtype)


def replace_lora_with_linear(model):
	for i, layer in enumerate(model.layers):
		if isinstance(layer, LoRaLayer):
			lora_update = layer.alpha * (layer.A @ layer.B)
			updated_weight = layer.original_layer.weight + lora_update
			use_bias = layer.original_layer.bias is not None
			updated_bias = layer.original_layer.bias

			new_linear_layer = nn.Linear(updated_weight.size(1), updated_weight.size(0), bias=use_bias)
			new_linear_layer.weight = updated_weight

			if use_bias:
				new_linear_layer.bias = updated_bias

			if isinstance(layer.original_layer, nn.QuantizedLinear):
				new_linear_layer = nn.QuantizedLinear.from_linear(new_linear_layer, new_linear_layer.group_size, new_linear_layer.bits)

			model.layers[i] = new_linear_layer


def get_module_by_name(model, name):
	parts = name.split(".")
	module = model
	for part in parts:
		if part.isdigit():
			module = module[int(part)]
		else:
			module = getattr(module, part)
	return module


def set_module_by_name(model, name, new_module):
	parts = name.split(".")
	module = model
	for part in parts[:-1]:
		if part.isdigit():
			module = module[int(part)]
		else:
			module = getattr(module, part)
	if parts[-1].isdigit():
		module[int(parts[-1])] = new_module
	else:
		setattr(module, parts[-1], new_module)


def get_peft_model(model, linear_layers, rank=10, alpha=0.1, dropout=0.1, freeze=True, verbose=True):
	if freeze:
		freeze_model(model)

	for name, module in model.language_model.named_modules():
		if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
			if name.split(".")[-1] in linear_layers:
				lora_layer = LoRaLayer(module, rank, alpha, dropout)
				set_module_by_name(model.language_model, name, lora_layer)

	model.config.lora = {}
	model.config.lora["rank"] = rank
	model.config.lora["alpha"] = alpha
	model.config.lora["dropout"] = dropout

	if verbose:
		print_trainable_parameters(model.language_model)

	return model


def freeze_model(model):
	for name, module in model.named_modules():
		name = name.split(".")[0]
		if name in ["language_model", "vision_model", "vision_tower", "aligner", "connector", "multi_modal_projector", "mm_projector"]:
			model[f"{name}"].freeze()


def find_all_linear_names(model):
	cls = nn.Linear
	quantized_cls = nn.QuantizedLinear
	lora_module_names = set()
	multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler", "aligner"]

	for name, module in model.named_modules():
		if any(mm_keyword in name for mm_keyword in multimodal_keywords):
			continue
		if isinstance(module, cls) or isinstance(module, quantized_cls):
			names = name.split(".")
			lora_module_names.add(names[0] if len(names) == 1 else names[-1])

	if "lm_head" in lora_module_names:
		lora_module_names.remove("lm_head")
	return list(lora_module_names)


def count_parameters(model):
	def nparams(m):
		if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
			return m.weight.size * (32 // m.bits)
		return sum(v.size for _, v in tree_flatten(m.parameters()))

	leaf_modules = tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
	total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
	return total_p


def print_trainable_parameters(model):
	def nparams(m):
		if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
			return m.weight.size * (32 // m.bits)
		return sum(v.size for _, v in tree_flatten(m.parameters()))

	leaf_modules = tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
	total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
	trainable_p = (sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6)

	print(f"#trainable params: {trainable_p} M || all params: {total_p} M || trainable%: {(trainable_p * 100 / total_p):.3f}%")


def apply_lora_layers(model, adapter_path):
	adapter_path = Path(adapter_path)

	if not adapter_path.exists():
		raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")

	with open(adapter_path / "adapter_config.json", "r") as f:
		config = json.load(f)
		if "rank" not in config:
			raise ValueError("The adapter does not have lora params in the config")

	list_of_modules = find_all_linear_names(model.language_model.model)
	if config is not None:
		model = get_peft_model(model, list_of_modules, **config)
	else:
		model = get_peft_model(model, list_of_modules)

	model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)

	return model
