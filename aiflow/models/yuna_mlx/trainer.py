import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .utils import load
from .trainer.lora import LoRALinear

def find_all_linear_names(model):
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Target attention and MLP layers in Yuna
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                 linear_layers.add(name)
    return list(linear_layers)

def apply_lora_layers(model, adapter_path: str):
    with open(Path(adapter_path) / "adapter_config.json", "r") as f:
        config = json.load(f)
        lora_rank = config["rank"]
        lora_alpha = config["alpha"]

    linear_layers = find_all_linear_names(model)

    for name in linear_layers:
        module = model
        path = name.split(".")
        for p in path[:-1]:
            module = getattr(module, p)

        linear_module = getattr(module, path[-1])
        lora_layer = LoRALinear.from_linear(linear_module, r=lora_rank, alpha=lora_alpha)
        setattr(module, path[-1], lora_layer)

    model.load_weights(str(Path(adapter_path) / "adapters.safetensors"), strict=False)
    return model

def main():
    parser = argparse.ArgumentParser(description="LoRA training for Yuna model.")
    # Add training arguments here, e.g., model path, data path, lora rank, etc.
    # This is a placeholder for a full training implementation.
    print("MLX training for Yuna is not fully implemented in this refactor.")
    print("This script sets up the structure for applying LoRA layers.")
    print("A full training loop would require a dataset loader and a loss function.")

if __name__ == "__main__":
    main()