import argparse
import shutil
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
import torch
import json

def torch_to_mx(a, *, dtype="float16"):
    a = a.to(torch.float32).numpy()
    return mx.array(a, getattr(mx, dtype))

def convert(pt_path: str, mlx_path: str, dtype: str = "float16", quantization: dict = None):
    pt_path = Path(pt_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading PyTorch weights...")
    weights = {}
    pt_weights_path = pt_path / "model.safetensors"
    if not pt_weights_path.exists():
        pt_weights_path = pt_path / "pytorch_model.bin"
        if not pt_weights_path.exists(): 
            raise FileNotFoundError(f"No model weights found in {pt_path}")
        print("[INFO] Loading from .bin file...")
        weights = torch.load(pt_weights_path, map_location="cpu")
    else:
        print("[INFO] Loading from .safetensors file...")
        with safe_open(pt_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys(): 
                weights[key] = f.get_tensor(key)

    print("[INFO] Converting weights to MLX...")
    mlx_weights = {}
    for key, value in weights.items():
        # General model name mapping
        key = key.replace("model.", "language_model.", 1)
        key = key.replace("visual.", "vision_tower.", 1)
        key = key.replace("audio_encoder.", "audio_tower.", 1)
        key = key.replace("audio_projector.proj.0", "audio_projector.proj_1")
        key = key.replace("audio_projector.proj.2", "audio_projector.proj_2")

        # Explicit mapping for PatchMerger's sequential layers
        if "vision_tower.merger.mlp.0" in key: 
            key = key.replace("vision_tower.merger.mlp.0", "vision_tower.merger.mlp_fc1")
        if "vision_tower.merger.mlp.2" in key: 
            key = key.replace("vision_tower.merger.mlp.2", "vision_tower.merger.mlp_fc2")

        # Handle Conv weight permutations
        if "vision_tower.patch_embed.proj.weight" in key: 
            value = value.permute(0, 2, 3, 4, 1) # (O, H, W, D, I)
        if "audio_tower.conv1.weight" in key or "audio_tower.conv2.weight" in key: 
            value = value.permute(2, 1, 0) # (W, I, O)

        mlx_weights[key] = torch_to_mx(value, dtype=dtype)

    # Save weights before quantization
    print("[INFO] Saving converted weights...")
    mx.save_safetensors(str(mlx_path / "model.safetensors"), mlx_weights)
    
    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        if (pt_path / config_file).exists(): 
            shutil.copyfile(pt_path / config_file, mlx_path / config_file)
    
    # Apply quantization if specified
    if quantization:
        print(f"[INFO] Applying quantization: {quantization}")
        
        # Load the model to apply quantization
        from yuna.yuna import Model, ModelConfig
        
        with open(pt_path / "config.json", "r") as f:
            config = json.load(f)
        
        model = Model(ModelConfig.from_dict(config))
        model.load_weights(str(mlx_path / "model.safetensors"), strict=False)
        
        # Define which layers should be quantized
        def class_predicate(p, m):
            # Quantize Linear layers in language model, but skip embeddings and final projection
            if isinstance(m, nn.Linear):
                # Skip embedding layer and lm_head
                if "embed_tokens" in p or "lm_head" in p:
                    return False
                # Quantize language model linear layers
                if "language_model" in p:
                    return True
            return False
        
        # Apply quantization
        nn.quantize(
            model,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 4),
            class_predicate=class_predicate,
        )
        
        print("[INFO] Quantization applied successfully")
        
        # Flatten quantized weights for saving
        from mlx.utils import tree_flatten
        mlx_weights = dict(tree_flatten(model.parameters()))
        
        print(f"[INFO] Flattened {len(mlx_weights)} weight tensors for saving")
        
        # Save quantized weights
        mx.save_safetensors(str(mlx_path / "model.safetensors"), mlx_weights)
        
        # Update config to indicate quantization
        with open(mlx_path / "config.json", "r") as f:
            config = json.load(f)
        
        config["quantization"] = {
            "group_size": quantization["group_size"],
            "bits": quantization["bits"],
        }
        
        with open(mlx_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("[INFO] Saved quantization config")
    
    print("[INFO] Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert Yuna PyTorch model to MLX.")
    parser.add_argument("--pt-path", required=True, type=str, help="Path to the PyTorch model directory.")
    parser.add_argument("--mlx-path", required=True, type=str, help="Path to save the MLX model.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Data type to save the MLX weights.")
    
    # Quantization arguments
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--q-group-size", type=int, default=64, help="Group size for quantization")
    parser.add_argument("--q-bits", type=int, default=4, choices=[2, 4, 8], help="Bits for quantization")
    
    args = parser.parse_args()
    
    quantization = None
    if args.quantize:
        quantization = {
            "group_size": args.q_group_size,
            "bits": args.q_bits,
        }
    
    convert(args.pt_path, args.mlx_path, args.dtype, quantization)

if __name__ == "__main__": 
    main()