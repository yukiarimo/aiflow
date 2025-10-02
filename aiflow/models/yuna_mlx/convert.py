import argparse
import json
import shutil
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from safetensors import safe_open

def torch_to_mx(a, *, dtype="float16"):
    import torch
    # Ensure float32 for conversion, then cast to target dtype in MLX
    a = a.to(torch.float32).numpy()
    return mx.array(a, getattr(mx, dtype))

def convert(pt_path: str, mlx_path: str, dtype: str = "float16"):
    pt_path = Path(pt_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading PyTorch weights...")
    weights = {}
    with safe_open(pt_path / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    print("[INFO] Converting weights to MLX...")
    mlx_weights = {}
    for key, value in weights.items():
        # General model name mapping
        if key.startswith("model."):
            key = key.replace("model.", "language_model.", 1)
        if key.startswith("visual."):
            key = key.replace("visual.", "vision_tower.", 1)

        # Explicit mapping for PatchMerger's sequential layers
        if "vision_tower.merger.mlp.0" in key:
            key = key.replace("vision_tower.merger.mlp.0", "vision_tower.merger.mlp_fc1")
        if "vision_tower.merger.mlp.2" in key:
            key = key.replace("vision_tower.merger.mlp.2", "vision_tower.merger.mlp_fc2")

        # Handle Conv3D weight permutation
        # This needs to be done *after* renaming to match the new MLX model structure
        if "vision_tower.patch_embed.proj.weight" in key:
            value = value.permute(0, 2, 3, 4, 1)

        mlx_weights[key] = torch_to_mx(value, dtype=dtype)

    print(f"[INFO] Saving MLX weights to {mlx_path / 'model.safetensors'}...")
    mx.save_safetensors(str(mlx_path / "model.safetensors"), mlx_weights)

    # Copy config and tokenizer files
    print("[INFO] Copying configuration and tokenizer files...")
    # Make sure we're copying the correct config and tokenizer for your model
    shutil.copyfile(pt_path / "config.json", mlx_path / "config.json")
    shutil.copyfile(pt_path / "tokenizer.json", mlx_path / "tokenizer.json")

    print("[INFO] Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert Yuna PyTorch model to MLX.")
    parser.add_argument(
        "--pt-path", required=True, type=str, help="Path to the PyTorch model directory."
    )
    parser.add_argument(
        "--mlx-path", required=True, type=str, help="Path to save the MLX model."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type to save the MLX weights.",
    )
    args = parser.parse_args()
    convert(args.pt_path, args.mlx_path, args.dtype)

if __name__ == "__main__":
    main()