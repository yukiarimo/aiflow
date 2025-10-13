import argparse
import shutil
from pathlib import Path
import mlx.core as mx
from safetensors import safe_open
import torch

def torch_to_mx(a, *, dtype="float16"):
    a = a.to(torch.float32).numpy()
    return mx.array(a, getattr(mx, dtype))

def convert(pt_path: str, mlx_path: str, dtype: str = "float16"):
    pt_path = Path(pt_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading PyTorch weights...")
    weights = {}
    pt_weights_path = pt_path / "model.safetensors"
    if not pt_weights_path.exists():
        pt_weights_path = pt_path / "pytorch_model.bin"
        if not pt_weights_path.exists(): raise FileNotFoundError(f"No model weights found in {pt_path}")
        print("[INFO] Loading from .bin file...")
        weights = torch.load(pt_weights_path, map_location="cpu")
    else:
        print("[INFO] Loading from .safetensors file...")
        with safe_open(pt_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys(): weights[key] = f.get_tensor(key)

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
        if "vision_tower.merger.mlp.0" in key: key = key.replace("vision_tower.merger.mlp.0", "vision_tower.merger.mlp_fc1")
        if "vision_tower.merger.mlp.2" in key: key = key.replace("vision_tower.merger.mlp.2", "vision_tower.merger.mlp_fc2")

        # Handle Conv weight permutations
        if "vision_tower.patch_embed.proj.weight" in key: value = value.permute(0, 2, 3, 4, 1) # (O, H, W, D, I)
        if "audio_tower.conv1.weight" in key or "audio_tower.conv2.weight" in key: value = value.permute(2, 1, 0) # (W, I, O)

        mlx_weights[key] = torch_to_mx(value, dtype=dtype)

    mx.save_safetensors(str(mlx_path / "model.safetensors"), mlx_weights)
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        if (pt_path / config_file).exists(): shutil.copyfile(pt_path / config_file, mlx_path / config_file)
    print("[INFO] Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert Yuna PyTorch model to MLX.")
    parser.add_argument("--pt-path", required=True, type=str, help="Path to the PyTorch model directory.")
    parser.add_argument("--mlx-path", required=True, type=str, help="Path to save the MLX model.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Data type to save the MLX weights.")
    args = parser.parse_args()
    convert(args.pt_path, args.mlx_path, args.dtype)

if __name__ == "__main__": main()