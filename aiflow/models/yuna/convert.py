import torch
from safetensors.torch import save_file
from model.yuna import Yuna

ckpt_path = "/content/drive/MyDrive/version_0/checkpoints/epoch=3-step=1544.ckpt"
output_path = "/content/drive/MyDrive/converted_model_epoch=3.safetensors"

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# If using Lightning, model weights are usually under 'state_dict'
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Remove 'model.' prefix if present
state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

# Save as safetensors
save_file(state_dict, output_path)
print(f"Saved safetensors to {output_path}")