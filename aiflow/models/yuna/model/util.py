import os
import json
from pathlib import Path
from typing import Optional, Union, Dict
import torch
from .vision import VisionConfig

def _rename_dict_keys(original_dict: dict, key_mapping: dict) -> dict:
    """
    Renames keys in a dictionary according to a provided mapping.

    Args:
        original_dict (dict): The original dictionary whose keys need to be renamed.
        key_mapping (dict): A mapping from old key names to new key names (old_key_name: new_key_name).

    Returns:
        dict: A new dictionary with keys renamed according to the mapping.
              Keys not present in the mapping remain unchanged.
    """
    new_dict = {}
    for key, value in original_dict.items():
        new_key_name = key_mapping[key] if key in key_mapping else key
        new_dict[new_key_name] = value
    return new_dict

def _convert_llm_config(hf_config: dict):
    llm_config_key_name_mapping = {
        "hidden_size": "n_embed",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_mlp",
        "rms_norm_eps": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "tie_word_embeddings": "tie_word_embeddings",
        "head_dim": "head_dim",
    }
    return _rename_dict_keys(
        original_dict=hf_config,
        key_mapping=llm_config_key_name_mapping,
    )

def _convert_vision_config(hf_config: dict):
    vision_config = hf_config["vision_config"]

    # Handle different naming conventions between Yuna and Yuna
    if "embed_dim" in vision_config:
        # Yuna format
        vision_config_key_name_mapping = {
            "depth": "n_layer",
            "embed_dim": "n_embed",
            "num_heads": "n_heads",
            "in_chans": "in_channels",
            "hidden_size": "output_n_embed",
            "spatial_patch_size": "spatial_patch_size",
            "temporal_patch_size": "temporal_patch_size",
            "spatial_merge_size": "spatial_merge_size",
        }
    else:
        # Yuna format
        vision_config_key_name_mapping = {
            "depth": "n_layer",
            "hidden_size": "n_embed",
            "num_heads": "n_heads",
            "out_hidden_size": "output_n_embed",
            "in_chans": "in_channels",  # Yuna also uses "in_chans"
            "patch_size": "spatial_patch_size",
            "temporal_patch_size": "temporal_patch_size",
            "spatial_merge_size": "spatial_merge_size",
            "intermediate_size": "intermediate_size",  # For gated MLP
            "hidden_act": "hidden_act",  # Activation function
        }

    return _rename_dict_keys(
        original_dict=vision_config,
        key_mapping=vision_config_key_name_mapping,
    )

def _filter_dict_by_dataclass(params: dict, dataclass_type) -> dict:
    return {k: v for k, v in params.items() if k in dataclass_type.__annotations__}

def load_pretrained_model(
    model_cls,
    repo_id: Union[str, Path],
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    force_download: bool = False,
    **kwargs,
):
    """
    Load a pretrained model using the same logic as the mixin, but as a standalone function.

    Args:
        model_cls: The model class (Yuna, Yuna, etc.)
        repo_id: local path
        device_map: Device mapping for multi-GPU

    Returns:
        Loaded model instance
    """
    # Determine model path
    if os.path.isdir(repo_id):
        model_path = Path(repo_id)
    else:
        raise ValueError("Only local directory loading is supported in this function.")

    # Load config
    with open(model_path / "config.json", "r") as f:
        config_data = json.load(f)

    # Get the appropriate config class from the model
    config_cls = model_cls.get_config_class()

    llm_config = _convert_llm_config(config_data)
    llm_config = _filter_dict_by_dataclass(llm_config, config_cls)

    # tie_word_embeddings might be missing in older models, set default to False
    if "tie_word_embeddings" not in llm_config:
        llm_config["tie_word_embeddings"] = True

    model_config = config_cls(**llm_config)
    if "vision_config" in config_data:
        vision_config = _convert_vision_config(config_data)
        vision_config = _filter_dict_by_dataclass(vision_config, VisionConfig)
        model_config.vision_config = VisionConfig(**vision_config)

    # Standard PyTorch loading (no accelerate)
    model = model_cls(model_config)
    # Try to find the weights file
    weights_path = model_path / "pytorch_model.bin"
    if not weights_path.exists():
        # Try safetensors if available
        weights_path = model_path / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"No model weights found in {model_path}")

    if str(weights_path).endswith(".bin"):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(weights_path))
            model.load_state_dict(state_dict, strict=False)
        except ImportError:
            raise ImportError("safetensors is required to load .safetensors files")

    # Move model to device if specified
    if device_map == "cuda" or (isinstance(device_map, str) and "cuda" in device_map):
        model = model.cuda()
    elif device_map == "mps":
        model = model.to("mps")
    elif device_map == "cpu" or device_map is None:
        model = model.cpu()
    return model