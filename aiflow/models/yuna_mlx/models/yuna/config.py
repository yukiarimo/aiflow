import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "yuna"
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 1536
    num_heads: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "yuna"
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Union[float, str, List[int]]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "yuna"
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vocab_size: int = 151936
    eos_token_id: int = 151643

    @classmethod
    def from_dict(cls, params):
        # Text config is built from top-level keys
        text_config = TextConfig.from_dict(params)

        # Vision config is built from the nested 'vision_config' dictionary
        vision_config = VisionConfig.from_dict(params.get("vision_config", {}))

        # Model config args are the remaining top-level keys
        model_args = {
            k: v for k, v in params.items()
            if k in inspect.signature(cls).parameters and k not in ["text_config", "vision_config"]
        }

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            **model_args,
        )