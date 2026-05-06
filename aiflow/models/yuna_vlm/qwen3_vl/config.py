import inspect
from ..base import BaseModelConfig


class VisionConfig(BaseModelConfig):
	def __init__(self, model_type="qwen3_vl", depth=32, hidden_size=1280, intermediate_size=3420, out_hidden_size=1536, num_heads=16, image_size=384, patch_size=14, vocab_size=32000, mlp_ratio=4.0, in_channels=3, layer_norm_eps=1e-6, spatial_patch_size=14, spatial_merge_size=2, tokens_per_second=2, temporal_patch_size=2, num_position_embeddings=2304, window_size=112, fullatt_block_indexes=None, deepstack_visual_indexes=None):
		self.model_type = model_type
		self.depth = depth
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.out_hidden_size = out_hidden_size
		self.num_heads = num_heads
		self.image_size = image_size
		self.patch_size = patch_size
		self.vocab_size = vocab_size
		self.mlp_ratio = mlp_ratio
		self.in_channels = in_channels
		self.layer_norm_eps = layer_norm_eps
		self.spatial_patch_size = spatial_patch_size
		self.spatial_merge_size = spatial_merge_size
		self.tokens_per_second = tokens_per_second
		self.temporal_patch_size = temporal_patch_size
		self.num_position_embeddings = num_position_embeddings
		self.window_size = window_size
		self.fullatt_block_indexes = fullatt_block_indexes if fullatt_block_indexes is not None else [7, 15, 23, 31]
		self.deepstack_visual_indexes = deepstack_visual_indexes if deepstack_visual_indexes is not None else []


class TextConfig(BaseModelConfig):
	def __init__(self, model_type, num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, rms_norm_eps, vocab_size, num_key_value_heads, head_dim, rope_theta=1000000.0, max_position_embeddings=32768, norm_topk_prob=True, rope_scaling=None, tie_word_embeddings=False, attention_bias=False, hidden_act="silu"):
		self.model_type = model_type
		self.num_hidden_layers = num_hidden_layers
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_attention_heads = num_attention_heads
		self.rms_norm_eps = rms_norm_eps
		self.vocab_size = vocab_size
		self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
		self.head_dim = head_dim
		self.rope_theta = rope_theta
		self.max_position_embeddings = max_position_embeddings
		self.norm_topk_prob = norm_topk_prob
		self.tie_word_embeddings = tie_word_embeddings
		self.attention_bias = attention_bias
		self.hidden_act = hidden_act

		self.rope_scaling = rope_scaling if rope_scaling is not None else {"type": "default", "mrope_section": [24, 20, 20]}

		if self.rope_scaling:
			if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
				self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")

			required_keys = {"mrope_section", "type"}
			if not all(key in self.rope_scaling for key in required_keys):
				raise ValueError(f"rope_scaling must contain keys {required_keys}")

			if not self.rope_scaling["type"] in ["mrope", "default"]:
				raise ValueError("rope_scaling type must be 'mrope' or 'default'")


class ModelConfig(BaseModelConfig):
	def __init__(self, text_config, vision_config, model_type, ignore_index=-100, image_token_id=151655, video_token_id=151656, image_token_index=None, video_token_index=None, vision_start_token_id=151652, vision_end_token_id=151653, vision_token_id=151654, vision_feature_select_strategy="default", vision_feature_layer=-2, vocab_size=32000, eos_token_id=None):
		self.text_config = text_config
		self.vision_config = vision_config
		self.model_type = model_type
		self.ignore_index = ignore_index
		self.image_token_id = image_token_id
		self.video_token_id = video_token_id
		self.image_token_index = image_token_index if image_token_index is not None else image_token_id
		self.video_token_index = video_token_index if video_token_index is not None else video_token_id
		self.vision_start_token_id = vision_start_token_id
		self.vision_end_token_id = vision_end_token_id
		self.vision_token_id = vision_token_id
		self.vision_feature_select_strategy = vision_feature_select_strategy
		self.vision_feature_layer = vision_feature_layer
		self.vocab_size = vocab_size
		self.eos_token_id = eos_token_id

	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
