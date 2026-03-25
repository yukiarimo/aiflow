import inspect
from ..base import BaseModelConfig
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig


class VisionConfig(Qwen3VLVisionConfig):
	def __init__(self, model_type="qwen3_5", **kwargs):
		super().__init__(**kwargs)
		self.model_type = model_type

		if getattr(self, "deepstack_visual_indexes", None) is not None and len(self.deepstack_visual_indexes) > 0:
			raise ValueError(f"deepstack is disabled for qwen3.5 temporally, but it is set to {self.deepstack_visual_indexes}")
		self.deepstack_visual_indexes = []


class TextConfig(BaseModelConfig):
	def __init__(self, model_type, hidden_size, intermediate_size, linear_num_value_heads, linear_num_key_heads, linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim, num_hidden_layers, num_attention_heads, rms_norm_eps, vocab_size, num_key_value_heads, max_position_embeddings, tie_word_embeddings=False, attention_bias=False, head_dim=None, rope_parameters=None, full_attention_interval=4):
		self.model_type = model_type
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.linear_num_value_heads = linear_num_value_heads
		self.linear_num_key_heads = linear_num_key_heads
		self.linear_key_head_dim = linear_key_head_dim
		self.linear_value_head_dim = linear_value_head_dim
		self.linear_conv_kernel_dim = linear_conv_kernel_dim
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.rms_norm_eps = rms_norm_eps
		self.vocab_size = vocab_size
		self.num_key_value_heads = num_key_value_heads
		self.max_position_embeddings = max_position_embeddings
		self.tie_word_embeddings = tie_word_embeddings
		self.attention_bias = attention_bias
		self.head_dim = head_dim
		self.full_attention_interval = full_attention_interval
		self.rope_parameters = rope_parameters if rope_parameters is not None else {"type": "default", "mrope_section": [11, 11, 10], "rope_theta": 100000, "partial_rotary_factor": 0.25, }

		if self.rope_parameters:
			if "type" not in self.rope_parameters and "rope_type" in self.rope_parameters:
				self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

			required_keys = {"mrope_section", "type", "rope_theta", "partial_rotary_factor"}
			if not all(key in self.rope_parameters for key in required_keys):
				raise ValueError(f"rope_parameters must contain keys {required_keys}")


class ModelConfig(BaseModelConfig):
	def __init__(self, text_config, vision_config, model_type, ignore_index=-100, image_token_id=248056, video_token_id=248057, image_token_index=None, video_token_index=None, vision_start_token_id=248045, vision_end_token_id=248046, vocab_size=248320, eos_token_id=None):
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
		self.vocab_size = vocab_size
		self.eos_token_id = eos_token_id

	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
