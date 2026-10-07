import inspect
from aiflow.models.yuna_audio.config import AudioEncoderConfig as AudioConfig
from ..base import BaseModelConfig
from ..qwen3_vl.config import TextConfig as _VLTextConfig
from ..qwen3_vl.config import VisionConfig


class TextConfig(_VLTextConfig):
	"""VL text config with HF `rope_parameters` → `rope_theta` / `rope_scaling` mapping."""
	@classmethod
	def from_dict(cls, params):
		params = dict(params or {})
		rope_parameters = params.pop("rope_parameters", None)
		if rope_parameters is not None:
			if "rope_theta" in rope_parameters and "rope_theta" not in params:
				params["rope_theta"] = rope_parameters["rope_theta"]
			if "rope_scaling" not in params:
				scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
				if "rope_type" in scaling and "type" not in scaling:
					scaling["type"] = scaling.pop("rope_type")
				params["rope_scaling"] = scaling
		return super().from_dict(params)


class ModelConfig(BaseModelConfig):
	def __init__(self, text_config, vision_config, audio_config=None, model_type="qwen3_avlm", ignore_index=-100, image_token_id=151655, video_token_id=151656, image_token_index=None, video_token_index=None, vision_start_token_id=151652, vision_end_token_id=151653, vision_token_id=151654, audio_token_id=151676, audio_start_token_id=151669, audio_end_token_id=151670, vision_feature_select_strategy="default", vision_feature_layer=-2, vocab_size=151936, eos_token_id=None, ):
		if isinstance(text_config, dict):
			text_config = _text_config_from_dict(text_config)
		if isinstance(vision_config, dict):
			vision_config = VisionConfig.from_dict(vision_config)
		if audio_config is None:
			audio_config = AudioConfig()
		elif isinstance(audio_config, dict):
			audio_config = AudioConfig.from_dict(audio_config)

		self.text_config = text_config
		self.vision_config = vision_config
		self.audio_config = audio_config
		self.model_type = model_type
		self.ignore_index = ignore_index
		self.image_token_id = image_token_id
		self.video_token_id = video_token_id
		self.image_token_index = image_token_index if image_token_index is not None else image_token_id
		self.video_token_index = video_token_index if video_token_index is not None else video_token_id
		self.vision_start_token_id = vision_start_token_id
		self.vision_end_token_id = vision_end_token_id
		self.vision_token_id = vision_token_id
		self.audio_token_id = audio_token_id
		self.audio_start_token_id = audio_start_token_id
		self.audio_end_token_id = audio_end_token_id
		self.vision_feature_select_strategy = vision_feature_select_strategy
		self.vision_feature_layer = vision_feature_layer
		self.vocab_size = vocab_size
		self.eos_token_id = eos_token_id

	@classmethod
	def from_dict(cls, params):
		params = dict(params)
		if "text_config" in params and isinstance(params["text_config"], dict):
			params["text_config"] = _text_config_from_dict(params["text_config"])
		if "vision_config" in params and isinstance(params["vision_config"], dict):
			params["vision_config"] = VisionConfig.from_dict(params["vision_config"])
		if "audio_config" in params and isinstance(params["audio_config"], dict):
			params["audio_config"] = AudioConfig.from_dict(params["audio_config"])
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


def _text_config_from_dict(params):
	return TextConfig.from_dict(params)
