import re
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ..qwen3_vl.language import LanguageModel
from ..qwen3_vl.vision import VisionModel
from .audio import AudioModel


def masked_scatter(final_embedding, image_mask_expanded, scaled_image_features):
	final_embedding_shape = final_embedding.shape
	scaled_image_features_flattened = mx.flatten(scaled_image_features)
	final_embedding_flattened = mx.flatten(final_embedding)
	image_mask_expanded_flattened = mx.flatten(image_mask_expanded)
	image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
	final_embedding_flattened[image_positions] = scaled_image_features_flattened
	final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)
	return final_embedding


def splice_audio_features(inputs_embeds, input_ids, audio_features, audio_token_id):
	"""Scatter audio features onto every <|audio_pad|> (supports multiple non-contiguous runs)."""
	audio_features = audio_features.astype(inputs_embeds.dtype)
	audio_token_mask = input_ids == audio_token_id
	if not audio_token_mask.any():
		return inputs_embeds

	n_audio_tokens = int(audio_token_mask.sum())
	n_audio_features = int(audio_features.shape[0])
	if n_audio_tokens != n_audio_features:
		raise ValueError(f"Audio features and audio tokens do not match: tokens={n_audio_tokens}, features={n_audio_features}")

	special_mask = mx.broadcast_to(audio_token_mask[..., None], inputs_embeds.shape)
	return masked_scatter(inputs_embeds, special_mask, audio_features)


class Model(nn.Module):
	"""Qwen3 AVL: Yuna VL 2B + Qwen3-ASR audio encoder/projector."""
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.vision_tower = VisionModel(config.vision_config)
		self.audio_tower = AudioModel(config.audio_config)
		self.language_model = LanguageModel(config.text_config, config)

	def get_audio_features(self, input_features, feature_attention_mask=None):
		return self.audio_tower(input_features, feature_attention_mask)

	def get_input_embeddings(self, input_ids=None, pixel_values=None, input_features=None, feature_attention_mask=None, **kwargs):
		image_grid_thw = kwargs.get("image_grid_thw", None)
		video_grid_thw = kwargs.get("video_grid_thw", None)
		grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
		inputs_embeds = self.language_model.model.embed_tokens(input_ids)
		visual_pos_masks = None
		deepstack_visual_embeds = None

		if pixel_values is not None:
			dtype = self.vision_tower.patch_embed.proj.weight.dtype
			pixel_values = pixel_values.astype(dtype)
			hidden_states, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw)
			inputs_embeds, image_mask = self.merge_input_ids_with_image_features(hidden_states, inputs_embeds, input_ids, self.config.image_token_index, self.config.video_token_index, )
			visual_pos_masks = image_mask[..., 0]
			deepstack_visual_embeds = deepstack_image_embeds

		if input_features is not None:
			audio_features = self.get_audio_features(input_features, feature_attention_mask)
			inputs_embeds = splice_audio_features(inputs_embeds, input_ids, audio_features, self.config.audio_token_id, )

		return {"inputs_embeds": inputs_embeds, "visual_pos_masks": visual_pos_masks, "deepstack_visual_embeds": deepstack_visual_embeds, }

	@staticmethod
	def merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, image_token_index, video_token_index):
		special_image_mask = input_ids == image_token_index
		special_video_mask = input_ids == video_token_index
		special_image_mask = special_image_mask | special_video_mask
		n_image_tokens = special_image_mask.sum()
		special_image_mask = special_image_mask[..., None]
		special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)
		n_image_features = image_features.shape[0]
		n_image_mask_elements = special_image_mask.sum()

		if n_image_mask_elements != image_features.size:
			raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

		inputs_embeds = masked_scatter(inputs_embeds, special_image_mask, image_features)
		return inputs_embeds, special_image_mask

	@property
	def layers(self):
		return self.language_model.model.layers

	def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, input_features=None, feature_attention_mask=None, **kwargs):
		cache_offset = 0  # only fuse multimodal features on the first (prefill) step
		if cache and cache[0] is not None:
			offset = cache[0].offset
			if isinstance(offset, int):
				cache_offset = offset
			elif isinstance(offset, mx.array):
				cache_offset = (offset if offset.ndim == 0 else offset[0]).item()

		if cache_offset > 0:
			pixel_values = None
			input_features = None
			feature_attention_mask = None
			kwargs.pop("image_grid_thw", None)
			kwargs.pop("video_grid_thw", None)

		embeds = self.get_input_embeddings(input_ids, pixel_values, input_features=input_features, feature_attention_mask=feature_attention_mask, **kwargs, )
		kwargs.update({"pixel_values": pixel_values, **embeds})
		logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
		return logits

	@staticmethod
	def sanitize(weights):
		sanitized_weights = {}
		for key, value in weights.items():
			if "multi_modal_projector.linear_1" in key:
				key = key.replace("model.multi_modal_projector.linear_1", "audio_tower.proj1")
				key = key.replace("multi_modal_projector.linear_1", "audio_tower.proj1")
			elif "multi_modal_projector.linear_2" in key:
				key = key.replace("model.multi_modal_projector.linear_2", "audio_tower.proj2")
				key = key.replace("multi_modal_projector.linear_2", "audio_tower.proj2")
			elif "audio_tower" in key:
				key = re.sub(r"^(?:model\.)?audio_tower", "audio_tower", key)
			elif "visual" in key:
				key = re.sub(r"^(?:model\.)?(?:language_model\.)*visual", "vision_tower", key)
			elif "model.language_model" in key:
				key = re.sub(r"model(?:\.language_model)+", "language_model.model", key)
			elif "lm_head" in key:
				key = key.replace("lm_head", "language_model.lm_head")

			if "conv2d" in key and key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 4:
				if value.shape[2] == value.shape[3] and value.shape[1] != value.shape[2]:  # PyTorch Conv2d (O,I,K,K)->MLX (O,K,K,I); already-MLX is (O,K,K,I) where shape[1]==shape[2]==K
					value = value.transpose(0, 2, 3, 1)

			sanitized_weights[key] = value

		return sanitized_weights
