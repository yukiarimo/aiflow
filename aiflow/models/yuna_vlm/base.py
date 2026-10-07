import inspect
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling


class LanguageModelOutput:
	def __init__(self, logits, hidden_states=None, cross_attention_states=None, encoder_outputs=None):
		self.logits = logits
		self.hidden_states = hidden_states
		self.cross_attention_states = cross_attention_states
		self.encoder_outputs = encoder_outputs


class BaseModelConfig:
	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

	def to_dict(self):
		res = {}
		for k, v in self.__dict__.items():
			if v is None:
				continue
			if isinstance(v, BaseModelConfig):
				res[k] = v.to_dict()
			else:
				res[k] = v
		return res


class BaseImageProcessor(ImageProcessor):
	def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size=None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
		crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
		crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
		self.image_mean = image_mean
		self.image_std = image_std
		self.size = size
		self.resample = resample
		self.rescale_factor = rescale_factor
		self.data_format = data_format
		self.crop_size = crop_size

	def preprocess(self, images):
		pass
