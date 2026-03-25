import inspect


class AudioEncoderConfig:
	def __init__(self, num_mel_bins=128, encoder_layers=24, encoder_attention_heads=16, encoder_ffn_dim=4096, d_model=1024, dropout=0.0, attention_dropout=0.0, activation_function="gelu", activation_dropout=0.0, scale_embedding=False, initializer_range=0.02, max_source_positions=1500, n_window=50, output_dim=2048, n_window_infer=800, conv_chunksize=500, downsample_hidden_size=480):
		self.num_mel_bins = num_mel_bins
		self.encoder_layers = encoder_layers
		self.encoder_attention_heads = encoder_attention_heads
		self.encoder_ffn_dim = encoder_ffn_dim
		self.d_model = d_model
		self.dropout = dropout
		self.attention_dropout = attention_dropout
		self.activation_function = activation_function
		self.activation_dropout = activation_dropout
		self.scale_embedding = scale_embedding
		self.initializer_range = initializer_range
		self.max_source_positions = max_source_positions
		self.n_window = n_window
		self.output_dim = output_dim
		self.n_window_infer = n_window_infer
		self.conv_chunksize = conv_chunksize
		self.downsample_hidden_size = downsample_hidden_size

	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


class TextConfig:
	def __init__(self, model_type="qwen3", vocab_size=151936, hidden_size=2048, intermediate_size=6144, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_act="silu", max_position_embeddings=65536, initializer_range=0.02, rms_norm_eps=1e-6, use_cache=True, tie_word_embeddings=True, rope_theta=1000000.0, rope_scaling=None, attention_bias=False, attention_dropout=0.0):
		self.model_type = model_type
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.num_key_value_heads = num_key_value_heads
		self.head_dim = head_dim
		self.hidden_act = hidden_act
		self.max_position_embeddings = max_position_embeddings
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.tie_word_embeddings = tie_word_embeddings
		self.rope_theta = rope_theta
		self.rope_scaling = rope_scaling
		self.attention_bias = attention_bias
		self.attention_dropout = attention_dropout

	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


class ModelConfig:
	def __init__(self, audio_config=None, text_config=None, model_type="qwen3_asr", model_repo=None, audio_token_id=151676, audio_start_token_id=151669, audio_end_token_id=151670, support_languages=None):
		self.model_type = model_type
		self.model_repo = model_repo
		self.audio_token_id = audio_token_id
		self.audio_start_token_id = audio_start_token_id
		self.audio_end_token_id = audio_end_token_id
		self.support_languages = support_languages if support_languages is not None else []

		if audio_config is None:
			self.audio_config = AudioEncoderConfig()
		elif isinstance(audio_config, dict):
			self.audio_config = AudioEncoderConfig.from_dict(audio_config)
		else:
			self.audio_config = audio_config

		if text_config is None:
			self.text_config = TextConfig()
		elif isinstance(text_config, dict):
			self.text_config = TextConfig.from_dict(text_config)
		else:
			self.text_config = text_config

	@classmethod
	def from_dict(cls, params):
		if "thinker_config" in params:
			thinker = params.get("thinker_config", {})
			if thinker.get("model_type") == "qwen3_forced_aligner":
				from .qwen3_forced_aligner import ForcedAlignerConfig
				return ForcedAlignerConfig.from_dict(params)

		params = params.copy()

		if "thinker_config" in params:
			thinker = params.pop("thinker_config")
			if "audio_config" in thinker:
				params["audio_config"] = thinker["audio_config"]
			if "text_config" in thinker:
				params["text_config"] = thinker["text_config"]
			if "audio_token_id" in thinker:
				params["audio_token_id"] = thinker["audio_token_id"]
			if "audio_start_token_id" in thinker:
				params["audio_start_token_id"] = thinker["audio_start_token_id"]
			if "audio_end_token_id" in thinker:
				params["audio_end_token_id"] = thinker["audio_end_token_id"]

		if "audio_config" in params and isinstance(params["audio_config"], dict):
			params["audio_config"] = AudioEncoderConfig.from_dict(params["audio_config"])
		elif "audio_config" not in params:
			params["audio_config"] = AudioEncoderConfig()

		if "text_config" in params and isinstance(params["text_config"], dict):
			params["text_config"] = TextConfig.from_dict(params["text_config"])
		elif "text_config" not in params:
			params["text_config"] = TextConfig()

		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


class STTOutput:
	def __init__(self, text, segments=None, language=None, prompt_tokens=0, generation_tokens=0, total_tokens=0, prompt_tps=0.0, generation_tps=0.0, total_time=0.0):
		self.text = text
		self.segments = segments
		self.language = language
		self.prompt_tokens = prompt_tokens
		self.generation_tokens = generation_tokens
		self.total_tokens = total_tokens
		self.prompt_tps = prompt_tps
		self.generation_tps = generation_tps
		self.total_time = total_time


class BaseModelArgs:
	@classmethod
	def from_dict(cls, params):
		return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
