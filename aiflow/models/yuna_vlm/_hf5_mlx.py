from transformers.models.auto.tokenization_auto import AutoTokenizer, REGISTERED_TOKENIZER_CLASSES  # mlx_lm AutoTokenizer.register("NewlineTokenizer", ...); transformers 5 TOKENIZER_MAPPING.register expects PreTrainedConfig and does key.__module__

if not getattr(AutoTokenizer.register, "_yuna_str_key", False):
	_register = AutoTokenizer.register

	def _register_str(config_class, tokenizer_class=None, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
		if isinstance(config_class, str):
			tok = tokenizer_class or fast_tokenizer_class or slow_tokenizer_class
			if tok is not None:
				REGISTERED_TOKENIZER_CLASSES[tok.__name__] = tok
			return
		return _register(config_class, tokenizer_class=tokenizer_class, slow_tokenizer_class=slow_tokenizer_class, fast_tokenizer_class=fast_tokenizer_class, exist_ok=exist_ok)

	_register_str._yuna_str_key = True
	AutoTokenizer.register = staticmethod(_register_str)
