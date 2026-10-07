import json
from transformers import AutoTokenizer


def _remove_space(x):
	if x and x[0] == " ":
		return x[1:]
	return x


class StreamingDetokenizer:
	__slots__ = ("text", "tokens", "offset")

	def reset(self):
		raise NotImplementedError()

	def add_token(self, token, skip_special_token_ids=[]):
		raise NotImplementedError()

	def finalize(self):
		raise NotImplementedError()

	@property
	def last_segment(self):
		text = self.text

		if text and text[-1] != "\ufffd":
			segment = text[self.offset:]
			self.offset = len(text)
			return segment
		return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer
		self._tokenizer.decode([0])
		self.reset()

	def reset(self):
		self.offset = 0
		self._tokens = []
		self._text = ""
		self._current_tokens = []
		self._current_text = ""

	def add_token(self, token, skip_special_token_ids=[]):
		if token in skip_special_token_ids:
			return
		self._current_tokens.append(token)

	def finalize(self):
		self._tokens.extend(self._current_tokens)
		self._text += self._tokenizer.decode(self._current_tokens)
		self._current_tokens = []
		self._current_text = ""

	@property
	def text(self):
		if self._current_tokens:
			self._current_text = self._tokenizer.decode(self._current_tokens)
		if self._current_text and self._current_text[-1] == "\n":
			self._tokens.extend(self._current_tokens)
			self._text += self._current_text
			self._current_tokens.clear()
			self._current_text = ""
		return self._text + self._current_text

	@property
	def tokens(self):
		return self._tokens


class BPEStreamingDetokenizer(StreamingDetokenizer):
	_byte_decoder = None

	def __init__(self, tokenizer, trim_space=False):
		self.trim_space = trim_space
		self.tokenmap = [None] * len(tokenizer.vocab)
		for value, tokenid in tokenizer.vocab.items():
			self.tokenmap[tokenid] = value
		self.reset()
		self.make_byte_decoder()

	def reset(self):
		self.offset = 0
		self._unflushed = ""
		self.text = ""
		self.tokens = []

	def add_token(self, token, skip_special_token_ids=[]):
		if token in skip_special_token_ids:
			return
		v = self.tokenmap[token]

		if self._byte_decoder[v[0]] == 32:
			current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode("utf-8")
			if self.text or not self.trim_space:
				self.text += current_text
			else:
				self.text += _remove_space(current_text)
			self._unflushed = v
		else:
			self._unflushed += v

	def finalize(self):
		current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode("utf-8")
		if self.text or not self.trim_space:
			self.text += current_text
		else:
			self.text += _remove_space(current_text)
		self._unflushed = ""

	@classmethod
	def make_byte_decoder(cls):
		if cls._byte_decoder is not None:
			return

		char_to_bytes = {}
		limits = [0, ord("!"), ord("~") + 1, ord("¡"), ord("¬") + 1, ord("®"), ord("ÿ") + 1]
		n = 0

		for i, (start, stop) in enumerate(zip(limits, limits[1:])):
			if i % 2 == 0:
				for b in range(start, stop):
					char_to_bytes[chr(2**8 + n)] = b
					n += 1
			else:
				for b in range(start, stop):
					char_to_bytes[chr(b)] = b
		cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
	def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer):
		self._tokenizer = tokenizer
		self._detokenizer = detokenizer_class(tokenizer)

	def __getattr__(self, attr):
		if attr == "detokenizer":
			return self._detokenizer
		else:
			return getattr(self._tokenizer, attr)


def _match(a, b):
	if type(a) != type(b):
		return False
	if isinstance(a, dict):
		return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
	if isinstance(a, list):
		return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))
	return a == b


def _is_bpe_decoder(decoder):
	_target_description = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
	return _match(_target_description, decoder)


def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra={}):
	detokenizer_class = NaiveStreamingDetokenizer

	tokenizer_file = model_path / "tokenizer.json"
	if tokenizer_file.exists():
		with open(tokenizer_file, "r") as f:
			try:
				tokenizer_content = json.load(f)
			except json.JSONDecodeError as e:
				raise json.JSONDecodeError("Failed to parse tokenizer.json", e.doc, e.pos)
		if "decoder" in tokenizer_content:
			if _is_bpe_decoder(tokenizer_content["decoder"]):
				detokenizer_class = BPEStreamingDetokenizer

	if return_tokenizer:
		return TokenizerWrapper(AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra), detokenizer_class)
	else:
		return detokenizer_class
