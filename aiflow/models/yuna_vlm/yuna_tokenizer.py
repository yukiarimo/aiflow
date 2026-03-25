import json
import re
import unicodedata
from collections import defaultdict
from transformers import AutoTokenizer


def _get_byte_to_char_map():
	bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)))
	cs = list(bs)
	n = 0

	for b in range(2**8):
		if b not in bs:
			bs.append(b)
			cs.append(2**8 + n)
			n += 1
	return {b: chr(c) for b, c in zip(bs, cs)}


_BYTES_TO_CHARS = _get_byte_to_char_map()
_CHARS_TO_BYTES = {c: b for b, c in _BYTES_TO_CHARS.items()}


def _get_pairs(word):
	pairs = set()
	if len(word) < 2:
		return pairs
	prev_char = word[0]
	for char in word[1:]:
		pairs.add((prev_char, char))
		prev_char = char
	return pairs


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


class YunaNormalizer:
	def normalize(self, text):
		return text


class YunaNFCNormalizer(YunaNormalizer):
	def normalize(self, text):
		return unicodedata.normalize("NFC", text)


class YunaPreTokenizer:
	def pre_tokenize(self, text):
		raise NotImplementedError


class YunaSplitPreTokenizer(YunaPreTokenizer):
	def __init__(self, pattern):
		if isinstance(pattern, str):
			self.pattern = re.compile(pattern)
		else:
			self.pattern = pattern

	def pre_tokenize(self, text):
		matches = list(self.pattern.finditer(text))
		if not matches:
			return [text]

		last_end = 0
		splits = []
		for match in matches:
			start, end = match.span()
			if last_end < start:
				splits.append(text[last_end:start])
			splits.append(text[start:end])
			last_end = end
		if last_end < len(text):
			splits.append(text[last_end:])

		return splits


class YunaByteLevelPreTokenizer(YunaPreTokenizer):
	def pre_tokenize(self, text):
		return ["".join(_BYTES_TO_CHARS[b] for b in text.encode("utf-8"))]


class YunaSequencePreTokenizer(YunaPreTokenizer):
	def __init__(self, pre_tokenizers):
		self.pre_tokenizers = pre_tokenizers

	def pre_tokenize(self, text):
		words = [text]
		for pre_tokenizer in self.pre_tokenizers:
			new_words = []
			for word in words:
				new_words.extend(pre_tokenizer.pre_tokenize(word))
			words = new_words
		return words


class YunaDecoder:
	def decode(self, tokens):
		raise NotImplementedError


class YunaByteLevelDecoder(YunaDecoder):
	def decode(self, tokens):
		text = "".join(tokens)
		buffer = bytearray()
		for char in text:
			if char in _CHARS_TO_BYTES:
				buffer.append(_CHARS_TO_BYTES[char])
		try:
			return buffer.decode("utf-8")
		except UnicodeDecodeError:
			return "\ufffd"


class YunaBPEModel:
	def __init__(self, vocab, merges, cache_capacity=10000, unk_token=None):
		self.vocab = vocab
		self.id_to_token_map = {i: t for t, i in self.vocab.items()}
		self.unk_token = unk_token
		self.merges = {}
		for i, merge_rule in enumerate(merges):
			if isinstance(merge_rule, str):
				p1, p2 = merge_rule.split()
			elif isinstance(merge_rule, (list, tuple)) and len(merge_rule) == 2:
				p1, p2 = merge_rule
			else:
				raise TypeError(f"Invalid merge rule format found: {merge_rule}")
			self.merges[(p1, p2)] = i

		self.cache = {}
		self.cache_capacity = cache_capacity

	def get_vocab_size(self):
		return len(self.vocab)

	def token_to_id(self, token):
		return self.vocab.get(token)

	def id_to_token(self, token_id):
		return self.id_to_token_map.get(token_id)

	def tokenize(self, word):
		if word in self.cache:
			return self.cache[word]

		if len(self.cache) > self.cache_capacity:
			self.cache.pop(next(iter(self.cache)))

		word_tuple = tuple(word)
		if not word_tuple:
			return []

		pairs = _get_pairs(word_tuple)
		if not pairs:
			return [word]

		while True:
			bigram = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
			if bigram not in self.merges:
				break

			first, second = bigram
			new_word = []
			i = 0
			while i < len(word_tuple):
				try:
					j = word_tuple.index(first, i)
					new_word.extend(word_tuple[i:j])
					i = j
				except ValueError:
					new_word.extend(word_tuple[i:])
					break

				if (i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second):
					new_word.append(first + second)
					i += 2
				else:
					new_word.append(word_tuple[i])
					i += 1
			word_tuple = tuple(new_word)
			if len(word_tuple) == 1:
				break
			pairs = _get_pairs(word_tuple)

		result = list(word_tuple)
		self.cache[word] = result
		return result


class YunaBpeTrainer:
	def __init__(self, vocab_size=30000, min_frequency=2, special_tokens=None, pre_tokenizer=None):
		self.vocab_size = vocab_size
		self.min_frequency = min_frequency
		self.special_tokens = special_tokens if special_tokens is not None else []
		self.pre_tokenizer = pre_tokenizer if pre_tokenizer is not None else YunaSplitPreTokenizer(r"\w+|[^\w\s]")

	def train(self, iterator):
		vocab = {token: i for i, token in enumerate(self.special_tokens)}
		for byte_char in _BYTES_TO_CHARS.values():
			if byte_char not in vocab:
				vocab[byte_char] = len(vocab)

		print("Stage 1: Counting word frequencies...")
		word_counts = defaultdict(int)
		for text in iterator:
			normalized = unicodedata.normalize("NFC", text)
			words = self.pre_tokenizer.pre_tokenize(normalized)
			for word in words:
				word_counts["".join(_BYTES_TO_CHARS[b] for b in word.encode("utf-8"))] += 1

		splits = {word: list(word) for word in word_counts}

		print("Stage 2: Merging pairs...")
		merges = []
		num_merges_needed = self.vocab_size - len(vocab)

		for i in range(num_merges_needed):
			pair_stats = defaultdict(int)
			for word, count in word_counts.items():
				symbols = splits[word]
				for j in range(len(symbols) - 1):
					pair_stats[symbols[j], symbols[j + 1]] += count

			if not pair_stats:
				break

			best_pair = max(pair_stats, key=pair_stats.get)

			if pair_stats[best_pair] < self.min_frequency:
				print(f"Stopping early. Best pair frequency {pair_stats[best_pair]} < min_frequency {self.min_frequency}.")
				break

			merges.append(list(best_pair))
			new_token = "".join(best_pair)
			vocab[new_token] = len(vocab)

			if (i + 1) % 100 == 0 or (i + 1) == num_merges_needed:
				print(f"  Merge {i + 1}/{num_merges_needed}: Merged '{best_pair[0]}' and '{best_pair[1]}' into '{new_token}' (count: {pair_stats[best_pair]})")

			for word in list(word_counts.keys()):
				symbols = splits[word]
				j = 0
				new_symbols = []

				while j < len(symbols):
					if j < len(symbols) - 1 and symbols[j] == best_pair[0] and symbols[j + 1] == best_pair[1]:
						new_symbols.append(new_token)
						j += 2
					else:
						new_symbols.append(symbols[j])
						j += 1
				splits[word] = new_symbols

		print(f"Training complete. Final vocab size: {len(vocab)}")
		return vocab, merges


class YunaTokenizer:
	def __init__(self, model, normalizer, pre_tokenizer, decoder, added_tokens, pad_token=None, eos_token=None):
		self.model = model
		self.normalizer = normalizer
		self.pre_tokenizer = pre_tokenizer
		self.decoder = decoder
		self.added_tokens = added_tokens
		self.added_tokens_decoder_map = {i: t for t, i in added_tokens.items()}
		self.pad_token = pad_token
		self.eos_token = eos_token
		special_tokens_pattern = "|".join(re.escape(token) for token in sorted(added_tokens.keys(), key=len, reverse=True))
		self.special_tokens_regex = re.compile(f"({special_tokens_pattern})") if special_tokens_pattern else None

	@classmethod
	def from_dict(cls, config):
		model_config = config["model"]
		bpe_model = YunaBPEModel(vocab=model_config["vocab"], merges=model_config["merges"], unk_token=model_config.get("unk_token"))
		normalizer_config = config.get("normalizer")
		normalizer = YunaNormalizer()

		if normalizer_config and normalizer_config["type"] == "NFC":
			normalizer = YunaNFCNormalizer()

		pre_tok_config = config["pre_tokenizer"]
		pre_tokenizers = []

		for pt_conf in pre_tok_config.get("pretokenizers", []):
			if pt_conf["type"] == "Split":
				pre_tokenizers.append(YunaSplitPreTokenizer(pt_conf["pattern"]["Regex"]))
			elif pt_conf["type"] == "ByteLevel":
				pre_tokenizers.append(YunaByteLevelPreTokenizer())
		pre_tokenizer = YunaSequencePreTokenizer(pre_tokenizers)
		decoder_config = config["decoder"]

		if decoder_config["type"] == "ByteLevel":
			decoder = YunaByteLevelDecoder()
		else:
			raise ValueError(f"Unsupported decoder: {decoder_config['type']}")

		added_tokens = {tok["content"]: tok["id"] for tok in config.get("added_tokens", [])}
		pad_token, eos_token = None, None

		if "padding" in config and config["padding"] is not None:
			pad_id = config["padding"].get("pad_id")
			pad_token = next((tok["content"] for tok in config["added_tokens"] if tok["id"] == pad_id), None)

		if "post_processor" in config and config["post_processor"] is not None:
			eos_token_id = config["post_processor"].get("eos_token_id")
			eos_token = next((tok["content"] for tok in config["added_tokens"] if tok["id"] == eos_token_id), None)
		return cls(model=bpe_model, normalizer=normalizer, pre_tokenizer=pre_tokenizer, decoder=decoder, added_tokens=added_tokens, pad_token=pad_token, eos_token=eos_token)

	@classmethod
	def from_file(cls, filepath):
		with open(filepath, "r", encoding="utf-8") as f:
			config = json.load(f)
		return cls.from_dict(config)

	def save(self, filepath, pretty=True):
		sorted_merges = sorted(self.model.merges.items(), key=lambda item: item[1])
		config = {"version": "1.0", "truncation": None, "padding": None, "post_processor": None, "added_tokens": [{"content": content, "id": id, "special": True} for content, id in self.added_tokens.items()], "model": {"type": "BPE", "vocab": self.model.vocab, "merges": [list(pair) for pair, rank in sorted_merges], "unk_token": self.model.unk_token, }, "normalizer": {"type": "NFC"}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": self.pre_tokenizer.pre_tokenizers[0].pattern.pattern}}, {"type": "ByteLevel"}]}, "decoder": {"type": "ByteLevel"}, }
		with open(filepath, "w", encoding="utf-8") as f:
			if pretty:
				json.dump(config, f, ensure_ascii=False, indent=2)
			else:
				json.dump(config, f, ensure_ascii=False)
		print(f"Yuna's tokenizer configuration saved to {filepath}")

	def encode(self, text):
		ids = []

		text_parts = [text]
		if self.special_tokens_regex:
			text_parts = self.special_tokens_regex.split(text)

		for part in text_parts:
			if not part:
				continue

			if part in self.added_tokens:
				ids.append(self.added_tokens[part])
			else:
				normalized = self.normalizer.normalize(part)
				words = self.pre_tokenizer.pre_tokenize(normalized)
				for word in words:
					tokens = self.model.tokenize(word)
					for token in tokens:
						token_id = self.model.token_to_id(token)
						if token_id is not None:
							ids.append(token_id)
						elif self.model.unk_token and self.model.token_to_id(self.model.unk_token) is not None:
							ids.append(self.model.token_to_id(self.model.unk_token))
		return ids

	def decode(self, ids, skip_special_tokens=False):
		tokens_to_decode = []
		for token_id in ids:
			if skip_special_tokens and self.added_tokens_decoder_map.get(token_id, None) is not None:
				continue
			token_str = self.model.id_to_token(token_id) or self.added_tokens_decoder_map.get(token_id)

			if token_str is not None:
				tokens_to_decode.append(token_str)
		return self.decoder.decode(tokens_to_decode)

	def train(self, iterator, vocab_size=30000, min_frequency=2):
		print("Starting training process for Yuna...")
		trainer = YunaBpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=list(self.added_tokens.keys()), pre_tokenizer=YunaSplitPreTokenizer(r"\w+|[^\w\s]"))
		new_vocab, new_merges_str = trainer.train(iterator)
		new_merges = [m.split(' ') for m in new_merges_str]

		self.model = YunaBPEModel(vocab=new_vocab, merges=new_merges, unk_token=self.model.unk_token)
		print("Yuna's tokenizer model has been updated with new training.")
