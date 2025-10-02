import json
import os
from functools import partial
from json import JSONDecodeError
from typing import List

from tokenizers import Tokenizer

REPLACEMENT_CHAR = "\ufffd"

def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x

class StreamingDetokenizer:
    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.offset = 0
        self._tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return
        self._current_tokens.append(token)
        self._tokens.append(token)


    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text.endswith(("\n", " ")):
            if self._current_text.endswith("\ufffd"):
                self._current_text = self._current_text[:-1]
            else:
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
        self._tokenizer = tokenizer
        self.vocab = self._tokenizer.get_vocab()
        self.tokenmap = [None] * len(self.vocab)
        for value, tokenid in self.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return
        v = self.tokenmap[token]
        # It's possible for v to be out of range for the byte_decoder if it's a special token
        if v[0] not in self._byte_decoder:
             self.finalize() # Flush previous content
             self.text += v
             return

        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", "replace")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8"
        )
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
    """A wrapper that combines a tokenizer and a streaming detokenizer."""

    def __init__(self, tokenizer: Tokenizer, model_path: str, detokenizer_class=BPEStreamingDetokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

        # FIX: Reliably load special tokens from tokenizer_config.json
        special_tokens = set()
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                config = json.load(f)
                for key in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token"]:
                    token_info = config.get(key)
                    if token_info:
                        # Sometimes it's a dict {'content': '...'}, sometimes a string
                        token_str = token_info.get("content") if isinstance(token_info, dict) else token_info
                        if token_str: special_tokens.add(token_str)

                # Handle additional special tokens
                added_tokens = config.get("additional_special_tokens", [])
                if isinstance(added_tokens, list):
                    for token in added_tokens:
                        token_str = token.get("content") if isinstance(token, dict) else token
                        if token_str: special_tokens.add(token_str)

        vocab = self._tokenizer.get_vocab()
        self.all_special_ids = [vocab[s] for s in special_tokens if s in vocab]

    @property
    def detokenizer(self):
        self._detokenizer.reset()
        return self._detokenizer

    # Forward all other attribute requests to the underlying tokenizer
    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self.detokenizer
        return getattr(self._tokenizer, attr)