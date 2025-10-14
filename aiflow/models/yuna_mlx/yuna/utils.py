import json
import os
from pathlib import Path
import mlx.core as mx
from PIL import Image
import numpy as np
from tokenizers import Tokenizer
import re
import cv2
from .audio import audio_to_mel_features, get_feat_extract_output_lengths
from .yuna import KVCache, Model, ModelConfig

def _remove_space(x):
    if x and x[0] == " ": return x[1:]
    return x

class StreamingDetokenizer:
    __slots__ = ("text", "tokens", "offset")
    def reset(self): raise NotImplementedError()
    def add_token(self, token, skip_special_token_ids=[]): raise NotImplementedError()
    def finalize(self): raise NotImplementedError()

    @property
    def last_segment(self):
        text = self.text
        if text and text[-1] != "\ufffd":
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""

class NaiveStreamingDetokenizer(StreamingDetokenizer):
    def __init__(self, tokenizer, trim_space=None):
        self._tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.offset = 0
        self._tokens, self._text, self._current_tokens, self._current_text = [], "", [], ""

    def add_token(self, token, skip_special_token_ids=[]):
        if token in skip_special_token_ids: return
        self._current_tokens.append(token)
        self._tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens, self._current_text = [], ""

    @property
    def text(self):
        if self._current_tokens: self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text.endswith(("\n", " ")):
            if self._current_text.endswith("\ufffd"): self._current_text = self._current_text[:-1]
            else:
                self._text += self._current_text
                self._current_tokens.clear()
                self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self): return self._tokens

class BPEStreamingDetokenizer(StreamingDetokenizer):
    _byte_decoder = None

    def __init__(self, tokenizer, trim_space=False):
        self.trim_space, self._tokenizer = trim_space, tokenizer
        self.vocab = self._tokenizer.get_vocab()
        self.tokenmap = [None] * len(self.vocab)
        for value, tokenid in self.vocab.items(): self.tokenmap[tokenid] = value
        self.reset()
        self.make_byte_decoder()

    def reset(self):
        self.offset, self._unflushed, self.text, self.tokens = 0, "", "", []

    def add_token(self, token, skip_special_token_ids=[]):
        if token in skip_special_token_ids: return
        v = self.tokenmap[token]

        if v[0] not in self._byte_decoder:
             self.finalize()
             self.text += v
             return

        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode("utf-8", "replace")
            self.text += current_text if self.text or not self.trim_space else _remove_space(current_text)
            self._unflushed = v
        else:
            # Check if v is a punctuation and _unflushed ends with a letter to avoid adding spaces after punctuation
            if v and self._unflushed and len(v) == 1 and v[0] in ",.:;!?" and len(self._unflushed) > 0 and self._unflushed[-1].isalnum():
                self._unflushed += v
            else:
                self._unflushed += v

    def finalize(self):
        if not self._unflushed: return

        try:
            current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode("utf-8")
        except:
            # If byte decoding fails, fall back to treating as raw text
            current_text = self._unflushed

        # Add proper spacing around punctuation
        if current_text and len(current_text) > 1:
            # Add space after periods, commas, etc. if followed by a letter
            current_text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', current_text)

        self.text += current_text if self.text or not self.trim_space else _remove_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        if cls._byte_decoder is not None: return
        char_to_bytes, n = {}, 0
        limits = [0, ord("!"), ord("~") + 1, ord("¡"), ord("¬") + 1, ord("®"), ord("ÿ") + 1]
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop): char_to_bytes[chr(2**8 + n)], n = b, n + 1
            else:
                for b in range(start, stop): char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes

class TokenizerWrapper:
    def __init__(self, tokenizer, model_path, detokenizer_class=NaiveStreamingDetokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer, trim_space=True)

        special_tokens = set()
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                config = json.load(f)
            for key in ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token"]:
                token_info = config.get(key)
                if token_info:
                    token_str = token_info.get("content") if isinstance(token_info, dict) else token_info
                    if token_str:
                        special_tokens.add(token_str)
            added_tokens = config.get("additional_special_tokens", [])
            if isinstance(added_tokens, list):
                for token in added_tokens:
                    token_str = token.get("content") if isinstance(token, dict) else token
                    if token_str:
                        special_tokens.add(token_str)

        tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
            for token_info in tokenizer_data.get("added_tokens", []):
                if isinstance(token_info, dict):
                    token_str = token_info.get("content")
                    if token_str and (token_info.get("special") or (token_str.startswith("<") and token_str.endswith(">"))):
                        special_tokens.add(token_str)
                elif isinstance(token_info, str) and token_info.startswith("<") and token_info.endswith(">"):
                    special_tokens.add(token_info)

        self.special_token_strings = sorted(special_tokens, key=len, reverse=True)
        self.stop_strings = [tok for tok in self.special_token_strings if tok.startswith("</") and tok.endswith(">")]
        if not self.stop_strings:
            self.stop_strings = [tok for tok in self.special_token_strings if tok.startswith("<") and tok.endswith(">")]

        vocab = self._tokenizer.get_vocab()
        self.all_special_ids = [vocab[s] for s in self.special_token_strings if s in vocab]

    @property
    def detokenizer(self):
        self._detokenizer.reset()
        return self._detokenizer

    def strip_special_tokens(self, text):
        cleaned = text
        for token in self.special_token_strings:
            cleaned = cleaned.replace(token, "")
        return cleaned

    def decode(self, token_ids, skip_special_tokens=True):
        decoded = self._tokenizer.decode(token_ids)
        return self.strip_special_tokens(decoded) if skip_special_tokens else decoded

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self.detokenizer
        return getattr(self._tokenizer, attr)

def load(model_path: str):
    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f: config = json.load(f)
    model = Model(ModelConfig.from_dict(config))
    model.load_weights(str(model_path / "model.safetensors"), strict=False)
    processor = YunaProcessor(model_path)
    return model, processor

def save_kv_cache(kv_cache, file_path: Path):
    if not kv_cache or kv_cache[0] is None: print("[Cache] KV cache is empty, nothing to save."); return
    tensor_dict = {f"key_{i}": layer.keys for i, layer in enumerate(kv_cache) if layer.keys is not None}
    tensor_dict.update({f"value_{i}": layer.values for i, layer in enumerate(kv_cache) if layer.values is not None})
    meta_dict = {"offsets": [layer.offset for layer in kv_cache]}

    if tensor_dict:
        mx.save_safetensors(str(file_path), tensor_dict)
        with open(file_path.with_suffix(".meta.json"), "w") as f: json.dump(meta_dict, f)
        print(f"[Cache] KV cache saved to {file_path}")

def load_kv_cache(model, file_path: Path):
    meta_path = file_path.with_suffix(".meta.json")

    if not file_path.exists() or not meta_path.exists(): return None
    with open(meta_path, "r") as f: meta_dict = json.load(f)
    tensor_dict = mx.load(str(file_path))
    reconstructed_cache = [KVCache() for _ in model.language_model.layers]
    for i, layer in enumerate(reconstructed_cache):
        key, value = tensor_dict.get(f"key_{i}"), tensor_dict.get(f"value_{i}")
        if key is not None and value is not None:
            layer.keys, layer.values = key, value
            layer.offset = meta_dict["offsets"][i]
    print(f"[Cache] KV cache loaded successfully from {file_path}")
    return reconstructed_cache

def get_model_path(path_or_hf_repo):
    model_path = Path(path_or_hf_repo)
    if not model_path.exists(): raise ValueError(f"Model path {model_path} does not exist.")
    return model_path

def load_config(model_path):
    with open(model_path / "config.json", "r") as f: return json.load(f)

class YunaProcessor:
    def __init__(self, model_path):
        raw_tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.tokenizer = TokenizerWrapper(raw_tokenizer, str(model_path))
        with open(model_path / "config.json", "r") as f: self.config = json.load(f)

        self.placeholders = {"image": "<|vision_start|><|image_pad|><|vision_end|>", "video": "<|vision_start|><|video_pad|><|vision_end|>", "audio": "<|vision_start|><|quad_start|><|vision_end|>"}
        self.placeholder_token_ids = {k: self.tokenizer.encode(v.split('><')[1]+'>').ids[0] for k, v in self.placeholders.items()}
        self.tokenizer.pad_token_id = self.tokenizer.encode("<|endoftext|>").ids[0]

        vis_conf = self.config.get("vision_config", {})
        self.IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.sps, self.tps, self.sms = vis_conf.get("spatial_patch_size", 14), vis_conf.get("temporal_patch_size", 2), vis_conf.get("spatial_merge_size", 2)

    def __call__(self, messages, image_paths=None, video_paths=None, audio_paths=None, add_generation_prompt=False):
        text = self.apply_chat_template(messages, add_generation_prompt) if isinstance(messages, list) else messages
        all_pixels, all_d_image, all_audio_features, all_audio_lens = [], [], [], []
        img_idx, vid_idx, aud_idx = 0, 0, 0
        final_text_parts = []
        last_idx = 0
        placeholder_pattern = re.compile("|".join(re.escape(p) for p in self.placeholders.values()))

        for match in placeholder_pattern.finditer(text):
            final_text_parts.append(text[last_idx:match.start()])
            placeholder = match.group(0)

            if placeholder == self.placeholders["image"] and img_idx < len(image_paths or []):
                pixels, d_image, num_tokens = self._process_image(Image.open(image_paths[img_idx]))
                all_pixels.append(pixels); all_d_image.append(d_image)
                final_text_parts.append(self.tokenizer.decode([self.placeholder_token_ids["image"]]) * num_tokens)
                img_idx += 1

            elif placeholder == self.placeholders["video"] and vid_idx < len(video_paths or []):
                frames, d_images, num_tokens_frame = self._process_video(video_paths[vid_idx])
                all_pixels.extend(frames); all_d_image.extend(d_images)
                final_text_parts.append(self.tokenizer.decode([self.placeholder_token_ids["video"]]) * num_tokens_frame * len(frames))
                vid_idx += 1

            elif placeholder == self.placeholders["audio"] and aud_idx < len(audio_paths or []):
                features, lens, num_tokens = self._process_audio(audio_paths[aud_idx])
                all_audio_features.append(features); all_audio_lens.append(lens)
                final_text_parts.append(self.tokenizer.decode([self.placeholder_token_ids["audio"]]) * num_tokens)
                aud_idx += 1

            last_idx = match.end()

        final_text_parts.append(text[last_idx:])
        input_ids = mx.array([self.tokenizer.encode("".join(final_text_parts)).ids], dtype=mx.int32)
        return {"input_ids": input_ids, "pixel_values": mx.concatenate(all_pixels, axis=0) if all_pixels else None, "d_image": mx.concatenate(all_d_image, axis=0) if all_d_image else None, "audio_features": mx.stack(all_audio_features, axis=0) if all_audio_features else None, "audio_feature_lens": mx.array(all_audio_lens, dtype=mx.int32) if all_audio_lens else None}

    def apply_chat_template(self, messages, add_generation_prompt=True):
        prompt = "<bos><dialog>"
        for msg in messages: prompt += f"<{msg['role']}>{msg['content']}</{msg['role']}>"
        if add_generation_prompt: prompt += "<yuna>"
        return prompt

    def _smart_resize(self, h, w, factor):
        h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
        return (int(h_bar) if h_bar > 0 else factor), (int(w_bar) if w_bar > 0 else factor)

    def _process_image(self, image):
        if image.mode != "RGB": image = image.convert("RGB")
        w, h = image.size
        rh, rw = self._smart_resize(h, w, factor=self.sps * self.sms)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        img_np = (np.array(image, dtype=np.float32) / 255.0 - self.IMAGE_MEAN) / self.IMAGE_STD
        img_np = np.expand_dims(img_np.transpose(2, 0, 1), axis=0)
        img_np = np.tile(img_np, (self.tps, 1, 1, 1))
        gt, gh, gw = img_np.shape[0] // self.tps, rh // self.sps, rw // self.sps
        pixels, d_image = mx.array(img_np), mx.array([[gt, gh, gw]], dtype=mx.int32)
        num_tokens = gt * (gh // self.sms) * (gw // self.sms)
        return pixels, d_image, num_tokens

    def _process_video(self, video_path, fps=1):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(video_fps / fps) if video_fps > fps else 1
        frames, d_images, num_tokens_frame, count = [], [], 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % frame_skip == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pixels, d_image, num_tokens = self._process_image(image)
                frames.append(pixels); d_images.append(d_image)
                if num_tokens_frame == 0: num_tokens_frame = num_tokens
            count += 1

        cap.release()
        return frames, d_images, num_tokens_frame

    def _process_audio(self, audio_path):
        features, L_frames = audio_to_mel_features(audio_path)
        feature_lens = mx.array([L_frames], dtype=mx.int32)
        _, L_tokens = get_feat_extract_output_lengths(feature_lens)
        return features, L_frames, L_tokens.item()

def load(path_or_hf_repo):
    model_path = get_model_path(path_or_hf_repo)
    config = load_config(model_path)
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)
    weights = mx.load(str(model_path / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=False)
    processor = YunaProcessor(model_path)
    return model, processor