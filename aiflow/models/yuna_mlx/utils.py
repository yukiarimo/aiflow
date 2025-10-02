import json
from pathlib import Path
from typing import Tuple
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
import numpy as np
from tokenizers import Tokenizer
from .tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer

from .models.yuna.yuna import Model, ModelConfig

def get_model_path(path_or_hf_repo: str) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists(): raise ValueError(f"Model path {model_path} does not exist.")
    return model_path

def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found in {model_path}")

class YunaProcessor:
    def __init__(self, model_path: Path, config: dict):
        tokenizer_lib_obj = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.tokenizer = TokenizerWrapper(tokenizer_lib_obj, str(model_path), BPEStreamingDetokenizer)

        self.config = config
        self.IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.MIN_PIXELS = 3136
        self.MAX_PIXELS = 12845056

    def __call__(self, prompt: str, images: list = None):
        if images:
            image = images[0]
            pixel_values, d_image = self._process_image(image)

            vision_config = self.config["vision_config"]
            merge_size = vision_config["spatial_merge_size"]
            pad_token_count = (d_image[0,0] * d_image[0,1] * d_image[0,2]) // (merge_size**2)
            image_pad_tokens = [self.config["image_token_id"]] * int(pad_token_count.item())

            # The calling function is now responsible for placing <image>
            if "<image>" in prompt:
                parts = prompt.split("<image>", 1)
                prompt_tokens_pre = self.tokenizer.encode(parts[0]).ids
                prompt_tokens_post = self.tokenizer.encode(parts[1]).ids
                token_ids = prompt_tokens_pre + image_pad_tokens + prompt_tokens_post
            else: # Fallback: append if no placeholder
                prompt_tokens = self.tokenizer.encode(prompt).ids
                token_ids = prompt_tokens + image_pad_tokens
        else:
            pixel_values, d_image = None, None
            token_ids = self.tokenizer.encode(prompt).ids

        input_ids = mx.array([token_ids])

        return { "input_ids": input_ids, "pixel_values": pixel_values, "d_image": d_image }

    def _smart_resize(self, h, w, factor=28):
        if h * w > self.MAX_PIXELS: scale = (self.MAX_PIXELS / (h * w)) ** 0.5
        elif h * w < self.MIN_PIXELS: scale = (self.MIN_PIXELS / (h*w)) ** 0.5
        else: scale = 1.0
        h, w = int(h * scale), int(w * scale)
        h_bar = round(h / factor) * factor
        w_bar = round(w / factor) * factor
        return h_bar, w_bar

    def _process_image(self, image: Image.Image):
        vision_config = self.config["vision_config"]
        sps = vision_config["spatial_patch_size"]
        tps = vision_config["temporal_patch_size"]
        sms = vision_config["spatial_merge_size"]

        if image.mode != "RGB": image = image.convert("RGB")

        w, h = image.size
        rh, rw = self._smart_resize(h, w, factor=sps * sms)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)

        img_np = (np.array(image, dtype=np.float32) / 255.0 - self.IMAGE_MEAN) / self.IMAGE_STD
        img_np = np.expand_dims(img_np.transpose(2, 0, 1), axis=0)
        img_np = np.tile(img_np, (tps, 1, 1, 1))

        grid_t = img_np.shape[0] // tps
        grid_h = rh // sps
        grid_w = rw // sps

        pixels = mx.array(img_np)
        d_image = mx.array([[grid_t, grid_h, grid_w]])

        return pixels, d_image

def load(path_or_hf_repo: str) -> Tuple[nn.Module, object]:
    model_path = get_model_path(path_or_hf_repo)
    config = load_config(model_path)
    model_config = ModelConfig.from_dict(config)

    model = Model(model_config)
    
    weights = mx.load(str(model_path / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=False)

    processor = YunaProcessor(model_path, config)

    return model, processor