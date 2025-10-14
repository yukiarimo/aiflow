import os
import json
from pathlib import Path
from typing import Optional, Union, Dict
import torch
from .vision import VisionConfig
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from .audio import get_feat_extract_output_lengths, audio_to_mel_features
import re
import cv2

class Processor:
    def __init__(self, repo_id, vision_config=None, audio_encoder=None, yuna_model=None):
        tokenizer_path = os.path.join(repo_id, "tokenizer.json") if os.path.isdir(repo_id) else repo_id
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vision_config = vision_config
        self.audio_encoder = audio_encoder
        self.yuna_model = yuna_model
        self.placeholders = {"image": "<|vision_start|><|image_pad|><|vision_end|>", "video": "<|vision_start|><|video_pad|><|vision_end|>", "audio": "<|vision_start|><|quad_start|><|vision_end|>"}
        self.chat_tokens = ["<dialog>", "<yuki>", "</yuki>", "<yuna>", "</yuna>", "<bos>", "<|endoftext|>"]
        self.image_pad_token_id = self.tokenizer.encode("<|image_pad|>").ids[0]
        self.video_pad_token_id = self.tokenizer.encode("<|video_pad|>").ids[0]
        self.audio_chunk_token_id = self.tokenizer.encode("<|quad_start|>").ids[0]

        if self.yuna_model:
            self.yuna_model.image_pad_token_id = self.image_pad_token_id
            self.yuna_model.video_pad_token_id = self.video_pad_token_id
            self.yuna_model.audio_chunk_token_id = self.audio_chunk_token_id

        if self.vision_config:
            self.IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            self.IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __call__(self, messages, image_paths=None, video_paths=None, audio_paths=None, add_generation_prompt=False, device=None):
        image_paths = image_paths or []
        video_paths = video_paths or []
        audio_paths = audio_paths or []

        if isinstance(messages, str): text_prompt = messages # Determine input mode (naked vs chat)
        else: text_prompt = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt) # Assumes list of dicts for chat

        placeholder_pattern = re.compile("|".join(re.escape(p) for p in self.placeholders.values())) # Find all modality placeholders in the text
        all_pixels, all_d_image, all_audio_features, all_audio_lens = [], [], [], []
        img_idx, vid_idx, aud_idx = 0, 0, 0
        final_text_parts = []
        last_idx = 0

        for match in placeholder_pattern.finditer(text_prompt):
            final_text_parts.append(text_prompt[last_idx:match.start()]) # Add text part before the placeholder
            placeholder = match.group(0)

            if placeholder == self.placeholders["image"]:
                if img_idx < len(image_paths):
                    pixels, d_image, num_tokens = self._process_visual(image_paths[img_idx])
                    all_pixels.append(pixels)
                    all_d_image.append(d_image)
                    final_text_parts.append(self.tokenizer.decode([self.image_pad_token_id]) * num_tokens)
                    img_idx += 1

            elif placeholder == self.placeholders["video"]:
                if vid_idx < len(video_paths):
                    # Process video as a sequence of images
                    video_frames, video_d_image, num_tokens_per_frame = self._process_video(video_paths[vid_idx])
                    all_pixels.extend(video_frames)
                    all_d_image.extend(video_d_image)

                    # For video, we use the image pad token for each frame
                    total_tokens = num_tokens_per_frame * len(video_frames)
                    final_text_parts.append(self.tokenizer.decode([self.image_pad_token_id]) * total_tokens)
                    vid_idx += 1

            elif placeholder == self.placeholders["audio"]:
                if aud_idx < len(audio_paths):
                    audio_feat, audio_len, num_tokens = self._process_audio(audio_paths[aud_idx])
                    all_audio_features.append(audio_feat)
                    all_audio_lens.append(audio_len)
                    final_text_parts.append(self.tokenizer.decode([self.audio_chunk_token_id]) * num_tokens)
                    aud_idx += 1

            last_idx = match.end()

        final_text_parts.append(text_prompt[last_idx:])  # Add the remaining text part
        final_prompt = "".join(final_text_parts) # Tokenize the final prompt with all placeholder tokens expanded
        input_ids = torch.tensor([self.tokenizer.encode(final_prompt).ids], dtype=torch.long)
        pixels_tensor = torch.cat(all_pixels, dim=0) if all_pixels else None
        d_image_tensor = torch.cat(all_d_image, dim=0) if all_d_image else None
        audio_features_tensor = torch.cat(all_audio_features, dim=1) if all_audio_features else None
        audio_lens_tensor = torch.cat(all_audio_lens, dim=0) if all_audio_lens else None
        result = {"input_ids": input_ids, "pixels": pixels_tensor, "d_image": d_image_tensor, "audio_features": audio_features_tensor, "audio_feature_lens": audio_lens_tensor}

        if device:
            for k, v in result.items():
                if v is not None and isinstance(v, torch.Tensor): result[k] = v.to(device)
        return result

    def apply_chat_template(self, messages, add_generation_prompt=True):
        prompt = "<bos><dialog>"
        for msg in messages: prompt += f"<{msg['role']}>{msg['content']}</{msg['role']}>"
        if add_generation_prompt: prompt += "<yuna>"
        return prompt

    def _process_visual(self, image_path):
        image = Image.open(image_path).convert('RGB')
        patches, t, h, w = self._process_image_to_patches(image)
        pixels = torch.from_numpy(patches).float().unsqueeze(0)
        d_image = torch.tensor([[t, h, w]], dtype=torch.long)
        sms = self.vision_config.spatial_merge_size
        num_tokens = t * (h // sms) * (w // sms)
        return pixels, d_image, num_tokens

    def _process_video(self, video_path, fps=1):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(video_fps / fps) if video_fps > fps else 1
        frame_pixels, frame_d_images = [], []
        num_tokens_per_frame = 0

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                pixels, d_image, num_tokens = self._process_visual(image)
                frame_pixels.append(pixels)
                frame_d_images.append(d_image)
                if num_tokens_per_frame == 0: num_tokens_per_frame = num_tokens

            frame_count += 1
        cap.release()
        return frame_pixels, frame_d_images, num_tokens_per_frame

    def _process_audio(self, audio_path):
        features, L_frames = audio_to_mel_features(audio_path)
        audio_feature_lens = torch.tensor([L_frames], dtype=torch.long)
        _, L_tokens = get_feat_extract_output_lengths(audio_feature_lens)
        num_audio_tokens = L_tokens.item()
        return features, audio_feature_lens, num_audio_tokens

    def _smart_resize(self, height, width, factor=14):
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        return int(h_bar) if h_bar > 0 else factor, int(w_bar) if w_bar > 0 else factor

    def _process_image_to_patches(self, image):
        sps = self.vision_config.spatial_patch_size
        tps = self.vision_config.temporal_patch_size
        sms = self.vision_config.spatial_merge_size
        resize_factor = sps * sms # The image must be resized to be divisible by BOTH patch size and merge size.

        image_np = np.array(image, dtype=np.float32)
        h, w = image_np.shape[:2]
        rh, rw = self._smart_resize(h, w, factor=resize_factor)
        image_resized = image.resize((rw, rh), resample=Image.BICUBIC)
        image_np_resized = np.array(image_resized, dtype=np.float32)
        image_np_resized = (image_np_resized / 255.0 - self.IMAGE_MEAN) / self.IMAGE_STD
        image_np_resized = np.transpose(image_np_resized, (2, 0, 1))[np.newaxis, ...]

        if image_np_resized.shape[0] == 1: image_np_resized = np.tile(image_np_resized, (tps, 1, 1, 1))
        b, c, h_resized, w_resized = image_np_resized.shape
        gt, gh, gw = b // tps, h_resized // sps, w_resized // sps # Use the RESIZED dimensions (rh, rw) to calculate the patch grid, not the original (h, w)
        patches = image_np_resized.reshape(gt, tps, c, gh, sps, gw, sps)
        patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        flatten_patches = patches.reshape(gt * gh * gw, c * tps * sps * sps)
        return flatten_patches.astype(np.float32), gt, gh, gw

def _filter_dict_by_dataclass(params: dict, dataclass_type) -> dict: return {k: v for k, v in params.items() if k in dataclass_type.__annotations__}

def load_pretrained_model(model_cls, repo_id: Union[str, Path], device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto", **kwargs):
    if not os.path.isdir(repo_id): raise ValueError("Only local directory loading is supported.")
    model_path = Path(repo_id)

    with open(model_path / "config.json", "r") as f: config_data = json.load(f)
    config_cls = model_cls.get_config_class()
    model_params = _filter_dict_by_dataclass(config_data, config_cls)
    model_config = config_cls(**model_params)

    if "vision_config" in config_data:
        vision_params = _filter_dict_by_dataclass(config_data["vision_config"], VisionConfig)
        model_config.vision_config = VisionConfig(**vision_params)

    if "audio_config" in config_data or "audio_config" in kwargs: model_config.audio_config = {}
    else: model_config.audio_config = None

    model = model_cls(model_config)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "pytorch_model.bin"
        if not weights_path.exists(): raise FileNotFoundError(f"No model weights found in {model_path}")

    if str(weights_path).endswith(".bin"): state_dict = torch.load(weights_path, map_location="cpu")
    else:
        import safetensors.torch
        state_dict = safetensors.torch.load_file(str(weights_path))

    model.load_state_dict(state_dict, strict=False)

    if device_map == "cuda" or (isinstance(device_map, str) and "cuda" in device_map): model = model.cuda()
    elif device_map == "mps": model = model.to("mps")
    elif device_map == "cpu" or device_map is None: model = model.cpu()
    return model