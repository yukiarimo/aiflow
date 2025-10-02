import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional
from tokenizers import Tokenizer
from .vision import VisionConfig
from tokenizers import Tokenizer

class Processor:
    def __init__(self, repo_id: str, vision_config: Optional[VisionConfig] = None):
        if repo_id.endswith(".json"):
            self.tokenizer = Tokenizer.from_file(repo_id)
        else:
            # Assume repo_id is a directory containing tokenizer.json
            self.tokenizer = Tokenizer.from_file(f"{repo_id}/tokenizer.json")
        self.vision_config = vision_config

        if self.vision_config is not None:
            # Vision-specific setup
            image_pad_token = "<|image_pad|>"
            vision_start_token = "<|vision_start|>"
            vision_end_token = "<|vision_end|>"
            self.tokenizer.add_special_tokens([image_pad_token])
            self.vision_start_token_id = self.tokenizer.encode(vision_start_token).ids[
                0
            ]
            self.image_pad_token_id = self.tokenizer.encode(image_pad_token).ids[0]
            self.vision_end_token_id = self.tokenizer.encode(vision_end_token).ids[0]

            # Constants for image processing
            self.MIN_PIXELS = 3136
            self.MAX_PIXELS = 12845056
            self.IMAGE_MEAN = np.array(
                [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
            )
            self.IMAGE_STD = np.array(
                [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
            )

    def __call__(
        self,
        messages: List,  # Accepts List[str] or List[dict]
        device: Optional[Union[str, torch.device]] = None,
    ) -> dict:
        """
        Process a list of text and/or image inputs.
        If vision_config is None, images are not allowed.
        """

        # Detect if input is chat-style (list of dicts) or raw (list of str/Image)
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            inputs = self.apply_chat_template(messages)
        else:
            inputs = messages

        #print(f"Processor received: {inputs}")
        # Data accumulators
        input_ids = []
        pixels_list = []
        d_image_list = []

        # Identify if we have vision support
        has_vision = self.vision_config is not None
        if not has_vision:
            # If we have no vision_config, do text-only
            for item in inputs:
                if isinstance(item, str):
                    input_ids.extend(self.tokenizer.encode(item).ids)
                else:
                    raise ValueError(
                        f"Images are not supported by a text-only model. Got {type(item)}"
                    )
        else:
            # Vision + text model
            merge_size = self.vision_config.spatial_merge_size
            image_pad_token_id = self.image_pad_token_id

            for item in inputs:
                if isinstance(item, str):
                    # Handle text
                    input_ids.extend(self.tokenizer.encode(item).ids)
                elif isinstance(item, Image.Image):
                    # Handle image
                    patches, t, h, w = self._process_image(item)
                    pixels_list.append(patches)
                    d_image_list.append([t, h, w])

                    pad_token_count = (t * h * w) // (merge_size**2)
                    pad_tokens = [image_pad_token_id] * pad_token_count
                    input_ids.extend(pad_tokens)
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

        # Convert accumulated ids to tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        # Convert all images to tensor if there are any
        if pixels_list:
            pixels_np = np.concatenate(pixels_list, axis=0)
            pixels = torch.tensor(pixels_np, dtype=torch.float)
            d_image = torch.tensor(d_image_list, dtype=torch.long)
        else:
            pixels = None
            d_image = None

        # Move to device (if given)
        if device is not None:
            input_ids = input_ids.to(device)
            if pixels is not None:
                pixels = pixels.to(device)

        # Return a dictionary consistent with common usage
        return {
            "input_ids": input_ids,
            "pixels": pixels,
            "d_image": d_image,
        }

    def apply_chat_template(
        self, messages: List[dict], add_vision_id: bool = False, add_generation_prompt: bool = True
    ) -> List[Union[str, Image.Image]]:
        """
        Apply a custom chat template to the messages.
        This template uses <yuki> for user, <yuna> for assistant, and handles images/videos.
        """
        result = []
        image_count = 0
        video_count = 0

        for message in messages:
            # Role tag
            if message["role"] == "user":
                result.append("<yuki>")
            elif message["role"] == "assistant":
                result.append("<yuna>")
            else:
                result.append(f"<{message['role']}>")

            # Content
            content = message["content"]
            if isinstance(content, str):
                result.append(content)
            else:
                for item in content:
                    # Handle image
                    if (
                        (isinstance(item, dict) and (item.get("type") == "image" or "image" in item or "image_url" in item))
                    ):
                        image_count += 1
                        if add_vision_id:
                            result.append(f"Picture {image_count}: ")
                        result.append("<|vision_start|><|image_pad|><|vision_end|>")
                    # Handle video
                    elif (
                        (isinstance(item, dict) and (item.get("type") == "video" or "video" in item))
                    ):
                        video_count += 1
                        if add_vision_id:
                            result.append(f"Video {video_count}: ")
                        result.append("<|vision_start|><|video_pad|><|vision_end|>")
                    # Handle text
                    elif isinstance(item, dict) and "text" in item:
                        result.append(item["text"])

            # Role closing tag
            if message["role"] == "user":
                result.append("</yuki>")
            elif message["role"] == "assistant":
                result.append("</yuna>")
            else:
                result.append(f"</{message['role']}>")

        # Add generation prompt if requested
        if add_generation_prompt:
            result.append("\n<yuna>")

        return result

    def _smart_resize(
        self, height: int, width: int, factor: int = 28
    ) -> Tuple[int, int]:
        if height < factor or width < factor:
            raise ValueError(
                f"height:{height} or width:{width} must be larger than factor:{factor}"
            )
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor

        if h_bar * w_bar > self.MAX_PIXELS:
            beta = np.sqrt((height * width) / self.MAX_PIXELS)
            h_bar = int(np.floor(height / beta / factor) * factor)
            w_bar = int(np.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < self.MIN_PIXELS:
            beta = np.sqrt(self.MIN_PIXELS / (height * width))
            h_bar = int(np.ceil(height * beta / factor) * factor)
            w_bar = int(np.ceil(width * beta / factor) * factor)

        return h_bar, w_bar

    def _process_image(self, image: Image.Image) -> Tuple[np.ndarray, int, int, int]:
        """Same logic as your existing _process_image method, returning the flatten patches + (t, h, w)."""

        # Example snippet:
        SPATIAL_PATCH_SIZE = self.vision_config.spatial_patch_size
        TEMPORAL_PATCH_SIZE = self.vision_config.temporal_patch_size
        SPATIAL_MERGE_SIZE = self.vision_config.spatial_merge_size

        image_np = np.array(image, dtype=np.float32)
        height, width = image_np.shape[:2]
        resized_height, resized_width = self._smart_resize(
            height,
            width,
            factor=SPATIAL_PATCH_SIZE * SPATIAL_MERGE_SIZE,
        )
        image_resized = image.resize(
            (resized_width, resized_height), resample=Image.BICUBIC
        )
        image_np_resized = np.array(image_resized, dtype=np.float32)

        # Normalize
        image_np_resized = image_np_resized / 255.0
        mean = self.IMAGE_MEAN.reshape(1, 1, -1)
        std = self.IMAGE_STD.reshape(1, 1, -1)
        image_np_resized = (image_np_resized - mean) / std

        # Convert to channels-first and add batch dimension
        image_np_resized = np.transpose(image_np_resized, (2, 0, 1))
        image_np_resized = image_np_resized[np.newaxis, ...]

        # Handle temporal dimension
        if image_np_resized.shape[0] == 1:
            image_np_resized = np.tile(image_np_resized, (TEMPORAL_PATCH_SIZE, 1, 1, 1))

        # Extract patches
        batch_size, channels, height, width = image_np_resized.shape
        grid_t = batch_size // TEMPORAL_PATCH_SIZE
        grid_h = resized_height // SPATIAL_PATCH_SIZE
        grid_w = resized_width // SPATIAL_PATCH_SIZE

        patches = image_np_resized.reshape(
            grid_t,
            TEMPORAL_PATCH_SIZE,
            channels,
            grid_h // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
            grid_w // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * TEMPORAL_PATCH_SIZE * SPATIAL_PATCH_SIZE * SPATIAL_PATCH_SIZE,
        )

        return flatten_patches.astype(np.float32), grid_t, grid_h, grid_w