import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .config import ModelConfig
from .vision import YunaVisionEncoder
from .language import YunaModel
from .cache import KVCache

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = YunaVisionEncoder(config.vision_config)
        self.language_model = YunaModel(config.text_config)
        if not config.text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size, config.vocab_size, bias=False
            )
        else:
            self.lm_head = None

    def _get_position_ids(self, input_ids, d_image=None, cache_offset=0):
        B, T = input_ids.shape

        if d_image is None:
            return mx.tile(mx.arange(cache_offset, cache_offset + T).reshape(1, T), (B, 1))

        all_pos_ids = mx.zeros((B, 3, T), dtype=mx.int32)
        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            seq_idx = 0
            image_idx = 0
            pos_chunks = []
            position_id = cache_offset

            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if d_image is not None and token_id == self.config.image_token_id:
                    t, h, w = d_image[image_idx]
                    # FIX: Correctly convert h to a Python integer with .item()
                    t, h, w = t.item(), h.item(), w.item()
                    sms = self.config.vision_config.spatial_merge_size
                    h, w = h // sms, w // sms
                    num_patches = t * h * w

                    t_idx = mx.tile(mx.arange(t).reshape(t, 1), (1, h * w)).flatten()
                    h_idx = mx.tile(mx.arange(h).reshape(1, h, 1), (t, 1, w)).flatten()
                    w_idx = mx.tile(mx.arange(w).reshape(1, 1, w), (t, h, 1)).flatten()

                    pos_vision = mx.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)

                    position_id = pos_vision.max().item() + 1
                    seq_idx += num_patches
                    image_idx += 1
                else:
                    pos_text = mx.tile(mx.array([[position_id]]), (3, 1))
                    pos_chunks.append(pos_text)
                    position_id += 1
                    seq_idx += 1

            if not pos_chunks: continue

            pos_ids_example = mx.concatenate(pos_chunks, axis=1)

            if pos_ids_example.shape[1] > T: pos_ids_example = pos_ids_example[:, :T]
            elif pos_ids_example.shape[1] < T:
                 pad_width = T - pos_ids_example.shape[1]
                 padding = mx.zeros((3, pad_width), dtype=mx.int32)
                 pos_ids_example = mx.concatenate([pos_ids_example, padding], axis=1)

            all_pos_ids[batch_idx] = pos_ids_example
        return all_pos_ids

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        d_image: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        cache_offset = cache[0].offset if cache is not None and cache[0] is not None else 0

        position_ids = self._get_position_ids(input_ids, d_image, cache_offset=cache_offset)
        input_embeds = self.language_model.embed_tokens(input_ids)

        if cache_offset == 0 and pixel_values is not None:
            image_embeds = self.vision_tower(pixels=pixel_values, d_image=d_image)
            image_positions = (input_ids == self.config.image_token_id)

            if image_positions.sum() > 0:
                input_embeds_flat = input_embeds.reshape(-1, input_embeds.shape[-1])
                image_positions_flat = image_positions.flatten()

                image_positions_list = image_positions_flat.tolist()
                indices = [i for i, is_image in enumerate(image_positions_list) if is_image]

                num_image_tokens = image_embeds.shape[0]

                if len(indices) == num_image_tokens:
                    for i, idx in enumerate(indices):
                        input_embeds_flat[idx] = image_embeds[i].astype(input_embeds_flat.dtype)
                    input_embeds = input_embeds_flat.reshape(input_embeds.shape)
                else:
                    print(f"Warning: Mismatch between image tokens ({len(indices)}) and vision features ({num_image_tokens}).")

        x, next_cache = self.language_model(x=input_embeds, position_ids=position_ids, cache=cache, mask=mask)

        if self.lm_head is None:
            logits = self.language_model.embed_tokens.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits, next_cache

    @property
    def layers(self):
        return self.language_model.layers