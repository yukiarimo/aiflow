# Copyright 2026 Apple Inc. / Yuna AVL gather extension.
#
# Qwen3-VL pipelined decoder + a second gather for AVL audio.
#
# Host contract:
#   image slots: vocab + 0 .. n_image-1          → image_embeds [N, h]
#   audio slots: vocab + n_image .. n_image+M-1  → audio_embeds [M, h]
#   deepstack + interleaved M-RoPE stay image-only
#   audio uses 1-D text RoPE (same as post-image text)

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from typing_extensions import Self, override

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
from coreai_models.models.macos.qwen3_vl import (
    DEFAULT_MROPE_SECTION,
    PIPELINED_STATE_NAMES,
    Qwen3VLPipelinedForCausalLM,
    Qwen3VLPipelinedModel,
    _trace_rope_shift,
)
from coreai_models.primitives.macos.cache import KVCache


def _trace_input_ids_avl(
    batch: int, query: int, vocab: int, n_image: int, n_audio: int
) -> torch.Tensor:
    """Keep both gathers live: S=1 is one image slot; longer traces mix image + audio."""
    ids = torch.randint(1, vocab, (batch, query), dtype=torch.int32)
    if query == 1:
        ids[0, 0] = vocab
        return ids
    img_run = min(n_image, max(2, query // 4))
    aud_run = min(n_audio, max(2, query // 4))
    used = img_run + aud_run
    if used >= query:
        img_run = max(1, query // 3)
        aud_run = max(1, query - img_run - 1)
    text_len = max(1, query - img_run - aud_run)
    ids[0, text_len : text_len + img_run] = torch.arange(
        vocab, vocab + img_run, dtype=torch.int32
    )
    ids[0, text_len + img_run : text_len + img_run + aud_run] = torch.arange(
        vocab + n_image, vocab + n_image + aud_run, dtype=torch.int32
    )
    return ids


class Qwen3AVLPipelinedModel(Qwen3VLPipelinedModel):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        n_image_tokens: int,
        n_audio_tokens: int,
        grid_w: int,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        n_deepstack_layers: int = 3,
    ) -> None:
        super().__init__(
            config,
            n_image_tokens=n_image_tokens,
            grid_w=grid_w,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            n_deepstack_layers=n_deepstack_layers,
        )
        self.n_audio_tokens = n_audio_tokens

    def _gather_embeds(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        vocab = self.vocab_size
        n_image = image_embeds.shape[0]
        n_audio = audio_embeds.shape[0]
        regular = input_ids.clamp(max=vocab - 1)
        h = self.embed_tokens(regular)
        is_image = (input_ids >= vocab) & (input_ids < vocab + n_image)
        is_audio = input_ids >= vocab + n_image
        img_slots = (input_ids - vocab).clamp(min=0, max=n_image - 1)
        aud_slots = (input_ids - vocab - n_image).clamp(min=0, max=n_audio - 1)
        img_h = image_embeds.index_select(0, img_slots.reshape(-1)).reshape(h.shape)
        aud_h = audio_embeds.index_select(0, aud_slots.reshape(-1)).reshape(h.shape)
        # Always index both buffers so S=1 export cannot DCE one gather.
        h = torch.where(is_image.unsqueeze(-1), img_h, h)
        return torch.where(is_audio.unsqueeze(-1), aud_h, h)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self._gather_embeds(input_ids, image_embeds, audio_embeds)
        vocab = self.vocab_size
        for layer_idx, layer in enumerate(self.layers):
            h = layer(
                h,
                input_ids,
                position_ids,
                rope_shift_start,
                rope_shift_amount,
                vocab,
                cache,
            )
            if layer_idx < self.n_deepstack_layers:
                h = self._deepstack_add(h, input_ids, deepstack_embeds, layer_idx)
        return self.norm(h)


class Qwen3AVLPipelinedForCausalLM(Qwen3VLPipelinedForCausalLM):
    """VL decoder + static `audio_embeds`. Same S=1 ship rule as Yuna VL."""

    def __init__(
        self,
        config: Qwen3Config,
        *,
        n_image_tokens: int,
        n_audio_tokens: int,
        grid_w: int,
        rope_theta: float = 5_000_000.0,
        mrope_section: tuple[int, int, int] = DEFAULT_MROPE_SECTION,
        model_device: str = "cpu",
    ) -> None:
        self.n_audio_tokens = n_audio_tokens
        super().__init__(
            config,
            n_image_tokens=n_image_tokens,
            grid_w=grid_w,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            model_device=model_device,
        )

    @override
    def _init_model(self, config: Qwen3Config) -> None:
        self.model = Qwen3AVLPipelinedModel(
            config,
            n_image_tokens=self.n_image_tokens,
            n_audio_tokens=self.n_audio_tokens,
            grid_w=self.grid_w,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(
            input_ids,
            position_ids,
            image_embeds,
            deepstack_embeds,
            audio_embeds,
            rope_shift_start,
            rope_shift_amount,
            cache,
        )
        return self.lm_head(out)

    @classmethod
    def from_hf(
        cls: type[Self],
        huggingface_model_id: str,
        *,
        grid_h: int = 14,
        grid_w: int = 14,
        n_audio_tokens: int = 832,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int = 4096,
        rope_theta: float | None = None,
        mrope_section: tuple[int, int, int] = DEFAULT_MROPE_SECTION,
    ) -> Self:
        n_image_tokens = grid_h * grid_w
        from transformers import AutoConfig

        raw = AutoConfig.from_pretrained(huggingface_model_id, trust_remote_code=True)
        arch = (getattr(raw, "architectures", None) or [""])[0]
        is_text_ckpt = arch == "Qwen3ForCausalLM" or getattr(raw, "model_type", None) == "qwen3"

        if is_text_ckpt:
            tc = raw
            theta = rope_theta or float(getattr(tc, "rope_theta", 5_000_000))
            base = Qwen3ForCausalLM.from_hf(
                huggingface_model_id,
                target_dtype=target_dtype,
                max_context_length=max_context_length,
            )
        else:
            tc = getattr(raw, "text_config", raw)
            rope = getattr(tc, "rope_parameters", None) or {}
            theta = rope_theta or float(rope.get("rope_theta", getattr(tc, "rope_theta", 5_000_000)))
            base = Qwen3ForCausalLM.from_hf_memory_efficient(
                huggingface_model_id,
                target_dtype=target_dtype,
                max_context_length=max_context_length,
                hf_config_attr="text_config",
                hf_state_dict_prefix="model.language_model.",
            )

        model = cls(
            base.config,
            n_image_tokens=n_image_tokens,
            n_audio_tokens=n_audio_tokens,
            grid_w=grid_w,
            rope_theta=theta,
            mrope_section=mrope_section,
            model_device="cpu",
        )
        model.load_state_dict(base.state_dict(), assign=True, strict=False)
        if base.config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model.to(dtype=target_dtype).eval()

    def build_export_spec(
        self,
        dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int = TRACE_KV_CACHE_SEQ_LEN,
        trace_query: int | None = None,
    ) -> dict:
        batch = 1
        query = trace_query if trace_query is not None else QUANT_TRACE_QUERY_LEN
        vocab = self.config.vocab_size
        hidden = self.config.hidden_size
        n = self.n_image_tokens
        m = self.n_audio_tokens

        input_ids = _trace_input_ids_avl(batch, query, vocab, n, m)
        position_ids = (
            torch.arange(query + QUANT_TRACE_OFFSET, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, query + QUANT_TRACE_OFFSET)
        )
        image_embeds = torch.zeros(n, hidden, dtype=dtype)
        deepstack_embeds = torch.zeros(3 * n, hidden, dtype=dtype)
        audio_embeds = torch.zeros(m, hidden, dtype=dtype)
        rope_shift_start, rope_shift_amount = _trace_rope_shift(query, n, self.grid_w)

        saved_max = self.config.max_position_embeddings
        self.config.max_position_embeddings = trace_kv_len
        k_cache, v_cache = KVCache.create_cache_tensors(self.config, dtype=dtype)
        self.config.max_position_embeddings = saved_max

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "image_embeds": image_embeds,
            "deepstack_embeds": deepstack_embeds,
            "audio_embeds": audio_embeds,
            "rope_shift_start": rope_shift_start,
            "rope_shift_amount": rope_shift_amount,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }

        kv_dyn = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=trace_kv_len, max=max_context_length
            )
        }
        vv_dyn = {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=trace_kv_len, max=max_context_length
            )
        }
        static_mm = {
            "image_embeds": {},
            "deepstack_embeds": {},
            "audio_embeds": {},
            "rope_shift_start": {},
            "rope_shift_amount": {},
        }

        if query == 1:
            dynamic_shapes = {
                "input_ids": {},
                "position_ids": {
                    1: torch.export.Dim(
                        "seq_pos", min=query + QUANT_TRACE_OFFSET, max=max_context_length - 1
                    )
                },
                **static_mm,
                "k_cache": kv_dyn,
                "v_cache": vv_dyn,
            }
        else:
            # `min=1` so this single graph also covers decode. Left at the Dim
            # default of 2, S=1 needs its own export, and two assets mean two
            # copies of the weights resident — which does not fit on an A17.
            dynamic_shapes = {
                "input_ids": {
                    1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)
                },
                "position_ids": {
                    1: torch.export.Dim("seq_pos", min=1, max=max_context_length - 1)
                },
                **static_mm,
                "k_cache": kv_dyn,
                "v_cache": vv_dyn,
            }

        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": (
                "input_ids",
                "position_ids",
                "image_embeds",
                "deepstack_embeds",
                "audio_embeds",
                "rope_shift_start",
                "rope_shift_amount",
            ),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }
