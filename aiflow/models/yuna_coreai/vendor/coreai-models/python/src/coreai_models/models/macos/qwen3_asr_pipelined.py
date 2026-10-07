# Copyright 2026 Apple Inc.
#
# Qwen3-ASR text decoder with a static `audio_embeds` input for the pipelined engine.
# Host contract: rewrite <|audio_pad|> ids to vocab + slot; the graph gathers
# audio rows in-graph for extension ids.

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
from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM, Qwen3Model, TransformerBlock
from coreai_models.primitives.macos.cache import KVCache

PIPELINED_STATE_NAMES = (KEY_CACHE_NAME, VALUE_CACHE_NAME)


class Qwen3ASRModel(Qwen3Model):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        audio_embeds: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        vocab = self.embed_tokens.num_embeddings
        regular = input_ids.clamp(max=vocab - 1)
        h = self.embed_tokens(regular)
        is_audio = input_ids >= vocab
        slots = (input_ids - vocab).clamp(min=0, max=audio_embeds.shape[0] - 1)
        audio_h = audio_embeds.index_select(0, slots.reshape(-1)).reshape(h.shape)
        h = torch.where(is_audio.unsqueeze(-1), audio_h, h)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Qwen3ASRPipelinedForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = Qwen3ForCausalLM._HF_MODEL_CLASS

    def __init__(
        self,
        config: Qwen3Config,
        n_audio_tokens: int,
        model_device: str = "cpu",
    ) -> None:
        self.n_audio_tokens = n_audio_tokens
        super().__init__(config, model_device)

    @override
    def _init_model(self, config: Qwen3Config) -> None:
        self.model = Qwen3ASRModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        audio_embeds: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, audio_embeds, cache)
        return self.lm_head(out)

    @classmethod
    def from_hf(
        cls: type[Self],
        huggingface_model_id: str,
        n_audio_tokens: int,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int = 4096,
    ) -> Self:
        base = Qwen3ForCausalLM.from_hf(
            huggingface_model_id,
            target_dtype=target_dtype,
            max_context_length=max_context_length,
        )
        model = cls(base.config, n_audio_tokens=n_audio_tokens, model_device="cpu")
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

        input_ids = torch.randint(1, vocab, (batch, query), dtype=torch.int32)
        position_ids = (
            torch.arange(query + QUANT_TRACE_OFFSET, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, query + QUANT_TRACE_OFFSET)
        )
        audio_embeds = torch.zeros(self.n_audio_tokens, hidden, dtype=dtype)

        saved_max = self.config.max_position_embeddings
        self.config.max_position_embeddings = trace_kv_len
        k_cache, v_cache = KVCache.create_cache_tensors(self.config, dtype=dtype)
        self.config.max_position_embeddings = saved_max

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "audio_embeds": audio_embeds,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }

        if query == 1:
            # S=1 pipelined bundle: static [1,1] query, growing position_ids + KV.
            dynamic_shapes = {
                "input_ids": {},
                "position_ids": {
                    1: torch.export.Dim(
                        "seq_pos", min=query + QUANT_TRACE_OFFSET, max=max_context_length - 1
                    )
                },
                "audio_embeds": {},
                "k_cache": {
                    KVCache.seq_len_dim(): torch.export.Dim(
                        "k_seq_len", min=trace_kv_len, max=max_context_length
                    )
                },
                "v_cache": {
                    KVCache.seq_len_dim(): torch.export.Dim(
                        "v_seq_len", min=trace_kv_len, max=max_context_length
                    )
                },
            }
        else:
            dynamic_shapes = {
                "input_ids": {1: torch.export.Dim("seq_ids", max=max_context_length - 2)},
                "position_ids": {
                    1: torch.export.Dim(
                        "seq_pos", min=query, max=max_context_length - 1
                    )
                },
                "audio_embeds": {},
                "k_cache": {
                    KVCache.seq_len_dim(): torch.export.Dim(
                        "k_seq_len", min=trace_kv_len, max=max_context_length
                    )
                },
                "v_cache": {
                    KVCache.seq_len_dim(): torch.export.Dim(
                        "v_seq_len", min=trace_kv_len, max=max_context_length
                    )
                },
            }

        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "audio_embeds"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        Qwen3ForCausalLM._mutate_state_dict(self, state_dict)
