# Copyright 2026 Apple Inc.
#
# Qwen3-VL text decoder with static multimodal inputs for the pipelined engine.
#
# Host contract (mirrors coreai-model-zoo export_qwen3_vl_pipelined.py):
#   * rewrite <|image_pad|> ids to vocab + slot
#   * image_embeds [N, h], deepstack_embeds [3N, h] static buffers
#   * rope_shift_start [1], rope_shift_amount [1] for post-image text positions
#   * with zero embeds and shift_start = 1<<30 the graph is a plain Qwen3 text LM

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from typing_extensions import Self, override

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM, USE_FUSED_KV
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

PIPELINED_STATE_NAMES = (KEY_CACHE_NAME, VALUE_CACHE_NAME)
DEFAULT_MROPE_SECTION = (24, 20, 20)


def _trace_input_ids(batch: int, query: int, vocab: int, n_image: int) -> torch.Tensor:
    """Include vocab+slot image placeholders so export captures gather/deepstack/M-RoPE."""
    ids = torch.randint(1, vocab, (batch, query), dtype=torch.int32)
    if query == 1:
        ids[0, 0] = vocab
        return ids
    img_run = min(n_image, max(4, query // 3))
    text_len = max(1, query - img_run)
    img_run = min(img_run, query - text_len)
    ids[0, text_len : text_len + img_run] = torch.arange(
        vocab, vocab + img_run, dtype=torch.int32
    )
    return ids


def _trace_rope_shift(query: int, n_image: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    if query == 1:
        return torch.tensor([1 << 30], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    img_run = min(n_image, max(4, query // 3))
    text_len = max(1, query - img_run)
    img_run = min(img_run, query - text_len)
    return (
        torch.tensor([text_len + img_run], dtype=torch.int32),
        torch.tensor([max(n_image - grid_w, 0)], dtype=torch.int32),
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE with cos/sin shaped [batch, seq, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class InterleavedMRoPE(nn.Module):
    """Interleaved M-RoPE cos/sin from 1-D engine positions + image extension ids."""

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        *,
        grid_w: int,
        n_image_tokens: int,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        self.n_image_tokens = n_image_tokens
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Baked 2-D slot layout (export avoids aten.remainder.Scalar from slot % grid_w).
        slots = torch.arange(n_image_tokens, dtype=torch.int32)
        slot_h = slots // grid_w
        slot_w = slots - slot_h * grid_w
        self.register_buffer("slot_h_lut", slot_h, persistent=False)
        self.register_buffer("slot_w_lut", slot_w, persistent=False)

        n_pairs = head_dim // 2
        mask_t = torch.zeros(n_pairs, dtype=torch.float32)
        mask_h = torch.zeros(n_pairs, dtype=torch.float32)
        mask_w = torch.zeros(n_pairs, dtype=torch.float32)
        for j in range(n_pairs):
            h_slice = slice(1, mrope_section[1] * 3, 3)
            w_slice = slice(2, mrope_section[2] * 3, 3)
            if h_slice.start <= j < h_slice.stop and (j - h_slice.start) % 3 == 0:
                mask_h[j] = 1.0
            elif w_slice.start <= j < w_slice.stop and (j - w_slice.start) % 3 == 0:
                mask_w[j] = 1.0
            else:
                mask_t[j] = 1.0
        self.register_buffer("mask_t", mask_t, persistent=False)
        self.register_buffer("mask_h", mask_h, persistent=False)
        self.register_buffer("mask_w", mask_w, persistent=False)

    def _freqs(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [batch, seq] -> [batch, seq, head_dim // 2]
        pos_f = pos.to(torch.float32).unsqueeze(-1)
        return pos_f * self.inv_freq.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        vocab: int,
        *,
        out_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids spans cached prefix + query; input_ids is query-only (pipelined contract).
        query_len = input_ids.shape[-1]
        seq_len = position_ids.shape[-1]
        offset = seq_len - query_len
        pos = position_ids.narrow(-1, offset, query_len)

        # Image slots are vocab .. vocab+N-1. Ids above that (AVL audio) stay 1-D text RoPE.
        is_image = (input_ids >= vocab) & (input_ids < vocab + self.n_image_tokens)
        slot = (input_ids - vocab).clamp(min=0, max=self.n_image_tokens - 1)
        s0 = pos - slot

        shift = rope_shift_amount.to(pos.dtype) * (
            pos >= rope_shift_start.to(pos.dtype)
        ).to(pos.dtype)
        text_pos = pos - shift

        flat = slot.reshape(-1)
        h_off = self.slot_h_lut.index_select(0, flat).reshape(slot.shape)
        w_off = self.slot_w_lut.index_select(0, flat).reshape(slot.shape)

        t_pos = torch.where(is_image, s0, text_pos)
        h_pos = torch.where(is_image, s0 + h_off, text_pos)
        w_pos = torch.where(is_image, s0 + w_off, text_pos)

        freqs_t = self._freqs(t_pos)
        freqs_h = self._freqs(h_pos)
        freqs_w = self._freqs(w_pos)
        freqs = (
            freqs_t * self.mask_t.view(1, 1, -1)
            + freqs_h * self.mask_h.view(1, 1, -1)
            + freqs_w * self.mask_w.view(1, 1, -1)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(out_dtype), emb.sin().to(out_dtype)


class VLAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        grid_w: int,
        n_image_tokens: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        if USE_FUSED_KV:
            self.qk_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads)
        else:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.mrope = InterleavedMRoPE(
            head_dim, rope_theta, mrope_section, grid_w=grid_w, n_image_tokens=n_image_tokens
        )
        self.sdpa = SDPA(is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        vocab: int,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if USE_FUSED_KV:
            query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
            query_key = self.qk_norm(query_key)
            query = query_key.narrow(1, 0, n_heads)
            key = query_key.narrow(1, n_heads, n_kv_heads)
        else:
            query = self.q_norm(qkv.narrow(1, 0, n_heads))
            key = self.k_norm(qkv.narrow(1, n_heads, n_kv_heads))

        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        cos, sin = self.mrope(
            input_ids, position_ids, rope_shift_start, rope_shift_amount, vocab, out_dtype=x.dtype
        )
        query, key = _apply_rotary(query, key, cos, sin)

        if cache is not None:
            seq_len = position_ids.shape[-1]
            offset = seq_len - query_len
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * self.head_dim)
        )
        return self.o_proj(output)


class VLTransformerBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        grid_w: int,
        n_image_tokens: int,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = VLAttention(
            config,
            layer_idx,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            grid_w=grid_w,
            n_image_tokens=n_image_tokens,
        )
        self.mlp = MLP(hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        vocab: int,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            input_ids,
            position_ids,
            rope_shift_start,
            rope_shift_amount,
            vocab,
            cache,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3VLPipelinedModel(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        n_image_tokens: int,
        grid_w: int,
        rope_theta: float,
        mrope_section: tuple[int, int, int],
        n_deepstack_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_image_tokens = n_image_tokens
        self.n_deepstack_layers = n_deepstack_layers
        hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                VLTransformerBlock(
                    config,
                    layer_idx,
                    rope_theta=rope_theta,
                    mrope_section=mrope_section,
                    grid_w=grid_w,
                    n_image_tokens=n_image_tokens,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def _gather_embeds(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        vocab = self.vocab_size
        regular = input_ids.clamp(max=vocab - 1)
        h = self.embed_tokens(regular)
        n_image = image_embeds.shape[0]
        is_image = (input_ids >= vocab) & (input_ids < vocab + n_image)
        slots = (input_ids - vocab).clamp(min=0, max=n_image - 1)
        img_h = image_embeds.index_select(0, slots.reshape(-1)).reshape(h.shape)
        return torch.where(is_image.unsqueeze(-1), img_h, h)

    def _deepstack_add(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        vocab = self.vocab_size
        n = self.n_image_tokens
        is_image = (input_ids >= vocab) & (input_ids < vocab + n)
        slot = (input_ids - vocab).clamp(min=0, max=n - 1)
        ds_idx = layer_idx * n + slot
        ds = deepstack_embeds.index_select(0, ds_idx.reshape(-1)).reshape(hidden_states.shape)
        return hidden_states + ds * is_image.unsqueeze(-1).to(hidden_states.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self._gather_embeds(input_ids, image_embeds)
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


class Qwen3VLPipelinedForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = Qwen3ForCausalLM._HF_MODEL_CLASS

    def __init__(
        self,
        config: Qwen3Config,
        *,
        n_image_tokens: int,
        grid_w: int,
        rope_theta: float = 5_000_000.0,
        mrope_section: tuple[int, int, int] = DEFAULT_MROPE_SECTION,
        model_device: str = "cpu",
    ) -> None:
        self.n_image_tokens = n_image_tokens
        self.grid_w = grid_w
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        super().__init__(config, model_device)

    @override
    def _init_model(self, config: Qwen3Config) -> None:
        self.model = Qwen3VLPipelinedModel(
            config,
            n_image_tokens=self.n_image_tokens,
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
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int = 4096,
        rope_theta: float | None = None,
        mrope_section: tuple[int, int, int] = DEFAULT_MROPE_SECTION,
    ) -> Self:
        n_image_tokens = grid_h * grid_w
        from transformers import AutoConfig

        raw = AutoConfig.from_pretrained(huggingface_model_id)
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

        input_ids = _trace_input_ids(batch, query, vocab, n)
        position_ids = (
            torch.arange(query + QUANT_TRACE_OFFSET, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch, query + QUANT_TRACE_OFFSET)
        )
        image_embeds = torch.zeros(n, hidden, dtype=dtype)
        deepstack_embeds = torch.zeros(3 * n, hidden, dtype=dtype)
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
            "rope_shift_start": rope_shift_start,
            "rope_shift_amount": rope_shift_amount,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }

        if query == 1:
            dynamic_shapes = {
                "input_ids": {},
                "position_ids": {
                    1: torch.export.Dim(
                        "seq_pos", min=query + QUANT_TRACE_OFFSET, max=max_context_length - 1
                    )
                },
                "image_embeds": {},
                "deepstack_embeds": {},
                "rope_shift_start": {},
                "rope_shift_amount": {},
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
                    1: torch.export.Dim("seq_pos", min=query, max=max_context_length - 1)
                },
                "image_embeds": {},
                "deepstack_embeds": {},
                "rope_shift_start": {},
                "rope_shift_amount": {},
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
            "input_names": (
                "input_ids",
                "position_ids",
                "image_embeds",
                "deepstack_embeds",
                "rope_shift_start",
                "rope_shift_amount",
            ),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        Qwen3ForCausalLM._mutate_state_dict(self, state_dict)


def _exportable_vision_attention_forward(attn_module, hidden_states, position_embeddings):
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attn_module.qkv(hidden_states)
        .reshape(seq_length, 3, attn_module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attn_output, _ = eager_attention_forward(
        attn_module,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=attn_module.scaling,
        dropout=0.0,
        is_causal=False,
    )
    return attn_module.proj(attn_output.reshape(seq_length, -1).contiguous())


def _patch_vision_for_export(visual_model, grid_thw: torch.Tensor) -> None:
    """Bake fixed-grid pos embed + 2D rotary for export (coreai-model-zoo recipe)."""
    with torch.no_grad():
        pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
        seq_len = pos_embeds.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        cu_seqlens = F.pad(
            torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            ),
            (1, 0),
            value=0,
        )

    visual_model.register_buffer("_export_pos_embeds", pos_embeds, persistent=False)
    visual_model.register_buffer("_export_cos", cos, persistent=False)
    visual_model.register_buffer("_export_sin", sin, persistent=False)
    visual_model.register_buffer("_export_cu_seqlens", cu_seqlens, persistent=False)

    position_embeddings = (visual_model._export_cos, visual_model._export_sin)
    for block in visual_model.blocks:

        def make_attn_forward(pos_emb):
            def forward(self, hidden_states, *_args, **_kwargs):
                return _exportable_vision_attention_forward(self, hidden_states, pos_emb)

            return forward

        block.attn.forward = types.MethodType(make_attn_forward(position_embeddings), block.attn)

    def exportable_visual_forward(self, hidden_states, grid_thw=None, **kwargs):
        hidden_states = self.patch_embed(hidden_states) + self._export_pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        pos_emb = (self._export_cos, self._export_sin)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self._export_cu_seqlens,
                position_embeddings=pos_emb,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[
                        self.deepstack_visual_indexes.index(layer_num)
                    ](hidden_states)
                )
        return self.merger(hidden_states), deepstack_feature_lists

    visual_model.forward = types.MethodType(exportable_visual_forward, visual_model)


class Qwen3VLVisionEncoder(nn.Module):
    """Fixed-grid Qwen3-VL vision tower: patches → (image_embeds, deepstack_embeds)."""

    def __init__(
        self,
        visual: nn.Module,
        *,
        grid_h: int,
        grid_w: int,
        patch_h: int,
        patch_w: int,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.patch_h = patch_h
        self.patch_w = patch_w

    @property
    def vcfg(self):
        return self.visual.config

    @property
    def n_patches(self) -> int:
        return self.patch_h * self.patch_w

    @property
    def n_image_tokens(self) -> int:
        return self.grid_h * self.grid_w

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeds, deepstack_list = self.visual(patches)
        deepstack_embeds = torch.cat(deepstack_list, dim=0)
        return image_embeds, deepstack_embeds

    @classmethod
    def from_hf(
        cls: type[Self],
        huggingface_model_id: str,
        *,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 14,
        grid_w: int = 14,
    ) -> Self:
        from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration

        patch_h = grid_h * 2
        patch_w = grid_w * 2
        cfg = AutoConfig.from_pretrained(
            huggingface_model_id, trust_remote_code=True
        )
        model_type = getattr(cfg, "model_type", "") or ""
        # Forced Qwen3VLForConditionalGeneration builds the stock 4B/8B graph even
        # when the folder is YunaAVL (2B / ViT-1024) and then dies on size mismatch.
        if model_type in ("qwen3_vl", "qwen3_vl_moe"):
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                huggingface_model_id,
                torch_dtype=target_dtype,
            )
        else:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    huggingface_model_id,
                    torch_dtype=target_dtype,
                    trust_remote_code=True,
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    huggingface_model_id,
                    torch_dtype=target_dtype,
                    trust_remote_code=True,
                )
        visual = getattr(model, "visual", None) or getattr(
            getattr(model, "model", None), "visual", None
        )
        if visual is None:
            raise AttributeError(
                f"{type(model).__name__} has no .visual / .model.visual"
            )
        grid_thw = torch.tensor([[1, patch_h, patch_w]], dtype=torch.long)
        _patch_vision_for_export(visual, grid_thw)
        enc = cls(visual, grid_h=grid_h, grid_w=grid_w, patch_h=patch_h, patch_w=patch_w)
        del model
        return enc.to(dtype=target_dtype).eval()
