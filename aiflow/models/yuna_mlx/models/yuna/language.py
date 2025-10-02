from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from .config import TextConfig

class YunaRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        self.inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )

    def __call__(self, x, position_ids):
        freqs = mx.expand_dims(position_ids, axis=-1) * self.inv_freq.astype(x.dtype)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = emb.cos().astype(x.dtype)
        sin = emb.sin().astype(x.dtype)
        return cos, sin

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.ndim == 3:
         cos = mx.expand_dims(cos, 1)
         sin = mx.expand_dims(sin, 1)
    elif cos.ndim == 4 and cos.shape[1] == 3:
        mrope_sections = [16, 24, 24]

        def process_component(c):
            splits = mx.split(c, [sum(mrope_sections[:i+1]) for i in range(len(mrope_sections)-1)], axis=-1)
            processed_splits = [splits[i][:, i % 3, :, :] for i in range(len(splits))]
            concatenated_splits = mx.concatenate(processed_splits, axis=-1)
            return mx.expand_dims(concatenated_splits, axis=1)

        cos = process_component(cos)
        sin = process_component(sin)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class YunaAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        n_embed = config.hidden_size
        head_dim = n_embed // self.n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(n_embed, n_embed, bias=True)
        self.k_proj = nn.Linear(n_embed, self.n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(n_embed, self.n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_embed, n_embed, bias=False)

    def __call__(self, x, cos, sin, mask, cache):
        B, T, C = x.shape
        head_dim = C // self.n_heads

        q = self.q_proj(x).reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        # FIX: The cache object is stateful and updated in-place. Do not return its contents.
        return self.o_proj(y)

class YunaMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class YunaBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = YunaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = YunaMLP(config)

    def __call__(self, x, cos, sin, mask, cache):
        # FIX: self_attn no longer returns the (k,v) tuple.
        attn_out = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class YunaModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = YunaRotaryEmbedding(config)
        self.layers = [YunaBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, position_ids, cache=None, mask=None):
        cos, sin = self.rotary_emb(x, position_ids)

        if mask is None and x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            # FIX: The layer updates the cache object internally and just returns the hidden state.
            x = layer(x, cos, sin, mask, cache[i])

        # FIX: Return the original list of cache objects, which have now been updated.
        return self.norm(x), cache