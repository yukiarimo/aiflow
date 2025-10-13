import mlx.core as mx
import mlx.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x): return self.weight * x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb_vision(tensor, freqs):
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos, sin = mx.cos(emb).astype(tensor.dtype), mx.sin(emb).astype(tensor.dtype)
    cos, sin = cos.reshape(1, 1, *cos.shape), sin.reshape(1, 1, *sin.shape) # Reshape for broadcasting
    return (tensor * cos) + (rotate_half(tensor) * sin)

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim, self.theta = dim, theta
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, seqlen):
        seq = mx.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = mx.outer(seq, self.inv_freq)
        return freqs

class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_patch_size = config.spatial_patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        kernel_size = (config.temporal_patch_size, config.spatial_patch_size, config.spatial_patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def __call__(self, x):
        x = x.reshape(-1, self.in_channels, self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)
        x = x.transpose(0, 2, 3, 4, 1) # (N, H, W, D, C) for MLX Conv3D
        x = self.proj(x)
        return x.reshape(-1, self.embed_dim)

class PatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        context_dim, dim = config.embed_dim, config.hidden_size
        spatial_merge_size = config.spatial_merge_size
        hidden_size = context_dim * (spatial_merge_size**2)

        if hasattr(config, 'intermediate_size') and getattr(config, 'intermediate_size', None) is not None: self.ln_q = RMSNorm(context_dim, eps=1e-6) # Add conditional norm layer for compatibility
        else: self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_fc1, self.mlp_act, self.mlp_fc2 = nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, dim)

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.mlp_fc1.weight.shape[1])
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        return x

class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embed_dim
        self.num_heads, self.head_dim = config.num_heads, dim // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, cu_seqlens, rotary_pos_emb):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).transpose(0, 2, 3, 1, 4)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # Unbind
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        mask = mx.full((L, L), -mx.inf, dtype=q.dtype)

        for i in range(1, len(cu_seqlens)):
            start, end = cu_seqlens[i - 1].item(), cu_seqlens[i].item()
            mask[start:end, start:end] = 0

        mask = mx.expand_dims(mask, (0, 1))
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(output)

class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, int(config.embed_dim * config.mlp_ratio))
        self.fc2 = nn.Linear(int(config.embed_dim * config.mlp_ratio), config.embed_dim)
        self.act_fn = nn.GELU(approx="fast")

    def __call__(self, x): return self.fc2(self.act_fn(self.fc1(x)))

class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = VisionMLP(config)

    def __call__(self, x, cu_seqlens, rotary_pos_emb):
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x

class YunaVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        self.head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(self.head_dim // 2)

    def rot_pos_emb(self, d_image):
        pos_ids_list = []
        sms = self.config.spatial_merge_size

        for t, h, w in d_image.tolist():
            hpos_ids = mx.tile(mx.arange(h).reshape(-1, 1), (1, w))
            hpos_ids = hpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            wpos_ids = mx.tile(mx.arange(w).reshape(1, -1), (h, 1))
            wpos_ids = wpos_ids.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            stacked_pos = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids_list.append(mx.tile(stacked_pos, (t, 1)))

        pos_ids = mx.concatenate(pos_ids_list, axis=0)
        max_grid_size = mx.max(d_image[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size.item())
        return rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)

    def __call__(self, pixels, d_image):
        x = self.patch_embed(pixels)
        rotary_pos_emb = self.rot_pos_emb(d_image)
        seq_lens = mx.repeat((d_image[:, 1] * d_image[:, 2]), d_image[:, 0])
        cu_seqlens = mx.cumsum(seq_lens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0))
        x = mx.expand_dims(x, 0)

        for blk in self.blocks: x = blk(x, cu_seqlens, rotary_pos_emb)
        x = x.squeeze(0)
        return self.merger(x)