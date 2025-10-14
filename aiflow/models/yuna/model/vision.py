import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class VisionConfig:
    n_embed: int
    n_layer: int
    n_heads: int
    output_n_embed: int
    in_channels: int
    spatial_merge_size: int
    spatial_patch_size: int
    temporal_patch_size: int
    n_mlp: int = None
    hidden_act: str = "quick_gelu"

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class PatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.spatial_patch_size = config.spatial_patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.n_embed
        kernel_size = [config.temporal_patch_size, config.spatial_patch_size, config.spatial_patch_size]
        self.proj = nn.Conv3d(config.in_channels, config.n_embed, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        target_dtype = self.proj.weight.dtype
        conv_input = hidden_states.view(B * L, self.in_channels, self.temporal_patch_size, self.spatial_patch_size, self.spatial_patch_size)
        x = self.proj(conv_input.to(target_dtype))
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        x = x.view(B, L, self.embed_dim)
        return x

class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, config = None) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.n_embed = context_dim * (spatial_merge_size**2)
        if config and hasattr(config, 'n_mlp') and config.n_mlp is not None: self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else: self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(self.n_embed, self.n_embed), nn.GELU(), nn.Linear(self.n_embed, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x) # Normalize the input features first. Input x shape: (B, L, C), e.g., (1, 1840, 1280). Output shape is still (1, 1840, 1280)
        B, L, C = x.shape # Reshape the NORMALIZED features for merging.
        merge_factor = self.spatial_merge_size ** 2
        x = x.view(B, L // merge_factor, merge_factor * C) # Shape becomes (1, 460, 5120)
        x = self.mlp(x)
        return x

class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        emb = torch.cat([freqs, freqs], dim=-1).to(tensor.device)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        output = (tensor * cos) + (VisionAttention._rotate_half(tensor) * sin)
        return output.to(orig_dtype)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if rotary_pos_emb is not None:
             q = self._apply_rotary_pos_emb_vision(q, rotary_pos_emb)
             k = self._apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        is_causal = (cu_seqlens is None)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.proj(attn_output)

class VisionMlp(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        dim = config.n_embed

        if config.n_mlp is not None:
            self.gate_proj = nn.Linear(dim, config.n_mlp, bias=True)
            self.up_proj = nn.Linear(dim, config.n_mlp, bias=True)
            self.down_proj = nn.Linear(config.n_mlp, dim, bias=True)
            self.act_fn = self._get_activation_fn(config.hidden_act)
            self.is_gated = True

        else:
            self.fc1 = nn.Linear(dim, 4 * dim)
            self.fc2 = nn.Linear(4 * dim, dim)
            self.is_gated = False

    def forward(self, x) -> torch.Tensor:
        target_dtype = x.dtype

        if self.is_gated:
            gate_out = self.act_fn(self.gate_proj(x.to(target_dtype)))
            up_out = self.up_proj(x.to(target_dtype))
            return self.down_proj(gate_out * up_out).to(target_dtype)
        else: return self.fc2(self._quick_gelu(self.fc1(x.to(target_dtype)))).to(target_dtype)

    @staticmethod
    def _quick_gelu(x): return x * torch.sigmoid(1.702 * x)

    def _get_activation_fn(self, act_name: str):
        if act_name == "silu": return F.silu
        elif act_name == "quick_gelu": return self._quick_gelu
        else: raise ValueError(f"Unsupported activation: {act_name}")

class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class YunaVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.n_mlp is not None:
            self.norm1 = RMSNorm(config.n_embed, eps=1e-6)
            self.norm2 = RMSNorm(config.n_embed, eps=1e-6)

        else:
            self.norm1 = nn.LayerNorm(config.n_embed, eps=1e-6)
            self.norm2 = nn.LayerNorm(config.n_embed, eps=1e-6)

        self.attn = VisionAttention(config.n_embed, config.n_heads)
        self.mlp = VisionMlp(config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class YunaVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config=config)
        self.blocks = nn.ModuleList([YunaVisionBlock(config) for _ in range(config.n_layer)])
        self.merger = PatchMerger(dim=config.output_n_embed, context_dim=config.n_embed, spatial_merge_size=config.spatial_merge_size, config=config)
        head_dim = config.n_embed // config.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        pos_ids_list = []
        max_h, max_w = d_image[:, 1].max(), d_image[:, 2].max()
        rotary_pos_emb_h = self.rotary_pos_emb(max_h)
        rotary_pos_emb_w = self.rotary_pos_emb(max_w)

        for t, h, w in d_image:
            h_grid = torch.arange(h, device=d_image.device)
            w_grid = torch.arange(w, device=d_image.device)
            pos_h = rotary_pos_emb_h[h_grid].unsqueeze(1).expand(-1, w, -1)
            pos_w = rotary_pos_emb_w[w_grid].unsqueeze(0).expand(h, -1, -1)
            pos_ids = torch.cat([pos_h, pos_w], dim=-1).flatten(0, 1)
            pos_ids_list.append(pos_ids.repeat(t, 1))

        return torch.cat(pos_ids_list, dim=0)

    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixels)
        cu_seqlens = None

        if pixels.shape[0] > 1:
            seq_lens = (d_image[:, 0] * d_image[:, 1] * d_image[:, 2]).to(torch.int32)
            cu_seqlens = F.pad(seq_lens.cumsum(0), (1, 0))
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        rotary_pos_emb = self.rot_pos_emb(d_image)
        for blk in self.blocks: hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        if cu_seqlens is not None: hidden_states = hidden_states.view(pixels.shape[0], -1, hidden_states.shape[-1])
        return self.merger(hidden_states)