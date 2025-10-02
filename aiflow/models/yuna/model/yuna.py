import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .vision import YunaVisionEncoder, VisionConfig

@dataclass
class YunaConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    vision_config: Optional[VisionConfig] = None
    head_dim: Optional[int] = None  # Explicit head dimension

class YunaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use explicit head_dim if provided, otherwise calculate
        d = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.inv_freq = 1.0 / (t ** (r / d)).float()

    def forward(self, x, position_ids):
        # Check the dimensionality of position_ids to decide shape
        # shape is typically B x T (2D) for text, or B x 3 x T (3D) for multimodal
        if position_ids.dim() == 3:
            inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0).to(x.device)
        else:
            inv_freq = self.inv_freq.to(x.device)

        position_ids = position_ids.unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

class YunaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head

        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=True)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos, sin, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        is_causal = past_kv is None and T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y, present_kv

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = YunaAttention._process_rotary_component(cos)
            sin = YunaAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (YunaAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (YunaAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class YunaRMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

class YunaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class YunaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = YunaRMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = YunaAttention(config)
        self.post_attention_layernorm = YunaRMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = YunaMLP(config)

    def forward(self, x, cos, sin, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        attn_out, present_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv

class YunaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = YunaRotaryEmbedding(config)
        self.layers = nn.ModuleList(YunaBlock(config) for _ in range(config.n_layer))
        self.norm = YunaRMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, use_cache: bool = False, past_key_values: Optional[List[Tuple]] = None):
        cos, sin = self.rotary_emb(x, position_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, cos, sin, past_kv=past_kv)
            if use_cache:
                next_cache.append(present_kv)

        x = self.norm(x)
        return x, (next_cache if use_cache else None)

class Yuna(nn.Module):
    def __init__(self, config: YunaConfig):
        super().__init__()
        self.config = config
        self.visual = YunaVisionEncoder(config.vision_config)
        self.model = YunaModel(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.vision_start_token_id = 151652
        self.image_pad_token_id = 151655
        self.video_pad_token_id = -1  # placeholder

    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        d_image: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        B, T = input_ids.shape
        device = input_ids.device
        all_pos_ids = torch.zeros(B, 3, T, dtype=torch.long, device=device)

        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            seq_idx = 0
            image_idx = 0
            pos_chunks = []
            position_id = 0

            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if d_image is not None and token_id == self.image_pad_token_id:
                    t, h, w = d_image[image_idx]
                    h = h // self.config.vision_config.spatial_merge_size
                    w = w // self.config.vision_config.spatial_merge_size

                    t_idx = torch.arange(t).view(t, 1).expand(t, h * w).flatten()
                    h_idx = torch.arange(h).view(1, h, 1).expand(t, h, w).flatten()
                    w_idx = torch.arange(w).view(1, 1, w).expand(t, h, w).flatten()

                    pos_vision = torch.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)
                    position_id = pos_vision.max().item() + 1
                    seq_idx += t * h * w
                    image_idx += 1
                else:
                    pos_text = torch.tensor([position_id])
                    pos_text = pos_text.unsqueeze(0).expand(3, 1)  # shape (3,1)
                    pos_chunks.append(pos_text)

                    position_id += 1
                    seq_idx += 1

            pos_ids_example = torch.cat(pos_chunks, dim=1).to(device)
            all_pos_ids[batch_idx] = pos_ids_example

        return all_pos_ids.transpose(0, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        input_embeds = self.model.embed_tokens(input_ids)

        if past_key_values is None and pixels is not None:
            image_embeds = self.visual(pixels=pixels, d_image=d_image)
            image_positions = (input_ids == self.image_pad_token_id).squeeze(0)
            input_embeds[0, image_positions] = image_embeds.to(input_embeds.dtype)

        x, next_cache = self.model(
            x=input_embeds, position_ids=position_ids, use_cache=use_cache, past_key_values=past_key_values
        )

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)

        return logits, next_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor],
        d_image: Optional[torch.Tensor],
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [151645, 151644, 151643]

        # Prefill phase
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)
        logits, past_key_values = self.forward(
            input_ids=input_ids,
            pixels=pixels,
            d_image=d_image,
            use_cache=True,
            past_key_values=None,
            position_ids=position_ids
        )

        # Get the very first next token
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # The position for the next token is one after the max of the prompt positions
        current_pos = position_ids.max().item() + 1

        all_generated_ids = [input_ids, next_token]

        if stream:
            yield next_token.item()

        # Decoding phase
        for _ in range(max_new_tokens - 1):
            if next_token.item() in stop_tokens:
                break

            # Create position_ids for the single new token
            pos_tensor = torch.tensor([current_pos], device=input_ids.device, dtype=torch.long)
            # Shape for (B=1, T=1) needs to be (3, 1, 1) to match RoPE expectations
            position_ids = pos_tensor.view(1, 1, 1).expand(1, 3, 1).transpose(0, 1)

            logits, past_key_values = self.forward(
                input_ids=next_token,
                pixels=None,
                d_image=None,
                use_cache=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
 
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            all_generated_ids.append(next_token)
            current_pos += 1

            if stream:
                yield next_token.item()

        if not stream:
            return torch.cat(all_generated_ids, dim=1)

    @classmethod
    def get_config_class(cls):
        return YunaConfig

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model
        return load_pretrained_model(cls, repo_id, device_map=device_map)