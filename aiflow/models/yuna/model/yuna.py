import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Any
from .vision import YunaVisionEncoder
from .audio import YunaAudioEncoder, AudioProjector
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
import json

@dataclass
class YunaConfig:
    n_embed: int = 1536
    n_heads: int = 12
    n_kv_heads: int = 2
    n_layer: int = 28
    n_mlp: int = 8960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    vision_config: Optional[Any] = None
    audio_config: Optional[Any] = None
    head_dim: Optional[int] = None

class YunaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.head_dim if config.head_dim is not None else (config.n_embed // config.n_heads)
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.inv_freq = 1.0 / (t ** (r / d)).float()

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0).to(x.device) if position_ids.dim() == 3 else self.inv_freq.to(x.device)
        position_ids = position_ids.unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        return cos, sin

class YunaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.n_embed = config.n_heads, config.n_kv_heads, config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head
        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=True)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, C = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None: past_k, past_v = past_kv; k = torch.cat((past_k, k), dim=2); v = torch.cat((past_v, v), dim=2)
        present_kv = (k, v)

        if self.n_kv_heads < self.n_heads: k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1); v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None and T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), present_kv

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4: cos = YunaAttention._process_rotary_component(cos); sin = YunaAttention._process_rotary_component(sin)
        else: cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (YunaAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (YunaAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3, :, :, :] for i, m in enumerate(sections)]
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
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

    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class YunaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = YunaRMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.self_attn = YunaAttention(config)
        self.post_attention_layernorm = YunaRMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.mlp = YunaMLP(config)

    def forward(self, x, cos, sin, past_kv=None):
        attn_out, present_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv

class YunaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = YunaRotaryEmbedding(config)
        self.layers = nn.ModuleList([YunaBlock(config) for _ in range(config.n_layer)])
        self.norm = YunaRMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self): self.gradient_checkpointing = True
    def gradient_checkpointing_disable(self): self.gradient_checkpointing = False

    def forward(self, x, position_ids, use_cache=False, past_key_values=None):
        cos, sin = self.rotary_emb(x, position_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training and not use_cache and past_kv is None:
                def custom_forward(tensor): return layer(tensor, cos, sin, past_kv=None)[0]
                x = checkpoint(custom_forward, x, use_reentrant=False)
                present_kv = None
            else: x, present_kv = layer(x, cos, sin, past_kv=past_kv)
            if use_cache: next_cache.append(present_kv)

        x = self.norm(x)
        return x, (next_cache if use_cache else None)

class Yuna(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = YunaVisionEncoder(config.vision_config) if config.vision_config else None

        if config.audio_config:
            self.audio_encoder = YunaAudioEncoder()
            self.audio_projector = AudioProjector(audio_tower_dim=self.audio_encoder.config.output_dim, llm_embed_dim=config.n_embed)

        else:
            self.audio_encoder = None
            self.audio_projector = None

        self.model = YunaModel(config)
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.image_pad_token_id = -1
        self.audio_chunk_token_id = -1

    def get_audio_tower(self): return self.audio_encoder
    def get_audio_projector(self): return self.audio_projector
    def get_vision_projector(self): return self.visual.merger if self.visual else None

    def _get_position_ids(self, input_ids, d_image=None, cache_offset=0):
        B, T = input_ids.shape
        device = input_ids.device
        all_pos_ids = torch.zeros(B, 3, T, dtype=torch.long, device=device)

        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            seq_idx, image_idx, pos_chunks, position_id = 0, 0, [], cache_offset
            while seq_idx < T:
                token_id = seq[seq_idx].item()

                if token_id == self.image_pad_token_id and d_image is not None and image_idx < d_image.shape[0]:
                    t, h, w = d_image[image_idx]
                    num_patches = t * h * w

                    t_idx = torch.arange(t, device=device).view(t, 1, 1).expand(t, h, w).flatten()
                    h_idx = torch.arange(h, device=device).view(1, h, 1).expand(t, h, w).flatten()
                    w_idx = torch.arange(w, device=device).view(1, 1, w).expand(t, h, w).flatten()

                    pos_vision = torch.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)

                    position_id = pos_vision.max().item() + 1
                    seq_idx += num_patches
                    image_idx += 1

                else:
                    pos = torch.full((3, 1), position_id, dtype=torch.long, device=device)
                    pos_chunks.append(pos)
                    position_id += 1
                    seq_idx += 1

            if not pos_chunks: continue
            pos_ids_example = torch.cat(pos_chunks, dim=1)

            if pos_ids_example.shape[1] > T: pos_ids_example = pos_ids_example[:, :T]
            elif pos_ids_example.shape[1] < T:
                padding = torch.zeros((3, T - pos_ids_example.shape[1]), dtype=torch.long, device=device)
                pos_ids_example = torch.cat([pos_ids_example, padding], dim=1)

            all_pos_ids[batch_idx] = pos_ids_example
        return all_pos_ids.transpose(0, 1)

    def forward(self, input_ids, pixels=None, d_image=None, audio_features=None, audio_feature_lens=None, use_cache=False, past_key_values=None):
        cache_offset = past_key_values[0][0].shape[2] if past_key_values else 0
        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image, cache_offset=cache_offset)
        input_embeds = self.model.embed_tokens(input_ids)

        if cache_offset == 0:
            if pixels is not None and self.visual:
                image_embeds = self.visual(pixels=pixels, d_image=d_image) # Shape (B, L_merged, C)
                image_positions = (input_ids == self.image_pad_token_id)
                if image_positions.sum() > 0: input_embeds[image_positions] = image_embeds.squeeze(0).to(input_embeds.dtype) # image_embeds is already (1, 460, 1536). No flatten needed.

            if audio_features is not None and self.audio_encoder and self.audio_projector:
                audio_tower_embeds = self.audio_encoder(audio_features, audio_feature_lens)
                audio_embeds = self.audio_projector(audio_tower_embeds)
                audio_positions = (input_ids == self.audio_chunk_token_id)
                if audio_positions.sum() > 0: input_embeds[audio_positions] = audio_embeds.to(input_embeds.dtype)

        x, next_cache = self.model(x=input_embeds, position_ids=position_ids, use_cache=use_cache, past_key_values=past_key_values)
        logits = torch.matmul(x, self.model.embed_tokens.weight.T) if self.lm_head is None else self.lm_head(x)
        return logits, next_cache

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens, temperature, top_p, top_k, repetition_penalty, repetition_context_size, negative_prompt_ids, eos_token_ids, cache=None):
        input_ids = inputs["input_ids"]
        pixels = inputs.get("pixels")
        d_image = inputs.get("d_image")
        audio_features = inputs.get("audio_features")
        audio_feature_lens = inputs.get("audio_feature_lens")

        if cache is None: logits, past_key_values = self.forward( input_ids=input_ids, pixels=pixels, d_image=d_image, audio_features=audio_features, audio_feature_lens=audio_feature_lens, use_cache=True)
        else: logits, past_key_values = self.forward(input_ids=input_ids, use_cache=True, past_key_values=cache)

        y = logits[:, -1:, :]
        generated_token_ids = []

        for _ in range(max_new_tokens):
            if repetition_penalty > 1.0: y = apply_repetition_penalty(y.squeeze(1), generated_token_ids[-repetition_context_size:], repetition_penalty).unsqueeze(1)
            y = apply_negative_prompt(y.squeeze(1), negative_prompt_ids).unsqueeze(1)
            next_token = sampler(y.squeeze(1), temperature, top_p, top_k)
            token_id = next_token.item()

            if token_id in eos_token_ids: break
            generated_token_ids.append(token_id)
            logits, past_key_values = self.forward(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
            y = logits

        all_tokens = torch.cat([inputs["input_ids"], torch.tensor([generated_token_ids], device=input_ids.device)], dim=1)
        return all_tokens, past_key_values

    @classmethod
    def get_config_class(cls): return YunaConfig

    @classmethod
    def from_pretrained(cls, repo_id, device_map="auto", **kwargs):
        from .utils import load_pretrained_model
        return load_pretrained_model(cls, repo_id, device_map=device_map, **kwargs)

    def save_to_safetensors(self, model_path="model.ckpt", save_path="model.safetensors"):
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt # If using Lightning, model weights are usually under 'state_dict'
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        save_file(state_dict, save_path) # Save as safetensors
        print(f"Saved safetensors!")

class LongTermMemory:
    def __init__(self, model, processor, device, memory_path="long_term_memory", max_tokens=8192):
        self.model = model
        self.processor = processor
        self.device = device
        self.memory_path = memory_path
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.entries = []      # List of text chunks
        self.embeddings = []   # List of corresponding embeddings
        self.memory_path.mkdir(exist_ok=True)
        self.load()

    @torch.inference_mode()
    def _get_embedding(self, text):
        """Creates an embedding for a chunk of text."""
        inputs = self.processor([text], device=self.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] == 0: return None # Ensure input is not empty

        hidden_states, _ = self.model.model(x=self.model.model.embed_tokens(input_ids), position_ids=self.model._get_position_ids(input_ids)) # Get hidden states
        embedding = hidden_states[:, -1, :].cpu() # Use the embedding of the last token as the representation
        return F.normalize(embedding.float(), p=2, dim=-1)

    def add(self, text_chunk):
        """Adds a new chunk to long-term memory and manages memory size."""
        embedding = self._get_embedding(text_chunk)
        if embedding is None: return
        chunk_token_count = len(self.processor.tokenizer.encode(text_chunk).ids)
        self.entries.append(text_chunk)
        self.embeddings.append(embedding)
        self.current_tokens += chunk_token_count

        print(f"[LTM] New memory added. Total LTM size: {self.current_tokens}/{self.max_tokens} tokens.")

        # Check if we need to compress memory
        if self.current_tokens > self.max_tokens:
            print(f"[LTM] Memory limit {self.max_tokens} exceeded. Compressing...")
            self._compress_memory(target_tokens=self.max_tokens // 2)
            self.save() # Save after compression

    def _compress_memory(self, target_tokens):
        """Reduces memory size by combining least-recently-used entries."""
        while self.current_tokens > target_tokens and len(self.entries) > 1:
            oldest_text_1 = self.entries.pop(0)
            oldest_text_2 = self.entries.pop(0)
            _ = self.embeddings.pop(0)
            _ = self.embeddings.pop(0)
            self.current_tokens -= len(self.processor.tokenizer.encode(oldest_text_1).ids)
            self.current_tokens -= len(self.processor.tokenizer.encode(oldest_text_2).ids)
            combined_text = f'"{oldest_text_1}"\n"{oldest_text_2}"'
            combined_embedding = self._get_embedding(combined_text)

            if combined_embedding is not None:
                self.entries.insert(0, combined_text) # Add the new compressed memory to the front (as it's now a summary of old events)
                self.embeddings.insert(0, combined_embedding)
                self.current_tokens += len(self.processor.tokenizer.encode(combined_text).ids)
                print(f"[LTM] Compressed two old memories. New LTM size: {self.current_tokens}/{self.max_tokens} tokens.")

    def retrieve(self, query_text, top_k=1):
        """Retrieves relevant formatted context from long-term memory."""
        if not self.entries: return ""

        query_embedding = self._get_embedding(query_text)
        if query_embedding is None: return ""

        memory_embeddings = torch.cat(self.embeddings, dim=0)
        similarities = (query_embedding @ memory_embeddings.T).squeeze()
        if similarities.dim() == 0: similarities = similarities.unsqueeze(0) # Handle single memory case

        top_k_indices = torch.topk(similarities, k=min(top_k, len(self.entries))).indices
        retrieved_texts = [self.entries[i] for i in top_k_indices]
        print(f"[LTM] Retrieved {len(retrieved_texts)} memories.")
        return "\n".join(f'"{text}"' for text in reversed(retrieved_texts))

    def save(self):
        """Saves memory entries and embeddings to disk."""
        with open(self.memory_path / "memory_entries.json", "w", encoding="utf-8") as f: json.dump(self.entries, f, ensure_ascii=False, indent=2)
        if self.embeddings: torch.save(torch.stack(self.embeddings), self.memory_path / "memory_embeddings.pt")
        print("[LTM] Save complete.")

    def load(self):
        """Loads memory from disk."""
        entries_path = self.memory_path / "memory_entries.json"
        embeddings_path = self.memory_path / "memory_embeddings.pt"

        if entries_path.exists() and embeddings_path.exists():
            with open(entries_path, "r", encoding="utf-8") as f: self.entries = json.load(f)
            stacked_embeddings = torch.load(embeddings_path)
            self.embeddings = [e.unsqueeze(0) for e in stacked_embeddings]
            self.current_tokens = sum(len(self.processor.tokenizer.encode(e).ids) for e in self.entries) # Recalculate token count
            print(f"[LTM] Load complete. LTM size: {self.current_tokens}/{self.max_tokens} tokens.")

def apply_repetition_penalty(logits, generated_tokens, penalty):
    """Apply repetition penalty to specific logits based on the given context."""
    if not generated_tokens: return logits

    vocab_size = logits.shape[-1]
    valid_tokens = [t for t in generated_tokens if 0 <= t < vocab_size]
    if not valid_tokens: return logits

    indices = torch.tensor(valid_tokens, device=logits.device)
    selected_logits = torch.gather(logits, 1, indices.unsqueeze(0)) # Use torch.gather to select logits for specified indices
    selected_logits = torch.where(selected_logits < 0, selected_logits * penalty, selected_logits / penalty) # Apply penalty
    logits.scatter_(1, indices.unsqueeze(0), selected_logits) # Use torch.scatter to update the original logits tensor
    return logits

def apply_negative_prompt(logits, negative_prompt_ids):
    """Penalize tokens from the negative prompt by setting their logits to -inf."""
    if not negative_prompt_ids: return logits
    indices_to_penalize = list(negative_prompt_ids)

    if indices_to_penalize:
        indices = torch.tensor(indices_to_penalize, device=logits.device)
        logits[:, indices] = -float("inf")
    return logits

def sampler(logits, temp, top_p, top_k):
    """Apply temperature, top-p, and top-k sampling to logits."""
    if temp == 0: return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temp

    if 0 < top_k < logits.shape[-1]:
        top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
        kth_val = top_k_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth_val, -torch.inf, logits)

    if 0 < top_p < 1.0:
        probs = torch.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p # Create a mask for tokens to remove

        # Shift the mask to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the removal mask back to the original logit positions
        indices_to_remove = torch.gather(sorted_indices, -1, torch.where(sorted_indices_to_remove)[1].unsqueeze(0))
        logits.scatter_(-1, indices_to_remove, -float("inf"))

    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)