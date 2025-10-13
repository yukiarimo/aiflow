import inspect
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from .vision import YunaVisionEncoder
from .audio import YunaAudioEncoder, AudioProjector
import json

@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params): return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

@dataclass
class VisionConfig(BaseModelConfig):
    n_layer: int = 32
    n_embed: int = 1280
    output_n_embed: int = 1536
    n_heads: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    intermediate_size: int = 5120
    hidden_act: str = "quick_gelu"

@dataclass
class TextConfig(BaseModelConfig):
    n_embed: int = 1536
    n_layer: int = 28
    n_heads: int = 12
    n_kv_heads: int = 2
    n_mlp: int = 8960
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

@dataclass
class ModelConfig(BaseModelConfig):
    text_config: object
    vision_config: object
    audio_config: object
    model_type: str = "yuna"
    vocab_size: int = 151936
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151653
    eos_token_id: int = 151643

    @classmethod
    def from_dict(cls, params): return cls(text_config=TextConfig.from_dict(params), vision_config=VisionConfig.from_dict(params.get("vision_config", {})), audio_config=params.get("audio_config", None), **{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

class YunaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        self.inv_freq = 1.0 / (config.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, position_ids):
        if position_ids.ndim > 2: inv_freq = mx.expand_dims(mx.expand_dims(self.inv_freq, 0), 0)
        else: inv_freq = self.inv_freq

        freqs = position_ids[..., None] * inv_freq.astype(x.dtype)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = emb.cos().astype(x.dtype), emb.sin().astype(x.dtype)
        return cos, sin

def _rotate_half(x): return mx.concatenate([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], axis=-1)

def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.ndim == 2: cos, sin = mx.expand_dims(cos, 1), mx.expand_dims(sin, 1)
    elif cos.ndim == 3: cos, sin = mx.expand_dims(cos, 0), mx.expand_dims(sin, 0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class YunaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(self, x, cos, sin, mask, cache):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None: k, v = cache.update_and_fetch(k, v)

        if self.n_kv_heads != self.n_heads:
            k = mx.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            v = mx.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))

class YunaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x): return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class YunaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = YunaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = YunaMLP(config)

    def __call__(self, x, cos, sin, mask, cache):
        attn_out = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class YunaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = YunaRotaryEmbedding(config)
        self.layers = [YunaBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, position_ids, cache=None, mask=None):
        cos, sin = self.rotary_emb(x, position_ids)
        if mask is None and x.shape[1] > 1: mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
        if cache is None: cache = [KVCache() for _ in self.layers]

        for i, layer in enumerate(self.layers): x = layer(x, cos, sin, mask, cache[i])
        return self.norm(x), cache

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Map from simple names to MLX model names
        text_config_mlx = TextConfig(
            hidden_size=config.text_config.n_embed,
            num_hidden_layers=config.text_config.n_layer,
            intermediate_size=config.text_config.n_mlp,
            num_attention_heads=config.text_config.n_heads,
            num_key_value_heads=config.text_config.n_kv_heads,
            **{k: v for k, v in config.text_config.__dict__.items() if k not in ['n_embed', 'n_layer', 'n_mlp', 'n_heads', 'n_kv_heads']}
        )
        vision_config_mlx = VisionConfig(
            embed_dim=config.vision_config.n_embed,
            hidden_size=config.vision_config.output_n_embed,
            depth=config.vision_config.n_layer,
            num_heads=config.vision_config.n_heads,
            **{k: v for k, v in config.vision_config.__dict__.items() if k not in ['n_embed', 'output_n_embed', 'n_layer', 'n_heads']}
        )

        self.vision_tower = YunaVisionEncoder(vision_config_mlx)
        if config.audio_config:
            self.audio_tower = YunaAudioEncoder()
            self.audio_projector = AudioProjector(self.audio_tower.config.output_dim, text_config_mlx.hidden_size)

        self.language_model = YunaModel(text_config_mlx)
        self.lm_head = nn.Linear(text_config_mlx.hidden_size, config.vocab_size, bias=False) if not text_config_mlx.tie_word_embeddings else None

    def _get_position_ids(self, input_ids, d_image=None, cache_offset=0):
        B, T = input_ids.shape
        is_multimodal = (d_image is not None) and (mx.sum((input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)) > 0)

        if not is_multimodal:
            positions = mx.arange(cache_offset, cache_offset + T, dtype=mx.int32)
            return mx.repeat(mx.expand_dims(positions, 0), repeats=B, axis=0)

        all_pos_ids = []
        for i in range(B):
            seq = input_ids[i]
            seq_idx, image_idx, pos_chunks, position_id = 0, 0, [], cache_offset

            while seq_idx < T:
                token_id = seq[seq_idx].item()
                is_image_token = (token_id == self.config.image_token_id or token_id == self.config.video_token_id)

                if is_image_token and image_idx < (d_image.shape[0] if d_image is not None else 0):
                    t, h, w = map(int, d_image[image_idx])
                    num_patches = t * h * w
                    t_idx = mx.repeat(mx.arange(t).reshape(t, 1, 1), h, axis=1).repeat(w, axis=2).flatten()
                    h_idx = mx.repeat(mx.arange(h).reshape(1, h, 1), t, axis=0).repeat(w, axis=2).flatten()
                    w_idx = mx.repeat(mx.arange(w).reshape(1, 1, w), t, axis=0).repeat(h, axis=1).flatten()
                    pos_vision = mx.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)
                    position_id = pos_vision.max().item() + 1
                    seq_idx += num_patches
                    image_idx += 1
                else:
                    pos = mx.full((3, 1), position_id, dtype=mx.int32)
                    pos_chunks.append(pos); position_id += 1; seq_idx += 1

            if not pos_chunks: continue
            pos_ids_example = mx.concatenate(pos_chunks, axis=1)

            if pos_ids_example.shape[1] > T: pos_ids_example = pos_ids_example[:, :T]
            elif pos_ids_example.shape[1] < T:
                 padding = mx.zeros((3, T - pos_ids_example.shape[1]), dtype=mx.int32)
                 pos_ids_example = mx.concatenate([pos_ids_example, padding], axis=1)
            all_pos_ids.append(pos_ids_example)

        return mx.stack(all_pos_ids, axis=0)

    def __call__(self, input_ids, pixel_values=None, d_image=None, audio_features=None, audio_feature_lens=None, cache=None):
        cache_offset = cache[0].offset if cache and cache[0] is not None else 0
        pos_ids = self._get_position_ids(input_ids, d_image, cache_offset)
        embeds = self.language_model.embed_tokens(input_ids)

        if cache_offset == 0:
            if pixel_values is not None and self.vision_tower is not None:
                img_embeds = self.vision_tower(pixel_values, d_image)
                img_pos = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
                if mx.sum(img_pos) > 0: embeds[img_pos] = img_embeds.astype(embeds.dtype)
            if audio_features is not None and hasattr(self, 'audio_tower'):
                audio_embeds = self.audio_projector(self.audio_tower(audio_features, audio_feature_lens))
                audio_pos = (input_ids == self.config.audio_token_id)
                if mx.sum(audio_pos) > 0: embeds[audio_pos] = audio_embeds.astype(embeds.dtype)

        x, cache = self.language_model(embeds, pos_ids, cache)
        return (self.language_model.embed_tokens.as_linear(x) if self.lm_head is None else self.lm_head(x)), cache

    @property
    def layers(self): return self.language_model.layers

class LongTermMemory:
    def __init__(self, model, processor, memory_path="long_term_memory_mlx", max_tokens=8192):
        self.model, self.processor, self.memory_path, self.max_tokens = model, processor, memory_path, max_tokens
        self.current_tokens, self.entries, self.embeddings = 0, [], []
        self.memory_path.mkdir(exist_ok=True)
        self.load()

    @mx.compile
    def _get_embedding_from_ids(self, input_ids): # JIT compiled
        input_embeds = self.model.language_model.embed_tokens(input_ids)
        position_ids = self.model._get_position_ids(input_ids)
        hidden_states, _ = self.model.language_model(input_embeds, position_ids)
        return hidden_states[:, -1, :]

    def _get_embedding(self, text):
        input_ids = self.processor(text, images=None)["input_ids"]

        if input_ids.size == 0: return None
        embedding = self._get_embedding_from_ids(input_ids)
        norm = mx.linalg.norm(embedding, ord=2, axis=-1, keepdims=True) # Use L2 normalization for similarity search, not just the norm
        normalized_embedding = embedding / norm
        mx.eval(normalized_embedding)
        return normalized_embedding

    def add(self, text_chunk):
        embedding = self._get_embedding(text_chunk)
        if embedding is None: print("[LTM] Skipped empty memory chunk."); return
        chunk_token_count = len(self.processor.tokenizer.encode(text_chunk).ids)
        self.entries.append(text_chunk)
        self.embeddings.append(embedding)
        self.current_tokens += chunk_token_count
        print(f"[LTM] New memory added. Total LTM size: {self.current_tokens}/{self.max_tokens} tokens.")

        if self.current_tokens > self.max_tokens:
            print(f"[LTM] Memory limit {self.max_tokens} exceeded. Compressing...")
            self._compress_memory(target_tokens=self.max_tokens // 2)
            self.save()

    def _compress_memory(self, target_tokens):
        while self.current_tokens > target_tokens and len(self.entries) > 1:
            oldest_text_1, oldest_text_2 = self.entries.pop(0), self.entries.pop(0)
            self.embeddings.pop(0); self.embeddings.pop(0)
            self.current_tokens -= len(self.processor.tokenizer.encode(oldest_text_1).ids)
            self.current_tokens -= len(self.processor.tokenizer.encode(oldest_text_2).ids)
            combined_text = f'"{oldest_text_1}"\n"{oldest_text_2}"'
            combined_embedding = self._get_embedding(combined_text)

            if combined_embedding is not None:
                self.entries.insert(0, combined_text)
                self.embeddings.insert(0, combined_embedding)
                self.current_tokens += len(self.processor.tokenizer.encode(combined_text).ids)
                print(f"[LTM] Compressed two old memories. New LTM size: {self.current_tokens}/{self.max_tokens} tokens.")

    def retrieve(self, query_text, top_k=1):
        if not self.entries: return ""
        query_embedding = self._get_embedding(query_text)

        if query_embedding is None: return ""
        memory_embeddings = mx.concatenate(self.embeddings, axis=0)
        similarities = (query_embedding @ memory_embeddings.T).squeeze() # Simplified similarity calculation to match PyTorch

        if similarities.ndim == 0: similarities = mx.expand_dims(similarities, 0)
        top_k_indices = mx.argsort(similarities)[-min(top_k, len(self.entries)):]
        top_k_indices = top_k_indices[::-1].tolist()
        retrieved_texts = [self.entries[i] for i in top_k_indices]
        print(f"[LTM] Retrieved {len(retrieved_texts)} memories.")
        return "\n".join(f'"{text}"' for text in reversed(retrieved_texts))

    def save(self):
        with open(self.memory_path / "memory_entries.json", "w", encoding="utf-8") as f: json.dump(self.entries, f, ensure_ascii=False, indent=2)
        if self.embeddings: mx.save_safetensors(str(self.memory_path / "memory_embeddings.safetensors"), {"embeddings": mx.concatenate(self.embeddings, axis=0)})
        print("[LTM] Saved long-term memory to disk.")

    def load(self):
        entries_path = self.memory_path / "memory_entries.json"
        embeddings_path = self.memory_path / "memory_embeddings.safetensors"

        if entries_path.exists() and embeddings_path.exists():
            print("[LTM] Loading long-term memory from disk...")
            with open(entries_path, "r", encoding="utf-8") as f: self.entries = json.load(f)
            data = mx.load(str(embeddings_path))

            if "embeddings" in data and data["embeddings"].shape[0] == len(self.entries):
                self.embeddings = [mx.expand_dims(e, axis=0) for e in data["embeddings"]]
                self.current_tokens = sum(len(self.processor.tokenizer.encode(e).ids) for e in self.entries)
                print(f"[LTM] Load complete. LTM size: {self.current_tokens}/{self.max_tokens} tokens.")
            else:
                print("[LTM] Mismatch in loaded memory files. Starting fresh.")
                self.entries, self.embeddings = [], []

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None: self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values

def apply_repetition_penalty(logits, generated_tokens, penalty):
    if not generated_tokens: return logits
    indices = mx.array([t for t in generated_tokens if 0 <= t < logits.shape[-1]])

    if indices.size == 0: return logits
    selected_logits = logits[:, indices]
    selected_logits = mx.where(selected_logits < 0, selected_logits * penalty, selected_logits / penalty)
    logits[:, indices] = selected_logits
    return logits

def apply_frequency_presence_penalty(logits, generated_tokens, frequency_penalty, presence_penalty):
    if not generated_tokens: return logits
    token_counts = {t: generated_tokens.count(t) for t in set(generated_tokens) if 0 <= t < logits.shape[-1]}
    if not token_counts: return logits
    indices = mx.array(list(token_counts.keys()))
    counts = mx.array(list(token_counts.values()))
    logits[:, indices] -= counts * frequency_penalty + presence_penalty
    return logits

def apply_logit_bias(logits, logit_bias):
    if not logit_bias: return logits

    indices = mx.array(list(logit_bias.keys()))
    values = mx.array(list(logit_bias.values()))
    logits[:, indices] += values
    return logits

def sampler(logits, temp, top_p, top_k):
    if temp == 0: return mx.argmax(logits, axis=-1)
    logits = logits / temp

    if 0 < top_k < logits.shape[-1]:
        kth_vals = -mx.partition(-logits, top_k - 1, axis=-1)[..., top_k - 1 : top_k]
        logits = mx.where(logits < kth_vals, -mx.inf, logits)

    if 0 < top_p < 1.0:
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[:, 1:] = remove_mask[:, :-1]
        remove_mask[:, 0] = False
        indices_to_remove = mx.zeros_like(remove_mask, dtype=mx.bool_)
        mx.scatter(indices_to_remove, sorted_indices, remove_mask, axis=-1)
        logits = mx.where(indices_to_remove, -mx.inf, logits)

    return mx.random.categorical(logits)

def generate(model, processor, prompt, image_paths, video_paths, audio_paths, max_tokens, temperature, top_p, top_k, repetition_penalty, repetition_context_size, frequency_penalty, presence_penalty, logit_bias, eos_token_ids, cache):
    inputs = processor(prompt, image_paths=image_paths, video_paths=video_paths, audio_paths=audio_paths)
    prompt_tokens = inputs["input_ids"]

    if cache is None: cache = [KVCache() for _ in model.language_model.layers]
    logits, _ = model(prompt_tokens, **({k: v for k, v in inputs.items() if k != 'input_ids'} if cache[0].offset == 0 else {}), cache=cache)
    y = logits[:, -1, :]
    generated_token_ids = []

    for _ in range(max_tokens):
        context = generated_token_ids[-repetition_context_size:]
        if repetition_penalty > 1.0: y = apply_repetition_penalty(y, context, repetition_penalty)
        if frequency_penalty > 0.0 or presence_penalty > 0.0: y = apply_frequency_presence_penalty(y, context, frequency_penalty, presence_penalty)
        y = apply_logit_bias(y, logit_bias)
        y = sampler(y, temperature, top_p, top_k)
        token_id = y.item()

        if token_id in eos_token_ids: break
        generated_token_ids.append(token_id)

        logits, cache = model(y[:, None], cache=cache)
        y = logits[:, -1, :]

    return processor.tokenizer.decode(generated_token_ids), cache