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
    n_mlp: int = 5120
    hidden_act: str = "quick_gelu"

    @property
    def embed_dim(self): return self.n_embed

    @property
    def depth(self): return self.n_layer

    @property
    def num_heads(self): return self.n_heads

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
    def from_dict(cls, params):
        text_config = TextConfig.from_dict(params.get("text_config", params))
        vision_config = VisionConfig.from_dict(params.get("vision_config", {}))
        audio_config = params.get("audio_config", None)
        model_params = {k: v for k, v in params.items() if k in inspect.signature(cls).parameters and k not in ['text_config', 'vision_config', 'audio_config']}
        return cls(text_config=text_config, vision_config=vision_config, audio_config=audio_config, **model_params)

class YunaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.n_embed // config.n_heads
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
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embed // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.n_embed, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.n_embed, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.n_embed, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.n_embed, bias=False)

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
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def __call__(self, x): return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class YunaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.self_attn = YunaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        self.mlp = YunaMLP(config)

    def __call__(self, x, cos, sin, mask, cache):
        attn_out = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class YunaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = YunaRotaryEmbedding(config)
        self.layers = [YunaBlock(config) for _ in range(config.n_layer)]
        self.norm = nn.RMSNorm(config.n_embed, eps=config.rms_norm_eps)

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

        self.vision_tower = YunaVisionEncoder(config.vision_config)
        if config.audio_config:
            self.audio_tower = YunaAudioEncoder()
            self.audio_projector = AudioProjector(self.audio_tower.config.output_dim, config.text_config.n_embed)

        self.language_model = YunaModel(config.text_config)
        self.lm_head = nn.Linear(config.text_config.n_embed, config.vocab_size, bias=False) if not config.text_config.tie_word_embeddings else None

    def _merge_modal_embeddings(self, input_ids, token_embeds, modal_embeds, modal_mask, modal_name="modal"):
        if modal_embeds is None:
            return token_embeds

        modal_mask = modal_mask.astype(mx.bool_)
        total_slots = int(mx.sum(modal_mask).item())
        if total_slots == 0:
            return token_embeds

        if modal_embeds.ndim > 2:
            modal_embeds = modal_embeds.reshape(-1, modal_embeds.shape[-1])
        modal_embeds = modal_embeds.astype(token_embeds.dtype)

        batch_outputs = []
        feature_start = 0
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            mask = modal_mask[batch_idx]
            num_slots = int(mx.sum(mask).item())

            if num_slots > 0:
                batch_features = modal_embeds[feature_start : feature_start + num_slots]
                if batch_features.shape[0] != num_slots:
                    raise ValueError(
                        f"Expected {num_slots} {modal_name} embeddings but received {batch_features.shape[0]} for batch index {batch_idx}."
                    )
                feature_start += num_slots

                cumsum = mx.cumsum(mask.astype(mx.int32))
                feature_indices = mx.where(mask, cumsum - 1, 0)
                gathered = batch_features[feature_indices]
                mask_expanded = mx.expand_dims(mask, axis=-1)
                merged = mx.where(mask_expanded, gathered, token_embeds[batch_idx])
            else:
                merged = token_embeds[batch_idx]

            batch_outputs.append(merged)

        if feature_start != modal_embeds.shape[0]:
            raise ValueError(
                f"Consumed {feature_start} of {modal_embeds.shape[0]} available {modal_name} embeddings; counts must match."
            )

        return mx.stack(batch_outputs, axis=0)

    def _get_position_ids(self, input_ids, d_image=None, cache_offset=0):
        B, T = input_ids.shape
        is_multimodal = (d_image is not None) and (mx.sum((input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)) > 0)

        if not is_multimodal:
            positions = mx.arange(cache_offset, cache_offset + T, dtype=mx.int32)
            return mx.repeat(mx.expand_dims(positions, 0), repeats=B, axis=0)

        d_image_list = d_image.astype(mx.int32).tolist() if d_image is not None else []
        image_meta_idx = 0
        all_pos_ids = []

        for i in range(B):
            seq = input_ids[i]
            seq_positions = []
            seq_idx = 0
            position_id = cache_offset
            local_image_offset = image_meta_idx

            while seq_idx < T:
                token_id = int(seq[seq_idx].item())
                is_image_token = (
                    token_id == self.config.image_token_id
                    or token_id == self.config.video_token_id
                )

                if is_image_token and local_image_offset < len(d_image_list):
                    t, h, w = map(int, d_image_list[local_image_offset])
                    num_patches = t * h * w
                    available_tokens = T - seq_idx
                    if available_tokens <= 0:
                        break

                    patch_len = min(num_patches, available_tokens)
                    patch_positions = mx.arange(patch_len, dtype=mx.int32) + position_id
                    seq_positions.append(patch_positions)
                    position_id += patch_len
                    seq_idx += patch_len
                    local_image_offset += 1
                else:
                    seq_positions.append(mx.array([position_id], dtype=mx.int32))
                    position_id += 1
                    seq_idx += 1

            image_meta_idx = local_image_offset

            if not seq_positions:
                all_pos_ids.append(mx.full((T,), cache_offset, dtype=mx.int32))
                continue

            pos_ids_example = mx.concatenate(seq_positions, axis=0)
            if pos_ids_example.shape[0] > T:
                pos_ids_example = pos_ids_example[:T]
            elif pos_ids_example.shape[0] < T:
                extra_len = T - pos_ids_example.shape[0]
                padding = mx.arange(extra_len, dtype=mx.int32) + position_id
                pos_ids_example = mx.concatenate([pos_ids_example, padding], axis=0)

            all_pos_ids.append(pos_ids_example)

        return mx.stack(all_pos_ids, axis=0)

    def __call__(self, input_ids, pixel_values=None, d_image=None, audio_features=None, audio_feature_lens=None, cache=None, style_embedding=None, style_strength=0.0):
        cache_offset = cache[0].offset if cache and cache[0] is not None else 0
        pos_ids = self._get_position_ids(input_ids, d_image, cache_offset)
        embeds = self.language_model.embed_tokens(input_ids)

        if cache_offset == 0:
            if pixel_values is not None and self.vision_tower is not None and d_image is not None:
                vision_embeds = self.vision_tower(pixel_values, d_image)
                vision_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
                embeds = self._merge_modal_embeddings(
                    input_ids,
                    embeds,
                    vision_embeds,
                    vision_mask,
                    modal_name="vision",
                )
            if audio_features is not None and hasattr(self, "audio_tower"):
                audio_embeds = self.audio_projector(self.audio_tower(audio_features, audio_feature_lens))
                audio_mask = input_ids == self.config.audio_token_id
                embeds = self._merge_modal_embeddings(
                    input_ids,
                    embeds,
                    audio_embeds,
                    audio_mask,
                    modal_name="audio",
                )

        x, cache = self.language_model(embeds, pos_ids, cache)
        if style_embedding is not None and style_strength > 0.0:
            x = apply_style_guidance(x, style_embedding, style_strength)

        logits = self.language_model.embed_tokens.as_linear(x) if self.lm_head is None else self.lm_head(x)
        return logits, cache

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

def sampler(logits, temp, top_p, top_k, typical_p=0.95, min_p=1e-1):
    if temp == 0: return mx.argmax(logits, axis=-1)

    logits = logits.astype(mx.float32)
    vocab = logits.shape[-1]
    eps = mx.array(1e-8, dtype=logits.dtype)

    raw_probs = mx.softmax(logits, axis=-1)
    entropy = -mx.sum(raw_probs * mx.log(raw_probs + eps), axis=-1, keepdims=True)
    max_entropy = mx.log(mx.array([vocab], dtype=logits.dtype))
    entropy_ratio = entropy / mx.maximum(max_entropy, eps)

    base_temp = mx.array(temp, dtype=logits.dtype)
    adaptive_temp = base_temp * (0.7 + 0.3 * entropy_ratio)
    adaptive_temp = mx.maximum(adaptive_temp, mx.array(1e-3, dtype=logits.dtype))
    logits = logits / adaptive_temp

    logits = logits + mx.random.normal(shape=logits.shape, scale=1e-4, dtype=logits.dtype)
    probs = mx.softmax(logits, axis=-1)

    # Fix: Create zeros array and then convert to boolean type
    remove_mask = mx.zeros_like(probs).astype(mx.bool_)
    if 0 < top_k < vocab:
        kth_vals = -mx.partition(-logits, top_k - 1, axis=-1)[..., top_k - 1 : top_k]
        remove_mask = remove_mask | (logits < kth_vals)

    batch_idx = mx.arange(probs.shape[0])[:, None]

    if 0 < top_p < 1.0:
        sorted_idx = mx.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[:, 0] = False
        top_p_mask = mx.zeros_like(remove_mask)
        top_p_mask[batch_idx, sorted_idx] = sorted_remove
        remove_mask = remove_mask | top_p_mask

    if 0 < typical_p < 1.0:
        log_probs = mx.log(probs + eps)
        shifted = mx.abs((-log_probs) - entropy)
        typical_idx = mx.argsort(shifted, axis=-1)
        typical_probs = mx.take_along_axis(probs, typical_idx, axis=-1)
        typical_cum = mx.cumsum(typical_probs, axis=-1)
        typical_remove = typical_cum > typical_p
        typical_remove[:, 0] = False
        typical_mask = mx.zeros_like(remove_mask)
        typical_mask[batch_idx, typical_idx] = typical_remove
        remove_mask = remove_mask | typical_mask

    if min_p > 0.0:
        remove_mask = remove_mask | (probs < min_p)

    filtered_logits = mx.where(remove_mask, -mx.inf, logits)
    finite_mask = mx.isfinite(filtered_logits)
    all_invalid = mx.logical_not(mx.any(finite_mask, axis=-1, keepdims=True))
    filtered_logits = mx.where(all_invalid, logits, filtered_logits)

    max_logits = mx.max(filtered_logits, axis=-1, keepdims=True)
    stabilized = filtered_logits - max_logits
    log_probs = filtered_logits - (max_logits + mx.log(mx.sum(mx.exp(stabilized), axis=-1, keepdims=True) + eps))

    return mx.random.categorical(log_probs.astype(mx.float32))

def apply_style_guidance(hidden_states, style_embedding, strength=0.3):
    if style_embedding is None or strength <= 0.0:
        return hidden_states

    style_vec = style_embedding.astype(hidden_states.dtype)
    if style_vec.ndim == 1:
        style_vec = style_vec[None, :]
    style_vec = style_vec / (mx.linalg.norm(style_vec, ord=2, axis=-1, keepdims=True) + 1e-8)
    style_vec = style_vec[:, None, :]  # (1, 1, D)

    target_hidden = hidden_states[:, -1:, :]
    target_norm = target_hidden / (mx.linalg.norm(target_hidden, ord=2, axis=-1, keepdims=True) + 1e-8)
    guided_last = (1.0 - strength) * target_norm + strength * style_vec
    guided_last = guided_last * mx.linalg.norm(target_hidden, ord=2, axis=-1, keepdims=True)

    prefix = hidden_states[:, :-1, :]
    return mx.concatenate([prefix, guided_last], axis=1)

def generate(model, processor, prompt, image_paths, video_paths, audio_paths, max_tokens, temperature, top_p, top_k, repetition_penalty, repetition_context_size, frequency_penalty, presence_penalty, logit_bias, eos_token_ids, cache, stop_strings=None, style_text=None, style_strength=0.3):
    inputs = processor(prompt, image_paths=image_paths, video_paths=video_paths, audio_paths=audio_paths)
    prompt_tokens = inputs["input_ids"]

    # Encode style guidance if provided
    style_embedding = None
    if style_text:
        style_inputs = processor(style_text, image_paths=None, video_paths=None, audio_paths=None)
        style_ids = style_inputs["input_ids"]
        style_embeds = model.language_model.embed_tokens(style_ids)
        style_pos_ids = model._get_position_ids(style_ids)
        style_hidden, _ = model.language_model(style_embeds, style_pos_ids)
        style_embedding = mx.mean(style_hidden, axis=1)
        mx.eval(style_embedding)

    if cache is None: cache = [KVCache() for _ in model.language_model.layers]

    initial_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"} if cache[0].offset == 0 else {}
    logits, cache = model(
        prompt_tokens,
        **initial_kwargs,
        cache=cache,
        style_embedding=style_embedding,
        style_strength=style_strength,
    )
    y = logits[:, -1, :]
    generated_token_ids = []

    for _ in range(max_tokens):
        context = generated_token_ids[-repetition_context_size:]
        if repetition_penalty > 1.0: y = apply_repetition_penalty(y, context, repetition_penalty)
        if frequency_penalty > 0.0 or presence_penalty > 0.0: y = apply_frequency_presence_penalty(y, context, frequency_penalty, presence_penalty)
        y = apply_logit_bias(y, logit_bias)
        y = sampler(y, temperature, top_p, top_k)
        token_id = int(y.item())

        if token_id in eos_token_ids:
            break
        generated_token_ids.append(token_id)

        next_ids = mx.array([[token_id]], dtype=prompt_tokens.dtype)
        logits, cache = model(
            next_ids,
            cache=cache,
            style_embedding=style_embedding,
            style_strength=style_strength,
        )
        y = logits[:, -1, :]
    raw_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    final_text = raw_text
    return final_text, cache

def stream_generate(model, processor, prompt, image_paths, video_paths, audio_paths, max_tokens, temperature, top_p, top_k, repetition_penalty, repetition_context_size, frequency_penalty, presence_penalty, logit_bias, eos_token_ids, cache, stop_strings=["</>"]):
    inputs = processor(prompt, image_paths=image_paths, video_paths=video_paths, audio_paths=audio_paths)
    prompt_tokens = inputs["input_ids"]

    if cache is None: cache = [KVCache() for _ in model.language_model.layers]
    logits, _ = model(prompt_tokens, **({k: v for k, v in inputs.items() if k != 'input_ids'} if cache[0].offset == 0 else {}), cache=cache)
    y = logits[:, -1, :]
    generated_token_ids = []
    previous_text = ""
    active_stops = [s for s in (stop_strings or processor.tokenizer.stop_strings) if s]
    max_stop_len = max((len(s) for s in active_stops), default=0)
    buffer = ""

    for _ in range(max_tokens):
        context = generated_token_ids[-repetition_context_size:]
        if repetition_penalty > 1.0: y = apply_repetition_penalty(y, context, repetition_penalty)
        if frequency_penalty > 0.0 or presence_penalty > 0.0: y = apply_frequency_presence_penalty(y, context, frequency_penalty, presence_penalty)
        y = apply_logit_bias(y, logit_bias)
        y = sampler(y, temperature, top_p, top_k)
        token_id = y.item()

        if token_id in eos_token_ids: break
        generated_token_ids.append(token_id)
        current_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        new_text = current_text[len(previous_text):]
        previous_text = current_text

        if not new_text:
            logits, cache = model(y[:, None], cache=cache)
            y = logits[:, -1, :]
            continue

        buffer += new_text

        stop_hit = None
        for stop in active_stops:
            idx = buffer.find(stop)
            if idx != -1 and (stop_hit is None or idx < stop_hit[0]): stop_hit = (idx, len(stop))

        if stop_hit:
            idx, _ = stop_hit
            if idx > 0: yield buffer[:idx]
            return

        if max_stop_len == 0:
            if buffer:
                yield buffer
                buffer = ""
        else:
            emit_len = len(buffer) - (max_stop_len - 1)
            if emit_len > 0:
                yield buffer[:emit_len]
                buffer = buffer[emit_len:]

        logits, cache = model(y[:, None], cache=cache)
        y = logits[:, -1, :]

    if buffer: yield buffer