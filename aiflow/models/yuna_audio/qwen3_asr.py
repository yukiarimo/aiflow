import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm
from .config import STTOutput, ModelConfig
from .audio_encoder import AudioEncoder, _get_feat_extract_output_lengths
from .utils import load_audio
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache
import transformers
from transformers import AutoTokenizer, WhisperFeatureExtractor


class StreamingResult:
	def __init__(self, text, is_final, start_time, end_time, language="en", prompt_tokens=0, generation_tokens=0):
		self.text = text
		self.is_final = is_final
		self.start_time = start_time
		self.end_time = end_time
		self.language = language
		self.prompt_tokens = prompt_tokens
		self.generation_tokens = generation_tokens


def split_audio_into_chunks(wav, sr, chunk_duration=1200.0, min_chunk_duration=1.0, search_expand_sec=5.0, min_window_ms=100.0):
	if wav.ndim > 1:
		wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

	total_samples = len(wav)
	total_sec = total_samples / sr

	if total_sec <= chunk_duration:
		if total_sec < min_chunk_duration:
			min_samples = int(min_chunk_duration * sr)
			wav = np.pad(wav, (0, min_samples - len(wav)))
		return [(wav, 0.0)]

	chunks = []
	start_sample = 0
	max_chunk_samples = int(chunk_duration * sr)
	search_samples = int(search_expand_sec * sr)
	min_window_samples = int(min_window_ms * sr / 1000)

	while start_sample < total_samples:
		end_sample = min(start_sample + max_chunk_samples, total_samples)

		if end_sample >= total_samples:
			chunk = wav[start_sample:total_samples]
			offset_sec = start_sample / sr
			if len(chunk) < min_chunk_duration * sr:
				min_samples = int(min_chunk_duration * sr)
				chunk = np.pad(chunk, (0, min_samples - len(chunk)))
			chunks.append((chunk, offset_sec))
			break

		search_start = max(start_sample, end_sample - search_samples)
		search_end = min(total_samples, end_sample + search_samples)
		search_region = wav[search_start:search_end]

		if len(search_region) > min_window_samples:
			energy = np.convolve(search_region**2, np.ones(min_window_samples) / min_window_samples, mode="valid")
			min_idx = np.argmin(energy) + min_window_samples // 2
			cut_sample = search_start + min_idx
		else:
			cut_sample = end_sample

		cut_sample = max(cut_sample, start_sample + sr)
		chunk = wav[start_sample:cut_sample]
		offset_sec = start_sample / sr

		if len(chunk) < min_chunk_duration * sr:
			min_samples = int(min_chunk_duration * sr)
			chunk = np.pad(chunk, (0, min_samples - len(chunk)))

		chunks.append((chunk, offset_sec))
		start_sample = cut_sample

	return chunks


def create_additive_causal_mask(N, offset=0):
	rinds = mx.arange(offset + N)
	linds = mx.arange(offset, offset + N) if offset else rinds
	mask = linds[:, None] < rinds[None]
	return mask * -1e9


class TextAttention(nn.Module):
	def __init__(self, config, layer_idx):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.num_kv_heads = config.num_key_value_heads
		self.head_dim = config.head_dim
		self.scale = self.head_dim**-0.5
		self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
		self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
		self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
		self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
		self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
		self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
		self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

	def __call__(self, hidden_states, cache=None):
		B, L, _ = hidden_states.shape
		queries = self.q_proj(hidden_states)
		keys = self.k_proj(hidden_states)
		values = self.v_proj(hidden_states)
		queries = queries.reshape(B, L, self.num_heads, self.head_dim)
		keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
		values = values.reshape(B, L, self.num_kv_heads, self.head_dim)
		queries = self.q_norm(queries)
		keys = self.k_norm(keys)
		queries = queries.transpose(0, 2, 1, 3)
		keys = keys.transpose(0, 2, 1, 3)
		values = values.transpose(0, 2, 1, 3)

		if cache is not None:
			offset = cache.offset
			queries = self.rope(queries, offset=offset)
			keys = self.rope(keys, offset=offset)
		else:
			offset = 0
			queries = self.rope(queries)
			keys = self.rope(keys)

		if cache is not None:
			keys, values = cache.update_and_fetch(keys, values)

		query_len = queries.shape[2]

		if query_len == 1:
			mask = None
		else:
			mask = create_additive_causal_mask(query_len, offset=offset).astype(queries.dtype)

		output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
		output = output.transpose(0, 2, 1, 3).reshape(B, query_len, -1)
		return self.o_proj(output)


class TextMLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size
		self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
		self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
		self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

	def __call__(self, x):
		return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
	def __init__(self, config, layer_idx):
		super().__init__()
		self.hidden_size = config.hidden_size
		self.self_attn = TextAttention(config, layer_idx)
		self.mlp = TextMLP(config)
		self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def __call__(self, hidden_states, cache=None):
		residual = hidden_states
		hidden_states = self.input_layernorm(hidden_states)
		hidden_states = self.self_attn(hidden_states, cache=cache)
		hidden_states = residual + hidden_states
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states
		return hidden_states


class TextModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.vocab_size = config.vocab_size
		self.num_hidden_layers = config.num_hidden_layers
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
		self.layers = [TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
		self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def __call__(self, input_ids=None, inputs_embeds=None, cache=None):
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		hidden_states = inputs_embeds
		if cache is None:
			cache = [None] * len(self.layers)

		for i, layer in enumerate(self.layers):
			hidden_states = layer(hidden_states, cache=cache[i])
		return self.norm(hidden_states)


class Qwen3ASRModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.vocab_size = config.text_config.vocab_size
		self.audio_tower = AudioEncoder(config.audio_config)
		self.model = TextModel(config.text_config)

		if config.text_config.tie_word_embeddings:
			self.lm_head = None
		else:
			self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

	def get_audio_features(self, input_features, feature_attention_mask=None):
		return self.audio_tower(input_features, feature_attention_mask)

	def _build_inputs_embeds(self, input_ids, audio_features):
		inputs_embeds = self.model.embed_tokens(input_ids)
		audio_features = audio_features.astype(inputs_embeds.dtype)
		audio_token_mask = input_ids == self.config.audio_token_id

		if audio_token_mask.any():
			batch_size, seq_len, hidden_dim = inputs_embeds.shape
			flat_mask = audio_token_mask.flatten()
			flat_mask_np = np.array(flat_mask)
			audio_indices = np.where(flat_mask_np)[0]

			if len(audio_indices) > 0 and audio_features.shape[0] > 0:
				num_to_replace = min(len(audio_indices), audio_features.shape[0])
				flat_embeds = inputs_embeds.reshape(-1, hidden_dim)
				start = int(audio_indices[0])  # contiguous `<|audio_pad|>`*N run → splice with concat (older MLX has no ArrayAt.set)
				flat_embeds = mx.concatenate([flat_embeds[:start], audio_features[:num_to_replace], flat_embeds[start + num_to_replace:]], axis=0)
				inputs_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_dim)

		return inputs_embeds

	def __call__(self, input_ids, input_embeddings=None, input_features=None, feature_attention_mask=None, cache=None):
		if input_features is not None and (cache is None or cache[0] is None or cache[0].offset == 0):
			audio_features = self.get_audio_features(input_features, feature_attention_mask)
			inputs_embeds = self._build_inputs_embeds(input_ids, audio_features)
		elif input_embeddings is not None:
			inputs_embeds = input_embeddings
		else:
			inputs_embeds = self.model.embed_tokens(input_ids)

		hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)
		if self.lm_head is not None:
			logits = self.lm_head(hidden_states)
		else:
			logits = self.model.embed_tokens.as_linear(hidden_states)
		return logits

	@property
	def layers(self):
		return self.model.layers

	@property
	def sample_rate(self):
		return 16000

	def make_cache(self):
		return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

	@staticmethod
	def sanitize(weights):
		sanitized = {}  # HF 4.57: thinker.*; HF 5.13: model.audio_tower / language_model / multi_modal_projector
		for k, v in weights.items():
			if k.startswith("thinker."):
				k = k[len("thinker."):]
			if k.startswith("model.language_model."):
				k = "model." + k[len("model.language_model."):]
			elif k.startswith("model.multi_modal_projector.linear_1."):
				k = "audio_tower.proj1." + k[len("model.multi_modal_projector.linear_1."):]
			elif k.startswith("model.multi_modal_projector.linear_2."):
				k = "audio_tower.proj2." + k[len("model.multi_modal_projector.linear_2."):]
			elif k.startswith("model.audio_tower."):
				k = k[len("model."):]
			if k == "lm_head.weight":
				continue
			if "conv2d" in k and k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 4:
				if v.shape[2] == v.shape[3] and v.shape[1] != v.shape[2]:
					v = v.transpose(0, 2, 3, 1)
			sanitized[k] = v
		return sanitized

	def model_quant_predicate(self, p, m):
		return not p.startswith("audio_tower")

	@classmethod
	def post_load_hook(cls, model, model_path):
		prev_verbosity = transformers.logging.get_verbosity()
		transformers.logging.set_verbosity_error()
		try:
			model._tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
			model._feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
		finally:
			transformers.logging.set_verbosity(prev_verbosity)

		if not hasattr(model.config, "model_repo") or model.config.model_repo is None:
			try:
				index = model_path.parts.index("hub")
				model.config.model_repo = (model_path.parts[index + 1].replace("models--", "").replace("--", "/"))
			except (ValueError, IndexError):
				model.config.model_repo = str(model_path)

		return model

	def _preprocess_audio(self, audio):
		audio_input = audio[0] if isinstance(audio, list) else audio
		if isinstance(audio_input, str):
			audio_input = load_audio(audio_input)
		audio_np = (np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input)
		audio_inputs = self._feature_extractor(audio_np, sampling_rate=16000, return_attention_mask=True, truncation=False, padding=True, return_tensors="np")
		input_features = mx.array(audio_inputs["input_features"])
		feature_attention_mask = mx.array(audio_inputs["attention_mask"])
		audio_lengths = feature_attention_mask.sum(axis=-1)
		aftercnn_lens = _get_feat_extract_output_lengths(audio_lengths)
		num_audio_tokens = int(aftercnn_lens[0].item())
		return input_features, feature_attention_mask, num_audio_tokens

	def extract_language(self, text):
		if "<asr_text>" in text and text.startswith("language "):
			return (text[len("language "):text.find("<asr_text>")].strip(), text[text.find("<asr_text>") + len("<asr_text>"):].strip())
		if "<|asr_text|>" in text and text.startswith("language "):
			return (text[len("language "):text.find("<|asr_text|>")].strip(), text[text.find("<|asr_text|>") + len("<|asr_text|>"):].strip())

		if text.startswith("language "):
			supported = getattr(self.config, "support_languages", []) or []
			if not supported:
				supported = ["English", "Chinese", "French", "Spanish", "German", "Russian", "Japanese", "Korean", "Italian", "Portuguese", "Dutch", "Arabic", "Turkish", "Vietnamese", "Indonesian", "Malay", "Thai", "Hindi", "Urdu", "Bengali"]

			supported = sorted(supported, key=len, reverse=True)
			text_lower = text.lower()

			for lang in supported:
				prefix = f"language {lang.lower()}"
				if text_lower.startswith(prefix):
					rest = text[len(prefix):].lstrip()
					if rest.startswith("<asr_text>") or rest.startswith("<|asr_text|>"):
						rest = rest.replace("<asr_text>", "", 1).replace("<|asr_text|>", "", 1).lstrip()
					return lang.capitalize(), rest

			return "English", text[len("language "):].lstrip()

		return "English", text

	def _build_prompt(self, num_audio_tokens, language=None, system_prompt=None):
		system_content = f"{system_prompt}\n" if system_prompt else ""

		if language is not None:
			supported = self.config.support_languages or []
			supported_lower = {lang.lower(): lang for lang in supported}
			lang_name = supported_lower.get(language.lower(), language)
			assistant_prefix = f"language {lang_name}<asr_text>"
		else:
			assistant_prefix = ""

		prompt = (f"<|im_start|>system\n{system_content}<|im_end|>\n"
		          f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
		          f"<|im_start|>assistant\n{assistant_prefix}")

		input_ids = self._tokenizer.encode(prompt, return_tensors="np")
		return mx.array(input_ids)

	def stream_generate(self, audio, max_tokens=8192, sampler=None, logits_processors=None, language=None, prefill_step_size=2048, verbose=False, system_prompt=None):
		if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
			raise RuntimeError("Tokenizer/FeatureExtractor not initialized. Call post_load_hook first.")

		input_features, feature_attention_mask, num_audio_tokens = (self._preprocess_audio(audio))
		input_ids = self._build_prompt(num_audio_tokens, language, system_prompt)
		eos_token_ids = [151645, 151643]

		with tqdm(total=1, desc="Encoding audio", disable=not verbose, leave=False) as pbar:
			audio_features = self.get_audio_features(input_features, feature_attention_mask)
			mx.eval(audio_features)
			pbar.update(1)

		del input_features, feature_attention_mask

		with tqdm(total=1, desc="Building embeddings", disable=not verbose, leave=False) as pbar:
			inputs_embeds = self._build_inputs_embeds(input_ids, audio_features)
			mx.eval(inputs_embeds)
			pbar.update(1)

		del audio_features

		input_embeddings = inputs_embeds[0]
		prompt = input_ids[0] if input_ids.ndim > 1 else input_ids

		del inputs_embeds

		prefill_pbar = None
		gen_pbar = None

		if verbose:
			prefill_pbar = tqdm(total=1, desc="Prefilling", unit="tok", leave=False)

		def prefill_progress(processed, total):
			nonlocal gen_pbar
			if prefill_pbar is not None:
				if prefill_pbar.total != total:
					prefill_pbar.total = total
					prefill_pbar.refresh()
				prefill_pbar.n = processed
				prefill_pbar.refresh()
				if processed >= total:
					prefill_pbar.close()
					if gen_pbar is None and verbose:
						gen_pbar = tqdm(total=max_tokens, desc="Generating", unit="tok", leave=False)

		for token, logprobs in generate_step(prompt=prompt, input_embeddings=input_embeddings, model=self, max_tokens=max_tokens, sampler=sampler, logits_processors=logits_processors, prefill_step_size=prefill_step_size, prompt_progress_callback=prefill_progress if verbose else None):
			if gen_pbar is not None:
				gen_pbar.update(1)
			if token in eos_token_ids:
				break
			yield token, logprobs

		if gen_pbar is not None:
			gen_pbar.close()

	def _generate_single_chunk(self, audio_chunk, max_tokens=8192, sampler=None, logits_processors=None, language=None, prefill_step_size=2048, verbose=False, system_prompt=None):
		generated_tokens = []
		prompt_tokens = 0

		for token, _ in self.stream_generate(audio_chunk, max_tokens=max_tokens, sampler=sampler, logits_processors=logits_processors, language=language, prefill_step_size=prefill_step_size, verbose=verbose, system_prompt=system_prompt):
			if prompt_tokens == 0:
				_, _, num_audio_tokens = self._preprocess_audio(audio_chunk)
				input_ids = self._build_prompt(num_audio_tokens, language, system_prompt)
				prompt_tokens = input_ids.shape[1]
			generated_tokens.append(int(token))

		text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
		return text, prompt_tokens, len(generated_tokens)

	def generate(self, audio, max_tokens=8192, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, min_tokens_to_keep=1, repetition_penalty=None, repetition_context_size=100, language=None, prefill_step_size=2048, chunk_duration=1200.0, min_chunk_duration=1.0, verbose=False, stream=False, system_prompt=None, **kwargs):
		if stream:
			return self.stream_transcribe(audio, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, min_tokens_to_keep=min_tokens_to_keep, repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size, language=language, prefill_step_size=prefill_step_size, chunk_duration=chunk_duration, min_chunk_duration=min_chunk_duration, verbose=verbose, system_prompt=system_prompt)

		del kwargs
		start_time = time.time()

		if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
			raise RuntimeError("Tokenizer/FeatureExtractor not initialized. Call post_load_hook first.")

		audio_input = audio[0] if isinstance(audio, list) else audio
		if isinstance(audio_input, str):
			audio_input = load_audio(audio_input)
		audio_np = (np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input)

		chunks = split_audio_into_chunks(audio_np, sr=self.sample_rate, chunk_duration=chunk_duration, min_chunk_duration=min_chunk_duration)

		sampler = make_sampler(temperature, top_p, min_p, min_tokens_to_keep=min_tokens_to_keep, top_k=top_k)
		logits_processors = (make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size) if repetition_penalty else None)

		all_texts = []
		segments = []
		total_prompt_tokens = 0
		total_generation_tokens = 0
		remaining_tokens = max_tokens

		chunk_iter = tqdm(chunks, desc="Processing chunks", disable=not verbose or len(chunks) == 1)
		for chunk_audio, offset_sec in chunk_iter:
			if remaining_tokens <= 0:
				break

			actual_chunk_duration = len(chunk_audio) / self.sample_rate
			text, prompt_toks, gen_toks = self._generate_single_chunk(chunk_audio, max_tokens=remaining_tokens, sampler=sampler, logits_processors=logits_processors, language=language, prefill_step_size=prefill_step_size, verbose=verbose and len(chunks) == 1, system_prompt=system_prompt, )

			if language is None:
				language, text = self.extract_language(text)

			all_texts.append(text)
			total_prompt_tokens += prompt_toks
			total_generation_tokens += gen_toks
			remaining_tokens -= gen_toks

			segments.append({"text": text, "language": language, "start": offset_sec, "end": offset_sec + actual_chunk_duration, })

		end_time = time.time()
		full_text = " ".join(all_texts)

		return STTOutput(text=full_text, segments=segments, language=[segment["language"] for segment in segments], prompt_tokens=total_prompt_tokens, generation_tokens=total_generation_tokens, total_tokens=total_prompt_tokens + total_generation_tokens, total_time=end_time - start_time, prompt_tps=(total_prompt_tokens / (end_time - start_time) if end_time > start_time else 0), generation_tps=(total_generation_tokens / (end_time - start_time) if end_time > start_time else 0))

	def stream_transcribe(self, audio, max_tokens=8192, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, min_tokens_to_keep=1, repetition_penalty=None, repetition_context_size=100, language=None, prefill_step_size=2048, chunk_duration=1200.0, min_chunk_duration=1.0, verbose=False, system_prompt=None):
		if not hasattr(self, "_tokenizer") or not hasattr(self, "_feature_extractor"):
			raise RuntimeError("Tokenizer/FeatureExtractor not initialized. Call post_load_hook first.")

		audio_input = audio[0] if isinstance(audio, list) else audio
		if isinstance(audio_input, str):
			audio_input = load_audio(audio_input)
		audio_np = (np.array(audio_input) if isinstance(audio_input, mx.array) else audio_input)

		total_duration = len(audio_np) / self.sample_rate
		chunks = split_audio_into_chunks(audio_np, sr=self.sample_rate, chunk_duration=chunk_duration, min_chunk_duration=min_chunk_duration)

		sampler = make_sampler(temperature, top_p, min_p, min_tokens_to_keep=min_tokens_to_keep, top_k=top_k)
		logits_processors = (make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size) if repetition_penalty else None)

		total_prompt_tokens = 0
		total_generation_tokens = 0
		remaining_tokens = max_tokens
		language_accumulator = ""

		chunk_iter = tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks", disable=not verbose or len(chunks) == 1)
		for chunk_idx, (chunk_audio, offset_sec) in chunk_iter:
			actual_chunk_duration = len(chunk_audio) / self.sample_rate
			is_last_chunk = chunk_idx == len(chunks) - 1
			token_count = 0

			_, _, num_audio_tokens = self._preprocess_audio(chunk_audio)
			input_ids = self._build_prompt(num_audio_tokens, language, system_prompt)
			chunk_prompt_tokens = input_ids.shape[1]
			total_prompt_tokens += chunk_prompt_tokens

			for i, (token, _) in enumerate(self.stream_generate(chunk_audio, max_tokens=remaining_tokens, sampler=sampler, logits_processors=logits_processors, language=language, prefill_step_size=prefill_step_size, verbose=verbose and len(chunks) == 1, system_prompt=system_prompt)):
				text = self._tokenizer.decode([int(token)])

				if language is None and i <= 2:
					language_accumulator += text
					if "<asr_text>" in language_accumulator:
						language, _ = self.extract_language(language_accumulator)
					continue

				prev_progress = token_count / max(remaining_tokens, 1)
				token_count += 1
				curr_progress = min(token_count / max(remaining_tokens, 1), 1.0)

				estimated_start = offset_sec + (actual_chunk_duration * prev_progress)
				estimated_end = offset_sec + (actual_chunk_duration * curr_progress)

				yield StreamingResult(text=text, is_final=False, start_time=estimated_start, end_time=estimated_end, language=language)

			total_generation_tokens += token_count
			remaining_tokens -= token_count

			yield StreamingResult(text="", is_final=is_last_chunk or remaining_tokens <= 0, start_time=offset_sec, end_time=offset_sec + actual_chunk_duration, language=language, prompt_tokens=total_prompt_tokens, generation_tokens=total_generation_tokens)

			if remaining_tokens <= 0:
				break


Model = Qwen3ASRModel
