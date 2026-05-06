import argparse
import contextlib
import functools
import time
import mlx.core as mx
from mlx.utils import tree_reduce
from .cache import make_prompt_cache, load_prompt_cache, save_prompt_cache, trim_cache
from .utils import StoppingCriteria, apply_repetition_penalty, prepare_inputs

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_QUANTIZED_KV_START = 5000


def parse_arguments():
	parser = argparse.ArgumentParser(description="Generate text from an image using a model.")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="The path to the local model directory.")
	parser.add_argument("--adapter-path", type=str, default=None, help="The path to the adapter weights.")
	parser.add_argument("--image", type=str, nargs="+", default=DEFAULT_IMAGE, help="URL or path of the image to process.")
	parser.add_argument("--audio", type=str, nargs="+", default=DEFAULT_AUDIO, help="URL or path of the audio to process.")
	parser.add_argument("--resize-shape", type=int, nargs="+", default=None, help="Resize shape for the image.")
	parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Message to be processed by the model.")
	parser.add_argument("--system", type=str, default=None, help="System message for the model.")
	parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum number of tokens to generate.")
	parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for sampling.")
	parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top P for sampling.")
	parser.add_argument("--mirostat-tau", type=float, default=0.0, help="Mirostat target entropy (tau). 0 disables.")
	parser.add_argument("--mirostat-eta", type=float, default=0.1, help="Mirostat learning rate.")
	parser.add_argument("--dynamic-temp-min", type=float, default=None, help="Dynamic temperature minimum.")
	parser.add_argument("--dynamic-temp-max", type=float, default=None, help="Dynamic temperature maximum.")
	parser.add_argument("--logit-noise", type=float, default=0.0, help="Amount of Gaussian noise added to logits.")
	parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
	parser.add_argument("--verbose", action="store_false", help="Detailed output.")
	parser.add_argument("--eos-tokens", type=str, nargs="+", default=None, help="EOS tokens to add to the tokenizer.")
	parser.add_argument("--stop-strings", type=str, nargs="+", default=None, help="A list of strings to stop generation on.")
	parser.add_argument("--max-kv-size", type=int, default=None, help="Maximum KV size for the prompt cache.")
	parser.add_argument("--kv-bits", type=int, default=None, help="Number of bits to quantize the KV cache to.")
	parser.add_argument("--kv-group-size", type=int, default=64, help="Group size for the KV cache.")
	parser.add_argument("--quantized-kv-start", type=int, default=DEFAULT_QUANTIZED_KV_START, help="Start index for the quantized KV cache.")
	parser.add_argument("--skip-special-tokens", action="store_true", help="Skip special tokens in the detokenizer.")
	parser.add_argument("--force-download", action="store_true", help="Force download the model (deprecated in local struct).")
	parser.add_argument("--cache-file", type=str, default=None, help="Path to save/load KV cache.")
	return parser.parse_args()


generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model, streams=None):
	if not mx.metal.is_available():
		try:
			yield
		finally:
			return

	model_bytes = tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0)
	max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
	if model_bytes > 0.9 * max_rec_size:
		model_mb = model_bytes // 2**20
		max_rec_mb = max_rec_size // 2**20
		print(f"[WARNING] Generating with a model that requires {model_mb} MB "
		      f"which is close to the maximum recommended size of {max_rec_mb} "
		      "MB. This can be slow.")
	old_limit = mx.set_wired_limit(max_rec_size)
	try:
		yield
	finally:
		if streams is not None:
			for s in streams:
				mx.synchronize(s)
		else:
			mx.synchronize()
		mx.set_wired_limit(old_limit)


class GenerationResult:
	def __init__(self, text="", token=None, logprobs=None, prompt_tokens=0, generation_tokens=0, total_tokens=0, prompt_tps=0.0, generation_tps=0.0, peak_memory=0.0):
		self.text = text
		self.token = token
		self.logprobs = logprobs
		self.prompt_tokens = prompt_tokens
		self.generation_tokens = generation_tokens
		self.total_tokens = total_tokens
		self.prompt_tps = prompt_tps
		self.generation_tps = generation_tps
		self.peak_memory = peak_memory


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
	if kv_bits is None:
		return
	for e, c in enumerate(prompt_cache):
		if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
			prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def top_p_sampling(logits, top_p, temperature):
	if (logits.dtype == mx.bfloat16):
		logits = logits.astype(mx.float32)

	probs = mx.softmax(logits / temperature, axis=-1)
	sorted_indices = mx.argsort(probs, axis=-1)
	sorted_probs = probs[..., sorted_indices.squeeze(0)]
	cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
	top_probs = mx.where(cumulative_probs > 1 - top_p, sorted_probs, mx.zeros_like(sorted_probs))
	sorted_token = mx.random.categorical(mx.log(top_probs))
	token = sorted_indices.squeeze(0)[sorted_token]
	return token


def generate_step(input_ids, model, pixel_values, mask, max_tokens=256, temperature=0.0, repetition_penalty=None, repetition_context_size=20, top_p=1.0, logit_bias=None, prompt_cache=None, max_kv_size=None, kv_bits=None, kv_group_size=64, quantized_kv_start=0, num_candidates=1, debug_candidates=False, candidate_index=0, candidate_min_prob=0.05, **kwargs):
	quantize_cache_fn = functools.partial(maybe_quantize_kv_cache, quantized_kv_start=quantized_kv_start, kv_group_size=kv_group_size, kv_bits=kv_bits)

	mirostat_tau = kwargs.get("mirostat_tau", 0.0)
	mirostat_eta = kwargs.get("mirostat_eta", 0.1)
	logit_noise = kwargs.get("logit_noise", 0.0)
	dynamic_temp_min = kwargs.get("dynamic_temp_min", None)
	dynamic_temp_max = kwargs.get("dynamic_temp_max", None)

	mu = 2.0 * mirostat_tau if mirostat_tau > 0 else 0.0

	def sample(logits):
		nonlocal mu
		
		if logit_noise > 0.0:
			noise = mx.random.normal(logits.shape, dtype=logits.dtype) * logit_noise
			logits = logits + noise

		if logit_bias:
			indices = mx.array(list(logit_bias.keys()))
			values = mx.array(list(logit_bias.values()))
			logits[:, indices] += values
		logprobs = logits - mx.logsumexp(logits)

		cur_temperature = temperature
		if dynamic_temp_min is not None and dynamic_temp_max is not None:
			probs = mx.exp(logprobs)
			entropy = -mx.sum(probs * logprobs, axis=-1).item()
			max_entropy = mx.log(mx.array(logits.shape[-1], dtype=logits.dtype)).item()
			entropy_ratio = min(1.0, entropy / (max_entropy * 0.5))
			cur_temperature = dynamic_temp_max - entropy_ratio * (dynamic_temp_max - dynamic_temp_min)
			cur_temperature = max(dynamic_temp_min, min(dynamic_temp_max, cur_temperature))

		top_candidates = None
		if debug_candidates or num_candidates > 1 or candidate_index > 0:
			top_k = max(num_candidates, candidate_index + 1)
			top_k = min(top_k, logits.shape[-1])
			sorted_indices = mx.argsort(logprobs[0], axis=-1)[::-1]
			top_indices = sorted_indices[:top_k]
			top_logprobs = logprobs[0][top_indices]
			top_candidates = (top_indices, top_logprobs)

		selected_index = candidate_index
		if candidate_index > 0 and top_candidates is not None:
			candidate_logprob = float(top_candidates[1][candidate_index])
			candidate_prob = mx.exp(candidate_logprob).item()
			if candidate_prob < candidate_min_prob:
				selected_index = 0
				if debug_candidates:
					print(f"  [FALLBACK] Candidate {candidate_index} prob={candidate_prob:.4f} < {candidate_min_prob}, selecting rank 1 instead")
			token = top_candidates[0][selected_index:selected_index + 1]
		elif mirostat_tau > 0:
			probs = mx.softmax(logits, axis=-1)
			sorted_indices = mx.argsort(probs, axis=-1)[..., ::-1]
			sorted_probs = probs[..., sorted_indices.squeeze(0)]
			
			surprise = -mx.log2(sorted_probs + 1e-10)
			mask = surprise <= mu
			mask = mx.logical_or(mask, mx.arange(mask.shape[-1]) == 0)
			
			top_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
			sorted_token_idx = mx.random.categorical(mx.log(top_probs + 1e-10))
			token = sorted_indices.squeeze(0)[sorted_token_idx]
			token = token.reshape(1)
			
			p_chosen = sorted_probs.squeeze(0)[sorted_token_idx]
			observed_surprise = -mx.log2(p_chosen + 1e-10).item()
			mu = mu - mirostat_eta * (observed_surprise - mirostat_tau)
		elif cur_temperature == 0:
			token = mx.argmax(logits, axis=-1)
		else:
			if top_p > 0 and top_p < 1.0:
				token = top_p_sampling(logits, top_p, cur_temperature)
			else:
				token = mx.random.categorical(logits * (1 / cur_temperature))
		return token, logprobs, top_candidates

	if repetition_penalty and (repetition_penalty < 0 or not isinstance(repetition_penalty, float)):
		raise ValueError(f"repetition_penalty must be a non-negative float, got {repetition_penalty}")

	y = input_ids
	if prompt_cache is None:
		prompt_cache = make_prompt_cache(model.language_model, max_kv_size=max_kv_size)

	repetition_context = input_ids.reshape(-1).tolist()
	if repetition_context_size:
		repetition_context = repetition_context[-repetition_context_size:]

	def _step(y, **kwargs):
		with mx.stream(generation_stream):
			nonlocal repetition_context
			if "decoder_input_ids" in kwargs:
				outputs = model.language_model(cache=prompt_cache, **kwargs)
			else:
				outputs = model.language_model(y[None], cache=prompt_cache, **kwargs)

			logits = outputs.logits[:, -1, :]

			if repetition_penalty:
				logits = apply_repetition_penalty(logits, repetition_context, repetition_penalty)
				y, logprobs, top_candidates = sample(logits)
				repetition_context.append(y.item())
			else:
				y, logprobs, top_candidates = sample(logits)

			if repetition_context_size:
				if len(repetition_context) > repetition_context_size:
					repetition_context = repetition_context[-repetition_context_size:]

			quantize_cache_fn(prompt_cache)
			return y, logprobs.squeeze(0), top_candidates

	outputs = model(input_ids, pixel_values, cache=prompt_cache, mask=mask, **kwargs)
	logits = outputs.logits[:, -1, :]
	quantize_cache_fn(prompt_cache)
	y, logprobs, top_candidates = sample(logits)
	mx.async_eval(y)

	if debug_candidates and top_candidates is not None:
		print(f"\n[Initial token candidates]")
		token_ids = top_candidates[0].tolist()
		logprob_vals = top_candidates[1].tolist()
		for i, (tok_id, logprob) in enumerate(zip(token_ids, logprob_vals)):
			prob = mx.exp(logprob).item()
			selected = " <-- SELECTED" if tok_id == y.item() else ""
			print(f"  Rank {i + 1}: token_id={tok_id}, logprob={logprob:.4f}, prob={prob:.4f}{selected}")

	if outputs.cross_attention_states is not None:
		kwargs = {k: v for k, v in zip(["cross_attention_states"], [outputs.cross_attention_states])}
	elif outputs.encoder_outputs is not None:
		kwargs = {"decoder_input_ids": y[None], "encoder_outputs": outputs.encoder_outputs}
	else:
		kwargs = {}

	n = 0
	while True:
		if n != max_tokens:
			next_y, next_logprobs, next_candidates = _step(y, **kwargs)
			mx.async_eval(next_y)

			if debug_candidates and next_candidates is not None:
				print(f"\n[Step {n + 1} token candidates]")
				token_ids = next_candidates[0].tolist()
				logprob_vals = next_candidates[1].tolist()
				for i, (tok_id, logprob) in enumerate(zip(token_ids, logprob_vals)):
					prob = mx.exp(logprob).item()
					selected = " <-- SELECTED" if tok_id == next_y.item() else ""
					print(f"  Rank {i + 1}: token_id={tok_id}, logprob={logprob:.4f}, prob={prob:.4f}{selected}")

			if "decoder_input_ids" in kwargs:
				kwargs["decoder_input_ids"] = next_y[None]
			yield y.item(), logprobs
			y, logprobs = next_y, next_logprobs
		if n == max_tokens:
			break

		n += 1
		if n % 256 == 0:
			mx.clear_cache()


def stream_generate(model, processor, prompt, image=None, audio=None, stop_strings=None, cache_file=None, **kwargs):
	tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

	stop_sequences = []
	if stop_strings:
		stop_sequences = ([stop_strings] if isinstance(stop_strings, str) else stop_strings)
	max_stop_len = max(len(s) for s in stop_sequences) if stop_sequences else 0

	skip_special_tokens = kwargs.pop("skip_special_tokens", False)
	skip_special_token_ids = (set(tokenizer.all_special_ids) if skip_special_tokens and hasattr(tokenizer, "all_special_ids") else [])

	add_special_tokens = (not hasattr(processor, "chat_template") if model.config.model_type in ["gemma3", "gemma3n"] else True)

	resize_shape = kwargs.pop("resize_shape", None)
	image_token_index = getattr(model.config, "image_token_index", None)

	num_candidates = kwargs.pop("num_candidates", 1)
	debug_candidates = kwargs.pop("debug_candidates", False)
	candidate_index = kwargs.pop("candidate_index", 0)
	candidate_min_prob = kwargs.pop("candidate_min_prob", 0.05)

	if kwargs.get("input_ids", None) is not None:
		input_ids = kwargs.pop("input_ids")
		pixel_values = kwargs.pop("pixel_values", None)
		mask = kwargs.pop("mask", None)
	else:
		inputs = prepare_inputs(processor, images=image, audio=audio, prompts=prompt, image_token_index=image_token_index, resize_shape=resize_shape, add_special_tokens=add_special_tokens)
		input_ids = inputs.get("input_ids", None)
		pixel_values = inputs.get("pixel_values", None)
		mask = inputs.get("attention_mask", None)
		data_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}
		kwargs.update(data_kwargs)

	prompt_cache = None
	full_input_ids = input_ids

	if cache_file:
		loaded_cache, cached_tokens = load_prompt_cache(cache_file, model.language_model)

		if loaded_cache is not None:
			curr_tokens = input_ids[0].tolist()
			common_len = 0
			min_len = min(len(curr_tokens), len(cached_tokens))
			for i in range(min_len):
				if curr_tokens[i] == cached_tokens[i]:
					common_len += 1
				else:
					break

			if common_len > 0:
				trim_cache(loaded_cache, common_len)
				prompt_cache = loaded_cache

				if common_len < len(curr_tokens):
					input_ids = input_ids[:, common_len:]

				if image_token_index is not None and pixel_values is not None:
					cached_part = curr_tokens[:common_len]
					if image_token_index in cached_part:
						count_total = curr_tokens.count(image_token_index)
						count_cached = cached_part.count(image_token_index)
						if count_cached == count_total:
							pixel_values = None
							if "image_grid_thw" in kwargs:
								del kwargs["image_grid_thw"]
							if "video_grid_thw" in kwargs:
								del kwargs["video_grid_thw"]

		if prompt_cache is None:
			prompt_cache = make_prompt_cache(model.language_model, kwargs.get("max_kv_size"))

	yielded_chars = 0
	generated_tokens_list = []

	with wired_limit(model, [generation_stream]):
		detokenizer = processor.detokenizer
		detokenizer.reset()
		tic = time.perf_counter()

		for n, (token, logprobs) in enumerate(generate_step(input_ids, model, pixel_values, mask, prompt_cache=prompt_cache, num_candidates=num_candidates, debug_candidates=debug_candidates, candidate_index=candidate_index, candidate_min_prob=candidate_min_prob, **kwargs, )):
			generated_tokens_list.append(token)

			if n == 0:
				prompt_time = time.perf_counter() - tic
				prompt_tps = input_ids.size / prompt_time
				tic = time.perf_counter()

			if tokenizer.stopping_criteria(token):
				break

			detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
			full_text = detokenizer.text

			stop_found = False
			stop_index = -1
			if stop_sequences:
				for seq in stop_sequences:
					idx = full_text.find(seq)
					if idx != -1:
						if stop_index == -1 or idx < stop_index:
							stop_index = idx
							stop_found = True

			segment_to_yield = ""
			if stop_found:
				if stop_index > yielded_chars:
					segment_to_yield = full_text[yielded_chars:stop_index]
				yield GenerationResult(text=segment_to_yield, token=token, logprobs=logprobs, prompt_tokens=input_ids.size, generation_tokens=n + 1, total_tokens=input_ids.size + n + 1, prompt_tps=prompt_tps, generation_tps=(n + 1) / (time.perf_counter() - tic), peak_memory=mx.get_peak_memory() / 1e9)
				break
			else:
				safe_length = len(full_text) - max_stop_len
				if safe_length > yielded_chars:
					segment_to_yield = full_text[yielded_chars:safe_length]
					yielded_chars = safe_length
					yield GenerationResult(text=segment_to_yield, token=token, logprobs=logprobs, prompt_tokens=input_ids.size, generation_tokens=n + 1, total_tokens=input_ids.size + n + 1, prompt_tps=prompt_tps, generation_tps=(n + 1) / (time.perf_counter() - tic), peak_memory=mx.get_peak_memory() / 1e9)

		detokenizer.finalize()
		full_text = detokenizer.text
		if yielded_chars < len(full_text) and not stop_found:
			final_segment = full_text[yielded_chars:]
			if stop_sequences:
				for seq in stop_sequences:
					if seq in final_segment:
						final_segment = final_segment.split(seq)[0]
			if final_segment:
				yield GenerationResult(text=final_segment, token=token, logprobs=logprobs, prompt_tokens=input_ids.size, generation_tokens=n + 1, total_tokens=input_ids.size + n + 1, prompt_tps=prompt_tps, generation_tps=(n + 1) / (time.perf_counter() - tic), peak_memory=mx.get_peak_memory() / 1e9)

		if cache_file and prompt_cache is not None:
			final_ids = full_input_ids[0].tolist() + generated_tokens_list
			save_prompt_cache(prompt_cache, cache_file, final_ids)

		mx.clear_cache()


def generate(model, processor, prompt, image=None, audio=None, verbose=False, stop_strings=None, cache_file=None, **kwargs):
	if verbose:
		print("=" * 10)
		files = []
		if image is not None:
			files.extend(image)
		if audio is not None:
			files.extend(audio)
		if kwargs.get("video") is not None:
			files.extend(kwargs.get("video"))
		print(f"Files: {files}", "\n")
		print("Prompt:", prompt)

	text = ""
	last_response = None

	eos_tokens = kwargs.get("eos_tokens", None)
	stopping_criteria = kwargs.get("stopping_criteria", None)
	tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

	if eos_tokens is not None:
		tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)
	elif stopping_criteria is not None:
		if isinstance(stopping_criteria, StoppingCriteria) or callable(stopping_criteria):
			tokenizer.stopping_criteria = stopping_criteria
		else:
			raise ValueError("stopping_criteria must be an instance of StoppingCriteria or a callable")
	else:
		tokenizer.stopping_criteria.reset(model.config.eos_token_id)

	generator = stream_generate(model, processor, prompt, image, audio, stop_strings=stop_strings, cache_file=cache_file, **kwargs)

	for response in generator:
		if verbose:
			print(response.text, end="", flush=True)
		text += response.text
		last_response = response

	if verbose:
		print("\n" + "=" * 10)

	if last_response is None or len(text) == 0:
		if verbose:
			print("No text generated for this prompt")
		return GenerationResult(text=text, token=None, logprobs=None, prompt_tokens=0, generation_tokens=0, total_tokens=0, prompt_tps=0.0, generation_tps=0.0, peak_memory=mx.get_peak_memory() / 1e9)

	return GenerationResult(text=text, token=last_response.token, logprobs=last_response.logprobs, prompt_tokens=last_response.prompt_tokens, generation_tokens=last_response.generation_tokens, total_tokens=last_response.total_tokens, prompt_tps=last_response.prompt_tps, generation_tps=last_response.generation_tps, peak_memory=last_response.peak_memory)
