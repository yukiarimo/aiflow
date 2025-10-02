import argparse
import time
from typing import Generator, List, Optional
import mlx.core as mx
from PIL import Image
from .utils import load
from .models.yuna.cache import KVCache

def apply_repetition_penalty(logits: mx.array, generated_tokens: List[int], penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.
    """
    if not generated_tokens:
        return logits
    
    vocab_size = logits.shape[-1]
    valid_tokens = [t for t in generated_tokens if 0 <= t < vocab_size]
    if not valid_tokens:
        return logits

    indices = mx.array(valid_tokens)
    
    selected_logits = logits[:, indices]
    selected_logits = mx.where(
        selected_logits < 0, selected_logits * penalty, selected_logits / penalty
    )
    logits[:, indices] = selected_logits
    return logits

def apply_negative_prompt(logits: mx.array, negative_prompt_ids: set):
    """
    Penalize tokens from the negative prompt by setting their logits to -inf.
    """
    if not negative_prompt_ids:
        return logits
    
    indices_to_penalize = list(negative_prompt_ids)
    if indices_to_penalize:
        logits[:, indices_to_penalize] = -mx.inf
    return logits

def sampler(logits: mx.array, temp: float, top_p: float, top_k: int):
    """
    Apply temperature, top-p, and top-k sampling to logits.
    """
    if temp == 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temp

    if 0 < top_k < logits.shape[-1]:
        kth_vals = -mx.partition(-logits, top_k - 1, axis=-1)[..., top_k - 1 : top_k]
        logits = mx.where(logits < kth_vals, -mx.inf, logits)

    # --- REVISED AND FIXED top_p SAMPLING LOGIC ---
    if 0 < top_p < 1.0:
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find the first index where cumulative probability exceeds top_p
        cutoff_index = mx.argmax(cumulative_probs > top_p, axis=-1)

        # Get the probability threshold from that index for each item in the batch
        prob_threshold = mx.take_along_axis(sorted_probs, cutoff_index[:, None], axis=-1)

        # Create a mask for tokens with probabilities greater than or equal to the threshold
        mask = probs >= prob_threshold

        # Apply the mask to the original logits
        logits = mx.where(mask, logits, -mx.inf)
    
    return mx.random.categorical(logits)


def stream_generate(
    model,
    processor,
    prompt,
    image,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    repetition_context_size,
    eos_tokens,
    skip_special_tokens,
    negative_prompt: Optional[str] = None,
    prefill_step_size=2048,
    ) -> Generator[str, None, None]:
    
    if image:
        image = Image.open(image)
        inputs = processor(prompt, images=[image])
    else:
        inputs = processor(prompt)

    cache = [KVCache() for _ in model.language_model.layers]
    
    negative_prompt_ids = set(processor.tokenizer.encode(negative_prompt).ids) if negative_prompt else set()

    prompt_tokens = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    d_image = inputs.get("d_image")
    
    while prompt_tokens.size > 0:
        chunk = prompt_tokens[:, :prefill_step_size]
        prompt_tokens = prompt_tokens[:, prefill_step_size:]

        logits, cache = model(
            chunk,
            pixel_values=pixel_values,
            d_image=d_image,
            cache=cache
        )
        pixel_values, d_image = None, None
        mx.eval(cache[0].keys)

    y = logits[:, -1, :]
    
    detokenizer = processor.tokenizer.detokenizer
    generated_token_ids = []
    
    skip_ids = set(processor.tokenizer.all_special_ids) if skip_special_tokens else set()

    for i in range(max_tokens):
        if repetition_penalty is not None and repetition_penalty > 1.0:
            context = generated_token_ids[-repetition_context_size:]
            y = apply_repetition_penalty(y, context, repetition_penalty)

        y = apply_negative_prompt(y, negative_prompt_ids)

        y = sampler(y, temperature, top_p, top_k)
        token_id = y.item()

        if token_id == model.config.eos_token_id:
            break

        generated_token_ids.append(token_id)
        detokenizer.add_token(token_id, skip_special_token_ids=skip_ids)

        current_text = detokenizer.text
        new_segment = detokenizer.last_segment
        if new_segment:
            if eos_tokens:
                for stop_str in eos_tokens:
                    idx = current_text.find(stop_str)
                    if idx != -1:
                        to_yield = current_text[:idx]
                        if to_yield:
                            yield to_yield[len(current_text) - len(new_segment):]
                        return
            yield new_segment

        logits, cache = model(y[:, None], cache=cache)
        y = logits[:, -1, :]

    detokenizer.finalize()
    final_text = detokenizer.last_segment
    if eos_tokens and final_text:
        for stop_str in eos_tokens:
            idx = final_text.find(stop_str)
            if idx != -1:
                final_text = final_text[:idx]
                break
    if final_text:
        yield final_text

def main():
    parser = argparse.ArgumentParser(description="Yuna model generation script.")
    parser.add_argument("--model", type=str, required=True, help="Path to the MLX model directory.")
    parser.add_argument("--prompt", type=str, default="hello", help="The text prompt (user part of chat).")
    parser.add_argument("--image", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling (disabled if -1).")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalty for repeating tokens.")
    parser.add_argument("--repetition-context-size", type=int, default=128, help="Context for repetition penalty.")
    parser.add_argument("--eos-tokens", type=str, nargs='+', help="List of strings to stop generation.")
    parser.add_argument("--skip-special-tokens", action="store_true", help="Skip special tokens in output.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="A string of words to penalize during generation.")
    
    args = parser.parse_args()

    print("[INFO] Loading model and processor...")
    model, processor = load(args.model)
    print("[INFO] Model loaded.")

    full_prompt = args.prompt

    print("="*20)
    print("Prompt:", full_prompt)
    if args.negative_prompt:
        print("Negative Prompt:", args.negative_prompt)
    print("Response: ", end="")


    start_time = time.time()
    for chunk in stream_generate(
        model=model,
        processor=processor,
        prompt=full_prompt,
        image=args.image,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
        eos_tokens=args.eos_tokens,
        skip_special_tokens=args.skip_special_tokens,
        negative_prompt=args.negative_prompt,
    ):
        print(chunk, end="", flush=True)

    end_time = time.time()
    print("\n" + "="*20)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()