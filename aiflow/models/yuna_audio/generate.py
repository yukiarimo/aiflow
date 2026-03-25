import contextlib
import inspect
import json
import os
import sys
import time
from pprint import pprint
import mlx.core as mx
from mlx.utils import tree_reduce
from .utils import load_model
from .config import STTOutput


def format_timestamp(seconds):
	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	seconds = seconds % 60
	return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def format_vtt_timestamp(seconds):
	return format_timestamp(seconds).replace(",", ".")


def save_as_txt(segments, output_path):
	with (open(f"{output_path}.txt", "w", encoding="utf-8") if output_path != "-" else contextlib.nullcontext(sys.stdout)) as f:
		f.write(segments.text)


def save_as_srt(segments, output_path):
	with (open(f"{output_path}.srt", "w", encoding="utf-8") if output_path != "-" else contextlib.nullcontext(sys.stdout)) as f:
		if hasattr(segments, "sentences"):
			for i, sentence in enumerate(segments.sentences, 1):
				f.write(f"{i}\n")
				f.write(f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}\n")
				f.write(f"{sentence.text}\n\n")
		else:
			for i, segment in enumerate(segments.segments, 1):
				f.write(f"{i}\n")
				f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
				f.write(f"{segment['text']}\n\n")


def save_as_vtt(segments, output_path):
	with (open(f"{output_path}.vtt", "w", encoding="utf-8") if output_path != "-" else contextlib.nullcontext(sys.stdout)) as f:
		f.write("WEBVTT\n\n")
		if hasattr(segments, "sentences"):
			sentences = segments.sentences
			for i, sentence in enumerate(sentences, 1):
				f.write(f"{i}\n")
				f.write(f"{format_vtt_timestamp(sentence.start)} --> {format_vtt_timestamp(sentence.end)}\n")
				f.write(f"{sentence.text}\n\n")
		else:
			sentences = segments.segments
			for i, token in enumerate(sentences, 1):
				f.write(f"{i}\n")
				f.write(f"{format_vtt_timestamp(token['start'])} --> {format_vtt_timestamp(token['end'])}\n")
				f.write(f"{token['text']}\n\n")


def save_as_json(segments, output_path):
	if hasattr(segments, "sentences"):
		result = {"text": segments.text, "sentences": [{"text": s.text, "start": s.start, "end": s.end, "duration": s.duration, "tokens": [{"text": t.text, "start": t.start, "end": t.end, "duration": t.duration, } for t in s.tokens], } for s in segments.sentences], }
		for i, s in enumerate(segments.sentences):
			if hasattr(s, "speaker_id"):
				result["sentences"][i]["speaker_id"] = s.speaker_id
	else:
		result = {"text": segments.text, "segments": [{"text": s["text"], "start": s["start"], "end": s["end"], "duration": s["end"] - s["start"], } for s in segments.segments], }
		for i, s in enumerate(segments.segments):
			if "speaker_id" in s:
				result["segments"][i]["speaker_id"] = s["speaker_id"]

	with (open(f"{output_path}.json", "w", encoding="utf-8") if output_path != "-" else contextlib.nullcontext(sys.stdout)) as f:
		json.dump(result, f, ensure_ascii=False, indent=2)


generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model, streams=None):
	if not mx.metal.is_available():
		try:
			yield
		finally:
			pass
	else:
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


def generate_transcription(model=None, audio=None, output_path="transcript", format="txt", verbose=False, text="", **kwargs):
	if model is None:
		raise ValueError("Model path or model instance must be provided.")

	if isinstance(model, str):
		model = load_model(model)

	mx.reset_peak_memory()
	start_time = time.time()
	if verbose:
		print("=" * 10)
		print(f"\033[94mAudio path:\033[0m {audio}")
		print(f"\033[94mOutput path:\033[0m {output_path}")
		print(f"\033[94mFormat:\033[0m {format}")

	gen_kwargs = kwargs.pop("gen_kwargs", None)
	if gen_kwargs:
		kwargs.update(gen_kwargs)

	if text:
		kwargs["text"] = text

	signature = inspect.signature(model.generate)
	kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

	if kwargs.get("stream", False):
		all_segments = []
		accumulated_text = ""
		language = "en"
		prompt_tokens = 0
		generation_tokens = 0
		for result in model.generate(audio, verbose=verbose, **kwargs):
			segment_dict = {"text": result.text, "start": result.start_time, "end": result.end_time, "is_final": result.is_final}

			all_segments.append(segment_dict)
			accumulated_text += result.text
			language = result.language

			if hasattr(result, "prompt_tokens") and result.prompt_tokens > 0:
				prompt_tokens = result.prompt_tokens
			if hasattr(result, "generation_tokens") and result.generation_tokens > 0:
				generation_tokens = result.generation_tokens

		stream_end_time = time.time()
		stream_duration = stream_end_time - start_time
		segments = STTOutput(text=accumulated_text.strip(), segments=all_segments, language=language, prompt_tokens=prompt_tokens, generation_tokens=generation_tokens, total_tokens=prompt_tokens + generation_tokens, total_time=stream_duration, prompt_tps=prompt_tokens / stream_duration if stream_duration > 0 else 0, generation_tps=(generation_tokens / stream_duration if stream_duration > 0 else 0))
	else:
		segments = model.generate(audio, verbose=verbose, generation_stream=generation_stream, **kwargs)

	if verbose:
		if hasattr(segments, "text"):
			print("\033[94mTranscription:\033[0m\n")
			print(f"{segments.text[:500]}...\n")

		if hasattr(segments, "segments") and segments.segments is not None:
			print("\033[94mSegments:\033[0m\n")
			pprint(segments.segments[:3] + ["..."])

	end_time = time.time()

	if verbose:
		print("\n" + "=" * 10)
		print(f"\033[94mSaving file to:\033[0m ./{output_path}.{format}")
		print(f"\033[94mProcessing time:\033[0m {end_time - start_time:.2f} seconds")

		if isinstance(segments, STTOutput):
			print(f"\033[94mPrompt:\033[0m {segments.prompt_tokens} tokens, "
			      f"{segments.prompt_tps:.3f} tokens-per-sec")
			print(f"\033[94mGeneration:\033[0m {segments.generation_tokens} tokens, "
			      f"{segments.generation_tps:.3f} tokens-per-sec")
		print(f"\033[94mPeak memory:\033[0m {mx.get_peak_memory() / 1e9:.2f} GB")

	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
	has_segments = hasattr(segments, "segments") and segments.segments is not None
	has_sentences = hasattr(segments, "sentences") and segments.sentences is not None

	if format == "txt" or (not has_segments and not has_sentences):
		if not has_segments and not has_sentences:
			print("[WARNING] No segments found, saving as plain text.")
		save_as_txt(segments, output_path)
	elif format == "srt":
		save_as_srt(segments, output_path)
	elif format == "vtt":
		save_as_vtt(segments, output_path)
	elif format == "json":
		save_as_json(segments, output_path)

	return segments
