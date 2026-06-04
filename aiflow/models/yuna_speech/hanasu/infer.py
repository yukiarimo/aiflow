import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from aiflow.models.yuna_speech.text import split_and_process_text
from .models import HanasuTTS
from .utils import PhonemeTokenizer, count_parameters, format_param_count, get_device, json_safe, load_checkpoint, load_config, load_vocos_vocoder, load_wav, phonemize_text, save_wav, wav_to_model_mel


@dataclass
class PromptSpec:
	mel: torch.Tensor
	prefix_token_len: int
	prefix_frames: int
	prompt_phonemes: str
	target_phonemes: str


def load_model(config, checkpoint, device, metadata_path=None):
	ckpt = load_checkpoint(checkpoint, map_location=device, metadata_path=metadata_path)
	ckpt_config = ckpt.get("config")
	if isinstance(ckpt_config, dict):
		config.clear()
		config.update(ckpt_config)
	tokenizer_data = ckpt.get("tokenizer")
	if not tokenizer_data:
		raise RuntimeError("Checkpoint has no tokenizer; cannot reconstruct the phoneme vocabulary.")
	tokenizer = PhonemeTokenizer.from_dict(tokenizer_data)
	if not config.get("audio", {}).get("mel_stats"):
		print("Warning: no audio.mel_stats found; mel denormalization may be wrong and the vocoder output degraded.")
	model = HanasuTTS(vocab_size=tokenizer.vocab_size, n_mels=int(config["audio"]["n_mels"]), config=config["model"], pad_id=tokenizer.pad_id, ).to(device)
	model.load_state_dict(ckpt["model"])
	model.eval()
	print(f"Loaded checkpoint: {checkpoint}")
	print(f"Acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
	return model, tokenizer


def resolve_id(value, name_map, kind):
	"""Resolve a speaker/language given as a name (e.g. 'Yuna', 'en-us') or an integer id."""
	if value is None:
		return 0
	if isinstance(value, str) and value in name_map:
		return int(name_map[value])
	try:
		return int(value)
	except (TypeError, ValueError):
		raise ValueError(f"Unknown {kind} '{value}'. Known: {sorted(name_map) if name_map else 'use an integer id'}")


def resolve_language_name(value, languages_map, language_id):
	"""Get the espeak language code (e.g. 'en-us') for phonemization from a name or id."""
	if isinstance(value, str) and value in languages_map:
		return value
	for name, idx in languages_map.items():
		if int(idx) == int(language_id):
			return name
	return "en-us"


def denormalize_mel(mel, config):
	mel_stats = config.get("audio", {}).get("mel_stats")
	if not mel_stats or not mel_stats.get("normalized_training", False):
		return mel
	mean = float(mel_stats["mean"])
	std = max(float(mel_stats["std"]), 1e-5)
	mel = mel * std + mean
	return mel.clamp(float(mel_stats.get("min", -14.0)), float(mel_stats.get("max", 4.0)))


def build_language_map(languages_map):
	if languages_map:
		return {name: name for name in languages_map}
	return {"en-us": "en-us", "ru": "ru", "ja": "ja"}


def split_phoneme_text(text):
	"""Split already-phonemized text into one chunk per sentence (. ? !)."""
	text = text.strip()
	sentences = re.split(r"([.!?]+)(?:\s+|$)", text)
	result = []
	for i in range(0, len(sentences) - 1, 2):
		sentence = sentences[i].strip()
		if sentence:
			result.append(f"{sentence}{sentences[i + 1]}")
	if len(sentences) % 2 == 1 and sentences[-1].strip():
		result.append(sentences[-1].strip())
	return result if result else ([text] if text else [])


def prepare_text_chunks(text, language_name, languages_map, phonemized=False):
	"""Split text into one synthesis chunk per sentence (. ? !)."""
	if phonemized:
		return [{"text": chunk, "language": language_name} for chunk in split_phoneme_text(text) if chunk.strip()]
	language_map = build_language_map(languages_map)
	return split_and_process_text(text, language=language_name, combine=False, language_map=language_map)


def resolve_chunk_language_id(chunk_language, languages_map, default_language_id):
	if chunk_language in languages_map:
		return int(languages_map[chunk_language])
	return int(default_language_id)


@torch.no_grad()
def synthesize_long_text(config, model, tokenizer, text, device, vocoder, speaker_id=0, language_id=0, language_name="en-us", languages_map=None, phonemized=False, max_steps=None, stop_threshold=None, attention_window=None, normalize_wav=False, debug=False, use_postnet=True, prompt=None, include_prompt_in_output=True, use_speaker_embedding=True, output=None, ):
	"""Synthesize long text one sentence at a time and concatenate the audio."""
	if prompt is not None:
		raise ValueError("Prompt conditioning does not support chunked synthesis")

	languages_map = languages_map or {}
	chunks = prepare_text_chunks(text, language_name, languages_map, phonemized=phonemized)
	if not chunks:
		raise ValueError("No text chunks to synthesize")

	if len(chunks) == 1:
		chunk = chunks[0]
		print(f"Phonemized [{chunk['language']}]: {chunk['text']}")
	elif len(chunks) > 1:
		print(f"Synthesizing {len(chunks)} text chunks")

	wav_parts = []
	for chunk_obj in tqdm(chunks, desc="Generating audio", unit="chunk"):
		chunk_text = chunk_obj["text"]
		chunk_language_id = resolve_chunk_language_id(chunk_obj["language"], languages_map, language_id)
		chunk_output = output
		wav = synthesize(config, model, tokenizer, chunk_text, device, vocoder, speaker_id=speaker_id, language_id=chunk_language_id, max_steps=max_steps, stop_threshold=stop_threshold, attention_window=attention_window, normalize_wav=False, debug=debug, use_postnet=use_postnet, output=chunk_output, prompt=prompt, include_prompt_in_output=include_prompt_in_output, use_speaker_embedding=use_speaker_embedding, )
		wav_parts.append(wav)

	wav = wav_parts[0] if len(wav_parts) == 1 else torch.cat(wav_parts)
	if normalize_wav:
		peak = wav.abs().max()
		if peak > 0:
			wav = wav / peak * 0.95
	return wav


def build_prompted_text(prompt_text, target_text, tokenizer, phonemized=False, language_name="en-us"):
	if phonemized:
		prompt_ph = prompt_text.strip()
		target_ph = target_text.strip()
	else:
		prompt_ph = phonemize_text(prompt_text, language_name)
		target_ph = phonemize_text(target_text, language_name)
	combined = f"{prompt_ph} {target_ph}".strip()
	prefix_token_len = len(tokenizer.encode(prompt_ph))
	return combined, PromptSpec(mel=None, prefix_token_len=prefix_token_len, prefix_frames=0, prompt_phonemes=prompt_ph, target_phonemes=target_ph, )


def load_prompt_mel(config, prompt_audio, device):
	wav, sr = load_wav(prompt_audio)
	mel = wav_to_model_mel(wav, sr, config).unsqueeze(0).to(device)
	return mel, mel.shape[1]


def prepare_prompt(config, prompt_audio, prompt_text, target_text, tokenizer, device, phonemized=False, language_name="en-us"):
	combined, prompt = build_prompted_text(prompt_text, target_text, tokenizer, phonemized=phonemized, language_name=language_name)
	mel, prefix_frames = load_prompt_mel(config, prompt_audio, device)
	prompt.mel = mel
	prompt.prefix_frames = prefix_frames
	return combined, prompt


def build_vocoder(device, vocos_dir):
	vocoder = load_vocos_vocoder(vocos_dir, device=device)
	print(f"Using Vocos vocoder from {vocos_dir}")
	return vocoder


@torch.no_grad()
def synthesize(config, model, tokenizer, text, device, vocoder, speaker_id=0, language_id=0, max_steps=None, stop_threshold=None, attention_window=None, normalize_wav=False, debug=False, use_postnet=True, output=None, prompt=None, include_prompt_in_output=True, use_speaker_embedding=True, ):
	token_ids = tokenizer.encode(text)
	tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
	token_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
	speaker_ids = torch.tensor([int(speaker_id)], dtype=torch.long, device=device)
	language_ids = torch.tensor([int(language_id)], dtype=torch.long, device=device)
	model_cfg = config["model"]
	infer_kwargs = dict(max_steps=int(max_steps or model_cfg.get("max_decoder_steps", 1200)), stop_threshold=float(stop_threshold or model_cfg.get("stop_threshold", 0.55)), min_steps=max(20, len(token_ids) * 3), attention_window=int(attention_window if attention_window is not None else model_cfg.get("attention_window", 12)), )
	if prompt is not None:
		infer_kwargs["mel_prefix"] = prompt.mel
		infer_kwargs["prefix_token_len"] = prompt.prefix_token_len
	infer_kwargs["use_speaker_embedding"] = use_speaker_embedding
	if output is None:
		output = model.infer(tokens, token_lens, speaker_ids, language_ids, **infer_kwargs)
	mel_source = output.mel_postnet if use_postnet else output.mel
	mel = denormalize_mel(mel_source[0].detach().cpu(), config)
	if prompt is not None and not include_prompt_in_output:
		mel = mel[prompt.prefix_frames:]
	if debug:
		label = "postnet" if use_postnet else "pre-postnet"
		print(f"mel source: {label}")
		hop = int(config["audio"]["hop_length"])
		sample_rate = int(config["audio"]["sample_rate"])
		if prompt is not None:
			print(f"prompt frames: {prompt.prefix_frames} (~{prompt.prefix_frames * hop / sample_rate:.2f}s)")
			print(f"prompt tokens: {prompt.prefix_token_len}")
			print(f"prompt phonemes: {prompt.prompt_phonemes}")
			print(f"target phonemes: {prompt.target_phonemes}")
		seconds = mel.shape[0] * hop / sample_rate
		print(f"generated frames: {mel.shape[0]} (~{seconds:.2f}s)")
		print("mel stats:", f"min={mel.min().item():.3f}", f"max={mel.max().item():.3f}", f"mean={mel.mean().item():.3f}", )
		align_path = output.alignments[0].argmax(dim=-1).detach().cpu()
		print(f"attention first/last: {int(align_path[0].item())} -> {int(align_path[-1].item())} / {len(token_ids) - 1}")
		print(f"attention max index reached: {int(align_path.max().item())} / {len(token_ids) - 1}")
		print(f"last stop prob: {float(torch.sigmoid(output.stop_logits[0, -1]).cpu()):.3f}")
	wav = vocoder(mel)
	if normalize_wav:
		peak = wav.abs().max()
		if peak > 0:
			wav = wav / peak * 0.95
	return wav


@torch.no_grad()
def infer_output(config, model, tokenizer, text, device, speaker_id=0, language_id=0, max_steps=None, stop_threshold=None, attention_window=None, prompt=None, use_speaker_embedding=True, ):
	token_ids = tokenizer.encode(text)
	tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
	token_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
	speaker_ids = torch.tensor([int(speaker_id)], dtype=torch.long, device=device)
	language_ids = torch.tensor([int(language_id)], dtype=torch.long, device=device)
	model_cfg = config["model"]
	infer_kwargs = dict(max_steps=int(max_steps or model_cfg.get("max_decoder_steps", 1200)), stop_threshold=float(stop_threshold or model_cfg.get("stop_threshold", 0.55)), min_steps=max(20, len(token_ids) * 3), attention_window=int(attention_window if attention_window is not None else model_cfg.get("attention_window", 12)), )
	if prompt is not None:
		infer_kwargs["mel_prefix"] = prompt.mel
		infer_kwargs["prefix_token_len"] = prompt.prefix_token_len
	infer_kwargs["use_speaker_embedding"] = use_speaker_embedding
	return model.infer(tokens, token_lens, speaker_ids, language_ids, **infer_kwargs)


def convert_checkpoint_to_safetensors(checkpoint, out, metadata_out=None):
	checkpoint_path = Path(checkpoint)
	out_path = Path(out)
	ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
	if "model" not in ckpt:
		raise KeyError(f"Checkpoint has no 'model' key: {checkpoint_path}")
	state_dict = ckpt["model"]
	tensors = {key: value.detach().cpu().contiguous() for key, value in state_dict.items() if torch.is_tensor(value)}
	if len(tensors) != len(state_dict):
		skipped = sorted(set(state_dict) - set(tensors))
		raise TypeError(f"Non-tensor model entries cannot be saved to safetensors: {skipped}")
	metadata = {"format": "pt", "model_type": str(ckpt.get("model_type", "hanasu_tts")), "source_checkpoint": checkpoint_path.name, "step": str(ckpt.get("step", "")), "epoch": str(ckpt.get("epoch", "")), }
	out_path.parent.mkdir(parents=True, exist_ok=True)
	save_file(tensors, out_path, metadata=metadata)
	metadata_out = Path(metadata_out) if metadata_out else out_path.with_suffix(".json")
	sidecar = {"model_type": ckpt.get("model_type", "hanasu_tts"), "step": ckpt.get("step"), "epoch": ckpt.get("epoch"), "config": json_safe(ckpt.get("config", {})), "tokenizer": json_safe(ckpt.get("tokenizer", {})), "weights": out_path.name, }
	with metadata_out.open("w", encoding="utf-8") as f:
		json.dump(sidecar, f, indent=2)
	print(f"Saved weights: {out_path}")
	print(f"Saved metadata: {metadata_out}")
	print(f"Tensors: {len(tensors)}")
	print(f"Epoch: {ckpt.get('epoch')} Step: {ckpt.get('step')}")


def main():
	parser = argparse.ArgumentParser(description="Synthesize speech with HanasuTTS + Vocos.")
	parser.add_argument("--config", default="config.json")
	parser.add_argument("--checkpoint", default=None)
	parser.add_argument("--metadata", default=None, help="JSON sidecar for .safetensors (config/tokenizer/mel_stats).")
	parser.add_argument("--text", default=None, help="Text to synthesize (normal text by default; it is phonemized for you).")
	parser.add_argument("--prompt-audio", default=None, help="Reference wav for optional pre-prompt conditioning.")
	parser.add_argument("--prompt-text", default=None, help="Transcript of --prompt-audio (phonemized unless --phonemes).")
	parser.add_argument("--no-prompt-in-output", action="store_true", help="Drop the reference mel from the written wav (conditioning only).")
	parser.add_argument("--phonemes", action="store_true", help="Treat --text and --prompt-text as already-phonemized IPA and skip phonemization.")
	parser.add_argument("--speaker", default="0", help="Speaker name (e.g. Yuna, Yuki) or integer id.")
	parser.add_argument("--no-speaker", action="store_true", help="Skip speaker embedding (use with --prompt-audio to clone arbitrary voices).")
	parser.add_argument("--language", default="0", help="Language code (e.g. en-us, ja, ru) or integer id.")
	parser.add_argument("--vocos-dir", default="vocos", help="Folder with the trained Vocos config.json + latest.ckpt.")
	parser.add_argument("--out", default="output.wav")
	parser.add_argument("--max-steps", type=int, default=None)
	parser.add_argument("--stop-threshold", type=float, default=None)
	parser.add_argument("--attention-window", type=int, default=None)
	parser.add_argument("--normalize-wav", action="store_true")
	parser.add_argument("--no-postnet", action="store_true", help="Feed Vocos the decoder mel before postnet (buzz A/B test).")
	parser.add_argument("--compare-postnet", action="store_true", help="Write two files: <stem>_postnet.wav and <stem>_prepostnet.wav.")
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--convert-checkpoint", default=None, help="Convert a .pt checkpoint to safetensors.")
	parser.add_argument("--convert-out", default="checkpoints/model.safetensors")
	parser.add_argument("--convert-metadata-out", default=None)
	args = parser.parse_args()

	if args.convert_checkpoint:
		convert_checkpoint_to_safetensors(args.convert_checkpoint, args.convert_out, args.convert_metadata_out)
		return

	if not args.checkpoint or not args.text or not args.out:
		parser.error("--checkpoint, --text, and --out are required for synthesis")

	config = load_config(args.config)
	device = get_device()
	model, tokenizer = load_model(config, args.checkpoint, device, metadata_path=args.metadata)
	speakers_map = config.get("dataset", {}).get("speakers", {})
	languages_map = config.get("dataset", {}).get("languages", {})
	speaker_id = resolve_id(args.speaker, speakers_map, "speaker")
	language_id = resolve_id(args.language, languages_map, "language")
	print(f"Speaker id: {speaker_id} | Language id: {language_id}")
	if args.no_speaker:
		print("Speaker embedding disabled")
	language_name = resolve_language_name(args.language, languages_map, language_id)

	if bool(args.prompt_audio) ^ bool(args.prompt_text):
		parser.error("--prompt-audio and --prompt-text must be used together")

	prompt = None
	raw_text = args.text
	phonemized = args.phonemes
	if args.prompt_audio:
		combined, prompt = prepare_prompt(config, args.prompt_audio, args.prompt_text, args.text, tokenizer, device, phonemized=args.phonemes, language_name=language_name, )
		raw_text = combined
		phonemized = True
		print(f"Prompt phonemes [{language_name}]: {prompt.prompt_phonemes}")
		print(f"Target phonemes [{language_name}]: {prompt.target_phonemes}")

	vocoder = build_vocoder(device, args.vocos_dir)
	synth_kwargs = dict(config=config, model=model, tokenizer=tokenizer, text=raw_text, device=device, vocoder=vocoder, speaker_id=speaker_id, language_id=language_id, language_name=language_name, languages_map=languages_map, phonemized=phonemized, max_steps=args.max_steps, stop_threshold=args.stop_threshold, attention_window=args.attention_window, normalize_wav=args.normalize_wav, debug=args.debug, prompt=prompt, include_prompt_in_output=not args.no_prompt_in_output, use_speaker_embedding=not args.no_speaker, )
	sample_rate = int(config["audio"]["sample_rate"])

	if args.compare_postnet:
		out_path = Path(args.out)
		if prompt is not None:
			output = infer_output(config, model, tokenizer, raw_text, device, speaker_id=speaker_id, language_id=language_id, max_steps=args.max_steps, stop_threshold=args.stop_threshold, attention_window=args.attention_window, prompt=prompt, use_speaker_embedding=not args.no_speaker, )
			for use_postnet, suffix in ((True, "postnet"), (False, "prepostnet")):
				path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix or '.wav'}")
				print(f"--- {suffix} ---")
				wav = synthesize(**synth_kwargs, use_postnet=use_postnet, output=output)
				save_wav(path, wav, sample_rate)
				print(f"Saved {path}")
		else:
			for use_postnet, suffix in ((True, "postnet"), (False, "prepostnet")):
				path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix or '.wav'}")
				print(f"--- {suffix} ---")
				wav = synthesize_long_text(**synth_kwargs, use_postnet=use_postnet)
				save_wav(path, wav, sample_rate)
				print(f"Saved {path}")
		return

	use_postnet = not args.no_postnet
	if args.no_postnet:
		print("Using pre-postnet decoder mel (postnet bypassed for Vocos)")
	if prompt is not None:
		wav = synthesize(**synth_kwargs, use_postnet=use_postnet)
	else:
		wav = synthesize_long_text(**synth_kwargs, use_postnet=use_postnet)
	save_wav(Path(args.out), wav, sample_rate)
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
