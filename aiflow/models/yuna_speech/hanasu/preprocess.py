import math
import wave
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import MelSpectrogram, PhonemeTokenizer, load_json, resample_linear, resolve_cache_dir, save_config, save_json, to_mono, validate_audio


def _load_audio_path(path):
	try:
		import soundfile as sf
		data, sr = sf.read(path, dtype="float32", always_2d=False)
		return np.asarray(data, dtype=np.float32), int(sr)
	except ImportError:
		pass
	try:
		import torchaudio
		wav, sr = torchaudio.load(str(path))
		return wav.squeeze(0).numpy(), int(sr)
	except ImportError:
		try:
			return _load_wav_stdlib(path)
		except Exception as wav_exc:
			raise RuntimeError("Loading audio paths requires soundfile, torchaudio, or a PCM WAV readable by wave") from wav_exc


def _load_wav_stdlib(path):
	with wave.open(str(path), "rb") as f:
		channels = f.getnchannels()
		sample_width = f.getsampwidth()
		sample_rate = f.getframerate()
		frames = f.readframes(f.getnframes())
	if sample_width == 1:
		data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
	elif sample_width == 2:
		data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
	elif sample_width == 4:
		data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
	else:
		raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
	if channels > 1:
		data = data.reshape(-1, channels)
	return data, sample_rate


def load_filelist(filelist_path, speakers_map, languages_map):
	"""Parse a phonemized filelist of the form: wav_path|speaker|language|phonemes. speaker/language strings are resolved to integer ids via the config maps. Rows with an unknown speaker/language or a malformed line are skipped with a warning."""
	path = Path(filelist_path)
	if not path.exists():
		raise FileNotFoundError(f"Missing filelist: {path}")

	records = []
	skipped = 0
	skip_warning_limit = 20
	with path.open("r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.rstrip("\n")
			if not line.strip():
				continue
			parts = line.split("|")
			if len(parts) < 4:
				skipped += 1
				if skipped <= skip_warning_limit:
					print(f"Warning: skipping malformed row {line_no}: expected wav|speaker|language|phonemes")
				continue
			wav_path = parts[0].strip()
			speaker = parts[1].strip()
			language = parts[2].strip()
			phonemes = "|".join(parts[3:]).strip()
			if speaker not in speakers_map:
				skipped += 1
				if skipped <= skip_warning_limit:
					print(f"Warning: row {line_no} unknown speaker '{speaker}' (known: {sorted(speakers_map)})")
				continue
			if language not in languages_map:
				skipped += 1
				if skipped <= skip_warning_limit:
					print(f"Warning: row {line_no} unknown language '{language}' (known: {sorted(languages_map)})")
				continue
			if not phonemes:
				skipped += 1
				continue
			records.append({"audio": wav_path, "text": phonemes, "speaker": speaker, "language": language, "speaker_id": int(speakers_map[speaker]), "language_id": int(languages_map[language]), })

	if skipped > skip_warning_limit:
		print(f"Warning: suppressed {skipped - skip_warning_limit} additional skipped-row warnings")
	if not records:
		raise RuntimeError(f"No usable rows parsed from {path}")
	print(f"{path.name}: {len(records)} usable rows, {skipped} skipped")
	return records


def select_records(records, limit, percent):
	total = len(records)
	count = total
	if percent is not None:
		count = max(1, int(total * float(percent) / 100.0))
	if limit is not None:
		count = min(count, int(limit))
	count = min(count, total)
	return records[:count]


def _prepare_records(records, tokenizer, mel_fn, samples_dir, cache_root, dataset_cfg, audio_cfg, start_index):
	prepared = []
	skipped = []
	examples = []
	stats = {"sum": 0.0, "sumsq": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")}
	min_seconds = float(dataset_cfg.get("min_audio_seconds", 0.25))
	max_seconds = float(dataset_cfg.get("max_audio_seconds", 20.0))
	sample_rate = int(audio_cfg["sample_rate"])
	max_tokens = int(dataset_cfg.get("max_tokens", 1024))
	for offset, record in enumerate(tqdm(records, desc="Preparing features")):
		idx = start_index + offset
		try:
			raw_text = record["text"]
			token_ids = tokenizer.encode(raw_text)
			if len(token_ids) > max_tokens:
				raise ValueError(f"text too long: {len(token_ids)} tokens")
			audio, sr = _load_audio_path(record["audio"])
			wav = to_mono(audio)
			wav = resample_linear(wav, sr, sample_rate)
			validate_audio(wav, sample_rate, min_seconds, max_seconds)
			mel = mel_fn(wav)
			if mel.shape[0] < 2:
				raise ValueError("mel has fewer than 2 frames")
			mel_float = mel.to(torch.float32)
			stats["sum"] += float(mel_float.sum().item())
			stats["sumsq"] += float(mel_float.square().sum().item())
			stats["count"] += int(mel_float.numel())
			stats["min"] = min(stats["min"], float(mel_float.min().item()))
			stats["max"] = max(stats["max"], float(mel_float.max().item()))
			out_path = samples_dir / f"{idx:06d}.pt"
			torch.save({"tokens": torch.tensor(token_ids, dtype=torch.long), "mel": mel_float, "speaker_id": int(record["speaker_id"]), "language_id": int(record["language_id"]), "text": raw_text, }, out_path, )
			rel = out_path.relative_to(cache_root).as_posix()
			prepared.append({"path": rel, "text": raw_text, "speaker": record["speaker"], "language": record["language"], "speaker_id": int(record["speaker_id"]), "language_id": int(record["language_id"]), "frames": int(mel.shape[0]), "tokens": len(token_ids), })
			if len(examples) < 3:
				examples.append({"text": raw_text, "speaker": record["speaker"], "language": record["language"]})
		except Exception as exc:
			skipped.append({"index": str(idx), "error": str(exc)})
	return prepared, skipped, examples, stats


def prepare_dataset(config, filelist_path=None, val_filelist_path=None, limit=None, percent=None, cache_root=None, force=False, ):
	dataset_cfg = config["dataset"]
	audio_cfg = config["audio"]
	speakers_map = dataset_cfg["speakers"]
	languages_map = dataset_cfg["languages"]
	filelist_path = filelist_path or dataset_cfg["filelist_path"]
	val_filelist_path = val_filelist_path if val_filelist_path is not None else dataset_cfg.get("val_filelist_path")
	cache_root = resolve_cache_dir(config, cache_root)
	samples_dir = cache_root / "samples"
	manifest_path = cache_root / "manifest.json"
	split_path = cache_root / "split.json"

	if manifest_path.exists() and split_path.exists() and not force:
		manifest = load_json(manifest_path)
		print(f"Using cached dataset at {cache_root} ({len(manifest['samples'])} samples)")
		return manifest

	train_records = load_filelist(filelist_path, speakers_map, languages_map)
	train_records = select_records(train_records, limit, percent)
	val_records = []
	if val_filelist_path:
		val_records = load_filelist(val_filelist_path, speakers_map, languages_map)

	tokenizer = PhonemeTokenizer()
	print(f"Phoneme vocabulary size: {tokenizer.vocab_size} (fixed text.py symbol table)")

	mel_fn = MelSpectrogram(audio_cfg)
	samples_dir.mkdir(parents=True, exist_ok=True)

	train_prepared, train_skipped, examples, stats = _prepare_records(train_records, tokenizer, mel_fn, samples_dir, cache_root, dataset_cfg, audio_cfg, start_index=0)
	if not train_prepared:
		raise RuntimeError("No training samples were prepared successfully")

	if val_records:
		val_prepared, val_skipped, _, _ = _prepare_records(val_records, tokenizer, mel_fn, samples_dir, cache_root, dataset_cfg, audio_cfg, start_index=len(train_prepared))
		prepared = train_prepared + val_prepared
		train_indices = list(range(len(train_prepared)))
		val_indices = list(range(len(train_prepared), len(prepared)))
		skipped = train_skipped + val_skipped
	else:
		prepared = train_prepared
		skipped = train_skipped
		val_ratio = float(dataset_cfg.get("validation_ratio", 0.02))
		val_count = max(1, int(math.ceil(len(prepared) * val_ratio))) if len(prepared) > 1 else 0
		indices = list(range(len(prepared)))
		train_indices = indices[:-val_count] if val_count else indices
		val_indices = indices[-val_count:] if val_count else []

	mel_mean = stats["sum"] / max(1, stats["count"])
	mel_var = max(1e-8, stats["sumsq"] / max(1, stats["count"]) - mel_mean * mel_mean)
	mel_std = math.sqrt(mel_var)
	manifest = {"source": str(filelist_path), "sample_rate": int(audio_cfg["sample_rate"]), "audio": audio_cfg, "tokenizer": tokenizer.to_dict(), "speakers": speakers_map, "languages": languages_map, "mel_stats": {"mean": mel_mean, "std": mel_std, "min": stats["min"], "max": stats["max"], "count": stats["count"], "normalized_training": True, }, "samples": prepared, "examples": examples, "skipped": skipped[:50], }
	split_data = {"train": train_indices, "val": val_indices}
	save_json(manifest, manifest_path)
	save_json(split_data, split_path)
	save_config({"audio": audio_cfg, "tokenizer": tokenizer.to_dict(), "speakers": speakers_map, "languages": languages_map}, cache_root / "feature_config.json")
	print(f"Prepared {len(prepared)} samples in {cache_root}; skipped {len(skipped)}")
	print(f"Train/val split: {len(train_indices)}/{len(val_indices)}")
	print(f"Mel stats: mean={mel_mean:.4f}, std={mel_std:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
	for ex in examples:
		print(f"Example [{ex['speaker']}/{ex['language']}]: {ex['text']}")
	return manifest


class HanasuDataset(Dataset):
	def __init__(self, cache_root, split="train"):
		self.cache_root = Path(cache_root)
		manifest_path = self.cache_root / "manifest.json"
		split_path = self.cache_root / "split.json"
		if not manifest_path.exists() or not split_path.exists():
			raise FileNotFoundError(f"Missing prepared dataset in {self.cache_root}; run python preprocess.py")
		self.manifest = load_json(manifest_path)
		self.split_data = load_json(split_path)
		self.indices = list(self.split_data[split])
		self.samples = self.manifest["samples"]
		self.mel_stats = self.manifest.get("mel_stats")

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample = self.samples[self.indices[idx]]
		data = torch.load(self.cache_root / sample["path"], map_location="cpu")
		if self.mel_stats and self.mel_stats.get("normalized_training", False):
			mean = float(self.mel_stats["mean"])
			std = max(float(self.mel_stats["std"]), 1e-5)
			data["mel"] = (data["mel"] - mean) / std
		return data


def collate_batch(batch, pad_id=0):
	token_lens = torch.tensor([item["tokens"].numel() for item in batch], dtype=torch.long)
	mel_lens = torch.tensor([item["mel"].shape[0] for item in batch], dtype=torch.long)
	max_tokens = int(token_lens.max().item())
	max_mels = int(mel_lens.max().item())
	n_mels = int(batch[0]["mel"].shape[1])
	tokens = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
	mels = torch.zeros((len(batch), max_mels, n_mels), dtype=torch.float32)
	speaker_ids = torch.zeros(len(batch), dtype=torch.long)
	language_ids = torch.zeros(len(batch), dtype=torch.long)
	texts = []
	for i, item in enumerate(batch):
		t_len = item["tokens"].numel()
		m_len = item["mel"].shape[0]
		tokens[i, :t_len] = item["tokens"]
		mels[i, :m_len] = item["mel"]
		speaker_ids[i] = int(item.get("speaker_id", 0))
		language_ids[i] = int(item.get("language_id", 0))
		texts.append(item.get("text", ""))
	return {"tokens": tokens, "token_lens": token_lens, "mels": mels, "mel_lens": mel_lens, "speaker_ids": speaker_ids, "language_ids": language_ids, "texts": texts, }


if __name__ == "__main__":
	import argparse
	from utils import load_config
	parser = argparse.ArgumentParser(description="Prepare phonemized multi-speaker features for HanasuTTS.")
	parser.add_argument("--config", default="config.json")
	parser.add_argument("--filelist", default=None, help="Path to the phonemized filelist (wav|speaker|language|phonemes).")
	parser.add_argument("--val-filelist", default=None, help="Optional explicit validation filelist.")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--percent", type=float, default=None)
	parser.add_argument("--cache-dir", default=None, help="Where to write prepared tensors (overrides config dataset.cache_dir). Example on Colab: /content/cache/hanasu_48k")
	parser.add_argument("--force", action="store_true")
	args = parser.parse_args()
	config = load_config(args.config)
	cache_root = resolve_cache_dir(config, args.cache_dir)
	if args.cache_dir:
		config["dataset"]["cache_dir"] = str(cache_root)
	print(f"Cache directory: {cache_root}")
	prepare_dataset(config, filelist_path=args.filelist, val_filelist_path=args.val_filelist, limit=args.limit, percent=args.percent, cache_root=cache_root, force=args.force, )
