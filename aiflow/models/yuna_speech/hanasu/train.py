import argparse
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models import HanasuTTS, MultiResolutionSTFTLoss, TacotronOutput, build_speaker_band_weights, hanasu_loss
from .preprocess import HanasuDataset, collate_batch, prepare_dataset
from .utils import PhonemeTokenizer, count_parameters, denormalize_mel_tensor, ensure_dir, format_param_count, get_device, load_checkpoint, load_config, load_json, load_vocos_generator, resolve_cache_dir, set_seed, vocos_mel_to_wav


def sync_conditioning_dims(config):
	"""Ensure the embedding tables cover every configured speaker/language (even unused ones)."""
	model_cfg = config.setdefault("model", {})
	dataset_cfg = config.get("dataset", {})
	speakers = dataset_cfg.get("speakers", {})
	languages = dataset_cfg.get("languages", {})
	if speakers:
		model_cfg["num_speakers"] = max(int(model_cfg.get("num_speakers", 0)), max(speakers.values()) + 1)
	if languages:
		model_cfg["num_languages"] = max(int(model_cfg.get("num_languages", 0)), max(languages.values()) + 1)
	return config


def build_model(config, tokenizer):
	return HanasuTTS(vocab_size=tokenizer.vocab_size, n_mels=int(config["audio"]["n_mels"]), config=config["model"], pad_id=tokenizer.pad_id, )


def move_batch(batch, device):
	return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(path, model, optimizer, scaler, step, epoch, config, tokenizer):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "config": config, "tokenizer": tokenizer.to_dict(), "model_type": "hanasu_tts", }, path, )


def load_cached_tokenizer(config):
	cache_root = resolve_cache_dir(config)
	manifest_path = cache_root / "manifest.json"
	if manifest_path.exists():
		manifest = load_json(manifest_path)
		if "tokenizer" in manifest:
			return PhonemeTokenizer.from_dict(manifest["tokenizer"])
	return None


def _index_output(output, mask):
	"""Select a subset of utterances (batch dim) from a TacotronOutput."""
	return TacotronOutput(mel=output.mel[mask], mel_postnet=output.mel_postnet[mask], stop_logits=output.stop_logits[mask], alignments=output.alignments[mask], )


def _index_batch(batch, mask):
	"""Select a subset of utterances from a collated batch dict (tensors and the texts list)."""
	keep = mask.nonzero(as_tuple=True)[0].tolist()
	out = {}
	for key, value in batch.items():
		if torch.is_tensor(value):
			out[key] = value[mask]
		elif isinstance(value, list):
			out[key] = [value[i] for i in keep]
		else:
			out[key] = value
	return out


@torch.no_grad()
def validate(model, loader, device, loss_fn, id_to_speaker=None):
	model.eval()
	agg = {}
	count = 0
	per_speaker = {}  # speaker_id -> {"loss": weighted_sum, "mel": ..., "post": ..., "n": samples}
	for batch in loader:
		batch = move_batch(batch, device)
		output = model(batch["tokens"], batch["token_lens"], batch["mels"], batch["speaker_ids"], batch["language_ids"])
		loss, metrics = loss_fn(output, batch)
		count += 1
		agg["val_loss"] = agg.get("val_loss", 0.0) + float(loss.cpu())
		for key, value in metrics.items():
			agg[f"val_{key}"] = agg.get(f"val_{key}", 0.0) + float(value)
		sids = batch["speaker_ids"]
		for sid in sids.unique().tolist():
			sub_mask = sids == sid
			n = int(sub_mask.sum().item())
			if n == 0:
				continue
			sub_loss, sub_metrics = loss_fn(_index_output(output, sub_mask), _index_batch(batch, sub_mask))
			bucket = per_speaker.setdefault(sid, {"loss": 0.0, "mel": 0.0, "post": 0.0, "n": 0})
			bucket["loss"] += float(sub_loss.cpu()) * n
			bucket["mel"] += float(sub_metrics["mel_loss"]) * n
			bucket["post"] += float(sub_metrics["postnet_mel_loss"]) * n
			bucket["n"] += n
	model.train()
	if count == 0:
		return {"val_loss": float("nan")}
	results = {key: value / count for key, value in agg.items()}
	id_to_speaker = id_to_speaker or {}
	for sid, bucket in sorted(per_speaker.items()):
		name = id_to_speaker.get(sid, str(sid))
		denom = max(1, bucket["n"])
		results[f"val_loss/{name}"] = bucket["loss"] / denom
		results[f"val_mel_loss/{name}"] = bucket["mel"] / denom
		results[f"val_postnet_mel_loss/{name}"] = bucket["post"] / denom
		results[f"val_samples/{name}"] = bucket["n"]
	return results


def build_speaker_f0_by_id(config):
	"""Map speaker_id -> fundamental frequency (Hz) from config (names resolved to ids)."""
	speakers_map = config.get("dataset", {}).get("speakers", {})
	speaker_f0 = config.get("dataset", {}).get("speaker_f0", {})
	return {int(speakers_map[name]): float(f0) for name, f0 in speaker_f0.items() if name in speakers_map and f0 is not None}


def build_speaker_loss_weights(config, num_speakers, device):
	"""Per-speaker scalar loss weights (default 1.0). Lets you rebalance a speaker the model
	under-serves even when clip counts are equal. Returns (vector, is_nonuniform)."""
	speakers_map = config.get("dataset", {}).get("speakers", {})
	weights_cfg = config.get("training", {}).get("speaker_loss_weights", {})
	vec = torch.ones(num_speakers, dtype=torch.float32)
	nonuniform = False
	for name, w in weights_cfg.items():
		if name in speakers_map and 0 <= int(speakers_map[name]) < num_speakers:
			vec[int(speakers_map[name])] = float(w)
			if abs(float(w) - 1.0) > 1e-9:
				nonuniform = True
	return vec.to(device), nonuniform


def run_count_params(config):
	sync_conditioning_dims(config)
	tokenizer = load_cached_tokenizer(config)
	if tokenizer is None:
		print("No prepared cache found; using the fixed text.py symbol table for the parameter count.")
		tokenizer = PhonemeTokenizer()
	acoustic = build_model(config, tokenizer)
	acoustic_params = count_parameters(acoustic)
	print(f"Acoustic model: {format_param_count(acoustic_params)} ({acoustic_params:,})")
	print("Vocoder: Vocos (frozen, loaded from the vocos/ folder at inference time)")


def run_train(args, config):
	set_seed(int(config["project"].get("seed", 1337)))
	sync_conditioning_dims(config)
	cache_root = resolve_cache_dir(config)
	if not (cache_root / "manifest.json").exists() or args.force_prepare:
		prepare_dataset(config, filelist_path=args.filelist, val_filelist_path=args.val_filelist, limit=args.limit, percent=args.percent, cache_root=cache_root, force=args.force_prepare, )

	in_memory = bool(config.get("dataset", {}).get("in_memory", False))
	train_ds = HanasuDataset(cache_root, "train", in_memory=in_memory)
	val_ds = HanasuDataset(cache_root, "val", in_memory=in_memory)
	tokenizer = PhonemeTokenizer.from_dict(train_ds.manifest["tokenizer"])
	if train_ds.mel_stats:
		config.setdefault("audio", {})["mel_stats"] = train_ds.mel_stats
		print("Using normalized mel targets:", f"mean={float(train_ds.mel_stats['mean']):.4f}", f"std={float(train_ds.mel_stats['std']):.4f}", )
	else:
		print("Warning: cache has no mel_stats. Re-run python preprocess.py --force for more stable training.")
	if train_ds.manifest.get("speaker_frame_counts"):
		print(f"Per-speaker frames (training signal balance): {train_ds.manifest['speaker_frame_counts']}")

	train_cfg = config["training"]
	collate = partial(collate_batch, pad_id=tokenizer.pad_id)
	num_workers = int(train_cfg.get("num_workers", 0))
	if in_memory and num_workers > 0:
		# Samples already live in RAM; extra workers would duplicate that cache per process.
		print("in_memory=true -> forcing num_workers=0 to avoid duplicating the cached dataset")
		num_workers = 0
	loader_kwargs = dict(batch_size=int(train_cfg["batch_size"]), collate_fn=collate, drop_last=False, pin_memory=torch.cuda.is_available(), num_workers=num_workers, )
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = True
		loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
	train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
	val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
	device = get_device()
	model = build_model(config, tokenizer).to(device)
	print(f"Acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")

	# Loss conditioning: f0 band emphasis + per-speaker balancing
	audio_cfg = config["audio"]
	n_mels = int(audio_cfg["n_mels"])
	f_max = float(audio_cfg["f_max"]) if audio_cfg.get("f_max") is not None else float(audio_cfg["sample_rate"]) / 2.0
	band_weights = build_speaker_band_weights(model.num_speakers, n_mels, float(audio_cfg.get("f_min", 0.0)), f_max, build_speaker_f0_by_id(config), emphasis=float(train_cfg.get("f0_emphasis_weight", 0.0)), harmonics=int(train_cfg.get("f0_emphasis_harmonics", 6)), )
	if band_weights is not None:
		band_weights = band_weights.to(device)
		print(f"f0 band emphasis enabled (weight={train_cfg.get('f0_emphasis_weight')}) for speaker f0s {build_speaker_f0_by_id(config)}")
	speaker_loss_weights, speaker_weights_nonuniform = build_speaker_loss_weights(config, model.num_speakers, device)
	id_to_speaker = {int(v): k for k, v in config.get("dataset", {}).get("speakers", {}).items()}
	loss_terms = dict(mel_weight=float(train_cfg.get("mel_loss_weight", 1.0)), stop_weight=float(train_cfg.get("stop_loss_weight", 0.5)), guided_attention_weight=float(train_cfg.get("guided_attention_weight", 0.2)), guided_attention_sigma=float(train_cfg.get("guided_attention_sigma", 0.4)), mel_mse_weight=float(train_cfg.get("mel_mse_weight", 0.0)), temporal_delta_weight=float(train_cfg.get("temporal_delta_weight", 0.0)), freq_delta_weight=float(train_cfg.get("freq_delta_weight", 0.0)), )

	def base_loss(output, batch):
		sids = batch["speaker_ids"]
		bw = band_weights[sids] if band_weights is not None else None
		sw = speaker_loss_weights[sids] if speaker_weights_nonuniform else None
		return hanasu_loss(output, batch["mels"], batch["mel_lens"], batch["token_lens"], band_weights=bw, sample_weights=sw, **loss_terms)

	# Optional Vocos waveform (multi-resolution STFT) loss
	mel_stats = config.get("audio", {}).get("mel_stats")
	vocos_weight = float(train_cfg.get("vocos_loss_weight", 0.0))
	vocos_interval = max(1, int(train_cfg.get("vocos_loss_interval", 1)))
	vocos_max_frames = int(train_cfg.get("vocos_loss_max_frames", 320))
	vocos_state = {"enabled": False, "backbone": None, "head": None, "mrstft": None}
	if args.vocos_dir and vocos_weight > 0:
		try:
			vb, vh = load_vocos_generator(args.vocos_dir, device)
			vocos_state.update(enabled=True, backbone=vb, head=vh, mrstft=MultiResolutionSTFTLoss().to(device))
			print(f"Vocos waveform loss enabled from {args.vocos_dir} (weight={vocos_weight}, max_frames={vocos_max_frames}, every {vocos_interval} steps)")
		except Exception as exc:
			print(f"Vocos waveform loss unavailable, continuing without it: {exc}")
	elif args.vocos_dir:
		print("Vocos dir given but vocos_loss_weight=0; skipping the waveform loss")

	def vocos_loss(output, batch):
		max_len = min(output.mel.shape[1], batch["mels"].shape[1])
		crop = min(max_len, vocos_max_frames)
		pred_feats = denormalize_mel_tensor(output.mel_postnet[:, :crop].float(), mel_stats).transpose(1, 2)
		tgt_feats = denormalize_mel_tensor(batch["mels"][:, :crop].float(), mel_stats).transpose(1, 2)
		with torch.no_grad():
			wav_tgt = vocos_mel_to_wav(vocos_state["backbone"], vocos_state["head"], tgt_feats)
		wav_pred = vocos_mel_to_wav(vocos_state["backbone"], vocos_state["head"], pred_feats)
		length = min(wav_pred.shape[-1], wav_tgt.shape[-1])
		return vocos_state["mrstft"](wav_pred[..., :length], wav_tgt[..., :length])

	optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)), )
	scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
	start_epoch = 0
	step = 0
	resume_path = args.resume or train_cfg.get("resume")
	if resume_path:
		ckpt = load_checkpoint(resume_path, map_location=device)
		model.load_state_dict(ckpt["model"])
		optimizer.load_state_dict(ckpt["optimizer"])
		scaler.load_state_dict(ckpt.get("scaler", {}))
		step = int(ckpt.get("step", 0))
		start_epoch = int(ckpt.get("epoch", 0))
		print(f"Resumed from {resume_path} at step {step}")

	writer = None
	try:
		from torch.utils.tensorboard import SummaryWriter

		writer = SummaryWriter(log_dir=str(ensure_dir(train_cfg.get("runs_dir", "runs")) / "train"))
	except Exception as exc:
		print(f"TensorBoard disabled: {exc}")

	ckpt_dir = ensure_dir(train_cfg.get("checkpoints_dir", "checkpoints"))
	model.train()
	for epoch in range(start_epoch, int(train_cfg["epochs"])):
		pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
		for batch in pbar:
			step += 1
			batch = move_batch(batch, device)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
				output = model(batch["tokens"], batch["token_lens"], batch["mels"], batch["speaker_ids"], batch["language_ids"])
				loss, metrics = base_loss(output, batch)
			if vocos_state["enabled"] and step % vocos_interval == 0:
				try:
					with torch.autocast(device_type=device.type, enabled=False):
						v_loss = vocos_loss(output, batch)
					loss = loss + vocos_weight * v_loss
					metrics["vocos_loss"] = float(v_loss.detach().cpu())
				except Exception as exc:
					print(f"Disabling Vocos waveform loss after runtime error: {exc}")
					vocos_state["enabled"] = False
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
			scaler.step(optimizer)
			scaler.update()
			pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", mel=f"{metrics['mel_loss']:.4f}", post=f"{metrics['postnet_mel_loss']:.4f}", attn=f"{metrics['guided_attention_loss']:.4f}", )

			if writer and step % int(train_cfg.get("log_interval", 20)) == 0:
				writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
				for key, value in metrics.items():
					writer.add_scalar(f"train/{key}", value, step)

			if step % int(train_cfg.get("val_interval", 500)) == 0 and len(val_ds) > 0:
				val_metrics = validate(model, val_loader, device, base_loss, id_to_speaker=id_to_speaker)
				print(f"step {step} validation: {val_metrics}")
				if writer:
					for key, value in val_metrics.items():
						writer.add_scalar(f"val/{key}", value, step)

			if step % int(train_cfg.get("save_interval", 1000)) == 0:
				save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler, step, epoch, config, tokenizer)

		save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler, step, epoch + 1, config, tokenizer)
	save_checkpoint(ckpt_dir / "final.pt", model, optimizer, scaler, step, int(train_cfg["epochs"]), config, tokenizer)
	if writer:
		writer.close()


def main():
	parser = argparse.ArgumentParser(description="Train HanasuTTS.")
	parser.add_argument("--config", default="config.json")
	parser.add_argument("--filelist", default=None, help="Phonemized filelist (wav|speaker|language|phonemes).")
	parser.add_argument("--val-filelist", default=None)
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--percent", type=float, default=None)
	parser.add_argument("--resume", default=None)
	parser.add_argument("--force-prepare", action="store_true")
	parser.add_argument("--cache-dir", default=None, help="Prepared dataset cache (overrides config dataset.cache_dir). Example on Colab: /content/cache/hanasu_48k")
	parser.add_argument("--vocos-dir", default=None, help="Folder with a trained Vocos config.json + latest.ckpt. When set (and training.vocos_loss_weight > 0), a frozen Vocos decodes predicted/target mels for a multi-resolution STFT waveform loss.")
	parser.add_argument("--count-params", action="store_true", help="Print model parameter count and exit.")
	args = parser.parse_args()
	config = load_config(args.config)
	if args.cache_dir:
		cache_root = resolve_cache_dir(config, args.cache_dir)
		config["dataset"]["cache_dir"] = str(cache_root)
		print(f"Cache directory: {cache_root}")
	if args.count_params:
		run_count_params(config)
		return
	run_train(args, config)


if __name__ == "__main__":
	main()
