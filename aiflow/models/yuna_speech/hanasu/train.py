import argparse
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models import HanasuTTS, hanasu_loss
from .preprocess import HanasuDataset, collate_batch, prepare_dataset
from .utils import PhonemeTokenizer, count_parameters, ensure_dir, format_param_count, get_device, load_checkpoint, load_config, load_json, resolve_cache_dir, set_seed


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


@torch.no_grad()
def validate(model, loader, device, config):
	model.eval()
	losses = []
	mel_losses = []
	post_losses = []
	stop_losses = []
	guide_losses = []
	train_cfg = config["training"]
	for batch in loader:
		batch = move_batch(batch, device)
		output = model(batch["tokens"], batch["token_lens"], batch["mels"], batch["speaker_ids"], batch["language_ids"])
		loss, metrics = hanasu_loss(output, batch["mels"], batch["mel_lens"], batch["token_lens"], float(train_cfg.get("mel_loss_weight", 1.0)), float(train_cfg.get("stop_loss_weight", 0.5)), float(train_cfg.get("guided_attention_weight", 0.2)), float(train_cfg.get("guided_attention_sigma", 0.4)), )
		losses.append(float(loss.cpu()))
		mel_losses.append(metrics["mel_loss"])
		post_losses.append(metrics["postnet_mel_loss"])
		stop_losses.append(metrics["stop_loss"])
		guide_losses.append(metrics["guided_attention_loss"])
	model.train()
	if not losses:
		return {"val_loss": float("nan")}
	return {"val_loss": sum(losses) / len(losses), "val_mel_loss": sum(mel_losses) / len(mel_losses), "val_postnet_mel_loss": sum(post_losses) / len(post_losses), "val_stop_loss": sum(stop_losses) / len(stop_losses), "val_guided_attention_loss": sum(guide_losses) / len(guide_losses), }


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

	train_ds = HanasuDataset(cache_root, "train")
	val_ds = HanasuDataset(cache_root, "val")
	tokenizer = PhonemeTokenizer.from_dict(train_ds.manifest["tokenizer"])
	if train_ds.mel_stats:
		config.setdefault("audio", {})["mel_stats"] = train_ds.mel_stats
		print("Using normalized mel targets:", f"mean={float(train_ds.mel_stats['mean']):.4f}", f"std={float(train_ds.mel_stats['std']):.4f}", )
	else:
		print("Warning: cache has no mel_stats. Re-run python preprocess.py --force for more stable training.")

	train_cfg = config["training"]
	collate = partial(collate_batch, pad_id=tokenizer.pad_id)
	train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), collate_fn=collate, drop_last=False, pin_memory=torch.cuda.is_available(), )
	val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)), collate_fn=collate, drop_last=False, pin_memory=torch.cuda.is_available(), )
	device = get_device()
	model = build_model(config, tokenizer).to(device)
	print(f"Acoustic parameters: {format_param_count(count_parameters(model))} ({count_parameters(model):,})")
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
				loss, metrics = hanasu_loss(output, batch["mels"], batch["mel_lens"], batch["token_lens"], float(train_cfg.get("mel_loss_weight", 1.0)), float(train_cfg.get("stop_loss_weight", 0.5)), float(train_cfg.get("guided_attention_weight", 0.2)), float(train_cfg.get("guided_attention_sigma", 0.4)), )
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
				val_metrics = validate(model, val_loader, device, config)
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
