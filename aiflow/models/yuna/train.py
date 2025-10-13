import os
import math
import argparse
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from model.processor import Processor
from model.yuna import Yuna
import json

class YunaDataset(Dataset):
    def __init__(self, jsonl_path, images_dir=None, audio_dir=None):
        self.data = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8") if line.strip()]
        self.images_dir = images_dir
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Create full paths for lists of files
        image_paths = [os.path.join(self.images_dir, p) for p in item.get('image_paths', [])] if self.images_dir else item.get('image_paths', [])
        audio_paths = [os.path.join(self.audio_dir, p) for p in item.get('audio_paths', [])] if self.audio_dir else item.get('audio_paths', [])

        return {"chat": item["chat"], "image_paths": image_paths, "audio_paths": audio_paths}

class RawTextDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8") if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"text": item["text"]}

def freeze_modules(model, train_modules):
    for name, param in model.named_parameters():
        param.requires_grad = False
    print(f"Unfreezing parameters for: {', '.join(train_modules)}")
    for module_name in train_modules:
        module_to_train = None
        if module_name == 'llm': module_to_train = model.model
        elif module_name == 'vision_projector': module_to_train = model.get_vision_projector()
        elif module_name == 'audio_tower': module_to_train = model.get_audio_tower()
        elif module_name == 'audio_projector': module_to_train = model.get_audio_projector()
        if module_to_train:
            for param in module_to_train.parameters():
                param.requires_grad = True
        else:
            print(f"Warning: Module '{module_name}' not found in model.")

def make_collate_fn(processor, max_seq_len):
    pad_id = processor.tokenizer.encode("<|endoftext|>").ids[0]

    def collate(batch):
        batch_inputs = []
        for ex in batch:
            try:
                inputs = processor(
                    messages=ex["chat"], image_paths=ex.get("image_paths"), audio_paths=ex.get("audio_paths"), add_generation_prompt=False
                )
                if inputs["input_ids"].shape[1] > max_seq_len: continue
                labels = inputs["input_ids"].clone()
                if processor.image_pad_token_id != -1: labels[labels == processor.image_pad_token_id] = -100
                if processor.video_pad_token_id != -1: labels[labels == processor.video_pad_token_id] = -100
                if processor.audio_chunk_token_id != -1: labels[labels == processor.audio_chunk_token_id] = -100
                inputs["labels"] = labels
                batch_inputs.append(inputs)
            except Exception as e:
                print(f"Skipping a sample due to error: {e}")
                continue

        if not batch_inputs: return None

        input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"].squeeze(0) for item in batch_inputs], batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence([item["labels"].squeeze(0) for item in batch_inputs], batch_first=True, padding_value=-100)

        pixels = torch.cat([item["pixels"] for item in batch_inputs if item["pixels"] is not None], dim=0) if any(item["pixels"] is not None for item in batch_inputs) else None
        d_image = torch.cat([item["d_image"] for item in batch_inputs if item["d_image"] is not None], dim=0) if any(item["d_image"] is not None for item in batch_inputs) else None
        audio_features = torch.cat([item["audio_features"] for item in batch_inputs if item["audio_features"] is not None], dim=1) if any(item["audio_features"] is not None for item in batch_inputs) else None
        audio_feature_lens = torch.cat([item["audio_feature_lens"] for item in batch_inputs if item["audio_feature_lens"] is not None], dim=0) if any(item["audio_feature_lens"] is not None for item in batch_inputs) else None

        return {"input_ids": input_ids, "labels": labels, "pixels": pixels, "d_image": d_image, "audio_features": audio_features, "audio_feature_lens": audio_feature_lens}
    return collate

def make_raw_text_collate_fn(processor, max_seq_len):
    pad_id = processor.tokenizer.encode("<|endoftext|>").ids[0]

    def collate(batch):
        batch_inputs = []
        for ex in batch:
            try:
                # Tokenize raw text
                encoded = processor.tokenizer.encode(ex["text"])
                input_ids = torch.tensor([encoded.ids], dtype=torch.long)

                if input_ids.shape[1] > max_seq_len:
                    input_ids = input_ids[:, :max_seq_len]

                labels = input_ids.clone()

                batch_inputs.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "pixels": None,
                    "d_image": None,
                    "audio_features": None,
                    "audio_feature_lens": None
                })
            except Exception as e:
                print(f"Skipping a sample due to error: {e}")
                continue

        if not batch_inputs: return None

        input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"].squeeze(0) for item in batch_inputs], batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence([item["labels"].squeeze(0) for item in batch_inputs], batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "labels": labels, "pixels": None, "d_image": None, "audio_features": None, "audio_feature_lens": None}
    return collate

def lm_ce_loss(logits, labels):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

class YunaLightningModule(L.LightningModule):
    def __init__(self, model, lr, weight_decay, total_steps, warmup_steps, empty_cache_steps=10):
        super().__init__()
        self.model = model
        self.lr, self.weight_decay = lr, weight_decay
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.empty_cache_steps = empty_cache_steps
        self.step_count = 0

    def training_step(self, batch, batch_idx):
        if batch is None: return None

        model_inputs = {
            "input_ids": batch["input_ids"],
            "pixels": batch.get("pixels"),
            "d_image": batch.get("d_image"),
            "audio_features": batch.get("audio_features"),
            "audio_feature_lens": batch.get("audio_feature_lens"),
        }

        logits = self.model(**model_inputs)[0]
        loss = lm_ce_loss(logits, batch["labels"])

        # Clear cache periodically
        self.step_count += 1
        if self.step_count % self.empty_cache_steps == 0:
            torch.cuda.empty_cache()

        self.log("train_loss", loss, prog_bar=True, on_step=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))

        def lr_lambda(step):
            if step < self.warmup_steps: return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

def main():
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_repo", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--train_modules", type=str, nargs='+', required=True)
    parser.add_argument("--raw_text", action="store_true", help="Use raw text-only training mode")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--strategy", type=str, default="auto")
    # Add new memory optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--empty_cache_steps", type=int, default=10, help="Clear cache every N steps")
    args = parser.parse_args()

    has_audio = 'audio_tower' in args.train_modules or 'audio_projector' in args.train_modules
    model = Yuna.from_pretrained(args.model_repo, audio_config={} if has_audio else None)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        if hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()

    freeze_modules(model, args.train_modules)
    processor = Processor(repo_id=args.model_repo, vision_config=model.config.vision_config, audio_encoder=model.audio_encoder, yuna_model=model)

    # Choose dataset and collate function based on mode
    if args.raw_text:
        print("Using raw text-only training mode")
        dataset = RawTextDataset(jsonl_path=args.jsonl_path)
        collate_fn = make_raw_text_collate_fn(processor, args.max_seq_len)
    else:
        print("Using multimodal training mode")
        dataset = YunaDataset(jsonl_path=args.jsonl_path, images_dir=args.images_dir, audio_dir=args.audio_dir)
        collate_fn = make_collate_fn(processor, args.max_seq_len)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    steps_per_epoch = len(dataloader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.03 * total_steps)

    lit_module = YunaLightningModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        empty_cache_steps=args.empty_cache_steps if hasattr(args, 'empty_cache_steps') else 10
    )
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="yuna-epoch={epoch}-step={step}", save_top_k=-1, every_n_epochs=1)
    trainer = L.Trainer(
        accelerator="auto",
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        gradient_clip_val=1.0  # Add gradient clipping
    )

    trainer.fit(lit_module, dataloader)

if __name__ == "__main__":
    main()

# --- EXAMPLE COMMAND to fine-tune the LLM on text, image, and audio data ---
# python train.py \
#     --model_repo Yuna-2B-ft-audio \
#     --jsonl_path mixed_dataset.jsonl \
#     --audio_dir wavs \
#     --images_dir images \
#     --train_modules llm audio_tower audio_projector \
#     --devices 1 \
#     --batch_size 2 \
#     --grad_accum 4 \
#     --epochs 1 \
#     --lr 1e-5
#     --max_seq_len 2048

# --- EXAMPLE COMMAND for raw text-only training ---
# python train.py \
#     --model_repo Yuna-2B-ft-audio \
#     --jsonl_path mixed_dataset.jsonl \
#     --raw_text \
#     --train_modules llm \
#     --devices 1 \
#     --batch_size 2 \
#     --grad_accum 4 \
#     --epochs 3 \
#     --lr 1e-5 \
#     --max_seq_len 2048