import os
import math
import argparse
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from model.processor import Processor
from model.yuna import Yuna
import json
from PIL import Image
from torch.utils.data import Dataset

def _transform_text_with_images(text, image_obj=None):
    """
    Splits text on <image> and inserts image_obj where needed.
    If image_obj is None, just returns text chunks.
    """
    parts = text.split("<image>")
    result = []
    for i, part in enumerate(parts):
        if part.strip():
            result.append({"type": "text", "content": part})
        if i < len(parts) - 1 and image_obj is not None:
            result.append({"type": "image", "content": image_obj})
    return result

class RawTextImageDataset(Dataset):
    def __init__(self, jsonl_path, images_dir=None):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        # If there is an <image> token and images_dir is provided, try to load the image
        if "<image>" in text and self.images_dir is not None:
            image_path = os.path.join(self.images_dir, f"{idx}.jpg")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                image = None
        else:
            image = None
        # Instead of wrapping, just return the raw text (and image if needed)
        return {"content": text, "id": idx}

# -------------------- utils --------------------
def freeze_except_language_model(model):
    # Freeze vision encoder
    for p in model.visual.parameters():
        p.requires_grad = False
    # for p in model.projection.parameters():   # <-- Remove or comment out this block
    #     p.requires_grad = False
    # Unfreeze language model
    for p in model.model.parameters():
        p.requires_grad = True
    model.visual.eval()
    # model.projection.eval()                  # <-- Remove or comment out this line
    model.model.train()

def make_collate_fn(processor: Processor, max_seq_len: int = 2048):
    pad_id = processor.tokenizer.encode("<|endoftext|>").ids[0]
    img_pad_id = processor.image_pad_token_id

    def collate(batch):
        ids_list, labels_list, pixels_list, dimg_list = [], [], [], []
        for ex in batch:
            out = processor(messages=ex["content"], device=None)  # ex["content"] is now a string
            ids = out["input_ids"].squeeze(0)
            pix = out["pixels"]
            dimg = out["d_image"]

            # avoid truncating through an image block
            if ids.numel() > max_seq_len:
                if (ids[max_seq_len:] == img_pad_id).any():
                    continue
                ids = ids[:max_seq_len]

            labels = ids.clone()
            labels[labels == img_pad_id] = -100
            ids_list.append(ids)
            labels_list.append(labels)
            if pix is not None:
                pixels_list.append(pix)
                dimg_list.append(dimg.squeeze(0))

        if not ids_list:
            return None

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        labels[input_ids == pad_id] = -100

        pixels = torch.cat(pixels_list, dim=0) if pixels_list else None
        d_image = torch.stack(dimg_list, dim=0) if dimg_list else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixels": pixels,
            "d_image": d_image,
        }

    return collate

def lm_ce_loss(logits, labels):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )

# -------------------- LightningModule --------------------
class ProjectionOnlyLM(L.LightningModule):
    def __init__(
        self,
        model,
        lr=5e-4,
        weight_decay=0.01,
        total_steps=1000,
        warmup_steps=100,
        log_every=50,
        use_act_ckpt=False,
        block_cls=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.log_every = int(log_every)
        self.use_act_ckpt = use_act_ckpt
        self.block_cls = block_cls
        freeze_except_language_model(self.model)  # <-- changed here

    def forward(self, input_ids, pixels=None, d_image=None):
        logits, _ = self.model(input_ids=input_ids, pixels=pixels, d_image=d_image)
        return logits

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        logits = self(
            input_ids=batch["input_ids"],
            pixels=batch["pixels"],
            d_image=batch["d_image"],
        )
        loss = lm_ce_loss(logits, batch["labels"])
        if (self.global_step + 1) % self.log_every == 0:
            self.log(
                "train/loss", loss, prog_bar=True, on_step=True, rank_zero_only=True
            )
            ppl = torch.exp(loss.clamp(0, 20))
            self.log("train/ppl", ppl, on_step=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        tsteps, wwarm = self.total_steps, self.warmup_steps

        def lr_lambda(step: int):
            if step < wwarm:
                return step / max(1, wwarm)
            prog = (step - wwarm) / max(1, tsteps - wwarm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

# -------------------- Callbacks --------------------
class SaveFullModelSafetensors(L.Callback):
    def __init__(self, path: str = "full_model.safetensors"):
        super().__init__()
        self.path = path

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ProjectionOnlyLM):
        if trainer.is_global_zero:
            state = {
                k: v.detach().cpu()
                for k, v in pl_module.model.state_dict().items()
            }
            save_file(state, self.path)
            print(f"[epoch {trainer.current_epoch}] saved {self.path}")

class PrintEpochSummary(L.Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ProjectionOnlyLM):
        if trainer.is_global_zero:
            loss = trainer.callback_metrics.get("train/loss")
            ppl = trainer.callback_metrics.get("train/ppl")
            print(
                f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | "
                f"loss={float(loss) if loss is not None else 'n/a'} | "
                f"ppl={float(ppl) if ppl is not None else 'n/a'}"
            )

# -------------------- DataModule --------------------
class VLDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        processor: Processor,
        batch_size=16,
        max_seq_len=1024,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.collate_fn = make_collate_fn(processor, max_seq_len=max_seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

# -------------------- main --------------------
def main():
    # helpful for Tensor Cores perf
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    ap.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp", "auto"])
    ap.add_argument("--proj_out", type=str, default="full_model.safetensors")
    ap.add_argument("--processor_repo", type=str, default="Yuna-2B")
    ap.add_argument("--cache_dir", type=str, default="./cache")
    args = ap.parse_args()

    # 1) Build model & processor on CPU (ensure your factory DOES NOT .to('cuda') internally)
    model = Yuna.from_pretrained("Yuna-2B")
    processor = Processor(
        repo_id=args.processor_repo, vision_config=model.config.vision_config
    )

    # 2) Data
    dataset = RawTextImageDataset(
        jsonl_path=os.path.join("output_combined_2048.jsonl"),
        #images_dir=os.path.join(args.cache_dir, "your_images_dir")  # or None if no images
        images_dir=None  # or None if no images
    )
    steps_per_epoch = max(1, math.ceil(len(dataset) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(100, int(0.02 * total_steps))
    dm = VLDataModule(
        dataset=dataset,
        processor=processor,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
    )

    # 3) Lightning module
    lit = ProjectionOnlyLM(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        log_every=1,
        block_cls=type(model.model.layers[0]),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch={epoch}-step={step}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,  # Save every epoch
        save_weights_only=False,  # Save full state (can set True if you want only weights)
    )

    # 4) Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="auto",
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        log_every_n_steps=1,
        callbacks=[SaveFullModelSafetensors(args.proj_out), PrintEpochSummary(), checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm)

if __name__ == "__main__":
    main()

"""
Example command to run:

!python train.py \
    --devices 1 \
    --batch_size 1 \
    --epochs 50 \
    --grad_accum 1 \
    --max_seq_len 2048 \
    --lr 1e-5 \
    --weight_decay 0.1 \
    --num_workers 2 \
    --precision bf16-mixed \
    --strategy auto \
    --proj_out full_model.safetensors \
    --processor_repo Yuna-2B \
    --cache_dir ./cache
"""