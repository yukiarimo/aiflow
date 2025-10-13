import argparse
import json
import os
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from .yuna.utils import load

class YunaDataset:
    def __init__(self, jsonl_path, images_dir=None, audio_dir=None):
        self.data = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8") if line.strip()]
        self.images_dir = images_dir
        self.audio_dir = audio_dir

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = [os.path.join(self.images_dir, p) for p in item.get('image_paths', [])] if self.images_dir else []
        audio_paths = [os.path.join(self.audio_dir, p) for p in item.get('audio_paths', [])] if self.audio_dir else []
        return {"chat": item["chat"], "image_paths": image_paths, "audio_paths": audio_paths}

def make_collate_fn(processor, max_seq_len):
    pad_id = processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0

    def collate(batch):
        batch_inputs = []
        for ex in batch:
            inputs = processor(messages=ex["chat"], image_paths=ex.get("image_paths"), audio_paths=ex.get("audio_paths"), add_generation_prompt=False) # Use add_generation_prompt=False for training
            if inputs["input_ids"].shape[1] > max_seq_len: continue
            labels = inputs["input_ids"].copy()

            for token_id in processor.placeholder_token_ids.values(): labels[labels == token_id] = -100
            inputs["labels"] = labels
            batch_inputs.append(inputs)

        if not batch_inputs: return None

        max_len_batch = max(item["input_ids"].shape[1] for item in batch_inputs)
        input_ids = np.full((len(batch_inputs), max_len_batch), pad_id, dtype=np.int32)
        labels = np.full((len(batch_inputs), max_len_batch), -100, dtype=np.int32)

        for i, item in enumerate(batch_inputs):
            seq_len = item["input_ids"].shape[1]
            input_ids[i, :seq_len] = item["input_ids"][0]
            labels[i, :seq_len] = item["labels"][0]

        pixels = mx.concatenate([item["pixel_values"] for item in batch_inputs if item["pixel_values"] is not None], axis=0) if any(item["pixel_values"] is not None for item in batch_inputs) else None
        d_image = mx.concatenate([item["d_image"] for item in batch_inputs if item["d_image"] is not None], axis=0) if any(item["d_image"] is not None for item in batch_inputs) else None
        audio_features = mx.concatenate([item["audio_features"] for item in batch_inputs if item["audio_features"] is not None], axis=0) if any(item["audio_features"] is not None for item in batch_inputs) else None
        audio_lens = mx.concatenate([item["audio_feature_lens"] for item in batch_inputs if item["audio_feature_lens"] is not None], axis=0) if any(item["audio_feature_lens"] is not None for item in batch_inputs) else None

        return {"input_ids": mx.array(input_ids), "labels": mx.array(labels), "pixel_values": pixels, "d_image": d_image, "audio_features": audio_features, "audio_feature_lens": audio_lens}
    return collate

def iterate_batches(dataset, processor, batch_size, max_seq_len):
    collate_fn = make_collate_fn(processor, max_seq_len)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        if len(batch_indices) < batch_size: continue
        batch_data = [dataset[idx] for idx in batch_indices]
        collated_batch = collate_fn(batch_data)
        if collated_batch: yield collated_batch

def loss_fn(model, batch):
    """A wrapper loss function that unpacks the batch and calls the model."""
    labels = batch.pop("labels")
    logits, _ = model(**batch)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return nn.losses.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="mean")

def main():
    parser = argparse.ArgumentParser(description="Yuna MLX Training")
    parser.add_argument("--model-repo", type=str, required=True)
    parser.add_argument("--jsonl-path", type=str, required=True)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--train-modules", type=str, nargs='+', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-path", type=str, default="checkpoints_mlx")
    args = parser.parse_args()

    print("[INFO] Loading model...")
    model, processor = load(args.model_repo)
    model.train()

    for name, param in model.named_parameters(): param.requires_grad = False
    print(f"[INFO] Unfreezing parameters for: {', '.join(args.train_modules)}")
    for module_name in args.train_modules:
        if module_name == 'llm': module = model.language_model
        elif module_name == 'vision_tower': module = model.vision_tower
        elif module_name == 'audio_tower': module = model.audio_tower
        elif module_name == 'audio_projector': module = model.audio_projector
        else: print(f"Warning: Module '{module_name}' not found."); continue
        for param in module.parameters(): param.requires_grad = True

    dataset = YunaDataset(args.jsonl_path, args.images_dir, args.audio_dir)
    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        for batch in iterate_batches(dataset, processor, args.batch_size, args.max_seq_len):
            # Pass the whole batch to the loss function
            loss, grads = loss_and_grad_fn(model, batch)
            for k, v in grads.items(): accumulated_grads[k] += v
            grad_accum_counter += 1

            if grad_accum_counter % args.grad_accum == 0:
                optimizer.update(model, {k: v / args.grad_accum for k, v in accumulated_grads.items()})
                mx.eval(model.parameters(), optimizer.state)

                loss_val = loss.item()
                total_loss += loss_val
                step_count += 1
                print(f"Epoch {epoch+1}, Step {step_count}, Loss: {loss_val:.4f}")
                accumulated_grads = {k: mx.zeros_like(v) for k, v in model.trainable_parameters().items()} # Reset accumulators

        avg_loss = total_loss / step_count if step_count > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_dir = Path(args.save_path) / f"epoch_{epoch+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(save_dir / "model.safetensors"))
        print(f"Saved checkpoint to {save_dir}")

if __name__ == "__main__": main()