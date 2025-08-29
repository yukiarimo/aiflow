import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import KokoroModel, EmbeddingGenerator

class EmotionalDataset(Dataset):
    """A clean dataset that returns embeddings and raw targets."""
    def __init__(self, config, embed_generator):
        self.config = config
        self.emotion_names = config["emotion_names"]
        with open(config["dataset_path"], 'r') as f:
            self.data = json.load(f)

        print("Pre-computing all embeddings for the dataset...")
        self.embeddings = [
            (
                embed_generator.get_embedding(item["memory"]),
                embed_generator.get_embedding(item["input"])
            )
            for item in tqdm(self.data, desc="Generating embeddings")
        ]
        self.targets = torch.tensor([[item["state"][name] for name in self.emotion_names] for item in self.data], dtype=torch.float32)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        memory_emb, input_emb = self.embeddings[idx]
        return torch.from_numpy(memory_emb), torch.from_numpy(input_emb), self.targets[idx]

def train(config):
    device = torch.device(config['device'] if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    embed_generator = EmbeddingGenerator(config)
    dataset = EmotionalDataset(config, embed_generator)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size

    print(f"Dataset size: {len(dataset)}. Training on {train_size}, validating on {val_size}.")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0)

    model = KokoroModel(config).to(device)
    # Load pre-trained weights if available
    if config.get("pretrained_model_path") and os.path.isfile(config["pretrained_model_path"]):
        model.load_state_dict(torch.load(config["pretrained_model_path"], map_location=device))
        print(f"Loaded pre-trained model from {config['pretrained_model_path']}")
    criterion = nn.MSELoss()

    base_lr = config["learning_rate"]
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=config["weight_decay"])

    warmup_epochs = config["lr_warmup_epochs"]
    t_max = config["epochs"] - warmup_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "lr": []}
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0

        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for memory_emb, input_emb, targets in train_bar:
            memory_emb, input_emb, targets = memory_emb.to(device), input_emb.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions_dict, _ = model(input_emb, memory_emb)
            predictions_tensor = torch.stack([predictions_dict[name] for name in config["emotion_names"]], dim=1)
            loss = criterion(predictions_tensor, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for memory_emb, input_emb, targets in val_loader:
                memory_emb, input_emb, targets = memory_emb.to(device), input_emb.to(device), targets.to(device)
                predictions_dict, _ = model(input_emb, memory_emb)
                predictions_tensor = torch.stack([predictions_dict[name] for name in config["emotion_names"]], dim=1)
                loss = criterion(predictions_tensor, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        history["val_loss"].append(avg_val_loss)

        if epoch >= warmup_epochs: scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            model_path = f"checkpoints/{config['model_name']}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    # Plotting training history
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch"), ax1.set_ylabel("MSE Loss"), ax1.legend(loc='upper left'), ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(history["lr"], label="Learning Rate", color='tab:green', linestyle='--')
    ax2.set_ylabel("Learning Rate"), ax2.legend(loc='upper right')
    plt.title(f"{config['model_name']} Training Progress")
    plt.savefig(f"{config['model_name']}-training-progress.png")
    print(f"Training plot saved to {config['model_name']}-training-progress.png")

if __name__ == "__main__":
    with open("config.json", 'r') as f: config = json.load(f)
    train(config)