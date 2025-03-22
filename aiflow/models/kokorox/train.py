import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ContentFilter
from aiflow.utils import load_config, get_text_embedding, load_kokoro_model
import argparse
from sklearn.model_selection import train_test_split

class KokoroXDataset(Dataset):
    """Dataset for training KokoroX content filter."""
    def __init__(self, data_path, config, embedding_function, cache_dir="embedding_cache", kokoro_model=None):
        self.config = config
        self.get_embedding = embedding_function
        self.emotion_names = config["emotion_names"]

        # Load the Kokoro model if not provided
        if kokoro_model is None:
            from utils import load_kokoro_model
            self.kokoro_model = load_kokoro_model(config)
        else:
            self.kokoro_model = kokoro_model

        # Move model to appropriate device
        self.device = torch.device(config["device"])
        self.kokoro_model.to(self.device)

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Setup cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create emotion scores cache directory
        self.emotion_cache_dir = os.path.join(self.cache_dir, "emotions")
        os.makedirs(self.emotion_cache_dir, exist_ok=True)

        # Pre-compute embeddings (optional)
        self._precompute_embeddings()

        # Pre-compute emotion scores
        self._precompute_emotions()

    def _precompute_embeddings(self):
            """Pre-compute and cache embeddings to speed up training."""
            for idx, item in enumerate(tqdm(self.data, desc="Caching embeddings")):
                # Cache question embedding
                q_cache_path = os.path.join(self.cache_dir, f"question_{idx}.npy")
                if not os.path.exists(q_cache_path):
                    q_emb = self.get_embedding(item["question"])
                    np.save(q_cache_path, q_emb)

                # Cache content embeddings (good and bad references)
                good_cache_path = os.path.join(self.cache_dir, f"good_{idx}.npy")
                if not os.path.exists(good_cache_path):
                    good_emb = self.get_embedding(item["good_reference"])
                    np.save(good_cache_path, good_emb)

                bad_cache_path = os.path.join(self.cache_dir, f"bad_{idx}.npy")
                if not os.path.exists(bad_cache_path):
                    bad_emb = self.get_embedding(item["bad_reference"])
                    np.save(bad_cache_path, bad_emb)

    def _precompute_emotions(self):
        """Pre-compute emotion scores using the Kokoro model"""
        for idx, _ in enumerate(tqdm(self.data, desc="Calculating emotion scores")):
            good_cache_path = os.path.join(self.emotion_cache_dir, f"good_emotions_{idx}.npy")
            bad_cache_path = os.path.join(self.emotion_cache_dir, f"bad_emotions_{idx}.npy")

            if not os.path.exists(good_cache_path) or not os.path.exists(bad_cache_path):
                # Load content embeddings
                good_emb_path = os.path.join(self.cache_dir, f"good_{idx}.npy")
                bad_emb_path = os.path.join(self.cache_dir, f"bad_{idx}.npy")

                good_emb = np.load(good_emb_path)
                bad_emb = np.load(bad_emb_path)

                # Process with Kokoro model
                with torch.no_grad():
                    # Process good content
                    good_tensor = torch.tensor(good_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                    good_emotion_outputs, _ = self.kokoro_model(good_tensor)
                    good_emotions = torch.cat([good_emotion_outputs[name] for name in self.emotion_names], dim=1)

                    # Process bad content
                    bad_tensor = torch.tensor(bad_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                    bad_emotion_outputs, _ = self.kokoro_model(bad_tensor)
                    bad_emotions = torch.cat([bad_emotion_outputs[name] for name in self.emotion_names], dim=1)

                    # Save to cache
                    np.save(good_cache_path, good_emotions.cpu().numpy())
                    np.save(bad_cache_path, bad_emotions.cpu().numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get cached embeddings
        q_cache_path = os.path.join(self.cache_dir, f"question_{idx}.npy")
        good_cache_path = os.path.join(self.cache_dir, f"good_{idx}.npy")
        bad_cache_path = os.path.join(self.cache_dir, f"bad_{idx}.npy")

        question_emb = np.load(q_cache_path)
        good_emb = np.load(good_cache_path)
        bad_emb = np.load(bad_cache_path)

        # Get real emotion scores from the cached Kokoro model predictions
        good_emotions_path = os.path.join(self.emotion_cache_dir, f"good_emotions_{idx}.npy")
        bad_emotions_path = os.path.join(self.emotion_cache_dir, f"bad_emotions_{idx}.npy")

        good_emotions = np.load(good_emotions_path).flatten()
        bad_emotions = np.load(bad_emotions_path).flatten()

        # Return as tensors
        return {
            "question": torch.tensor(question_emb, dtype=torch.float32),
            "good_content": torch.tensor(good_emb, dtype=torch.float32),
            "bad_content": torch.tensor(bad_emb, dtype=torch.float32),
            "good_emotions": torch.tensor(good_emotions, dtype=torch.float32),
            "bad_emotions": torch.tensor(bad_emotions, dtype=torch.float32),
            "target": item["target"],
        }

def train_content_filter(config_path="config.json"):
    # Load config
    config = load_config(config_path)
    device = torch.device(config["device"])

    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize content filter model
    model = ContentFilter(config).to(device)

    # Create dataset splits
    with open(config["dataset_path"], 'r') as f:
        all_data = json.load(f)

    # Split data
    train_data, val_data = train_test_split(all_data, test_size=0.01, random_state=42)

    # Save splits for reference
    with open("train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")

    # Create datasets with the shared model
    kokoro_model = load_kokoro_model(config)
    train_dataset = KokoroXDataset("train_data.json", config, get_text_embedding, kokoro_model=kokoro_model)
    val_dataset = KokoroXDataset("val_data.json", config, get_text_embedding, kokoro_model=kokoro_model)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0  # Adjust based on your system
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=0
    )

    # Define loss function and optimizer
    # Using BCELoss for binary classification of content quality
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training variables
    epochs = config["epochs"]
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Get data from batch
            question = batch["question"].to(device)
            good_content = batch["good_content"].to(device)
            bad_content = batch["bad_content"].to(device)
            good_emotions = batch["good_emotions"].to(device)
            bad_emotions = batch["bad_emotions"].to(device)

            batch_size = question.size(0)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass for good content (should get high score)
            good_scores = model(question, good_content, good_emotions)
            good_target = torch.ones_like(good_scores)

            # Forward pass for bad content (should get low score)
            bad_scores = model(question, bad_content, bad_emotions)
            bad_target = torch.zeros_like(bad_scores)

            # Calculate loss
            good_loss = criterion(good_scores, good_target)
            bad_loss = criterion(bad_scores, bad_target)
            loss = good_loss + bad_loss

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * batch_size

            # Calculate accuracy
            good_correct = (good_scores > 0.5).sum().item()
            bad_correct = (bad_scores < 0.5).sum().item()
            train_correct += good_correct + bad_correct
            train_total += batch_size * 2

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "acc": f"{(good_correct + bad_correct)/(batch_size*2):.4f}"
            })

        # Calculate average training loss and accuracy
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                # Get data from batch
                question = batch["question"].to(device)
                good_content = batch["good_content"].to(device)
                bad_content = batch["bad_content"].to(device)
                good_emotions = batch["good_emotions"].to(device)
                bad_emotions = batch["bad_emotions"].to(device)

                batch_size = question.size(0)

                # Forward pass for good content
                good_scores = model(question, good_content, good_emotions)
                good_target = torch.ones_like(good_scores)

                # Forward pass for bad content
                bad_scores = model(question, bad_content, bad_emotions)
                bad_target = torch.zeros_like(bad_scores)

                # Calculate loss
                good_loss = criterion(good_scores, good_target)
                bad_loss = criterion(bad_scores, bad_target)
                loss = good_loss + bad_loss

                # Update statistics
                val_loss += loss.item() * batch_size

                # Calculate accuracy
                good_correct = (good_scores > 0.5).sum().item()
                bad_correct = (bad_scores < 0.5).sum().item()
                val_correct += good_correct + bad_correct
                val_total += batch_size * 2

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "acc": f"{(good_correct + bad_correct)/(batch_size*2):.4f}"
                })

        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'config': config
            }, "checkpoints/kokorox_best_model.pth")

            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Yuna Ai Kokoro V1 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/kokorox_training_progress.png')

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/kokorox_best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train KokoroX content filter')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    train_content_filter(args.config)