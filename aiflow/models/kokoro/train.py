import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import EmotionalModel
from utils import load_config, create_dataloader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

class EmotionalLoss(nn.Module):
    """Loss function for emotional prediction with emotion-specific weighting"""
    def __init__(self, emotion_names, weights=None):
        super().__init__()
        self.emotion_names = emotion_names
        self.num_emotions = len(emotion_names)
        
        # Default to equal weights if not provided
        if weights is None:
            self.weights = {name: 1.0 for name in emotion_names}
        else:
            self.weights = weights
            
    def forward(self, predictions, targets):
        """Calculate weighted loss across emotions"""
        # Ensure both inputs have same shape [batch_size, num_emotions]
        batch_size = predictions.shape[0]
        target_batch_size = targets.shape[0]
        
        # Handle batch size mismatch by adapting targets to match predictions
        if batch_size != target_batch_size:
            print(f"Warning: Batch size mismatch! Predictions: {batch_size}, Targets: {target_batch_size}")
            if batch_size == 1 and target_batch_size > 1:
                # Broadcast single prediction to match target batch size
                predictions = predictions.expand(target_batch_size, -1)
            elif target_batch_size == 1 and batch_size > 1:
                # Broadcast single target to match prediction batch size
                targets = targets.expand(batch_size, -1)
            else:
                # More complex mismatch - use only the common elements
                min_batch = min(batch_size, target_batch_size)
                predictions = predictions[:min_batch]
                targets = targets[:min_batch]
        
        # Reshape targets if necessary
        if targets.dim() == 1 and predictions.dim() == 2:
            # If target is [num_emotions] and pred is [batch_size, num_emotions]
            targets = targets.unsqueeze(0).expand(batch_size, -1)
        elif targets.dim() == 3 and predictions.dim() == 2:
            # If target is [batch_size, 1, num_emotions]
            targets = targets.squeeze(1)
        
        # Use MSE loss for simplicity
        mse_loss = nn.functional.mse_loss(predictions, targets, reduction='none')
        
        # Apply weights for each emotion dimension
        weighted_loss = 0.0
        for i, emotion in enumerate(self.emotion_names):
            emotion_loss = mse_loss[:, i].mean()
            weighted_loss += self.weights[emotion] * emotion_loss
            
        return weighted_loss / self.num_emotions

def prepare_dataset(data_path, test_size=0.01, random_state=42, max_samples=1000):
    """Split dataset into train and validation sets"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Limit dataset size for faster testing
    if max_samples and len(data) > max_samples:
        print(f"Limiting dataset to {max_samples} samples (out of {len(data)})")
        data = data[:max_samples]
    
    # For very small datasets, ensure at least one sample in each split
    if len(data) <= 3:
        test_size = 1/len(data)
    
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Save the split datasets
    with open("train_data.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f, indent=4)
        
    print(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples")
    return "train_data.json", "val_data.json"

def train(config):
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Get model name for checkpoints
    model_name = config.get("model_name")
    model_name_inside = config.get("model_name")
    
    # Prepare datasets
    train_path, val_path = prepare_dataset("dataset.json", max_samples=1000000)
    
    # Setup device
    device = config["device"]
    print(f"Using device: {device}")
    
    # Calculate appropriate batch sizes based on dataset sizes
    with open(train_path, 'r') as f:
        train_size = len(json.load(f))
    with open(val_path, 'r') as f:
        val_size = len(json.load(f))
    
    train_batch_size = max(8, min(config["batch_size"], train_size // 100))
    val_batch_size = max(8, min(config["batch_size"], val_size // 10))
    
    print(f"Using batch sizes: {train_batch_size} (train), {val_batch_size} (val)")
    
    # Create dataloaders (only once)
    train_dataloader = create_dataloader(train_path, config, shuffle=True, batch_size=train_batch_size)
    val_dataloader = create_dataloader(val_path, config, shuffle=False, batch_size=val_batch_size)
    
    # Initialize model
    model = EmotionalModel(config).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * 0.01)
    
    # Custom weighted loss - can adjust weights if needed
    emotion_weights = {name: 1.0 for name in config["emotion_names"]}
    criterion = EmotionalLoss(config["emotion_names"], emotion_weights)
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 10  # Early stopping after 10 epochs without improvement
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        batch_count = 0
        for memory_emb_batch, input_emb_batch, target_state_batch in progress_bar:
            batch_count += 1
            
            try:
                # Move tensors to device with proper shape handling
                memory_emb_batch = memory_emb_batch.to(torch.float32).to(device)
                input_emb_batch = input_emb_batch.to(torch.float32).to(device)
                target_state_batch = target_state_batch.to(torch.float32).to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                emotion_outputs, _ = model(input_emb_batch)
                predicted_states = torch.cat([emotion_outputs[emotion_name] for emotion_name in config["emotion_names"]], dim=1)
                
                # Calculate loss - with improved handling
                loss = criterion(predicted_states, target_state_batch)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("Warning: NaN loss detected! Using alternate loss calculation.")
                    # Try a simpler loss function as fallback
                    loss = torch.nn.functional.l1_loss(predicted_states, target_state_batch)
                    
                    if torch.isnan(loss):
                        print("Still getting NaN loss. Skipping this batch.")
                        continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Update weights
                optimizer.step()
                
                # Update stats
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except RuntimeError as e:
                print(f"Error in batch {batch_count}: {str(e)}")
                continue
        
        if batch_count > 0:
            avg_train_loss = train_loss / batch_count
            train_losses.append(avg_train_loss)
        else:
            print("Warning: No valid batches in training!")
            avg_train_loss = float('nan')
            train_losses.append(avg_train_loss)
        
        # Validation phase with the same error checking
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            for memory_emb_batch, input_emb_batch, target_state_batch in progress_bar:
                try:
                    val_batch_count += 1
                    # Move tensors to device with proper type conversion
                    memory_emb_batch = memory_emb_batch.to(torch.float32).to(device)
                    input_emb_batch = input_emb_batch.to(torch.float32).to(device)
                    
                    # Ensure target is a 2D tensor [batch_size, num_emotions]
                    if target_state_batch.dim() == 3:
                        target_state_batch = target_state_batch.squeeze(1)
                    target_state_batch = target_state_batch.to(torch.float32).to(device)
                    
                    # Forward pass
                    emotion_outputs, _ = model(input_emb_batch)
                    predicted_states = torch.cat([emotion_outputs[emotion_name] for emotion_name in config["emotion_names"]], dim=1)
                    
                    # Calculate loss - safely
                    loss = criterion(predicted_states, target_state_batch)
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                    
                except RuntimeError as e:
                    print(f"Error in validation batch {val_batch_count}: {str(e)}")
                    continue
        
        if val_batch_count > 0:
            avg_val_loss = val_loss / val_batch_count
            val_losses.append(avg_val_loss)
        else:
            print("Warning: No valid batches in validation!")
            avg_val_loss = float('nan')
            val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Only save checkpoint if loss is valid
        if not torch.isnan(torch.tensor(avg_val_loss)) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model with model name
            torch.save({
                'model_name': model_name_inside,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, f"checkpoints/{model_name}_best_model.pth")
            
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_name': model_name_inside,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, f"checkpoints/{model_name}_epoch_{epoch+1}.pth")
    
    # Save final model with model name
    torch.save({
        'model_name': model_name_inside,
        'model_state_dict': model.state_dict(),
        'config': config
    }, f"{model_name}_final.pth")
    
    # Plot training progress (filter out NaN values for plotting)
    plt.figure(figsize=(10, 6))
    train_losses_clean = [x for x in train_losses if not np.isnan(x)]
    val_losses_clean = [x for x in val_losses if not np.isnan(x)]
    plt.plot(range(len(train_losses_clean)), train_losses_clean, label='Training Loss')
    plt.plot(range(len(val_losses_clean)), val_losses_clean, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/{model_name}_training_progress.png')
    
    print(f"Training complete. Final model saved to {model_name}_final.pth")
    if not np.isnan(best_val_loss):
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print("Warning: Best validation loss was NaN. Model may not have trained properly.")

if __name__ == "__main__":
    config = load_config()
    train(config)