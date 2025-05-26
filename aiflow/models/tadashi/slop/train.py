import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import librosa
import argparse
from model.model import MultiModalAIDetector
from model.model import process_audio_chunk
from model.image_processing import ImageProcessor
from model.video_processing import VideoProcessor
import argparse

class AIContentDataset(Dataset):
    def __init__(self, root_dir, modality, transform=None):
        self.root_dir = root_dir
        self.modality = modality
        self.transform = transform
        self.samples = []

        # Set up file extensions based on modality
        if modality in ['speech', 'instrumental', 'mixed_audio']:
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        elif modality == 'image':
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        elif modality == 'video':
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Find all files with the specified extensions
        human_files = []
        ai_files = []

        for ext in extensions:
            human_files.extend(glob.glob(os.path.join(root_dir, 'human', f'*{ext}')))
            ai_files.extend(glob.glob(os.path.join(root_dir, 'ai', f'*{ext}')))

        # Create sample list with labels (0 for human, 1 for AI)
        for file_path in human_files:
            self.samples.append((file_path, 0))

        for file_path in ai_files:
            self.samples.append((file_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # Process based on modality
        if self.modality in ['speech', 'instrumental', 'mixed_audio']:
            # Load audio file (30-second chunk)
            audio, sr = librosa.load(file_path, sr=48000, duration=30)
            # Process audio
            features = process_audio_chunk(audio).squeeze(0)

        elif self.modality == 'image':
            # Load and process image
            image_processor = ImageProcessor()
            features = image_processor.process_image(file_path).squeeze(0)

        elif self.modality == 'video':
            # Use VideoProcessor to extract frames
            video_processor = VideoProcessor()
            processed_data = video_processor.process_video(file_path)
            features = processed_data['frames']  # Extract frames tensor

        return features, torch.tensor(label, dtype=torch.long)

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, modality='speech'):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            print(f"Input shape: {inputs.shape}")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, modality)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return model

def main():
    parser = argparse.ArgumentParser(description='Train AI Content Detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--modality', type=str, required=True, 
                        choices=['speech', 'instrumental', 'mixed_audio', 'image', 'video'],
                        help='Content modality to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='model/model.pth', help='Path to save model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')

    args = parser.parse_args()

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = AIContentDataset(args.data_dir, args.modality)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    model = MultiModalAIDetector().to(device)

    # Load checkpoint if specified
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    model = train_model(model, dataloader, criterion, optimizer, device, args.epochs, args.modality)

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()