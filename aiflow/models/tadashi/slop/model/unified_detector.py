import torch
import torch.nn as nn
import os
import glob
from .model import MultiModalAIDetector
from .audio_processing import classify_audio, batch_process_audio
from .image_processing import AIImageDetector
from .video_processing import AIVideoDetector
from .model import process_audio_chunk
from .image_processing import ImageProcessor
from .video_processing import VideoProcessor
import librosa

class AIContentDetector:
    """Unified interface for AI-generated content detection across multiple modalities"""
    def __init__(self, model_path=None, device=None):
        # Determine device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = device

        # Initialize model
        self.model = MultiModalAIDetector().to(self.device)

        # Load model weights if provided
        if model_path is not None:
            self.load_model(model_path)

        # Initialize detectors
        self.image_detector = AIImageDetector(self.model, self.device)
        self.video_detector = AIVideoDetector(self.model, self.device)

    def load_model(self, model_path):
        """Load model weights from file"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def save_model(self, model_path):
        """Save model weights to file"""
        try:
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def classify_file(self, file_path, modality=None):
        """
        Classify a file as human or AI generated

        Args:
            file_path: Path to the file
            modality: Optional modality override ('speech', 'instrumental', 'mixed_audio', 'image', 'video')
                      If None, will be determined from file extension

        Returns:
            Dictionary with Human and AI percentages
        """
        # Determine modality from file extension if not provided
        if modality is None:
            file_ext = os.path.splitext(file_path)[1].lower()

            # Audio files
            if file_ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                modality = 'mixed_audio'

            # Image files
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                modality = 'image'

            # Video files
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                modality = 'video'

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        # Process based on modality
        if modality in ['speech', 'instrumental', 'mixed_audio']:
            return classify_audio(file_path, self.model, self.device, modality)

        elif modality == 'image':
            return self.image_detector.classify_image(file_path)

        elif modality == 'video':
            return self.video_detector.classify_video(file_path)

        else:
            raise ValueError(f"Unknown modality: {modality}")

    def classify_directory(self, directory_path, modality=None, file_types=None):
        """
        Classify all supported files in a directory

        Args:
            directory_path: Path to the directory
            modality: Optional modality override (if None, determined per file)
            file_types: Optional list of file patterns to match (e.g., ['*.jpg', '*.png'])

        Returns:
            Dictionary mapping filenames to classification results
        """
        if file_types is None:
            if modality in ['speech', 'instrumental', 'mixed_audio']:
                file_types = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
            elif modality == 'image':
                file_types = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            elif modality == 'video':
                file_types = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            else:
                # All supported formats
                file_types = [
                    # Audio
                    '*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a',
                    # Images
                    '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                    # Videos
                    '*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm'
                ]

        # Process based on modality
        if modality in ['speech', 'instrumental', 'mixed_audio']:
            return batch_process_audio(directory_path, self.model, self.device, modality, file_types)

        elif modality == 'image':
            return self.image_detector.batch_process_images(directory_path, file_types)

        elif modality == 'video':
            return self.video_detector.batch_process_videos(directory_path, file_types)

        else:
            # Process all files with appropriate detector
            results = {}

            # Get all matching files
            files = []
            for file_type in file_types:
                files.extend(glob.glob(os.path.join(directory_path, file_type)))

            for file_path in files:
                try:
                    file_result = self.classify_file(file_path)
                    results[os.path.basename(file_path)] = file_result
                except Exception as e:
                    results[os.path.basename(file_path)] = f"Error: {str(e)}"

            return results

    def train_model(self, train_data, modality, epochs=10, batch_size=32, learning_rate=0.001):
        """
        Train the model on a specific modality
        """
        # Create dataset and dataloader from the train_data dictionary
        train_dataset = []
        train_labels = []
        
        # Process each file in train_data
        for file_path, label in train_data.items():
            if modality in ['speech', 'instrumental', 'mixed_audio']:
                audio, sr = librosa.load(file_path, sr=48000, duration=30)
                features = process_audio_chunk(audio)
            elif modality == 'image':
                image_processor = ImageProcessor()
                features = image_processor.process_image(file_path)
            elif modality == 'video':
                video_processor = VideoProcessor()
                processed_data = video_processor.process_video(file_path)
                features = processed_data['frames']
            
            train_dataset.append(features)
            train_labels.append(label)
        
        # Convert to appropriate tensors
        X = torch.stack(train_dataset)
        y = torch.tensor(train_labels, dtype=torch.long)
        
        # Create TensorDataset and DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Set model to training mode
        self.model.train()
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        print("Training complete.")
        # Save the model after training
        self.save_model("trained_model_big.pth")
        print("Model saved after training.")