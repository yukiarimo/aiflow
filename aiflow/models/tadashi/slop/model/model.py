import torch
import torch.nn as nn
import numpy as np
import librosa
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

class BaseFeatureExtractor(nn.Module):
    """Base class for all feature extractors"""
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class SpeechFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor specialized for speech audio"""
    def __init__(self):
        super(SpeechFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Attention mechanism for speech patterns
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv(x)
        attention_weights = self.attention(features)
        features = features * attention_weights
        # Global average pooling along frequency dimension
        features = features.mean(dim=2)
        # Output shape: [batch_size, time_steps, channels]
        features = features.permute(0, 2, 1)
        return features

class InstrumentalFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor specialized for instrumental audio"""
    def __init__(self):
        super(InstrumentalFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # Wider kernel for broader frequency patterns
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        features = self.conv(x)
        # Global average pooling along frequency dimension
        features = features.mean(dim=2)
        # Output shape: [batch_size, time_steps, channels]
        features = features.permute(0, 2, 1)
        return features

class MixedAudioFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for general audio (mixed speech and instrumental)"""
    def __init__(self):
        super(MixedAudioFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        features = self.conv(x)
        # Global average pooling along frequency dimension
        features = features.mean(dim=2)
        # Output shape: [batch_size, time_steps, channels]
        features = features.permute(0, 2, 1)
        return features

class ImageFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for images"""
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        # MobileNet-inspired architecture for efficiency
        self.conv = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Depthwise separable convolutions
            # Block 1
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Depthwise
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),  # Depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),  # Depthwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Final block
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),  # Depthwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        # Reshape to match the format expected by the classifier
        # [batch_size, 1, feature_dim]
        return features.unsqueeze(1)

class VideoFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for video frames"""
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        # Shared CNN for frame processing
        self.frame_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        # Check if input has an extra dimension
        if x.dim() == 6:  # [batch_size, 1, num_frames, channels, height, width]
            # Remove the extra dimension
            x = x.squeeze(1)
        
        # Now x shape should be: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.size()

        # Process each frame individually
        frame_features = []
        for i in range(num_frames):
            # Extract single frame [batch_size, channels, height, width]
            frame = x[:, i, :, :, :]

            # Process through frame processor
            features = self.frame_processor(frame)
            features = features.view(features.size(0), -1)
            features = self.fc(features)

            frame_features.append(features)

        # Stack frame features [batch_size, num_frames, features]
        features = torch.stack(frame_features, dim=1)

        return features

class TemporalProcessor(nn.Module):
    """Processes temporal/sequential data"""
    def __init__(self, input_size=128, hidden_size=128):
        super(TemporalProcessor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        output, _ = self.gru(x)
        # Use last timestep from both directions
        last_forward = output[:, -1, :self.gru.hidden_size]
        last_backward = output[:, 0, self.gru.hidden_size:]
        combined = torch.cat([last_forward, last_backward], dim=1)
        return combined

class ModalityAdapter(nn.Module):
    """Adapts features from different modalities to a common space"""
    def __init__(self, input_size, output_size=128):
        super(ModalityAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.fc(x)

class ClassificationHead(nn.Module):
    """Final classification layers"""
    def __init__(self, input_size=128, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class MultiModalAIDetector(nn.Module):
    """Unified model for detecting AI-generated content across multiple modalities"""
    def __init__(self):
        super(MultiModalAIDetector, self).__init__()

        # Feature extractors for different modalities
        self.speech_extractor = SpeechFeatureExtractor()
        self.instrumental_extractor = InstrumentalFeatureExtractor()
        self.mixed_audio_extractor = MixedAudioFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor()

        # Temporal processor for sequential data
        self.temporal_processor = TemporalProcessor(input_size=128, hidden_size=128)

        # Modality adapters
        self.audio_adapter = ModalityAdapter(input_size=256, output_size=128)  # 256 from bidirectional GRU
        self.image_adapter = ModalityAdapter(input_size=128, output_size=128)
        self.video_adapter = ModalityAdapter(input_size=256, output_size=128)  # 256 from bidirectional GRU

        # Classification head
        self.classifier = ClassificationHead(input_size=128, num_classes=2)

        # Track active modality
        self.active_modality = None

    def forward(self, x, modality):
        """
        Forward pass through the model

        Args:
            x: Input data
            modality: One of 'speech', 'instrumental', 'mixed_audio', 'image', 'video'

        Returns:
            Classification logits
        """
        self.active_modality = modality

        # Memory optimization: Only load components needed for current modality
        with torch.no_grad():
            # Clear unused components from CUDA memory if possible
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if modality == 'speech':
            features = self.speech_extractor(x)
            features = self.temporal_processor(features)
            features = self.audio_adapter(features)

        elif modality == 'instrumental':
            features = self.instrumental_extractor(x)
            features = self.temporal_processor(features)
            features = self.audio_adapter(features)

        elif modality == 'mixed_audio':
            features = self.mixed_audio_extractor(x)
            features = self.temporal_processor(features)
            features = self.audio_adapter(features)

        elif modality == 'image':
            features = self.image_extractor(x)
            features = features.squeeze(1)  # Remove sequence dimension
            features = self.image_adapter(features)

        elif modality == 'video':
            features = self.video_extractor(x)
            features = self.temporal_processor(features)
            features = self.video_adapter(features)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        return self.classifier(features)

# Preprocessing functions
def process_audio_chunk(audio_chunk, sr=48000, fixed_length=1400):
    """Process audio chunk into mel spectrogram"""
    # Generate Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk, 
        sr=sr, 
        n_fft=4096,
        hop_length=256,
        n_mels=256,
        fmax=24000,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize the spectrogram
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    # Pad or truncate to fixed length
    if mel_spec_db.shape[1] < fixed_length:
        pad_width = fixed_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :fixed_length]

    # Convert to PyTorch tensor
    return torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def process_image(image_path, target_size=(224, 224)):
    """Process image for the model"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def extract_frames(video_path, num_frames=30, target_size=(224, 224)):
    """Extract frames from video for processing"""
    import cv2

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    # Calculate frame indices to extract (evenly spaced)
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]

    # Extract frames
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transformations
            frame_tensor = transform(frame)
            frames.append(frame_tensor)

    cap.release()

    # If we couldn't extract enough frames, pad with zeros
    while len(frames) < num_frames:
        frames.append(torch.zeros_like(frames[0]))

    # Stack frames into a single tensor [num_frames, channels, height, width]
    return torch.stack(frames).unsqueeze(0)  # Add batch dimension

def extract_audio_from_video(video_path, sr=48000):
    """Extract audio from video file"""
    import moviepy.editor as mp

    try:
        video = mp.VideoFileClip(video_path)
        if video.audio is not None:
            # Extract audio to temporary file
            temp_audio_path = video_path + ".temp.wav"
            video.audio.write_audiofile(temp_audio_path, fps=sr, nbytes=2, verbose=False, logger=None)

            # Load audio
            audio, _ = librosa.load(temp_audio_path, sr=sr)

            # Remove temporary file
            os.remove(temp_audio_path)

            return audio
        else:
            return None
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

# Classification functions
def classify_audio(file_path, model, device, modality='mixed_audio'):
    """Classify audio file as human or AI generated"""
    # Load audio file at 48kHz
    audio, sr = librosa.load(file_path, sr=48000)

    # Calculate number of samples for 30-second chunks
    chunk_samples = 30 * sr

    # Split audio into 30-second chunks
    chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

    # Ensure last chunk has enough samples (discard if less than 5 seconds)
    if len(chunks[-1]) < 5 * sr:
        chunks = chunks[:-1]

    if not chunks:
        return {"Human": 0, "AI": 0}

    # Process each chunk and average results
    human_prob_sum = 0
    ai_prob_sum = 0

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            # Process the chunk
            input_tensor = process_audio_chunk(chunk).to(device)

            # Perform inference
            logits = model(input_tensor, modality)
            probabilities = torch.nn.functional.softmax(logits, dim=1)

            human_prob_sum += probabilities[0, 0].item()
            ai_prob_sum += probabilities[0, 1].item()

    # Average the probabilities
    chunk_count = len(chunks)
    human_percent = (human_prob_sum / chunk_count) * 100
    ai_percent = (ai_prob_sum / chunk_count) * 100

    return {"Human": human_percent, "AI": ai_percent}

def classify_image(file_path, model, device):
    """Classify image file as human or AI generated"""
    # Process the image
    input_tensor = process_image(file_path).to(device)

    model.eval()
    with torch.no_grad():
        # Perform inference
        logits = model(input_tensor, 'image')
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        human_percent = probabilities[0, 0].item() * 100
        ai_percent = probabilities[0, 1].item() * 100

    return {"Human": human_percent, "AI": ai_percent}

def classify_video(file_path, model, device):
    """Classify video file as human or AI generated"""
    # Extract frames from video
    frames = extract_frames(file_path).to(device)

    # Extract audio if available
    audio = extract_audio_from_video(file_path)

    model.eval()
    with torch.no_grad():
        # Process video frames
        video_logits = model(frames, 'video')
        video_probs = torch.nn.functional.softmax(video_logits, dim=1)

        # Initialize results
        human_percent = video_probs[0, 0].item() * 100
        ai_percent = video_probs[0, 1].item() * 100

        # If audio is available, process it too and combine results
        if audio is not None:
            # Calculate number of samples for 30-second chunks
            chunk_samples = 30 * 48000

            # Split audio into 30-second chunks
            chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

            # Ensure last chunk has enough samples (discard if less than 5 seconds)
            if len(chunks[-1]) < 5 * 48000:
                chunks = chunks[:-1]

            if chunks:
                # Process each chunk and average results
                audio_human_sum = 0
                audio_ai_sum = 0

                for chunk in chunks:
                    # Process the chunk
                    input_tensor = process_audio_chunk(chunk).to(device)

                    # Perform inference
                    audio_logits = model(input_tensor, 'mixed_audio')
                    audio_probs = torch.nn.functional.softmax(audio_logits, dim=1)

                    audio_human_sum += audio_probs[0, 0].item()
                    audio_ai_sum += audio_probs[0, 1].item()

                # Average the audio probabilities
                chunk_count = len(chunks)
                audio_human_percent = (audio_human_sum / chunk_count) * 100
                audio_ai_percent = (audio_ai_sum / chunk_count) * 100

                # Combine video and audio results (simple average)
                human_percent = (human_percent + audio_human_percent) / 2
                ai_percent = (ai_percent + audio_ai_percent) / 2

    return {"Human": human_percent, "AI": ai_percent}

def classify_file(file_path, model, device):
    """Classify a file based on its extension"""
    file_ext = os.path.splitext(file_path)[1].lower()

    # Audio files
    if file_ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        return classify_audio(file_path, model, device, 'mixed_audio')

    # Image files
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        return classify_image(file_path, model, device)

    # Video files
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return classify_video(file_path, model, device)

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

def classify_directory(directory_path, model, device, file_types=None):
    """Classify all supported files in a directory"""
    if file_types is None:
        file_types = [
            # Audio
            '*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a',
            # Images
            '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
            # Videos
            '*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm'
        ]

    results = {}

    # Get all matching files
    files = []
    for file_type in file_types:
        files.extend(glob.glob(os.path.join(directory_path, file_type)))

    for file_path in files:
        try:
            # Clear CUDA cache if available
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()

            file_result = classify_file(file_path, model, device)
            results[os.path.basename(file_path)] = file_result
        except Exception as e:
            results[os.path.basename(file_path)] = f"Error: {str(e)}"

    return results
