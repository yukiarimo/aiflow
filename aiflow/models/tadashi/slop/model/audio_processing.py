import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
import glob

class AudioProcessor:
    """Base class for audio processing"""
    def __init__(self, sr=48000, n_fft=4096, hop_length=256, n_mels=256, fmax=24000, fixed_length=1400):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.fixed_length = fixed_length

    def process_chunk(self, audio_chunk):
        """Process audio chunk into mel spectrogram"""
        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk, 
            sr=self.sr, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize the spectrogram
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        # Pad or truncate to fixed length
        if mel_spec_db.shape[1] < self.fixed_length:
            pad_width = self.fixed_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :self.fixed_length]

        # Convert to PyTorch tensor
        return torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def load_and_process_file(self, file_path, chunk_duration=30):
        """Load audio file and split into chunks for processing"""
        # Load audio file
        audio, sr = librosa.load(file_path, sr=self.sr)

        # Calculate number of samples for chunks
        chunk_samples = chunk_duration * sr

        # Split audio into chunks
        chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

        # Ensure last chunk has enough samples (discard if less than 5 seconds)
        if len(chunks[-1]) < 5 * sr:
            chunks = chunks[:-1]

        if not chunks:
            return []

        # Process each chunk
        processed_chunks = [self.process_chunk(chunk) for chunk in chunks]

        return processed_chunks

class SpeechProcessor(AudioProcessor):
    """Specialized processor for speech audio"""
    def __init__(self, sr=48000, n_fft=2048, hop_length=256, n_mels=128, fmax=8000, fixed_length=1400):
        # Speech typically has lower frequency content, so we adjust parameters
        super(SpeechProcessor, self).__init__(sr, n_fft, hop_length, n_mels, fmax, fixed_length)

    def extract_speech_features(self, audio_chunk):
        """Extract additional speech-specific features"""
        # Basic mel spectrogram processing
        mel_spec_tensor = self.process_chunk(audio_chunk)

        # Additional speech-specific features could be added here
        # For example, pitch tracking, formant analysis, etc.

        return mel_spec_tensor

class InstrumentalProcessor(AudioProcessor):
    """Specialized processor for instrumental audio"""
    def __init__(self, sr=48000, n_fft=4096, hop_length=256, n_mels=256, fmax=24000, fixed_length=1400):
        # Instrumental music typically has wider frequency range
        super(InstrumentalProcessor, self).__init__(sr, n_fft, hop_length, n_mels, fmax, fixed_length)

    def extract_instrumental_features(self, audio_chunk):
        """Extract additional instrumental-specific features"""
        # Basic mel spectrogram processing
        mel_spec_tensor = self.process_chunk(audio_chunk)

        # Additional instrumental-specific features could be added here
        # For example, chroma features, spectral contrast, etc.

        return mel_spec_tensor

class MixedAudioProcessor(AudioProcessor):
    """Processor for general mixed audio"""
    def __init__(self, sr=48000, n_fft=4096, hop_length=256, n_mels=256, fmax=24000, fixed_length=1400):
        super(MixedAudioProcessor, self).__init__(sr, n_fft, hop_length, n_mels, fmax, fixed_length)

    def extract_mixed_features(self, audio_chunk):
        """Extract features for mixed audio content"""
        # Basic mel spectrogram processing
        mel_spec_tensor = self.process_chunk(audio_chunk)

        # Additional mixed audio features could be added here

        return mel_spec_tensor

def classify_audio(file_path, model, device, modality='mixed_audio'):
    """Classify audio file as human or AI generated"""
    # Select appropriate processor based on modality
    if modality == 'speech':
        processor = SpeechProcessor()
    elif modality == 'instrumental':
        processor = InstrumentalProcessor()
    else:  # mixed_audio
        processor = MixedAudioProcessor()

    # Load and process audio
    processed_chunks = processor.load_and_process_file(file_path)

    if not processed_chunks:
        return {"Human": 0, "AI": 0}

    # Process each chunk and average results
    human_prob_sum = 0
    ai_prob_sum = 0

    model.eval()
    with torch.no_grad():
        for input_tensor in processed_chunks:
            # Move tensor to device
            input_tensor = input_tensor.to(device)

            # Perform inference
            logits = model(input_tensor, modality)
            probabilities = F.softmax(logits, dim=1)

            human_prob_sum += probabilities[0, 0].item()
            ai_prob_sum += probabilities[0, 1].item()

    # Average the probabilities
    chunk_count = len(processed_chunks)
    human_percent = (human_prob_sum / chunk_count) * 100
    ai_percent = (ai_prob_sum / chunk_count) * 100

    return {"Human": human_percent, "AI": ai_percent}

def batch_process_audio(directory_path, model, device, modality='mixed_audio', file_types=None):
    """Process all audio files in a directory"""
    if file_types is None:
        file_types = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']

    results = {}

    # Get all matching files
    files = []
    for file_type in file_types:
        files.extend(glob.glob(os.path.join(directory_path, file_type)))

    for file_path in files:
        try:
            file_result = classify_audio(file_path, model, device, modality)
            results[os.path.basename(file_path)] = file_result
        except Exception as e:
            results[os.path.basename(file_path)] = f"Error: {str(e)}"

    return results
