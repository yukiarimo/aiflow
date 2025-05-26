import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import moviepy.editor as mp
import torchvision.transforms as transforms
from .audio_processing import MixedAudioProcessor
import librosa

class VideoProcessor:
    """Processor for video data"""
    def __init__(self, target_size=(224, 224), num_frames=30, frame_rate=30):
        self.target_size = target_size
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.audio_processor = MixedAudioProcessor()

    def extract_frames(self, video_path):
        """Extract frames from video for processing"""
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps

        # Calculate frame indices to extract (evenly spaced)
        if frame_count <= self.num_frames:
            frame_indices = list(range(frame_count))
        else:
            frame_indices = [int(i * frame_count / self.num_frames) for i in range(self.num_frames)]

        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transformations
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)

        cap.release()

        # If we couldn't extract enough frames, pad with zeros
        while len(frames) < self.num_frames:
            if frames:
                frames.append(torch.zeros_like(frames[0]))
            else:
                # Create a blank frame if no frames were extracted
                blank = torch.zeros(3, self.target_size[0], self.target_size[1])
                frames.append(blank)

        # Stack frames into a single tensor [num_frames, channels, height, width]
        return torch.stack(frames).unsqueeze(0)  # Add batch dimension

    def extract_audio(self, video_path, sr=48000):
        """Extract audio from video file"""
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

    def process_video(self, video_path):
        """Process video for the model, extracting both frames and audio if available"""
        # Extract frames
        frames = self.extract_frames(video_path)

        # Extract audio if available
        audio = self.extract_audio(video_path)

        # Process audio if available
        audio_features = None
        if audio is not None:
            # Calculate number of samples for 30-second chunks
            chunk_samples = 30 * 48000

            # Split audio into 30-second chunks
            chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

            # Ensure last chunk has enough samples (discard if less than 5 seconds)
            if len(chunks[-1]) < 5 * 48000:
                chunks = chunks[:-1]

            if chunks:
                # Process each chunk
                audio_features = [self.audio_processor.process_chunk(chunk) for chunk in chunks]

        return {
            'frames': frames,
            'audio_features': audio_features
        }

class AIVideoDetector:
    """Detector for AI-generated videos"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.processor = VideoProcessor()

    def classify_video(self, file_path):
        """Classify video file as human or AI generated"""
        # Process the video
        processed_data = self.processor.process_video(file_path)
        frames = processed_data['frames'].to(self.device)
        audio_features = processed_data['audio_features']

        self.model.eval()
        with torch.no_grad():
            # Process video frames
            video_logits = self.model(frames, 'video')
            video_probs = F.softmax(video_logits, dim=1)

            # Initialize results
            human_percent = video_probs[0, 0].item() * 100
            ai_percent = video_probs[0, 1].item() * 100

            # If audio is available, process it too and combine results
            if audio_features is not None and len(audio_features) > 0:
                audio_human_sum = 0
                audio_ai_sum = 0

                for audio_tensor in audio_features:
                    # Move tensor to device
                    audio_tensor = audio_tensor.to(self.device)

                    # Perform inference
                    audio_logits = self.model(audio_tensor, 'mixed_audio')
                    audio_probs = F.softmax(audio_logits, dim=1)

                    audio_human_sum += audio_probs[0, 0].item()
                    audio_ai_sum += audio_probs[0, 1].item()

                # Average the audio probabilities
                chunk_count = len(audio_features)
                audio_human_percent = (audio_human_sum / chunk_count) * 100
                audio_ai_percent = (audio_ai_sum / chunk_count) * 100

                # Combine video and audio results (simple average)
                human_percent = (human_percent + audio_human_percent) / 2
                ai_percent = (ai_percent + audio_ai_percent) / 2

        return {"Human": human_percent, "AI": ai_percent}

    def batch_process_videos(self, directory_path, file_types=None):
        """Process all video files in a directory"""
        if file_types is None:
            file_types = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']

        results = {}

        # Get all matching files
        files = []
        for file_type in file_types:
            files.extend(glob.glob(os.path.join(directory_path, file_type)))

        for file_path in files:
            try:
                file_result = self.classify_video(file_path)
                results[os.path.basename(file_path)] = file_result
            except Exception as e:
                results[os.path.basename(file_path)] = f"Error: {str(e)}"

        return results

# Additional video analysis functions

def analyze_frame_consistency(video_path, sample_rate=5):
    """Analyze consistency between frames to detect AI generation artifacts"""
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample frames at regular intervals
    frame_indices = list(range(0, frame_count, sample_rate))

    if len(frame_indices) < 2:
        return {"error": "Video too short for analysis"}

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale for simpler analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

    cap.release()

    if len(frames) < 2:
        return {"error": "Failed to extract frames"}

    # Calculate frame differences
    diffs = []
    for i in range(1, len(frames)):
        # Calculate absolute difference between consecutive frames
        diff = cv2.absdiff(frames[i], frames[i-1])
        mean_diff = np.mean(diff)
        diffs.append(mean_diff)

    # Analyze the differences
    results = {
        "mean_diff": np.mean(diffs),
        "std_diff": np.std(diffs),
        "min_diff": np.min(diffs),
        "max_diff": np.max(diffs)
    }

    # Calculate consistency score (lower means more consistent/smoother)
    results["consistency_score"] = results["std_diff"] / (results["mean_diff"] + 1e-5)

    return results

def detect_temporal_artifacts(video_path, window_size=10):
    """Detect temporal artifacts that might indicate AI generation"""
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < window_size:
        return {"error": "Video too short for temporal analysis"}

    # Extract consecutive frames for analysis
    frames = []
    for i in range(min(frame_count, 100)):  # Limit to first 100 frames for efficiency
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        else:
            break

    cap.release()

    if len(frames) < window_size:
        return {"error": "Failed to extract enough frames"}

    # Calculate optical flow between consecutive frames
    flows = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i-1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Calculate magnitude of flow vectors
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(np.mean(mag))

    # Analyze flow consistency
    results = {
        "mean_flow": np.mean(flows),
        "std_flow": np.std(flows),
        "flow_consistency": np.std(flows) / (np.mean(flows) + 1e-5)
    }

    # Check for unnatural motion patterns
    # (AI-generated videos often have inconsistent motion)
    window_flows = []
    for i in range(len(flows) - window_size + 1):
        window = flows[i:i+window_size]
        window_flows.append(np.std(window) / (np.mean(window) + 1e-5))

    results["window_flow_consistency"] = np.mean(window_flows)

    return results

def analyze_audio_video_sync(video_path):
    """Analyze audio-video synchronization for potential AI generation artifacts"""
    try:
        video = mp.VideoFileClip(video_path)

        if video.audio is None:
            return {"error": "No audio track found"}

        # For now, return basic video properties
        results = {
            "duration": video.duration,
            "fps": video.fps,
            "audio_fps": video.audio.fps,
            "has_audio": video.audio is not None
        }

        return results

    except Exception as e:
        return {"error": str(e)}