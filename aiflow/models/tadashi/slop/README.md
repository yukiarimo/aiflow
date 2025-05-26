# Tadashi - AI-Generated Content Detector
Tadashi is a multimodal AI-generated content detector capable of distinguishing between human-created and AI-generated content across multiple modalities including audio, images, and videos. Named after the Japanese word for "correct" (正), Tadashi aims to provide accurate detection of AI-generated "slop" content.

## Model Overview

### Architecture
Tadashi uses a unified neural network architecture with specialized feature extractors for different content modalities:

#### Supported Modalities
- **Audio Processing**
  - Speech detection (human speech vs AI-generated speech)
  - Instrumental music detection (real vs AI-generated music)
  - Mixed audio detection (general audio content)
- **Image Processing**
  - AI-generated image detection with artifact analysis
  - Support for common formats: JPG, PNG, WebP, BMP
- **Video Processing**
  - Frame-by-frame analysis with temporal consistency checks
  - Audio-visual fusion for videos with sound tracks
  - Motion pattern analysis for detecting AI generation artifacts

#### Technical Specifications
- **Audio**: Processes 48kHz 16-bit audio with 30-second chunks
- **Images**: Handles images up to 1024px resolution
- **Videos**: Supports 1080p 30fps video processing
- **Memory Optimized**: Designed for M1 MacBook with 16GB RAM
- **Device Support**: CUDA, MPS (Apple Silicon), and CPU inference

### Key Features
- **Memory Efficient**: Dynamic component loading based on active modality
- **Modular Design**: Each modality can be trained independently
- **Temporal Awareness**: Considers sequential patterns in audio and video
- **Artifact Detection**: Specialized detection of AI generation artifacts
- **Batch Processing**: Efficient directory-level processing

### Requirements
**System Requirements:**
- Python 3.8+
- PyTorch 1.12+
- 8GB+ RAM (16GB recommended for video processing)
- Optional: CUDA-compatible GPU or Apple Silicon for acceleration

## Training

### Dataset Preparation
Tadashi requires labeled datasets organized in a specific directory structure for each modality:

#### Audio Dataset Structure
```
dataset/
├── speech/
│   ├── human/
│   │   ├── human_speech_001.wav
│   │   ├── human_speech_002.mp3
│   │   └── ...
│   └── ai/
│       ├── ai_speech_001.wav
│       ├── ai_speech_002.mp3
│       └── ...
├── instrumental/
│   ├── human/
│   │   └── ...
│   └── ai/
│       └── ...
└── mixed_audio/
    ├── human/
    │   └── ...
    └── ai/
        └── ...
```

#### Image Dataset Structure
```
dataset/
├── images/
│   ├── human/
│   │   ├── real_photo_001.jpg
│   │   ├── real_photo_002.png
│   │   └── ...
│   └── ai/
│       ├── ai_generated_001.jpg
│       ├── ai_generated_002.png
│       └── ...
```

#### Video Dataset Structure
```
dataset/
├── videos/
│   ├── human/
│   │   ├── real_video_001.mp4
│   │   ├── real_video_002.mov
│   │   └── ...
│   └── ai/
│       ├── ai_video_001.mp4
│       ├── ai_video_002.mov
│       └── ...
```

### Training Individual Modalities
Use the [`train.py`](train.py) script to train Tadashi on specific modalities:

#### Audio Modalities
```bash
# Train speech detector
python train.py --data_dir dataset/speech --modality speech --epochs 50 --batch_size 16 --lr 0.001 --output model/speech_model.pth

# Train instrumental music detector
python train.py --data_dir dataset/instrumental --modality instrumental --epochs 50 --batch_size 16 --lr 0.001 --output model/instrumental_model.pth

# Train mixed audio detector
python train.py --data_dir dataset/mixed_audio --modality mixed_audio --epochs 50 --batch_size 16 --lr 0.001 --output model/mixed_audio_model.pth
```

#### Image Modality
```bash
# Train image detector
python train.py --data_dir dataset/images --modality image --epochs 30 --batch_size 32 --lr 0.0005 --output model/image_model.pth
```

#### Video Modality
```bash
# Train video detector
python train.py --data_dir dataset/videos --modality video --epochs 40 --batch_size 8 --lr 0.0008 --output model/video_model.pth
```

### Continuing Training from Checkpoints
Resume training from a saved checkpoint:

```bash
python train.py --data_dir dataset/speech --modality speech --epochs 25 --checkpoint model/speech_model.pth --output model/speech_model_continued.pth
```

### Creating a Unified Model
After training individual modality models, merge them into a single unified model using [`merge_models.py`](merge_models.py):

```bash
python merge_models.py \
    --speech model/speech_model.pth \
    --instrumental model/instrumental_model.pth \
    --mixed_audio model/mixed_audio_model.pth \
    --image model/image_model.pth \
    --video model/video_model.pth \
    --output model/tadashi_unified.pth
```

The merge process:
1. Loads each modality-specific model
2. Checks for and fixes NaN values in model weights
3. Combines modality-specific components into a unified architecture
4. Validates the merged model with test inputs
5. Saves the unified model for inference

## Inference

### Quick Start
The simplest way to use Tadashi is through the [`detector.py`](detector.py) command-line interface:

```bash
# Classify a single file (auto-detects modality)
python detector.py --input path/to/content.mp4

# Specify modality explicitly
python detector.py --input path/to/audio.wav --modality mixed_audio

# Process entire directory
python detector.py --input path/to/media_folder/
```

### Command-Line Interface

#### Basic Usage
```bash
python detector.py --input <file_or_directory_path> [OPTIONS]
```

#### Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | File or directory path | Required |
| `--modality` | Force specific modality | auto (detect from extension) |
| `--model` | Path to model file | model/model.pth |
| `--benchmark` | Show performance metrics | False |
| `--batch_size` | Batch size for directories | 1 |

#### Modality Options
- `speech`: Human speech vs AI-generated speech
- `instrumental`: Real music vs AI-generated music  
- `mixed_audio`: General audio content
- `image`: Real photos vs AI-generated images
- `video`: Real videos vs AI-generated videos
- `auto`: Automatically detect based on file extension

### Programmatic Usage

#### Using the Unified Detector

```python
from model.unified_detector import AIContentDetector

# Initialize detector
detector = AIContentDetector(model_path='model/tadashi_unified.pth')

# Classify single file
result = detector.classify_file('suspicious_audio.wav')
print(f"Human: {result['Human']:.1f}%, AI: {result['AI']:.1f}%")

# Process directory
results = detector.classify_directory('media_folder/')
for filename, result in results.items():
    print(f"{filename}: Human {result['Human']:.1f}% | AI {result['AI']:.1f}%")
```

#### Using Individual Processors

```python
from model.model import MultiModalAIDetector
from model.image_processing import AIImageDetector
from model.video_processing import AIVideoDetector
import torch

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MultiModalAIDetector().to(device)
model.load_state_dict(torch.load('model/tadashi_unified.pth', map_location=device))

# Image detection
image_detector = AIImageDetector(model, device)
result = image_detector.classify_image('suspicious_image.jpg')

# Video detection  
video_detector = AIVideoDetector(model, device)
result = video_detector.classify_video('suspicious_video.mp4')
```

### Supported File Formats

#### Audio
- WAV, MP3, FLAC, OGG, M4A
- Minimum duration: 5 seconds
- Processed in 30-second chunks

#### Images
- JPG, JPEG, PNG, BMP, WebP
- Automatically resized to 224x224 for processing
- Supports up to 4K resolution input

#### Video
- MP4, AVI, MOV, MKV, WebM
- Extracts 30 frames for analysis
- Audio track processed separately if available
- Results combined for final classification

### Performance Benchmarking
Monitor Tadashi's performance on your system:

```bash
# Benchmark single file
python detector.py --input test_video.mp4 --benchmark

# Benchmark directory processing
python detector.py --input test_folder/ --benchmark
```

Benchmark output includes:
- Processing time per file
- Memory usage (GPU and system)
- Classification accuracy metrics
- Throughput (files per second)

### Output Format

Tadashi returns classification results as percentages:

```python
{
    "Human": 75.3,    # Confidence that content is human-created
    "AI": 24.7        # Confidence that content is AI-generated
}
```

### Advanced Analysis Features

#### Video Temporal Analysis
```python
from model.video_processing import analyze_frame_consistency, detect_temporal_artifacts

# Analyze frame-to-frame consistency
consistency = analyze_frame_consistency('video.mp4')
print(f"Consistency Score: {consistency['consistency_score']:.3f}")

# Detect temporal artifacts
artifacts = detect_temporal_artifacts('video.mp4')
print(f"Motion Consistency: {artifacts['flow_consistency']:.3f}")
```

#### Image Artifact Detection
```python
from model.image_processing import analyze_image_artifacts, detect_image_boundaries

# Analyze pixel-level artifacts
artifacts = analyze_image_artifacts('image.jpg')
print(f"Symmetry Score: {artifacts['symmetry_score']:.3f}")

# Detect unnatural boundaries
boundaries = detect_image_boundaries('image.jpg')
print(f"Edge Consistency: {boundaries['mean_magnitude']:.3f}")
```

### Integration Examples

#### Web API Integration
```python
from flask import Flask, request, jsonify
from model.unified_detector import AIContentDetector

app = Flask(__name__)
detector = AIContentDetector(model_path='model/tadashi_unified.pth')

@app.route('/detect', methods=['POST'])
def detect_content():
    file = request.files['content']
    file.save('temp_file')
    result = detector.classify_file('temp_file')
    return jsonify(result)
```

#### Batch Processing Script
```python
import os
import json
from model.unified_detector import AIContentDetector

def process_media_library(library_path, output_file):
    detector = AIContentDetector(model_path='model/tadashi_unified.pth')
    results = {}
    
    for root, dirs, files in os.walk(library_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.jpg', '.wav', '.mp3')):
                file_path = os.path.join(root, file)
                try:
                    result = detector.classify_file(file_path)
                    results[file_path] = result
                except Exception as e:
                    results[file_path] = {"error": str(e)}
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Process entire media library
results = process_media_library('/path/to/media', 'detection_results.json')
```