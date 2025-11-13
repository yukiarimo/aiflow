# HiFi-GAN
HiFi-GAN is a generative adversarial network (GAN) for high-fidelity spectrogram-to-waveform synthesis.

## Inference
To use HiFi-GAN for inference, you can load a pre-trained model and generate audio from mel-spectrograms. Below is an example of how to do this using PyTorch:

```python
import torch
import numpy as np
import soundfile as sf
from utils import hifigan

# Load the HiFi-GAN model using the utility function
model = hifigan(
    checkpoint_path="/Users/yuki/Documents/Github/yuna-ai/lib/models/agi/hifigan-v1/hifigan-v1.pth",
    map_location="cpu"
)

# Load mel-spectrogram
mel_np = np.load("/Users/yuki/Documents/AI/autoencoder/test_converted_to_yuna.npy")
print(f"Original mel shape: {mel_np.shape}")

# Convert to tensor and ensure correct shape [batch_size, n_mels, time_frames]
mel = torch.from_numpy(mel_np)
if mel.dim() == 2:  # If shape is [n_mels, time_frames]
    mel = mel.unsqueeze(0)  # Add batch dimension -> [1, n_mels, time_frames]
elif mel.dim() == 4:  # If shape is [1, 1, n_mels, time_frames]
    mel = mel.squeeze(1)  # Remove extra dimension -> [1, n_mels, time_frames]

print(f"Final mel shape for model: {mel.shape}")

# Generate audio - HiFi-GAN models typically just do forward pass
with torch.inference_mode():
    wav = model(mel)

# Save wav - assuming 48kHz sample rate based on your generator config
sf.write("original_spectrogram.wav", wav.squeeze().cpu().numpy(), 48000)
```

## Training
To train HiFi-GAN, you need to prepare your dataset and follow the training procedure. Below are the steps to get started.

### Step 1: Dataset Preparation
Split your dataset into training and validation sets, and organize it in the following structure:

```
└───wavs
    ├───dev
    │   ├───1.wav
    │   └───2.wav
    └───train
        ├───3.wav
        └───4.wav
```

> The `train` and `dev` directories should contain the training and validation splits respectively.

### Step 2: Train HifiGAN
To train HiFi-GAN, you can use the provided `train.py` script. Make sure you have the required dependencies installed, and then run the following command:

```
usage: train.py [-h] [--resume RESUME] [--finetune] dataset-dir checkpoint-dir

Train or finetune HiFi-GAN.

positional arguments:
  dataset-dir      path to the preprocessed data directory
  checkpoint-dir   path to the checkpoint directory

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from
  --finetune       whether to finetune (note that a resume path must be given)
```