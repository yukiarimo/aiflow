# AiFlow
AiFlow is a lightweight Python package for inferencing, evaluating, and training machine learning models.

[![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/YukiArimo)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yukiarimo)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/users/1131657390752800899)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/yukiarimo)

## Table of Contents
- [AiFlow](#aiflow)
	- [Table of Contents](#table-of-contents)
	- [Installation](#installation)
	- [Usage](#usage)
		- [1. Hanasu TTS](#1-hanasu-tts)
		- [2. Hifi Gan (Vocoder)](#2-hifi-gan-vocoder)
		- [3. Yuna Audio (ASR)](#3-yuna-audio-asr)
		- [4. Yuna VLM](#4-yuna-vlm)
	- [License](#license)
	- [Contact](#contact)

## Installation
To install AiFlow, run the following command:

```bash
pip install -e .
```

## Usage
AiFlow contains multiple powerful models: **Hanasu TTS**, **Hifi Gan**, **Yuna Audio**, and **Yuna VLM**. Here is a simple but detailed guide on how to integrate and use each model in your project.

### 1. Hanasu TTS
Hanasu is a lightweight Text-to-Speech synthesis model. You can generate speech from text directly using the `SynthesizerTrn` model.

```python
import torch
from aiflow.models.hanasu.models import SynthesizerTrn

# 1. Initialize the model (adjust the properties according to your config)
model = SynthesizerTrn(
	n_vocab=256,
	spec_channels=80,
	segment_size=8192,
	inter_channels=192,
	hidden_channels=192,
	filter_channels=768,
	n_heads=2,
	n_layers=6,
	kernel_size=3,
	p_dropout=0.1,
	resblock="1",
	resblock_kernel_sizes=[3, 7, 11],
	resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
	upsample_rates=[8, 8, 2, 2],
	upsample_initial_channel=512,
	upsample_kernel_sizes=[16, 16, 4, 4],
	n_speakers=0,
	gin_channels=0,
	use_sdp=True,
)

# 2. Load the model weights and set to evaluation mode
# model.load_state_dict(torch.load("path/to/hanasu.pth"))
# model.eval()

# 3. Provide a text tensor of ids and generate the audio
# audio_output = model(text_tensor, text_lengths)
```

### 2. Hifi Gan (Vocoder)
Hifi Gan is a high-fidelity vocoder that generates an audio waveform from mel-spectrograms. Use the `HifiganGenerator`.

```python
import torch
from aiflow.models.hifigan.models import HifiganGenerator

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

# 1. Create a configuration dictionary
config = AttrDict({
	"resblock": "1",
	"num_gpus": 1,
	"batch_size": 16,
	"learning_rate": 0.0002,
	"upsample_rates": [8, 8, 2, 2],
	"upsample_kernel_sizes": [16, 16, 4, 4],
	"upsample_initial_channel": 512,
	"resblock_kernel_sizes": [3, 7, 11],
	"resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
})

# 2. Initialize the Vocoder and load the weights
vocoder = HifiganGenerator(config)
# vocoder.load_state_dict(torch.load("path/to/hifigan.pth"))
# vocoder.eval()

# 3. Generate audio from a generated mel-spectrogram
# audio = vocoder(mel_spectrogram)
```

### 3. Yuna Audio (ASR)
Yuna Audio is our Automatic Speech Recognition (ASR) model based on Qwen3. You can use it to transcribe audio or process speech tasks.

```python
from aiflow.models.yuna_audio.qwen3_asr import Model

# 1. Load the configuration and instantiate the ASR model
# config = ...
# asr_model = Model(config)
# asr_model.load_weights("path/to/yuna_audio_weights")

# 2. Transcribe an audio file using the generate method
# result = asr_model.generate(
#     audio="path/to/audio/file.wav",
#     language="English",
#     max_tokens=8192
# )
# print("Transcription:", result.text)
```

### 4. Yuna VLM
Yuna VLM is our Vision-Language Model. It allows processing either images or audio alongside text prompts, returning generated text sequences.

```python
from aiflow.models.yuna_vlm.generate import generate

# 1. Load your Yuna VLM model and processor
# model, processor = ...

# 2. Generate text from an image with a question/prompt
# result = generate(
#     model=model,
#     processor=processor,
#     prompt="What are these?",
#     image=["path/to/image.jpg"],
#     max_tokens=256,
#     temperature=0.5
# )
# print("Response:", result.text)
```

## License
AiFlow is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0).

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.