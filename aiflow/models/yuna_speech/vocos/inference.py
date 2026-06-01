import argparse
import torch
import torchaudio
import json
from torch import nn
from .models import VocosBackbone, MelSpectrogramFeatures, ISTFTHead


def auto_device():
	"""Pick the best available device: cuda > mps > cpu."""
	if torch.cuda.is_available():
		return "cuda"
	if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def build_feature_extractor(cfg):
	return MelSpectrogramFeatures(**cfg)


def build_backbone(cfg):
	return VocosBackbone(**cfg)


def build_head(cfg):
	return ISTFTHead(**cfg)


class Vocos(nn.Module):
	"""Fourier-based neural vocoder for audio synthesis. Designed for inference, reconstructing a waveform from a mel-spectrogram via a ConvNeXt backbone and an ISTFT head."""
	def __init__(self, feature_extractor, backbone, head):
		super().__init__()
		self.feature_extractor = feature_extractor
		self.backbone = backbone
		self.head = head

	@classmethod
	def from_config(cls, config_path):
		"""Build an (untrained) Vocos model from a config.json file."""
		with open(config_path) as f:
			config = json.load(f)
		feature_extractor = build_feature_extractor(config["feature_extractor"])
		backbone = build_backbone(config["backbone"])
		head = build_head(config["head"])
		return cls(feature_extractor=feature_extractor, backbone=backbone, head=head)

	@classmethod
	def from_pretrained(cls, config_path, checkpoint_path, device=None):
		"""Build a Vocos model from a config.json and load weights from a checkpoint. Accepts both Lightning checkpoints (`.ckpt` containing a `state_dict`) and plain state dicts."""
		if device is None:
			device = auto_device()
		model = cls.from_config(config_path)
		ckpt = torch.load(checkpoint_path, map_location="cpu")
		state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
		wanted = ("feature_extractor.", "backbone.", "head.")  # Keep only the generator weights (feature_extractor / backbone / head), drop discriminators, etc.
		filtered = {k: v for k, v in state_dict.items() if k.startswith(wanted)}
		model.load_state_dict(filtered, strict=False)  # strict=False: deterministic buffers (mel filterbank, STFT window) are rebuilt at init.
		model.eval()
		model.to(device)
		return model

	@torch.inference_mode()
	def forward(self, audio_input):
		"""Copy-synthesis from an audio waveform of shape (B, T)."""
		features = self.feature_extractor(audio_input)
		return self.decode(features)

	@torch.inference_mode()
	def decode(self, features_input):
		"""Decode a waveform from pre-computed log-mel features of shape (B, C, L)."""
		x = self.backbone(features_input)
		return self.head(x)


def parse_args():
	parser = argparse.ArgumentParser(description="Vocos copy-synthesis inference.")
	parser.add_argument("--config", default="config.json", help="Path to config.json.")
	parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.ckpt or .pt).")
	parser.add_argument("--input", required=True, help="Path to the input audio file.")
	parser.add_argument("--output", default="output.wav", help="Path to write the reconstructed audio.")
	parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Inference device. 'auto' picks cuda > mps > cpu.", )
	return parser.parse_args()


def load_config_sample_rate(config_path):
	with open(config_path) as f:
		config = json.load(f)
	return config["feature_extractor"]["sample_rate"]


def main():
	args = parse_args()
	device = auto_device() if args.device == "auto" else args.device
	print(f"Using device: {device}")
	sample_rate = load_config_sample_rate(args.config)
	model = Vocos.from_pretrained(args.config, args.checkpoint, device=device)
	y, sr = torchaudio.load(args.input)
	if y.size(0) > 1:  # mix to mono
		y = y.mean(dim=0, keepdim=True)
	if sr != sample_rate:
		y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)
	y = y.to(device)
	y_hat = model(y).cpu()
	torchaudio.save(args.output, y_hat, sample_rate)
	print(f"Saved reconstructed audio to {args.output}")


if __name__ == "__main__":
	main()
