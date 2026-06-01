import argparse
import json
import torch
from pytorch_lightning import Trainer, seed_everything, LightningDataModule, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from .models import VocosBackbone, MelSpectrogramFeatures, safe_log, ISTFTHead
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
import math
import pytorch_lightning as pl
from torch import nn
from einops import rearrange
from torch.nn import Conv2d
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram
import matplotlib
from matplotlib import pyplot as plt

torch.set_num_threads(1)
matplotlib.use("Agg")


def save_figure_to_numpy(fig):
	"""Save a matplotlib figure to an RGB numpy array."""
	fig.canvas.draw()
	data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (4, ))
	return data[..., :3]


def plot_spectrogram_to_numpy(spectrogram):
	"""Plot a spectrogram and convert it to a numpy array."""
	spectrogram = spectrogram.astype(np.float32)
	fig, ax = plt.subplots(figsize=(12, 3))
	im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
	plt.colorbar(im, ax=ax)
	plt.xlabel("Frames")
	plt.ylabel("Channels")
	plt.tight_layout()
	data = save_figure_to_numpy(fig)
	plt.close()
	return data


class GradNormCallback(Callback):
	"""Callback to log the gradient norm."""
	def on_after_backward(self, trainer, model):
		model.log("grad_norm", gradient_norm(model))


def gradient_norm(model, norm_type=2.0):
	grads = [p.grad for p in model.parameters() if p.grad is not None]
	total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
	return total_norm


class MultiPeriodDiscriminator(nn.Module):
	"""Multi-Period Discriminator module adapted from HiFi-Gan"""
	def __init__(self, periods=(2, 3, 5, 7, 11)):
		super().__init__()
		self.discriminators = nn.ModuleList([DiscriminatorP(period=p) for p in periods])

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for d in self.discriminators:
			y_d_r, fmap_r = d(x=y)
			y_d_g, fmap_g = d(x=y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)
		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
	def __init__(self, period, in_channels=1, kernel_size=5, stride=3, lrelu_slope=0.1):
		super().__init__()
		self.period = period
		self.convs = nn.ModuleList([weight_norm(Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))), weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))), weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))), weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))), weight_norm(Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(kernel_size // 2, 0))), ])
		self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
		self.lrelu_slope = lrelu_slope

	def forward(self, x):
		x = x.unsqueeze(1)
		fmap = []
		# 1d to 2d
		b, c, t = x.shape
		if t % self.period != 0:  # pad first
			n_pad = self.period - (t % self.period)
			x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
			t = t + n_pad
		x = x.view(b, c, t // self.period, self.period)
		for i, l in enumerate(self.convs):
			x = l(x)
			x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
			if i > 0:
				fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = torch.flatten(x, 1, -1)
		return x, fmap


class MultiResolutionDiscriminator(nn.Module):
	"""Multi-Resolution Discriminator module adapted from DAC"""
	def __init__(self, fft_sizes=(2048, 1024, 512)):
		super().__init__()
		self.discriminators = nn.ModuleList([DiscriminatorR(window_length=w) for w in fft_sizes])

	def forward(self, y, y_hat):
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for d in self.discriminators:
			y_d_r, fmap_r = d(x=y)
			y_d_g, fmap_g = d(x=y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)
		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
	def __init__(self, window_length, channels=32, hop_factor=0.25, bands=((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)), ):
		super().__init__()
		self.window_length = window_length
		self.hop_factor = hop_factor
		self.spec_fn = Spectrogram(n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None)
		n_fft = window_length // 2 + 1
		bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
		self.bands = bands
		convs = lambda: nn.ModuleList([weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))), weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))), ])
		self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
		self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

	def spectrogram(self, x):
		x = x - x.mean(dim=-1, keepdims=True)  # Remove DC offset
		x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)  # Peak normalize the volume of input audio
		x = self.spec_fn(x)
		x = torch.view_as_real(x)
		x = rearrange(x, "b f t c -> b c t f")
		x_bands = [x[..., b[0]:b[1]] for b in self.bands]  # Split into bands
		return x_bands

	def forward(self, x):
		x_bands = self.spectrogram(x)
		fmap = []
		x = []
		for band, stack in zip(x_bands, self.band_convs):
			for i, layer in enumerate(stack):
				band = layer(band)
				band = torch.nn.functional.leaky_relu(band, 0.1)
				if i > 0:
					fmap.append(band)
			x.append(band)
		x = torch.cat(x, dim=-1)
		x = self.conv_post(x)
		fmap.append(x)
		return x, fmap


class MelSpecReconstructionLoss(nn.Module):
	"""L1 distance between the mel-scaled magnitude spectrograms of the ground truth and the generated sample."""
	def __init__(self, sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128, f_min=0.0, f_max=None):
		super().__init__()
		self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max, center=True, power=1, )

	def forward(self, y_hat, y):
		mel_hat = safe_log(self.mel_spec(y_hat))
		mel = safe_log(self.mel_spec(y))
		loss = torch.nn.functional.l1_loss(mel, mel_hat)
		return loss


class GeneratorLoss(nn.Module):
	"""Generator Loss module. Calculates the loss for the generator based on discriminator outputs."""
	def forward(self, disc_outputs):
		loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
		gen_losses = []
		for dg in disc_outputs:
			l = torch.mean(torch.clamp(1 - dg, min=0))
			gen_losses.append(l)
			loss += l
		return loss, gen_losses


class DiscriminatorLoss(nn.Module):
	"""Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs."""
	def forward(self, disc_real_outputs, disc_generated_outputs):
		loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
		r_losses = []
		g_losses = []
		for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
			r_loss = torch.mean(torch.clamp(1 - dr, min=0))
			g_loss = torch.mean(torch.clamp(1 + dg, min=0))
			loss += r_loss + g_loss
			r_losses.append(r_loss)
			g_losses.append(g_loss)
		return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
	"""Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators."""
	def forward(self, fmap_r, fmap_g):
		loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
		for dr, dg in zip(fmap_r, fmap_g):
			for rl, gl in zip(dr, dg):
				loss += torch.mean(torch.abs(rl - gl))
		return loss


def match_audio_lengths(y, y_hat):
	"""Trim both waveforms to the same length (ISTFT reconstruction is often a few samples shorter)."""
	n = min(y.shape[-1], y_hat.shape[-1])
	return y[..., :n], y_hat[..., :n]


def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
	"""Cosine learning-rate schedule with a linear warmup, implemented without external deps."""
	def lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class VocosExp(pl.LightningModule):
	def __init__(self, feature_extractor, backbone, head, sample_rate, initial_learning_rate, mel_loss=None, num_warmup_steps=0, mel_loss_coeff=45, mrd_loss_coeff=0.1, pretrain_mel_steps=0, decay_mel_coeff=False, ):
		"""
        Args:
            feature_extractor: Extracts mel features from audio signals.
            backbone: Backbone model.
            head: Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate: Sampling rate of the audio signals.
            initial_learning_rate: Initial learning rate for the optimizer.
            num_warmup_steps: Number of warmup steps for the learning-rate scheduler.
            mel_loss_coeff: Coefficient for the mel-spectrogram reconstruction loss.
            mrd_loss_coeff: Coefficient for the multi-resolution discriminator loss.
            pretrain_mel_steps: Number of steps to pre-train without the GAN objective.
            decay_mel_coeff: If True, the mel-spectrogram loss coefficient is decayed during training.
        """
		super().__init__()
		self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])
		self.feature_extractor = feature_extractor
		self.backbone = backbone
		self.head = head
		self.multiperioddisc = MultiPeriodDiscriminator()
		self.multiresddisc = MultiResolutionDiscriminator()
		self.disc_loss = DiscriminatorLoss()
		self.gen_loss = GeneratorLoss()
		self.feat_matching_loss = FeatureMatchingLoss()
		mel_cfg = mel_loss or {}
		self.melspec_loss = MelSpecReconstructionLoss(sample_rate=mel_cfg.get("sample_rate", sample_rate), n_fft=mel_cfg.get("n_fft", 2048), hop_length=mel_cfg.get("hop_length", 512), n_mels=mel_cfg.get("n_mels", 128), f_min=mel_cfg.get("f_min", 0.0), f_max=mel_cfg.get("f_max", None), )
		self.train_discriminator = False
		self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

	def configure_optimizers(self):
		disc_params = [{"params": self.multiperioddisc.parameters()}, {"params": self.multiresddisc.parameters()}, ]
		gen_params = [{"params": self.feature_extractor.parameters()}, {"params": self.backbone.parameters()}, {"params": self.head.parameters()}, ]
		opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
		opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
		max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
		scheduler_disc = cosine_schedule_with_warmup(opt_disc, self.hparams.num_warmup_steps, max_steps)
		scheduler_gen = cosine_schedule_with_warmup(opt_gen, self.hparams.num_warmup_steps, max_steps)
		return ([opt_disc, opt_gen], [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}], )

	def forward(self, audio_input):
		features = self.feature_extractor(audio_input)
		x = self.backbone(features)
		audio_output = self.head(x)
		return audio_output

	def training_step(self, batch, batch_idx, optimizer_idx):
		audio_input = batch

		# train discriminator
		if optimizer_idx == 0 and self.train_discriminator:
			with torch.no_grad():
				audio_hat = self(audio_input)
			audio_input, audio_hat = match_audio_lengths(audio_input, audio_hat)
			real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat)
			real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat)
			loss_mp, loss_mp_real, _ = self.disc_loss(disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp)
			loss_mrd, loss_mrd_real, _ = self.disc_loss(disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd)
			loss_mp /= len(loss_mp_real)
			loss_mrd /= len(loss_mrd_real)
			loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd
			self.log("discriminator/total", loss, prog_bar=True)
			self.log("discriminator/multi_period_loss", loss_mp)
			self.log("discriminator/multi_res_loss", loss_mrd)
			return loss

		# train generator
		if optimizer_idx == 1:
			audio_hat = self(audio_input)
			audio_input, audio_hat = match_audio_lengths(audio_input, audio_hat)
			if self.train_discriminator:
				_, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio_input, y_hat=audio_hat)
				_, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio_input, y_hat=audio_hat)
				loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
				loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
				loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
				loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
				loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
				loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
				self.log("generator/multi_period_loss", loss_gen_mp)
				self.log("generator/multi_res_loss", loss_gen_mrd)
				self.log("generator/feature_matching_mp", loss_fm_mp)
				self.log("generator/feature_matching_mrd", loss_fm_mrd)
			else:
				loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

			mel_loss = self.melspec_loss(audio_hat, audio_input)
			loss = (loss_gen_mp + self.hparams.mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + self.hparams.mrd_loss_coeff * loss_fm_mrd + self.mel_loss_coeff * mel_loss)
			self.log("generator/total_loss", loss, prog_bar=True)
			self.log("mel_loss_coeff", self.mel_loss_coeff)
			self.log("generator/mel_loss", mel_loss)

			if self.global_step % 1000 == 0 and self.global_rank == 0:
				self.logger.experiment.add_audio("train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate)
				self.logger.experiment.add_audio("train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate)
				with torch.no_grad():
					mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
					mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
				self.logger.experiment.add_image("train/mel_target", plot_spectrogram_to_numpy(mel.data.cpu().numpy()), self.global_step, dataformats="HWC")
				self.logger.experiment.add_image("train/mel_pred", plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()), self.global_step, dataformats="HWC")

			return loss

	def validation_step(self, batch, batch_idx):
		audio_input = batch
		audio_hat = self(audio_input)
		audio_input, audio_hat = match_audio_lengths(audio_input, audio_hat)
		mel_loss = self.melspec_loss(audio_hat, audio_input)
		return {"val_loss": mel_loss, "audio_input": audio_input[0], "audio_pred": audio_hat[0]}

	def validation_epoch_end(self, outputs):
		if self.global_rank == 0:
			audio_in = outputs[0]["audio_input"]
			audio_pred = outputs[0]["audio_pred"]
			self.logger.experiment.add_audio("val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
			self.logger.experiment.add_audio("val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate)
			mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
			mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
			self.logger.experiment.add_image("val_mel_target", plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()), self.global_step, dataformats="HWC")
			self.logger.experiment.add_image("val_mel_hat", plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()), self.global_step, dataformats="HWC")
		avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
		self.log("val_loss", avg_loss, sync_dist=True)

	@property
	def global_step(self):
		"""Override global_step so that it returns the total number of batches processed."""
		return self.trainer.fit_loop.epoch_loop.total_batch_idx

	def on_train_batch_start(self, *args):
		if self.global_step >= self.hparams.pretrain_mel_steps:
			self.train_discriminator = True
		else:
			self.train_discriminator = False

	def on_train_batch_end(self, *args):
		def mel_loss_coeff_decay(current_step, num_cycles=0.5):
			max_steps = self.trainer.max_steps // 2
			if current_step < self.hparams.num_warmup_steps:
				return 1.0
			progress = float(current_step - self.hparams.num_warmup_steps) / float(max(1, max_steps - self.hparams.num_warmup_steps))
			return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

		if self.hparams.decay_mel_coeff:
			self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class VocosDataModule(LightningDataModule):
	def __init__(self, train_params, val_params):
		super().__init__()
		self.train_config = train_params
		self.val_config = val_params

	def _get_dataloder(self, cfg, train):
		dataset = VocosDataset(cfg, train=train)
		dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=train, pin_memory=True, )
		return dataloader

	def train_dataloader(self):
		return self._get_dataloder(self.train_config, train=True)

	def val_dataloader(self):
		return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
	def __init__(self, cfg, train):
		with open(cfg["filelist_path"]) as f:
			self.filelist = f.read().splitlines()
		self.sampling_rate = cfg["sampling_rate"]
		self.num_samples = cfg["num_samples"]
		self.train = train

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, index):
		audio_path = self.filelist[index]
		y, sr = torchaudio.load(audio_path)
		if y.size(0) > 1:
			y = y.mean(dim=0, keepdim=True)  # mix to mono
		# Peak-normalize to 0 dB, then apply a random gain (pure torch, no sox dependency).
		gain_db = np.random.uniform(-6.0, -1.0) if self.train else -3.0
		y = y / y.abs().max().clamp(min=1e-8)
		y = y * (10**(gain_db / 20.0))
		if sr != self.sampling_rate:
			y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
		if y.size(-1) < self.num_samples:
			pad_length = self.num_samples - y.size(-1)
			padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
			y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
		elif self.train:
			start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
			y = y[:, start:start + self.num_samples]
		else:
			y = y[:, :self.num_samples]  # During validation, always take the first segment for determinism

		return y[0]


def parse_args():
	parser = argparse.ArgumentParser(description="Train Vocos.")
	parser.add_argument("-c", "--config", default="config.json", help="Path to config.json.")
	return parser.parse_args()


def main():
	args = parse_args()
	with open(args.config) as f:
		config = json.load(f)
	seed_everything(config["seed"], workers=True)
	datamodule = VocosDataModule(train_params=config["data"]["train"], val_params=config["data"]["val"])
	feature_extractor = MelSpectrogramFeatures(**config["feature_extractor"])
	backbone = VocosBackbone(**config["backbone"])
	head = ISTFTHead(**config["head"])
	model = VocosExp(feature_extractor=feature_extractor, backbone=backbone, head=head, mel_loss=config["feature_extractor"], **config["model"], )
	trainer_cfg = config["trainer"]
	logger = TensorBoardLogger(save_dir=trainer_cfg["save_dir"])
	callbacks = [LearningRateMonitor(), ModelSummary(max_depth=2), ModelCheckpoint(monitor="val_loss", filename="vocos_{epoch}_{step}_{val_loss:.4f}", save_top_k=trainer_cfg["save_top_k"], save_last=True, ), GradNormCallback(), ]
	use_gpu = torch.cuda.is_available()
	trainer = Trainer(logger=logger, callbacks=callbacks, accelerator="gpu" if use_gpu else "cpu", devices=1, precision=trainer_cfg["precision"] if use_gpu else 32, max_steps=trainer_cfg["max_steps"], limit_val_batches=trainer_cfg["limit_val_batches"], log_every_n_steps=trainer_cfg["log_every_n_steps"], )
	trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
	main()
