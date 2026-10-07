import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import N_MELS, RMVPE_DIM

COREML_CHUNK_FRAMES = 1024  # ANE wants fixed chunk + ReLU + BatchNorm (no RNN/GELU/grouped conv); math takes any T; long audio is stitched from chunks
COREML_OVERLAP = 256
CONTEXT = 256
STOP_DELAY = 8
QDR_MARGIN = 0.2


class ConvBN(nn.Module):
	def __init__(self, c_in, c_out, k=3, dilation=1):
		super().__init__()
		self.pad = dilation * (k - 1) // 2
		self.conv = nn.Conv1d(c_in, c_out, k, padding=0, dilation=dilation, bias=False)
		self.bn = nn.BatchNorm1d(c_out)

	def forward(self, x):
		if self.pad:
			x = F.pad(x, (self.pad, self.pad), mode='replicate')
		return F.relu(self.bn(self.conv(x)))


class LiquidBlock(nn.Module):
	"""Input-dependent mix of a dilated conv and the skip. The gate is a closed-form stand-in for a liquid time constant: each frame chooses how much new context to take and how much of the running features to keep. A pause and a real stop move that gate differently because the dilated conv sees the labeled end, not only the local dip."""
	def __init__(self, channels, dilation, dropout=0.1):
		super().__init__()
		self.mix = ConvBN(channels, channels, 3, dilation=dilation)
		self.pad = dilation
		self.gate_conv = nn.Conv1d(channels, channels, 3, padding=0, dilation=dilation, bias=False)
		self.gate_bn = nn.BatchNorm1d(channels)
		self.drop = nn.Dropout(dropout)
		self.out_bn = nn.BatchNorm1d(channels)
		self._gate = None

	def forward(self, x):
		y = self.mix(x)
		g_in = F.pad(x, (self.pad, self.pad), mode='replicate') if self.pad else x
		g = torch.clamp(self.gate_bn(self.gate_conv(g_in)), 0.0, 1.0)
		if self.training:
			self._gate = g.detach()
		z = g * self.drop(y) + (1.0 - g) * x
		return F.relu(self.out_bn(z))


class LiquidVAD(nn.Module):
	"""Liquid-gated conv VAD. Any length T. Pitch is an optional input, not a teacher."""
	def __init__(self, n_mels=N_MELS, rmvpe_dim=RMVPE_DIM, hidden=64, dropout=0.12, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
		super().__init__()
		self.n_mels = n_mels
		self.rmvpe_dim = rmvpe_dim
		self.hidden = hidden
		self.dropout = dropout
		self.dilations = list(dilations)
		self.mel_in = nn.Sequential(ConvBN(n_mels, hidden, 5), ConvBN(hidden, hidden, 3, dilation=2))
		self.pitch_in = ConvBN(rmvpe_dim, hidden, 3)
		self.aux_net = ConvBN(3, 16, 3)
		self.fuse = ConvBN(hidden + hidden + 16, hidden, 1)
		self.blocks = nn.ModuleList([LiquidBlock(hidden, d, dropout) for d in self.dilations])
		self.dense = ConvBN(hidden * len(self.dilations), hidden, 1)
		self.head = nn.Conv1d(hidden, 3, 1)
		self.pitch_head = nn.Conv1d(hidden, 1, 1)
		self.apply(self._init)

	@staticmethod
	def _init(m):
		if isinstance(m, nn.Conv1d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def config(self):
		return {'n_mels': self.n_mels, 'rmvpe_dim': self.rmvpe_dim, 'hidden': self.hidden, 'dropout': self.dropout, 'dilations': self.dilations}

	def _aux(self, mel):
		energy = mel.mean(1, keepdim=True)
		high = mel[:, -32:, :].mean(1, keepdim=True)
		low = mel[:, :24, :].mean(1, keepdim=True)
		return torch.cat([energy, high, low], 1)

	def gate_stats(self):
		vals = [blk._gate for blk in self.blocks if blk._gate is not None]
		if not vals:
			return None
		flat = torch.cat([v.flatten() for v in vals])
		return flat[:8192].float().cpu()

	def forward(self, mel, rmvpe=None, aux=False):
		t = mel.shape[-1]
		if rmvpe is None:
			rmvpe = mel.new_zeros(mel.shape[0], self.rmvpe_dim, t)
		mel = F.pad(mel, (CONTEXT, CONTEXT), mode='replicate')
		rmvpe = F.pad(rmvpe, (CONTEXT, CONTEXT), mode='replicate')
		h = self.fuse(torch.cat([self.mel_in(mel), self.pitch_in(rmvpe), self.aux_net(self._aux(mel))], 1))
		skips = []
		for blk in self.blocks:
			h = blk(h)
			skips.append(h)
		h = self.dense(torch.cat(skips, 1))
		logits = self.head(h).transpose(1, 2)[:, CONTEXT:CONTEXT + t]
		if not aux:
			return logits
		pitch = torch.sigmoid(self.pitch_head(h)).squeeze(1)[:, CONTEXT:CONTEXT + t]
		return logits, pitch


def masked_bce(logits, target, mask, neg_w=1.0):
	loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
	if neg_w != 1.0:
		loss = loss * torch.where(target >= 0.5, torch.ones_like(loss), loss.new_full((), neg_w))
	return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def qdr_loss(logits, target, mask, margin=QDR_MARGIN, k=16):
	"""Quadratic disparity ranking: speech frames should outrank silence."""
	prob = torch.sigmoid(logits)
	losses = []
	for b in range(prob.shape[0]):
		m = mask[b] > 0.5
		p = prob[b][m]
		y = target[b][m] >= 0.5
		pos = p[y]
		neg = p[~y]
		if pos.numel() < 1 or neg.numel() < 1:
			continue
		ip = torch.randint(0, pos.numel(), (min(k, pos.numel()), ), device=p.device)
		inn = torch.randint(0, neg.numel(), (min(k, neg.numel()), ), device=p.device)
		diff = pos[ip][:, None] - neg[inn][None, :]
		losses.append(F.relu(margin - diff).pow(2).mean())
	if not losses:
		return prob.sum() * 0.0
	return torch.stack(losses).mean()


def _shift_later(x, delay):
	if delay <= 0:
		return x
	return torch.cat([x[:, delay:], x.new_zeros(x.shape[0], delay)], 1)


def stop_loss(logits, speech, stop, mask, delay=STOP_DELAY):
	"""Stop target sits a little after the click. Stop mass on an in-speech pause is penalized."""
	delayed = torch.maximum(stop, _shift_later(stop, delay))
	bce = masked_bce(logits, delayed, mask)
	prob = torch.sigmoid(logits)
	pause = (speech >= 0.5) & (stop < 0.2) & (mask > 0.5)
	late = (prob * pause.float()).sum() / pause.float().sum().clamp_min(1.0)
	return bce + late, bce, late


def voiced_pitch_loss(pred, rmvpe, mask):
	"""f0 error on voiced frames only. Silent and unvoiced frames are not graded."""
	f0 = rmvpe[:, 0]
	voiced = (rmvpe[:, 1] >= 0.5).float() * mask
	err = (pred - f0).pow(2)
	return (err * voiced).sum() / voiced.sum().clamp_min(1.0)


def vad_loss(logits, speech, start, stop, mask, w_speech=1.0, w_start=2.0, w_stop=3.0):
	speech_bce = masked_bce(logits[..., 0], speech, mask, neg_w=1.15)
	qdr = qdr_loss(logits[..., 0], speech, mask)
	speech_l = speech_bce + qdr
	start_l = masked_bce(logits[..., 1], start, mask)
	stop_l, stop_bce, late = stop_loss(logits[..., 2], speech, stop, mask)
	total = w_speech * speech_l + w_start * start_l + w_stop * stop_l
	with torch.no_grad():
		prob = torch.sigmoid(logits[..., 2])
		end = (stop >= 0.5) & (mask > 0.5)
		pause = (speech >= 0.5) & (stop < 0.2) & (mask > 0.5)
		end_p = (prob * end.float()).sum() / end.float().sum().clamp_min(1.0)
		pause_p = (prob * pause.float()).sum() / pause.float().sum().clamp_min(1.0)
	parts = {'speech': speech_l.item(), 'bce': speech_bce.item(), 'qdr': qdr.item(), 'start': start_l.item(), 'stop': stop_l.item(), 'stop_bce': stop_bce.item(), 'late': late.item(), 'stop_on_end': end_p.item(), 'stop_on_pause': pause_p.item(), }
	return total, parts


def save_checkpoint(path, model, epoch, loss, extra=None):
	os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
	payload = {'model': model.state_dict(), 'config': model.config(), 'epoch': epoch, 'loss': loss, 'arch': 'liquid_gate'}
	if extra:
		payload.update(extra)
	torch.save(payload, path)


def load_vad(path, device=None):
	ckpt = torch.load(path, map_location='cpu', weights_only=False)
	cfg = ckpt.get('config') or {}
	model = LiquidVAD(**{k: cfg[k] for k in ('n_mels', 'rmvpe_dim', 'hidden', 'dropout', 'dilations') if k in cfg})
	model.load_state_dict(ckpt['model'])
	if device is not None:
		model = model.to(device)
	return model.eval(), ckpt


class ExportVAD(nn.Module):
	"""mel [B, 128, T] and rmvpe [B, 3, T] (zeros if pitch failed) → logits [B, 3, T]."""
	def __init__(self, net):
		super().__init__()
		self.net = net

	def forward(self, mel, rmvpe):
		return self.net(mel, rmvpe).transpose(1, 2).contiguous()


def load_vad_coreml(path, units=None):
	import coremltools as ct
	if units is None:
		units = ct.ComputeUnit.ALL
	return ct.models.MLModel(path, compute_units=units)


def _coreml_inputs(ml):
	spec = ml.get_spec()
	return [i.name for i in spec.description.input]


def _coreml_fixed_t(ml):
	spec = ml.get_spec()
	for inp in spec.description.input:
		if inp.name != 'mel':
			continue
		shape = list(inp.type.multiArrayType.shape)
		if shape and int(shape[-1]) > 1:
			return int(shape[-1])
	return None


def infer_vad_coreml(ml, mel, rmvpe=None, frames=None, overlap=COREML_OVERLAP):
	"""mel [128, T] -> logits [T, 3]. Fixed-chunk models overlap-add; flexible models take T as-is."""
	mel = np.ascontiguousarray(np.asarray(mel, dtype=np.float32))
	t = mel.shape[-1]
	names = _coreml_inputs(ml)
	fixed = _coreml_fixed_t(ml)
	if frames is None:
		frames = fixed or COREML_CHUNK_FRAMES

	def _pitch_chunk(a, n):
		if 'rmvpe' not in names:
			return None
		if rmvpe is None:
			return np.zeros((RMVPE_DIM, n), dtype=np.float32)
		pitch = np.ascontiguousarray(np.asarray(rmvpe, dtype=np.float32))
		part = pitch[:, a:a + n]
		if part.shape[-1] < n:
			part = np.pad(part, ((0, 0), (0, n - part.shape[-1])))
		return part

	def _run(chunk, a0):
		feed = {'mel': chunk[None]}
		pitch = _pitch_chunk(a0, chunk.shape[-1])
		if pitch is not None:
			feed['rmvpe'] = pitch[None]
		out = ml.predict(feed)
		logits = np.array(out['logits'], dtype=np.float32)
		if logits.ndim == 3:
			logits = logits[0]
		if logits.shape[0] == 3:
			return logits
		return logits.T

	if fixed is None or t <= frames:
		x = np.pad(mel, ((0, 0), (0, frames - t))) if (fixed is not None and t < frames) else mel
		return _run(x, 0)[:, :t].T
	acc = np.zeros((3, t), dtype=np.float32)
	w = np.zeros(t, dtype=np.float32)
	step = max(1, frames - overlap)
	win = np.clip(np.hanning(frames).astype(np.float32), 0.05, None)
	a = 0
	while a < t:
		b = min(a + frames, t)
		chunk = np.pad(mel[:, a:b], ((0, 0), (0, frames - (b - a)))) if (b - a) < frames else mel[:, a:b]
		y = _run(chunk, a)[:, :b - a]
		ww = win[:b - a]
		acc[:, a:b] += y * ww
		w[a:b] += ww
		if b >= t:
			break
		a += step
	return (acc / np.clip(w, 1e-6, None)).T
