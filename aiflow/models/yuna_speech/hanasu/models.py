import math
import torch
from torch import nn
import torch.nn.functional as F


def make_padding_mask(lengths, max_len=None):
	max_len = int(max_len or lengths.max().item())
	return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


class Prenet(nn.Module):
	def __init__(self, in_dim, sizes, dropout):
		super().__init__()
		layers = []
		last = in_dim
		for size in sizes:
			layers.append(nn.Linear(last, size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			last = size
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class ConvBlock(nn.Module):
	def __init__(self, channels, kernel_size, dropout):
		super().__init__()
		self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
		self.bn = nn.BatchNorm1d(channels)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.dropout(F.relu(self.bn(self.conv(x))))


class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, pad_id, dropout, conv_layers, num_languages):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
		self.language_embedding = nn.Embedding(num_languages, hidden_size)  # Language identity is tied to the text, so the language embedding is mixed into the token embeddings (element-wise add, broadcast over time) before the conv stack.
		self.convs = nn.ModuleList([ConvBlock(hidden_size, 5, dropout) for _ in range(conv_layers)])
		self.lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True, )

	def forward(self, tokens, language_ids):
		x = self.embedding(tokens)
		x = x + self.language_embedding(language_ids).unsqueeze(1)
		x = x.transpose(1, 2)
		for conv in self.convs:
			x = conv(x)
		x = x.transpose(1, 2)
		self.lstm.flatten_parameters()
		out, _ = self.lstm(x)
		return out


class LocationSensitiveAttention(nn.Module):
	def __init__(self, query_dim, memory_dim, attention_dim, location_channels, location_kernel):
		super().__init__()
		self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
		self.memory_layer = nn.Linear(memory_dim, attention_dim, bias=False)
		self.location_conv = nn.Conv1d(2, location_channels, location_kernel, padding=location_kernel // 2, bias=False)
		self.location_layer = nn.Linear(location_channels, attention_dim, bias=False)
		self.v = nn.Linear(attention_dim, 1, bias=True)

	def forward(self, query, memory, processed_memory, attention_weights, attention_cum, mask, window_mask=None, ):
		location = torch.stack([attention_weights, attention_cum], dim=1)
		location = self.location_conv(location).transpose(1, 2)
		energies = self.v(torch.tanh(self.query_layer(query).unsqueeze(1) + processed_memory + self.location_layer(location))).squeeze(-1)
		energies = energies.masked_fill(mask, -1e4)
		if window_mask is not None:
			energies = energies.masked_fill(window_mask, -1e4)
		weights = F.softmax(energies, dim=-1)
		context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
		return context, weights


class Postnet(nn.Module):
	def __init__(self, n_mels, channels, layers, dropout):
		super().__init__()
		modules = []
		in_channels = n_mels
		for idx in range(layers):
			out_channels = n_mels if idx == layers - 1 else channels
			modules.append(nn.Conv1d(in_channels, out_channels, 5, padding=2))
			modules.append(nn.BatchNorm1d(out_channels))
			if idx < layers - 1:
				modules.append(nn.Tanh())
			modules.append(nn.Dropout(dropout))
			in_channels = out_channels
		self.net = nn.Sequential(*modules)

	def forward(self, mel):
		return self.net(mel.transpose(1, 2)).transpose(1, 2)


class TacotronOutput:
	def __init__(self, mel, mel_postnet, stop_logits, alignments):
		self.mel = mel
		self.mel_postnet = mel_postnet
		self.stop_logits = stop_logits
		self.alignments = alignments


class HanasuTTS(nn.Module):
	def __init__(self, vocab_size, n_mels, config, pad_id=0):
		super().__init__()
		hidden = int(config.get("hidden_size", 256))
		dropout = float(config.get("dropout", 0.5))
		prenet_sizes = list(config.get("prenet_sizes", [256, 128]))
		attention_dim = int(config.get("attention_dim", 128))
		decoder_dim = int(config.get("decoder_dim", 512))
		self.n_mels = n_mels
		self.pad_id = pad_id
		self.num_speakers = int(config.get("num_speakers", 2))
		self.num_languages = int(config.get("num_languages", 3))
		self.reduction_factor = int(config.get("reduction_factor", 4))
		self.encoder = Encoder(vocab_size, hidden, pad_id, dropout=float(config.get("encoder_dropout", 0.15)), conv_layers=int(config.get("encoder_conv_layers", 3)), num_languages=self.num_languages, )
		self.speaker_embedding = nn.Embedding(self.num_speakers, hidden)  # Speaker identity is a global property of the utterance, so it is broadcast across the encoder memory (added to every text position) before attention/decoding.
		self.prenet = Prenet(n_mels, prenet_sizes, dropout=dropout)
		prenet_dim = prenet_sizes[-1]
		self.attention_rnn = nn.GRUCell(prenet_dim + hidden, decoder_dim)
		self.attention = LocationSensitiveAttention(query_dim=decoder_dim, memory_dim=hidden, attention_dim=attention_dim, location_channels=int(config.get("location_channels", 32)), location_kernel=int(config.get("location_kernel_size", 31)), )
		self.decoder_rnn = nn.GRUCell(decoder_dim + hidden, decoder_dim)
		proj_dim = decoder_dim + hidden
		self.mel_proj = nn.Linear(proj_dim, n_mels * self.reduction_factor)
		self.stop_proj = nn.Linear(proj_dim, self.reduction_factor)
		self.postnet = Postnet(n_mels, channels=int(config.get("postnet_channels", 512)), layers=int(config.get("postnet_layers", 5)), dropout=float(config.get("postnet_dropout", 0.5)), )

	def initialize_decoder_states(self, memory):
		batch, text_len, hidden = memory.shape
		attention_hidden = memory.new_zeros(batch, self.attention_rnn.hidden_size)
		decoder_hidden = memory.new_zeros(batch, self.decoder_rnn.hidden_size)
		context = memory.new_zeros(batch, hidden)
		attention_weights = memory.new_zeros(batch, text_len)
		attention_weights[:, 0] = 1.0
		attention_cum = memory.new_zeros(batch, text_len)
		attention_cum[:, 0] = 1.0
		return attention_hidden, decoder_hidden, context, attention_weights, attention_cum

	def decode_step(self, decoder_input, memory, processed_memory, mask, attention_hidden, decoder_hidden, context, attention_weights, attention_cum, window_mask=None, ):
		prenet_out = self.prenet(decoder_input)
		attention_hidden = self.attention_rnn(torch.cat([prenet_out, context], dim=-1), attention_hidden)
		context, attention_weights = self.attention(attention_hidden, memory, processed_memory, attention_weights, attention_cum, mask, window_mask, )
		attention_cum = attention_cum + attention_weights
		decoder_hidden = self.decoder_rnn(torch.cat([attention_hidden, context], dim=-1), decoder_hidden)
		proj_input = torch.cat([decoder_hidden, context], dim=-1)
		mel = self.mel_proj(proj_input).view(proj_input.shape[0], self.reduction_factor, self.n_mels)
		stop = self.stop_proj(proj_input)
		return mel, stop, attention_hidden, decoder_hidden, context, attention_weights, attention_cum

	def forward(self, tokens, token_lens, mels, speaker_ids, language_ids):
		memory = self.encoder(tokens, language_ids)
		memory = memory + self.speaker_embedding(speaker_ids).unsqueeze(1)
		mask = make_padding_mask(token_lens, tokens.shape[1])
		processed_memory = self.attention.memory_layer(memory)
		states = self.initialize_decoder_states(memory)
		r = self.reduction_factor
		pad_frames = (r - (mels.shape[1] % r)) % r
		if pad_frames:
			mels_padded = F.pad(mels, (0, 0, 0, pad_frames))
		else:
			mels_padded = mels
		decoder_steps = mels_padded.shape[1] // r
		go = mels.new_zeros(mels.shape[0], self.n_mels)
		mel_outputs = []
		stop_outputs = []
		alignments = []
		decoder_input = go
		for t in range(decoder_steps):
			mel, stop, *states = self.decode_step(decoder_input, memory, processed_memory, mask, *states)
			mel_outputs.append(mel)
			stop_outputs.append(stop)
			alignments.append(states[-2])
			decoder_input = mels_padded[:, (t + 1) * r - 1]
		mel = torch.stack(mel_outputs, dim=1).reshape(mels.shape[0], decoder_steps * r, self.n_mels)
		stop_logits = torch.stack(stop_outputs, dim=1).reshape(mels.shape[0], decoder_steps * r)
		align = torch.stack(alignments, dim=1)
		mel_postnet = mel + self.postnet(mel)
		return TacotronOutput(mel=mel, mel_postnet=mel_postnet, stop_logits=stop_logits, alignments=align)

	@torch.no_grad()
	def infer(self, tokens, token_lens, speaker_ids, language_ids, max_steps=1200, stop_threshold=0.55, min_steps=30, attention_window=12, mel_prefix=None, prefix_token_len=0, use_speaker_embedding=True, ):
		memory = self.encoder(tokens, language_ids)
		if use_speaker_embedding:
			memory = memory + self.speaker_embedding(speaker_ids).unsqueeze(1)
		mask = make_padding_mask(token_lens, tokens.shape[1])
		processed_memory = self.attention.memory_layer(memory)
		states = self.initialize_decoder_states(memory)
		decoder_input = memory.new_zeros(tokens.shape[0], self.n_mels)
		mel_outputs = []
		stop_outputs = []
		alignments = []
		r = self.reduction_factor
		prefix_frames = 0
		prefix_decoder_steps = 0
		prev_attention_index = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
		if mel_prefix is not None:
			prefix_frames = int(mel_prefix.shape[1])
			pad_frames = (r - (prefix_frames % r)) % r
			mel_prefix_padded = F.pad(mel_prefix, (0, 0, 0, pad_frames)) if pad_frames else mel_prefix
			prefix_decoder_steps = mel_prefix_padded.shape[1] // r
			prompt_token_limit = max(1, int(prefix_token_len))
			for step in range(prefix_decoder_steps):
				window_mask = None
				if attention_window > 0:
					positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
					left = (prev_attention_index - 1).clamp_min(0).unsqueeze(1)
					right = (prev_attention_index + attention_window).clamp_max(prompt_token_limit - 1).unsqueeze(1)
					window_mask = (positions < left) | (positions > right)
					window_mask = window_mask | mask
				mel, stop, *states = self.decode_step(decoder_input, memory, processed_memory, mask, *states, window_mask=window_mask)
				ref_mel = mel_prefix_padded[:, step * r:(step + 1) * r].view(mel.shape[0], r, self.n_mels)
				mel_outputs.append(ref_mel)
				stop_outputs.append(stop)
				alignments.append(states[-2])
				prev_attention_index = states[-2].argmax(dim=-1).clamp_min(prev_attention_index).clamp_max(prompt_token_limit - 1)
				next_idx = min((step + 1) * r - 1, mel_prefix_padded.shape[1] - 1)
				decoder_input = mel_prefix_padded[:, next_idx]
		target_max_decoder_steps = max(1, max_steps // r)
		max_decoder_steps = prefix_decoder_steps + target_max_decoder_steps
		min_decoder_steps = prefix_decoder_steps + max(1, min_steps // r)
		if not mel_outputs:
			prev_attention_index = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
		for step in range(prefix_decoder_steps, max_decoder_steps):
			window_mask = None
			if attention_window > 0:
				positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
				left = (prev_attention_index - 1).clamp_min(0).unsqueeze(1)
				right = (prev_attention_index + attention_window).clamp_max(tokens.shape[1] - 1).unsqueeze(1)
				window_mask = (positions < left) | (positions > right)
				window_mask = window_mask | mask
			mel, stop, *states = self.decode_step(decoder_input, memory, processed_memory, mask, *states, window_mask=window_mask)
			mel_outputs.append(mel)
			stop_outputs.append(stop)
			alignments.append(states[-2])
			prev_attention_index = states[-2].argmax(dim=-1).clamp_min(prev_attention_index)
			decoder_input = mel[:, -1]
			if step >= min_decoder_steps and torch.sigmoid(stop[:, -1]).min().item() > stop_threshold:
				break
		mel = torch.stack(mel_outputs, dim=1).reshape(tokens.shape[0], -1, self.n_mels)
		stop_logits = torch.stack(stop_outputs, dim=1).reshape(tokens.shape[0], -1)
		align = torch.stack(alignments, dim=1)
		if prefix_frames > 0:
			gen_mel = mel[:, prefix_frames:]
			gen_postnet = gen_mel + self.postnet(gen_mel) if gen_mel.shape[1] > 0 else gen_mel
			mel_postnet = torch.cat([mel[:, :prefix_frames], gen_postnet], dim=1)
		else:
			mel_postnet = mel + self.postnet(mel)
		return TacotronOutput(mel=mel, mel_postnet=mel_postnet, stop_logits=stop_logits, alignments=align)


def hanasu_loss(output, target_mel, mel_lens, token_lens, mel_weight=1.0, stop_weight=0.5, guided_attention_weight=0.2, guided_attention_sigma=0.4, mel_mse_weight=0.0, temporal_delta_weight=0.0, freq_delta_weight=0.0, band_weights=None, sample_weights=None, ):
	"""Multi-term acoustic loss.
	band_weights: optional (B, n_mels) per-sample mel-band emphasis (e.g. boosting each speaker's f0/harmonic region) so the model is pushed to reproduce pitch/expressive detail rather than averaging it away.
	sample_weights: optional (B,) per-sample scalar (speaker balancing).
	Beyond the standard L1 reconstruction it adds: an L2 term (discourages the blurry, muffled mels L1 alone tolerates), a temporal-delta term (matches frame-to-frame motion -> less jittery, more natural prosody) and a frequency-delta term (sharper formants).
	"""
	max_len = min(output.mel.shape[1], target_mel.shape[1])
	pred = output.mel[:, :max_len]
	post = output.mel_postnet[:, :max_len]
	target = target_mel[:, :max_len]
	frame_valid = (~make_padding_mask(mel_lens.clamp(max=max_len), max_len)).float()  # (B, T)
	weight = frame_valid.unsqueeze(-1)  # (B, T, 1)
	if band_weights is not None:
		weight = weight * band_weights.unsqueeze(1)  # (B, T, n_mels)
	if sample_weights is not None:
		weight = weight * sample_weights.view(-1, 1, 1)
	denom = weight.sum().clamp(min=1.0)

	mel_loss = (weight * (pred - target).abs()).sum() / denom
	post_loss = (weight * (post - target).abs()).sum() / denom
	mel_mse = (weight * (post - target).square()).sum() / denom if mel_mse_weight > 0 else pred.new_tensor(0.0)

	if temporal_delta_weight > 0 and max_len > 1:
		valid_pair = (frame_valid[:, 1:] * frame_valid[:, :-1]).unsqueeze(-1)  # (B, T-1, 1)
		w_t = valid_pair * (band_weights.unsqueeze(1) if band_weights is not None else 1.0)
		if sample_weights is not None:
			w_t = w_t * sample_weights.view(-1, 1, 1)
		d_pred = post[:, 1:] - post[:, :-1]
		d_tgt = target[:, 1:] - target[:, :-1]
		temporal_loss = (w_t * (d_pred - d_tgt).abs()).sum() / (w_t.sum().clamp(min=1.0) if torch.is_tensor(w_t) else 1.0)
	else:
		temporal_loss = pred.new_tensor(0.0)

	if freq_delta_weight > 0 and post.shape[-1] > 1:
		fw = frame_valid.unsqueeze(-1)
		df_pred = post[:, :, 1:] - post[:, :, :-1]
		df_tgt = target[:, :, 1:] - target[:, :, :-1]
		freq_loss = (fw * (df_pred - df_tgt).abs()).sum() / (fw.expand_as(df_pred).sum().clamp(min=1.0))
	else:
		freq_loss = pred.new_tensor(0.0)

	stop_target = torch.zeros_like(output.stop_logits[:, :max_len])
	for i, length in enumerate(mel_lens.tolist()):
		stop_target[i, max(0, min(length, max_len) - 1):] = 1.0
	stop_loss = F.binary_cross_entropy_with_logits(output.stop_logits[:, :max_len], stop_target)
	guide_loss = guided_attention_loss(output.alignments, token_lens, mel_lens, reduction_factor=max(1, output.stop_logits.shape[1] // max(1, output.alignments.shape[1])), sigma=guided_attention_sigma, )
	loss = mel_weight * (mel_loss + post_loss) + mel_mse_weight * mel_mse + temporal_delta_weight * temporal_loss + freq_delta_weight * freq_loss + stop_weight * stop_loss + guided_attention_weight * guide_loss
	return loss, {"mel_loss": float(mel_loss.detach().cpu()), "postnet_mel_loss": float(post_loss.detach().cpu()), "mel_mse": float(mel_mse.detach().cpu()), "temporal_delta_loss": float(temporal_loss.detach().cpu()), "freq_delta_loss": float(freq_loss.detach().cpu()), "stop_loss": float(stop_loss.detach().cpu()), "guided_attention_loss": float(guide_loss.detach().cpu()), }


def guided_attention_loss(alignments, token_lens, mel_lens, reduction_factor, sigma=0.4):
	if alignments.numel() == 0:
		return alignments.new_tensor(0.0)
	losses = []
	steps = alignments.shape[1]
	text_steps = alignments.shape[2]
	for batch_idx in range(alignments.shape[0]):
		text_len = max(1, min(int(token_lens[batch_idx].item()), text_steps))
		mel_len = int(mel_lens[batch_idx].item())
		dec_len = max(1, min(steps, (mel_len + reduction_factor - 1) // reduction_factor))
		t = torch.arange(dec_len, device=alignments.device, dtype=alignments.dtype) / max(1, dec_len)
		n = torch.arange(text_len, device=alignments.device, dtype=alignments.dtype) / max(1, text_len)
		weights = 1.0 - torch.exp(-((t[:, None] - n[None, :])**2) / (2.0 * sigma * sigma))
		losses.append((alignments[batch_idx, :dec_len, :text_len] * weights).sum(dim=-1).mean())
	return torch.stack(losses).mean()


def _hz_to_mel(hz):
	return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
	return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def mel_band_center_freqs(n_mels, f_min, f_max):
	"""Center frequency (Hz) of each mel band, matching torchaudio's htk mel layout."""
	m_min, m_max = _hz_to_mel(f_min), _hz_to_mel(f_max)
	m_pts = [m_min + (m_max - m_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
	centers = [_mel_to_hz(m) for m in m_pts[1:-1]]
	return torch.tensor(centers, dtype=torch.float32)


def build_speaker_band_weights(num_speakers, n_mels, f_min, f_max, speaker_f0_by_id, emphasis=0.5, harmonics=6):
	"""Per-speaker (num_speakers, n_mels) mel-band weights that emphasize the fundamental and first few harmonics of each voice. This nudges the mel loss to care more about the pitch/harmonic structure that carries expressiveness, per speaker (e.g. a high-f0 girl vs a low-f0 boy). Weights are mean-normalized to ~1 per speaker so the overall loss scale is unchanged. Returns None when emphasis<=0 so callers can skip the work entirely."""
	if emphasis <= 0 or not speaker_f0_by_id:
		return None
	centers = mel_band_center_freqs(n_mels, f_min, f_max)
	weights = torch.ones(num_speakers, n_mels, dtype=torch.float32)
	for sid, f0 in speaker_f0_by_id.items():
		if f0 is None or float(f0) <= 0:
			continue
		f0 = float(f0)
		w = torch.ones(n_mels, dtype=torch.float32)
		for k in range(1, int(harmonics) + 1):
			hf = k * f0
			if hf > f_max:
				break
			width = max(0.5 * f0, 1.0)
			w = w + emphasis * torch.exp(-0.5 * ((centers - hf) / width)**2)
		w = w * (n_mels / w.sum().clamp(min=1e-6))  # mean-normalize to keep loss scale stable
		if 0 <= int(sid) < num_speakers:
			weights[int(sid)] = w
	return weights


class MultiResolutionSTFTLoss(nn.Module):
	"""Spectral-convergence + log-magnitude STFT loss at several resolutions, computed on waveforms (e.g. Vocos-decoded predicted vs. target mel). Captures periodic/harmonic fine structure that frame-wise mel L1 misses, without training the vocoder itself."""
	def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512), win_sizes=(512, 1024, 2048)):
		super().__init__()
		self.fft_sizes = list(fft_sizes)
		self.hop_sizes = list(hop_sizes)
		self.win_sizes = list(win_sizes)
		self._windows = {}

	def _window(self, win, device, dtype):
		key = (win, device, dtype)
		if key not in self._windows:
			self._windows[key] = torch.hann_window(win, device=device, dtype=dtype)
		return self._windows[key]

	def _stft_mag(self, x, fft, hop, win):
		window = self._window(win, x.device, x.dtype)
		spec = torch.stft(x, n_fft=fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
		return spec.abs().clamp(min=1e-7)

	def forward(self, pred, target):
		pred = pred.float()
		target = target.float()
		total = pred.new_tensor(0.0)
		for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
			x = self._stft_mag(pred, fft, hop, win)
			y = self._stft_mag(target, fft, hop, win)
			sc = torch.norm(y - x, p="fro") / torch.norm(y, p="fro").clamp(min=1e-7)
			mag = F.l1_loss(torch.log(x), torch.log(y))
			total = total + sc + mag
		return total / max(1, len(self.fft_sizes))
