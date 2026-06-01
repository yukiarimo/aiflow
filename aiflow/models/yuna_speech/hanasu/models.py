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


def hanasu_loss(output, target_mel, mel_lens, token_lens, mel_weight=1.0, stop_weight=0.5, guided_attention_weight=0.2, guided_attention_sigma=0.4, ):
	max_len = min(output.mel.shape[1], target_mel.shape[1])
	target = target_mel[:, :max_len]
	mel_mask = ~make_padding_mask(mel_lens.clamp(max=max_len), max_len)
	mel_loss = F.l1_loss(output.mel[:, :max_len][mel_mask], target[mel_mask])
	post_loss = F.l1_loss(output.mel_postnet[:, :max_len][mel_mask], target[mel_mask])
	stop_target = torch.zeros_like(output.stop_logits[:, :max_len])
	for i, length in enumerate(mel_lens.tolist()):
		stop_target[i, max(0, min(length, max_len) - 1):] = 1.0
	stop_loss = F.binary_cross_entropy_with_logits(output.stop_logits[:, :max_len], stop_target)
	guide_loss = guided_attention_loss(output.alignments, token_lens, mel_lens, reduction_factor=max(1, output.stop_logits.shape[1] // max(1, output.alignments.shape[1])), sigma=guided_attention_sigma, )
	loss = mel_weight * (mel_loss + post_loss) + stop_weight * stop_loss + guided_attention_weight * guide_loss
	return loss, {"mel_loss": float(mel_loss.detach().cpu()), "postnet_mel_loss": float(post_loss.detach().cpu()), "stop_loss": float(stop_loss.detach().cpu()), "guided_attention_loss": float(guide_loss.detach().cpu()), }


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
