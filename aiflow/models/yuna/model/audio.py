import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class YunaAudioEncoderConfig:
    def __init__(self):
        self.num_mel_bins = 128
        self.encoder_layers = 32
        self.encoder_attention_heads = 20
        self.encoder_ffn_dim = 5120
        self.d_model = 1280
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.activation_function = "gelu"
        self.max_source_positions = 1500
        self.n_window = 100
        self.output_dim = 2048
        self.head_dim = self.d_model // self.encoder_attention_heads
        self.dtype = torch.bfloat16

def gelu(x): return F.gelu(x)

class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0: raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]

        positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("positional_embedding", positional_embedding, persistent=False)

    def forward(self, seqlen): return self.positional_embedding[:seqlen, :]

class YunaAudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states, cu_seqlens, attention_mask):
        batch_size = len(cu_seqlens) - 1
        device = hidden_states.device

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_outputs = []
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            l = end - start
            if l == 0: continue

            q = query_states[start:end].reshape(l, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            k = key_states[start:end].reshape(l, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            v = value_states[start:end].reshape(l, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, start:end, start:end]
                attn_weights = attn_weights + mask_chunk

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(l, -1).contiguous()
            attn_outputs.append(attn_output)

        if not attn_outputs:return torch.empty((0, self.embed_dim), dtype=hidden_states.dtype, device=device)
        attn_output_full = torch.cat(attn_outputs, dim=0)
        attn_output_full = self.out_proj(attn_output_full)
        return attn_output_full

class YunaAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn = YunaAudioAttention(config)
        self.activation_fn = gelu
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens, attention_mask):
        residual = hidden_states
        normed_h = self.self_attn_layer_norm(hidden_states)
        attn_output = self.self_attn(hidden_states=normed_h, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_h = self.final_layer_norm(hidden_states)
        ffn_output = self.fc1(normed_h)
        ffn_output = self.activation_fn(ffn_output)
        ffn_output = self.fc2(ffn_output)
        hidden_states = residual + ffn_output
        return hidden_states

class YunaAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = YunaAudioEncoderConfig()
        self.dtype = self.config.dtype
        self.n_window = self.config.n_window
        self.conv1 = nn.Conv1d(self.config.num_mel_bins, self.config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.config.d_model, self.config.d_model, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.config.max_source_positions, self.config.d_model)
        self.layers = nn.ModuleList([YunaAudioEncoderLayer(self.config) for _ in range(self.config.encoder_layers)])
        self.ln_post = nn.LayerNorm(self.config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(self.config.d_model, self.config.output_dim)

    def _create_attention_mask(self, seq_length, cu_seqlens, dtype, device):
        min_dtype = torch.finfo(dtype).min
        attention_mask = torch.full([1, 1, seq_length, seq_length], min_dtype, device=device, dtype=dtype)

        for i in range(1, len(cu_seqlens)):
            start, end = cu_seqlens[i - 1].item(), cu_seqlens[i].item()
            attention_mask[..., start:end, start:end] = 0
        return attention_mask

    def _padded_and_mask_function(self, chunk_list, tensor_len):
        max_len = tensor_len.max().item()
        dim = chunk_list[0].shape[0]
        device = chunk_list[0].device
        padded_feature = torch.full((len(chunk_list), dim, max_len), 0.0, dtype=self.dtype, device=device)
        batch_mask = torch.zeros((len(tensor_len), max_len), dtype=torch.long, device=device)

        for i, length in enumerate(tensor_len):
            length = length.item()
            batch_mask[i, :length] = 1
            padded_feature[i, :, :length] = chunk_list[i].to(self.dtype)

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        batch_mask_after_cnn = torch.zeros((len(tensor_len), max_len_after_cnn), dtype=torch.long, device=device)
        for i, length in enumerate(feature_lens_after_cnn): batch_mask_after_cnn[i, :length.item()] = 1
        return padded_feature, batch_mask.unsqueeze(1), batch_mask_after_cnn.bool()

    def forward(self, input_features, feature_lens):
        device = input_features.device
        all_chunk_lengths = []
        for L_segment in feature_lens.tolist():
            n_full_chunks = L_segment // (2 * self.n_window)
            tail_chunk_len = L_segment % (2 * self.n_window)
            if n_full_chunks > 0: all_chunk_lengths.extend([2 * self.n_window] * n_full_chunks)
            if tail_chunk_len > 0: all_chunk_lengths.append(tail_chunk_len)

        chunk_list = input_features.to(self.dtype).split(all_chunk_lengths, dim=1)
        chunk_lengths_tensor = torch.tensor(all_chunk_lengths, dtype=torch.long, device=device)
        padded_feature, padded_mask, padded_mask_after_cnn = self._padded_and_mask_function(chunk_list, chunk_lengths_tensor)

        padded_embed = self.conv1(padded_feature)
        padded_embed = F.gelu(padded_embed) * padded_mask.to(padded_embed.dtype)
        padded_embed = self.conv2(padded_embed)
        padded_embed = F.gelu(padded_embed)
        hidden_states_padded = padded_embed.transpose(1, 2)
        hidden_states_padded = hidden_states_padded + self.positional_embedding(hidden_states_padded.shape[1]).unsqueeze(0).to(hidden_states_padded.dtype)

        hidden_states = hidden_states_padded[padded_mask_after_cnn]
        aftercnn_lens = padded_mask_after_cnn.sum(1).to(torch.int32)
        cu_seqlens = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), aftercnn_lens.cumsum(0)))
        attention_mask = self._create_attention_mask(hidden_states.shape[0], cu_seqlens, hidden_states.dtype, device)

        for encoder_layer in self.layers: hidden_states = encoder_layer(hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        hidden_states_list_chunks = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        _, audio_output_lengths = get_feat_extract_output_lengths(feature_lens)

        chunk_idx = 0
        for i, L_segment in enumerate(feature_lens.tolist()):
            expected_output_len = audio_output_lengths[i].item()
            n_full_chunks = L_segment // (2 * self.n_window)
            n_chunks = n_full_chunks + (1 if L_segment % (2 * self.n_window) > 0 else 0)
            segment_chunks = hidden_states_list_chunks[chunk_idx : chunk_idx + n_chunks]
            chunk_idx += n_chunks

            segment_states = torch.cat(segment_chunks, dim=0)
            segment_states = segment_states.transpose(0, 1).unsqueeze(0)
            segment_states = self.avg_pooler(segment_states).squeeze(0).transpose(0, 1)
            segment_states = self.ln_post(segment_states)
            segment_states = self.proj(segment_states)

            if segment_states.shape[0] != expected_output_len: raise RuntimeError(f"Output length mismatch for segment {i}. Expected {expected_output_len}, got {segment_states.shape[0]}.")
            token_audio_list.append(segment_states)

        return torch.cat(token_audio_list, dim=0)

class AudioProjector(nn.Module):
    """Projects audio embeddings from the YunaAudioEncoder tower to the LLM's embedding space."""
    def __init__(self, audio_tower_dim=2048, llm_embed_dim=1536):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(audio_tower_dim, llm_embed_dim), nn.GELU(), nn.Linear(llm_embed_dim, llm_embed_dim))

    def forward(self, x): return self.proj(x)

def get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers (after stride 2 and pooling).
    """
    input_lengths = (input_lengths.long() - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths

def audio_to_mel_features(audio_path, sr=48000, n_fft=400, hop_length=160, n_mels=128):
    """
    Loads audio and computes log-mel spectrogram features.
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=1.0)
    features = torch.from_numpy(log_mel).float()
    return features, features.shape[1]