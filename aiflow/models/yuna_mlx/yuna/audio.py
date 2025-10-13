import librosa
import numpy as np
import mlx.core as mx
import mlx.nn as nn

class YunaAudioEncoderConfig:
    d_model = 1280
    encoder_layers = 32
    encoder_ffn_dim = 5120
    encoder_attention_heads = 20
    head_dim = d_model // encoder_attention_heads
    output_dim = 2048
    n_window = 100
    num_mel_bins = 128
    max_source_positions = 1500

class YunaAudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.scaling = config.head_dim**-0.5

    def __call__(self, x, mask):
        B, L, C = x.shape
        q = self.q_proj(x).reshape(B, L, self.config.encoder_attention_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.config.encoder_attention_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.config.encoder_attention_heads, self.config.head_dim).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scaling, mask=mask)
        return self.out_proj(attn.transpose(0, 2, 1, 3).reshape(B, L, C))

class YunaAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = YunaAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        return x

class YunaAudioEncoder(nn.Module):
    def __init__(self, config=YunaAudioEncoderConfig()):
        super().__init__(); self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        self.pos_conv_embed = nn.SinusoidalPositionalEncoding(config.d_model, max_len=config.max_source_positions)
        self.layers = [YunaAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)

    def _pad_and_mask(self, chunks, lengths):
        max_len = lengths.max().item()
        padded = mx.zeros((len(chunks), self.config.num_mel_bins, max_len))
        mask = mx.zeros((len(chunks), 1, 1, max_len))

        for i, (chunk, length) in enumerate(zip(chunks, lengths.tolist())):
            padded[i, :, :length] = chunk
            mask[i, :, :, :length] = 1

        return padded, mask

    def __call__(self, features, feature_lens):
        all_chunks, all_lengths = [], []

        for i, L in enumerate(feature_lens.tolist()):
            pos = 0

            while pos < L:
                chunk_len = min(2 * self.config.n_window, L - pos)
                all_chunks.append(features[i, :chunk_len, :])
                all_lengths.append(chunk_len); pos += chunk_len

        all_lengths = mx.array(all_lengths, dtype=mx.int32)
        padded, mask = self._pad_and_mask(all_chunks, all_lengths)

        x = self.conv2(nn.gelu(self.conv1(padded.transpose(0, 2, 1)))).transpose(0, 2, 1)
        x = x + self.pos_conv_embed(x)

        mask_after_cnn = (mask > 0).squeeze(1).squeeze(1)[:, ::2]
        attn_mask = mx.expand_dims(mask_after_cnn, 1) * mx.expand_dims(mask_after_cnn, 2)
        attn_mask = mx.expand_dims(attn_mask, 1).astype(x.dtype)

        for layer in self.layers: x = layer(x, mask=attn_mask)

        token_audio_list = []
        _, output_lens = get_feat_extract_output_lengths(feature_lens)
        chunk_idx = 0

        for i, L_segment in enumerate(feature_lens.tolist()):
            n_chunks = (L_segment + (2 * self.config.n_window) - 1) // (2 * self.config.n_window)
            segment = x[chunk_idx : chunk_idx + n_chunks].reshape(-1, x.shape[-1])
            chunk_idx += n_chunks

            segment = self.avg_pooler(mx.expand_dims(segment.transpose(), 0)).transpose().squeeze(0)
            segment = self.proj(self.ln_post(segment))

            # Ensure correct length
            expected_len = output_lens[i].item()
            if segment.shape[0] > expected_len: segment = segment[:expected_len]
            token_audio_list.append(segment)

        return mx.concatenate(token_audio_list, axis=0)

class AudioProjector(nn.Module):
    def __init__(self, audio_tower_dim, llm_embed_dim):
        super().__init__()
        self.proj_1 = nn.Linear(audio_tower_dim, llm_embed_dim)
        self.proj_2 = nn.Linear(llm_embed_dim, llm_embed_dim)

    def __call__(self, x): return self.proj_2(nn.gelu(self.proj_1(x)))

def get_feat_extract_output_lengths(input_lengths):
    input_lengths = (input_lengths.astype(mx.int64) - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths

def audio_to_mel_features(audio_path, sr=48000, n_fft=400, hop_length=160, n_mels=128):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mx.array(log_mel, dtype=mx.float32).T.copy(), log_mel.shape[1]