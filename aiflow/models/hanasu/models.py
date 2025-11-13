import math
import queue
import threading
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from . import commons
from . import monotonic_align
from .commons import get_padding, init_weights
from . import utils
from .text import symbols, split_and_process_text, cleaned_text_to_sequence
import numpy as np
from tqdm import tqdm
import soundfile as sf
import tempfile
import os
from .mel_processing import mel_spectrogram_torch
from .utils import load_wav_to_torch
from . import commons
from .transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1
AVAILABLE_FLOW_TYPES = ["pre_conv", "pre_conv2", "fft", "mono_layer_inter_residual", "mono_layer_post_residual"]

class Encoder(nn.Module):  # backward compatible vits2 encoder
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                self.cond_layer_idx = (kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2) # vits2 says 3rd block, so idx is 2 by default
                assert self.cond_layer_idx < self.n_layers, "cond_layer_idx should be less than n_layers"

        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, proximal_bias=False, proximal_init=True, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """x: decoder input, h: encoder output"""
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.inference_mode():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = (torch.ones_like(scores).triu(-self.block_length).tril(self.block_length))
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = (output.transpose(2, 3).contiguous().view(b, d, t_t))  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]"""
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """x: [b, h, l, d], y: [h or 1, m, d], ret: [b, h, l, m]"""
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0: padded_relative_embeddings = F.pad(relative_embeddings, commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else: padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """x: [b, h, l, 2*l-1], ret: [b, h, l, l]"""
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """x: [b, h, l, l], ret: [b, h, l, 2*l-1]"""
        batch, heads, length, _ = x.size()

        # padd along column
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])

        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions. Args: length: an integer scalar. Returns: a Tensor with shape [1, 1, length, length]"""
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal: self.padding = self._causal_padding
        else: self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu": x = x * torch.sigmoid(1.702 * x)
        else: x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1: return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1: return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        self.depth_conv = remove_weight_norm(self.depth_conv, name="weight")
        self.point_conv = remove_weight_norm(self.point_conv, name="weight")

class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode="zeros", device=None, dtype=None):
        super().__init__()
        self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, output_padding=output_padding, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        remove_weight_norm(self.depth_conv, name="weight")
        remove_weight_norm(self.point_conv, name="weight")

def weight_norm_modules(module, name="weight", dim=0):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D):
        module.weight_norm()
        return module
    else: return weight_norm(module, name, dim)

def remove_weight_norm_modules(module, name="weight"):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D): module.remove_weight_norm()
    else: remove_weight_norm(module, name)

class FFT(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.0, proximal_bias=False, proximal_init=True, isflow=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow and "gin_channels" in kwargs and kwargs["gin_channels"] > 0:
            cond_layer = torch.nn.Conv1d(kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1)
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
            self.gin_channels = kwargs["gin_channels"]
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """x: decoder input, h: encoder output"""
        if g is not None: g = self.cond_layer(g)

        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(x, g_l, torch.IntTensor([self.hidden_channels]))
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

class DDSConv(nn.Module):
    """Dialted and Depth-Separable Convolution"""
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None: x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask

class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
        else: self.cond_layer = None

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1: res_skip_channels = 2 * hidden_channels
            else: res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None and self.cond_layer is not None: g = self.cond_layer(g)
        elif g is not None and self.cond_layer is None: g = None

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else: g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else: output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        for l in self.in_layers: remove_weight_norm(l)
        for l in self.res_skip_layers: remove_weight_norm(l)
        if self.cond_layer is not None: remove_weight_norm(self.cond_layer)

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None: x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)

class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None: x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)

class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else: return torch.exp(x) * x_mask

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else: return x

class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else: return (x - self.m) * torch.exp(-self.logs) * x_mask

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

class ConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=reverse, tails="linear", tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse: return x, logdet
        else: return x

class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask)
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = (torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q)

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot)
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale)
            for flow in flows: z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class DurationDiscriminatorV1(nn.Module):  # vits2
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_2 = LayerNorm(filter_channels)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = self.pre_out_conv_2(x * x_mask)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = self.conv_2(x * x_mask)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs

class DurationDiscriminatorV2(nn.Module):  # vits2
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_1 = LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_2 = LayerNorm(filter_channels)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append([output_prob])

        return output_probs

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=self.gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class ResidualCouplingTransformersLayer2(nn.Module):  # vits2
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.pre_transformer = Encoder(hidden_channels, hidden_channels, n_heads=2, n_layers=1, kernel_size=kernel_size, p_dropout=p_dropout)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = h + self.pre_transformer(h * x_mask, x_mask)  # vits2 residual connection
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

class ResidualCouplingTransformersLayer(nn.Module):  # vits2
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre_transformer = Encoder(self.half_channels, self.half_channels, n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, window_size=None) # vits2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post_transformer = Encoder(self.hidden_channels, self.hidden_channels, n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, window_size=None) # vits2

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
        x0_ = x0_ + x0  # vits2 residual connection
        h = self.pre(x0_) * x_mask  # changed from x0 to x0_ to retain x0 for the flow
        h = self.enc(h, x_mask, g=g)

        stats = self.post(h) * x_mask
        if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

class FFTransformerCouplingLayer(nn.Module):  # vits2
    def __init__(self, channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout=0, filter_channels=768, mean_only=False, gin_channels=0):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow=True, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h_ = self.enc(h, x_mask, g=g)
        h = h_ + h
        stats = self.post(h) * x_mask
        if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

class MonoTransformerFlowLayer(nn.Module):  # vits2
    def __init__(self, channels, hidden_channels, mean_only=False, residual_connection=False):
        # according to VITS-2 paper fig 1B set residual_connection=True
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.residual_connection = residual_connection
        self.pre_transformer = Encoder(self.half_channels, self.half_channels, n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, window_size=None) # vits2

        self.post = nn.Conv1d(self.half_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        if self.residual_connection:
            if not reverse:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1 = m + x1 * torch.exp(logs) * x_mask
                x_ = torch.cat([x0, x1], 1)
                x = x + x_
                logdet = torch.sum(torch.log(torch.exp(logs) + 1), [1, 2])
                logdet = logdet + torch.log(torch.tensor(2)) * (x0.shape[1] * x0.shape[2])
                return x, logdet
            else:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0 = x0 / 2
                x0_ = x0 * x_mask
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1_ = ((x1 - m) / (1 + torch.exp(-logs))) * x_mask
                x = torch.cat([x0, x1_], 1)
                return x
        else:
            x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
            h = x0_ + x0  # vits2
            stats = self.post(h) * x_mask
            if not self.mean_only: m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)
            if not reverse:
                x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], 1)
                logdet = torch.sum(logs, [1, 2])
                return x, logdet
            else:
                x1 = (x1 - m) * torch.exp(-logs) * x_mask
                x = torch.cat([x0, x1], 1)
                return x

class ResidualCouplingTransformersBlock(nn.Module):  # vits2
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0, use_transformer_flows=False, transformer_flow_type="pre_conv"):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        if use_transformer_flows:
            if transformer_flow_type == "pre_conv":
                for i in range(n_flows):
                    self.flows.append(ResidualCouplingTransformersLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                    self.flows.append(Flip())
            elif transformer_flow_type == "pre_conv2":
                for i in range(n_flows):
                    self.flows.append(ResidualCouplingTransformersLayer2(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                    self.flows.append(Flip())
            elif transformer_flow_type == "fft":
                for i in range(n_flows):
                    self.flows.append(FFTransformerCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                    self.flows.append(Flip())
            elif transformer_flow_type == "mono_layer_inter_residual":
                for i in range(n_flows):
                    self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                    self.flows.append(Flip())
                    self.flows.append(MonoTransformerFlowLayer(channels, hidden_channels, mean_only=True))
            elif transformer_flow_type == "mono_layer_post_residual":
                for i in range(n_flows):
                    self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                    self.flows.append(Flip())
                    self.flows.append(MonoTransformerFlowLayer(channels, hidden_channels, mean_only=True, residual_connection=True))
        else:
            for i in range(n_flows):
                self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
                self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows: x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows): x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows: x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows): x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)): self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0: self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None: x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i * self.num_kernels + j](x)
                else: xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS()]
        discs = discs + [DiscriminatorP(i) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

from .text import sequence_to_text

def build_duration_multipliers_from_ids(ids, device,
                                        comma=1.5, period=12.0, question=2.8, exclam=3.0,
                                        dash=2.0, ellipsis=3.0,
                                        stress_primary=1.3, stress_secondary=1.2,
                                        clamp_min=0.6, clamp_max=20.5):  # Updated clamp_max to accommodate period=12.0
    """
    ids: LongTensor [1, T] of token ids
    returns: [1, 1, T] multipliers to apply to w (durations) per token
    """
    seq = sequence_to_text(ids[0].tolist())
    T = len(seq)
    scales = torch.ones(1, 1, T, device=device)
    punct_map = {
        ',': comma, ';': comma, ':': comma,
        '.': period, '?': question, '!': exclam,
        '—': dash, '–': dash, '…': ellipsis
    }
    # Boost punctuation durations
    for i, ch in enumerate(seq):
        if ch in punct_map:
            scales[0, 0, i] = punct_map[ch]
    # Apply stress marks to next non-punct token
    def is_skip(c):
        return (c in punct_map) or (c in ['ˈ', 'ˌ', ' '])
    i = 0
    while i < T:
        if seq[i] in ['ˈ', 'ˌ']:
            j = i + 1
            while j < T and is_skip(seq[j]):
                j += 1
            if j < T:
                factor = stress_primary if seq[i] == 'ˈ' else stress_secondary
                scales[0, 0, j] = scales[0, 0, j] * factor
        i += 1
    return torch.clamp(scales, clamp_min, clamp_max)

def build_noise_profile_from_ids(ids, device, base=1.0, around_punct=1.25, window=3,  # Increased noise and window for broader emotional range
                                 clamp_min=0.75, clamp_max=1.5):  # Wider clamp for more variation
    """
    returns: [1, 1, T] multiplicative profile for noise at each token (for gentle intonation)
    """
    seq = sequence_to_text(ids[0].tolist())
    T = len(seq)
    prof = torch.ones(1, 1, T, device=device) * base
    puncts = set([',', '.', ';', ':', '!', '?', '—', '–', '…'])
    for i, ch in enumerate(seq):
        if ch in puncts:
            for j in range(max(0, i - window), min(T, i + window + 1)):
                if j != i:
                    prof[0, 0, j] *= around_punct
    return torch.clamp(prof, clamp_min, clamp_max)

class SynthesizerTrn(nn.Module):
    """Synthesizer for Training"""
    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, use_sdp=True, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", False)
        self.use_transformer_flows = kwargs.get("use_transformer_flows", False)
        self.transformer_flow_type = kwargs.get("transformer_flow_type", "mono_layer_post_residual")
        if self.use_transformer_flows: assert self.transformer_flow_type in AVAILABLE_FLOW_TYPES, f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        self.use_sdp = use_sdp
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        self.enc_gin_channels = gin_channels if self.use_spk_conditioned_encoder and gin_channels > 0 else 0
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=self.enc_gin_channels)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingTransformersBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels, use_transformer_flows=self.use_transformer_flows, transformer_flow_type=self.transformer_flow_type)

        if use_sdp: self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else: self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1: self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None  # [b, h, 1]

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.inference_mode():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            epsilon = (torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale)
            neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach())

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
            logw_ = torch.log(w + 1e-6) * x_mask
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return (o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (x, logw, logw_))

    def infer(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1, noise_scale_w=0.8, temperature=1.0, duration_blur_sigma=0.0, duration_multipliers=None, noise_profile=None, **kwargs):
        with torch.autocast(device_type=x.device.type):
            sid = torch.LongTensor([sid]).to(x.device) if sid is not None else None
            g = self.emb_g(sid).unsqueeze(-1) if sid is not None else None  # [b, h, 1]
            
            # Store original token IDs before encoding
            x_tokens = x
            
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)

            if duration_blur_sigma > 0.0: logw = commons.gaussian_blur_1d(logw * x_mask, kernel_size=5, sigma=duration_blur_sigma) * x_mask

            if temperature != 1.0: logw = logw / temperature
            w = torch.exp(logw) * x_mask * length_scale

            if duration_multipliers is not None:
                w = w * duration_multipliers.to(w.dtype)
                
                # Apply absolute override for periods (set to exactly 20 frames)
                from .text import sequence_to_text
                # Use the original token IDs, not the encoded features
                x_list = x_tokens[0].tolist()
                
                seq = sequence_to_text(x_list)
                for i, ch in enumerate(seq):
                    if ch == '.':
                        w[0, 0, i] = 12.0

            w = torch.clamp(w, min=0.1)
            w_ceil = torch.ceil(w)

            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            if duration_multipliers is not None:
                duration_multipliers = torch.matmul(attn.squeeze(1), duration_multipliers.transpose(1, 2)).transpose(1, 2)
            if noise_profile is not None:
                noise_profile = torch.matmul(attn.squeeze(1), noise_profile.transpose(1, 2)).transpose(1, 2)

            if temperature != 1.0: logs_p = logs_p * (1.0 / temperature)

            noise = torch.randn_like(m_p)
            noise_scalar = torch.tensor(noise_scale, dtype=m_p.dtype, device=m_p.device)
            factor = noise_scalar
            if noise_profile is not None:
                factor = factor * noise_profile.to(m_p.dtype)
            z_p = m_p + noise * torch.exp(logs_p) * factor

            # g is a speaker embedding
            # swap speaker by conditioning the flow with different g
            # g = self.emb_g(torch.LongTensor([2]).to(x.device)).unsqueeze(-1) # this line forces g to be speaker 3's embedding
            
            # to mix speakers, uncomment the following lines and adjust the alpha value
            # g_src = self.emb_g(torch.LongTensor([0]).to(x.device)).unsqueeze(-1)
            # g_tgt = self.emb_g(torch.LongTensor([1]).to(x.device)).unsqueeze(-1)
            # alpha = 0.5  # adjust this value between 0.0 and 1.0
            # g = (1 - alpha) * g_src + alpha * g_tgt

            z = self.flow(z_p, y_mask, g=g, reverse=True)
            o = self.dec(z, g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p), w_ceil

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, _, _, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
    
    def infer_mel_from_flow(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1, noise_scale_w=0.8, temperature=1.0, duration_blur_sigma=0.0, duration_multipliers=None, noise_profile=None, **kwargs):
        """
        Runs inference and returns the latent representation 'z' from the flow network,
        instead of the final audio waveform. This is the model's internal mel-spectrogram.
        """
        with torch.autocast(device_type=x.device.type):
            sid = torch.LongTensor([sid]).to(x.device) if sid is not None else None
            g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None  # [b, h, 1]

            x_tokens = x
            
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)

            if duration_blur_sigma > 0.0: logw = commons.gaussian_blur_1d(logw * x_mask, kernel_size=5, sigma=duration_blur_sigma) * x_mask

            if temperature != 1.0: logw = logw / temperature
            w = torch.exp(logw) * x_mask * length_scale

            if duration_multipliers is not None:
                w = w * duration_multipliers.to(w.dtype)
                
                from .text import sequence_to_text
                x_list = x_tokens[0].tolist()
                
                seq = sequence_to_text(x_list)
                for i, ch in enumerate(seq):
                    if ch == '.':
                        w[0, 0, i] = 12.0

            w = torch.clamp(w, min=0.1)
            w_ceil = torch.ceil(w)

            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            if duration_multipliers is not None:
                duration_multipliers = torch.matmul(attn.squeeze(1), duration_multipliers.transpose(1, 2)).transpose(1, 2)
            if noise_profile is not None:
                noise_profile = torch.matmul(attn.squeeze(1), noise_profile.transpose(1, 2)).transpose(1, 2)

            if temperature != 1.0: logs_p = logs_p * (1.0 / temperature)

            noise = torch.randn_like(m_p)
            noise_scalar = torch.tensor(noise_scale, dtype=m_p.dtype, device=m_p.device)
            factor = noise_scalar
            if noise_profile is not None:
                factor = factor * noise_profile.to(m_p.dtype)
            z_p = m_p + noise * torch.exp(logs_p) * factor
            
            z = self.flow(z_p, y_mask, g=g, reverse=True)
            z_masked = z * y_mask

            # Instead of decoding to audio, we return the latent 'z'
            return z_masked, attn, y_mask, (z, z_p, m_p, logs_p), w_ceil

def load_model(config_path, model_path, device="mps"):
    """Load the model from the specified path."""
    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(len(symbols), 128, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    net_g.hps = hps

    return net_g

def inference(model=None, text=None, sid=0, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0, device="mps", stream=False, output_file=None, language="en-us", duration_blur_sigma=0.0, temperature=1.0):
    processed_chunks = split_and_process_text(text, language=language, max_length=300, combine=True)
    print(f"Split into {len(processed_chunks)} chunks")

    if not stream:
        if output_file:
            temp_files = []
            for i, chunk in enumerate(tqdm(processed_chunks, desc="Generating audio", unit="chunk")):
                stn_tst = cleaned_text_to_sequence(chunk)
                phoneme_text = sequence_to_text(stn_tst)  # Get phoneme representation
                
                with torch.inference_mode():
                    x_tst = torch.LongTensor(stn_tst).to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([len(stn_tst)]).to(device)
                    dur_mult = build_duration_multipliers_from_ids(x_tst, device)
                    noise_prof = build_noise_profile_from_ids(x_tst, device)
                    result = model.infer(
                        x_tst, x_tst_lengths,
                        sid=sid if sid is not None else None,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=length_scale,
                        duration_blur_sigma=duration_blur_sigma,
                        temperature=temperature,
                        duration_multipliers=dur_mult,
                        noise_profile=noise_prof
                    )
                    audio_chunk = result[0][0, 0].data.cpu().float().numpy()
                    durations = result[4][0, 0].data.cpu().numpy()  # Get durations
                    
                    # Print phoneme durations
                    """print(f"\n=== Chunk {i+1} Phoneme Durations ===")
                    for phoneme, duration in zip(phoneme_text, durations):
                        print(f"{phoneme}: {duration:.2f} frames")
                    print("=" * 40 + "\n")"""
                    
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_files.append(temp_file.name)
                    sf.write(temp_file.name, audio_chunk, 48000)
                    temp_file.close()

            print("Concatenating audio chunks...")
            all_audio = []
            for temp_file in temp_files:
                chunk_audio, _ = sf.read(temp_file)
                all_audio.append(chunk_audio)

            final_audio = np.concatenate(all_audio)
            sf.write(output_file, final_audio, 48000)

            for temp_file in temp_files: os.unlink(temp_file)

            print(f"Audio saved to {output_file}")
            return final_audio

        else:
            all_audio = []
            for i, chunk in enumerate(tqdm(processed_chunks, desc="Generating audio", unit="chunk")):
                stn_tst = cleaned_text_to_sequence(chunk)
                with torch.inference_mode():
                    x_tst = torch.LongTensor(stn_tst).to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([len(stn_tst)]).to(device)
                    dur_mult = build_duration_multipliers_from_ids(x_tst, device)
                    noise_prof = build_noise_profile_from_ids(x_tst, device)
                    audio_chunk = (model.infer(
                        x_tst, x_tst_lengths,
                        sid=sid if sid is not None else None,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=length_scale,
                        duration_blur_sigma=duration_blur_sigma,
                        temperature=temperature,
                        duration_multipliers=dur_mult,
                        noise_profile=noise_prof
                    )[0][0, 0].data.cpu().float().numpy())
                    all_audio.append(audio_chunk)
            audio = np.concatenate(all_audio)
            return audio
    else:
        def audio_generator():
            audio_queue = queue.Queue(maxsize=2)
            pbar = tqdm(total=len(processed_chunks), desc="Generating audio", unit="chunk")

            def generate_audio_worker():
                for i, chunk in enumerate(processed_chunks):
                    stn_tst = cleaned_text_to_sequence(chunk)
                    with torch.inference_mode():
                        x_tst = torch.LongTensor(stn_tst).to(device).unsqueeze(0)
                        x_tst_lengths = torch.LongTensor([len(stn_tst)]).to(device)
                        dur_mult = build_duration_multipliers_from_ids(x_tst, device)
                        noise_prof = build_noise_profile_from_ids(x_tst, device)
                        audio_chunk = (model.infer(
                            x_tst, x_tst_lengths,
                            sid=sid if sid is not None else None,
                            noise_scale=noise_scale,
                            noise_scale_w=noise_scale_w,
                            length_scale=length_scale,
                            duration_blur_sigma=duration_blur_sigma,
                            temperature=temperature,
                            duration_multipliers=dur_mult,
                            noise_profile=noise_prof
                        )[0][0, 0].data.cpu().float().numpy())
                        audio_queue.put(audio_chunk)
                        pbar.update(1)

            worker_thread = threading.Thread(target=generate_audio_worker, daemon=True)
            worker_thread.start()

            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=30)
                    if audio_chunk is None: break
                    yield audio_chunk
                except queue.Empty: break

            worker_thread.join(timeout=5)

        return audio_generator()

def voice_conversion_inference(model=None, source_wav_path=None, source_speaker_id=0, target_speaker_id=1, device="mps", hps=None):
    hps = model.hps.data if hasattr(model, 'hps') else hps
    max_wav_value = hps.data.max_wav_value
    filter_length = hps.data.filter_length
    hop_length = hps.data.hop_length
    win_length = hps.data.win_length
    n_mel_channels = hps.data.n_mel_channels
    sampling_rate = hps.data.sampling_rate
    mel_fmin = hps.data.mel_fmin
    mel_fmax = hps.data.mel_fmax

    audio, sr = load_wav_to_torch(source_wav_path)
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    mel = mel_spectrogram_torch(audio_norm, filter_length, n_mel_channels, sampling_rate, hop_length, win_length, mel_fmin, mel_fmax, center=False)

    with torch.inference_mode():
        y = mel.to(device)
        y_lengths = torch.LongTensor([y.shape[2]]).to(device)
        sid_src = torch.LongTensor([source_speaker_id]).to(device)
        sid_tgt = torch.LongTensor([target_speaker_id]).to(device)
        print(f"Performing voice conversion from speaker {source_speaker_id} to {target_speaker_id}...")
        audio_out, _, _ = model.voice_conversion(y, y_lengths, sid_src=sid_src, sid_tgt=sid_tgt)
        print("Voice conversion complete.")

    return audio_out[0, 0].data.cpu().float().numpy()

def convert_to_single_speaker(ckpt_paths, new_emb_size=512, output_suffix="_single"):
    """Convert multi-speaker model checkpoints to single-speaker versions.
    Args: ckpt_paths: Path to checkpoint file or list of checkpoint paths, new_emb_size: Size for new speaker embedding (e.g., 512). If None, no new embedding is added., output_suffix: Suffix to add to output filenames (default: "_single")
    Returns: List of output file paths
    """

    def remove_speaker_embeddings(model_dict):
        """Remove speaker embedding keys from model dictionary."""
        possible_keys = ["emb_g.weight", "speaker_embedding.weight", "speaker_emb.weight", "spk_embed.weight"]
        removed = []
        for key in possible_keys:
            if key in model_dict:
                del model_dict[key]
                removed.append(key)
        return removed

    def add_new_embedding(model_dict, key, size):
        """Add new speaker embedding to model dictionary."""
        model_dict[key] = torch.randn(size, dtype=torch.float32)
        print(f"Added new embedding: {key} with shape {model_dict[key].shape}")

    if isinstance(ckpt_paths, str): ckpt_paths = [ckpt_paths] # Handle single path or list of paths

    output_paths = []

    for ckpt_path in ckpt_paths:
        print(f"Processing {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "model" in checkpoint:
            model_dict = checkpoint["model"]
            removed = remove_speaker_embeddings(model_dict)
            print(f"Removed keys: {removed}")

            if new_emb_size is not None and "emb_g.weight" in removed:
                add_new_embedding(model_dict, "emb_g.weight", (new_emb_size,))

            checkpoint["model"] = model_dict
            out_path = ckpt_path.replace(".pth", f"{output_suffix}.pth")
            torch.save(checkpoint, out_path)
            output_paths.append(out_path)
            print(f"Saved single-speaker checkpoint to {out_path}\n")
        else: print("No 'model' key found in checkpoint, skipping.\n")

    return output_paths