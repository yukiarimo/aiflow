import math
import torch
from torch.nn import functional as F

def gaussian_blur_1d(x, kernel_size=3, sigma=1.0):
    padding = kernel_size // 2

    # Create Gaussian Kernel
    kernel = torch.arange(-padding, padding + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.flip(0).unsqueeze(0).unsqueeze(0)

    # Pad the last dimension (Time dimension)
    x = F.pad(x, (padding, padding), mode='replicate')
    x = F.conv1d(x, kernel, padding=0)

    return x

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Conv" in classname: m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1): return (kernel_size * dilation - dilation) // 2
def convert_pad_shape(pad_shape): return [item for sublist in pad_shape[::-1] for item in sublist]

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5 + (0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q))
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protecting from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x): return rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)

def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        if x.size(2) == 0:
            print(f"Warning: tensor has zero time dimension at batch {i}")
            continue
        if idx_end > x.size(2):
            available_size = x.size(2) - idx_str
            if available_size > 0: ret[i, :, :available_size] = x[i, :, idx_str:x.size(2)]
        else: ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None: x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size), ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    return signal.view(1, channels, length)

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

def subsequent_mask(length): return torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act

def shift_1d(x): return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]

def sequence_mask(length, max_length=None):
    if max_length is None: max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None: clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None: p.grad.data.clamp_(min=-clip_value, max=clip_value)

    return total_norm ** (1.0 / norm_type)