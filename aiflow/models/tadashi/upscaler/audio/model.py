import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d)):
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0)

class SEBlock(nn.Module):
    """Channel‐wise Squeeze‐and‐Excitation."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResidualDownBlock(nn.Module):
    """Residual block with stride‐2 downsampling + SE."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 2, dilation: int = 1, groups: int = 8):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad,
                               dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1,
                              stride=stride, bias=False)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.gn1(out); out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + identity
        out = self.act2(out)
        out = self.se(out)
        return out

class ResidualUpBlock(nn.Module):
    """Residual block with upsampling + SE."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 2, groups: int = 8):
        super().__init__()
        pad = kernel_size // 2
        self.upsample_factor = stride
        self.conv_pre = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  padding=pad, bias=False)
        self.gn_pre = nn.GroupNorm(groups, out_ch)
        self.act_pre = nn.LeakyReLU(0.2, inplace=True)
        self.conv_post = nn.Conv1d(out_ch * 2, out_ch, kernel_size,
                                   padding=pad, bias=False)
        self.gn_post = nn.GroupNorm(groups, out_ch)
        self.act_post = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='nearest')
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode='nearest')
        out = self.conv_pre(x)
        out = self.gn_pre(out); out = self.act_pre(out)
        out = torch.cat([out, skip], dim=1)
        out = self.conv_post(out)
        out = self.gn_post(out); out = self.act_post(out)
        out = self.se(out)
        return out

class AudioUpsampler(nn.Module):
    """
    Improved U‐Net style 1D upsampler with residual SE blocks.
    Handles very short transients and long contexts,
    reduces artifacts on high‐frequency anime voices.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 depth: int = 4,
                 upsample_factor: int = 4):
        super().__init__()
        self.depth = depth
        self.upsample_factor = upsample_factor

        # Initial conv + GN + act
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=True),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            self.down_blocks.append(ResidualDownBlock(
                in_ch=ch, out_ch=ch * 2,
                kernel_size=5, stride=2,
                dilation=1 if i < 2 else 2  # increase dilation deeper
            ))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(ch)
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(ResidualUpBlock(
                in_ch=ch, out_ch=ch // 2,
                kernel_size=5, stride=2
            ))
            ch //= 2

        # Final ConvTranspose + tanh
        final_k = upsample_factor * 2
        final_pad = upsample_factor // 2
        self.final_upsample = nn.ConvTranspose1d(
            ch, out_channels,
            kernel_size=final_k, stride=upsample_factor, padding=final_pad,
            bias=True
        )
        self.final_act = nn.Tanh()

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, L]
        skips: List[torch.Tensor] = []
        out = self.initial(x)

        # Downsample path
        for db in self.down_blocks:
            skips.append(out)
            out = db(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Upsample path
        for ub, skip in zip(self.up_blocks, reversed(skips)):
            out = ub(out, skip)

        # Final upsampling
        out = self.final_upsample(out)
        out = self.final_act(out)
        return out