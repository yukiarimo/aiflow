# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    iOS-optimized Multi-Layer Perceptron with SiLU gated activation.

    This module implements a gated feed-forward network optimized for iOS.
    It uses Conv2d layers instead of Linear layers for better iOS performance.

    Args:
        dim: Input and output dimension
        hidden_dim: Intermediate (hidden/up projection) dimension
        bias: Whether to use bias in conv layers (default: False)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _, dim = x.shape

        # Conv2d expects NCHW format. Fuse batch and seq_len dimensions.
        x = x.reshape(batch_size * query_len, dim, 1, 1)

        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.silu(self.gate_proj(x))
        down_tensor = self.down_proj(up_tensor * gate_tensor)

        return down_tensor.reshape(batch_size, query_len, 1, dim)
