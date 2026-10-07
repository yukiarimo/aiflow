# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SiLU gated activation.

    This module implements a gated feed-forward network commonly used in
    transformer architectures. It uses a gate projection with SiLU activation
    and an up projection, followed by a down projection.

    Args:
        dim: Input and output dimension
        hidden_dim: Hidden layer dimension
        bias: Whether to use bias in linear layers (default: False)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: we compute the up projection before the gate projection
        # in order to get better performance on macOS
        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.silu(self.gate_proj(x))
        return self.down_proj(up_tensor * gate_tensor)
