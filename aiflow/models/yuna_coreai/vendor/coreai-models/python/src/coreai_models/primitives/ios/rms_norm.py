# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch
import torch.nn as nn


# iOS-friendly RMS norm requires that the reduction-mean occurs on dim 1
class RMSNorm(nn.Module):
    """
    Apply root mean square normalization (RMSNorm) to input tensor, with attributes pre-determined.

    The RMSNorm operation is defined as:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
    """

    def __init__(self, dim: int, eps: float = 1e-6, n_heads: int | None = None) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
            self._eps = nn.Buffer(torch.tensor(eps), persistent=False)
        self._use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform root mean square layer normalization on input.

        Args:
            input: Input tensor of shape (batch_size, seq_len, 1, dim)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, 1, dim)
        """
        if self._use_hf_impl:
            input_dtype = input.dtype
            input = input.to(torch.float32)

        square = input * input
        mean_square = square.mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self._eps)
        x_2normalized = input * inv_rms

        if self._use_hf_impl:
            x_2normalized = x_2normalized.to(input_dtype)

        return x_2normalized * self.weight
