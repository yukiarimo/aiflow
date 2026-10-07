# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import coreai_torch
import coreai_torch.composite_ops
import torch
from typing_extensions import Self


class RMSNorm(coreai_torch.composite_ops.RMSNorm):
    """Apply root mean square normalization (RMSNorm) to input tensor."""

    def __init__(
        self: Self,
        dim: int,
        eps: float = 1e-5,
        n_heads: int | None = None,
    ) -> None:
        super().__init__(dim=dim, eps=eps, n_heads=n_heads)


class RMSNormPlusOne(RMSNorm):
    """
    RMSNorm variant where 1.0 is added to the scaling weight during the forward pass.
    Used by Gemma3.
    """

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm with +1.0 offset to the weight."""
        # .float() matches HuggingFace transformers numerics for parity testing
        weight_plus_one = self.weight.float() + 1.0
        return self.rmsnorm_impl(x, weight_plus_one)


class RMSNormGated(torch.nn.Module):
    """
    Gated RMSNorm variant that optionally applies SiLU gating after normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self._rmsnorm_impl = coreai_torch.composite_ops.RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """Apply RMSNorm, optionally with SiLU gating."""
        input_dtype = x.dtype
        x = self._rmsnorm_impl(x, self.weight)
        if gate is not None:
            x = x * torch.nn.functional.silu(gate.to(torch.float32))
            x = x.to(input_dtype)
        return x
