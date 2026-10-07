# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn


class SDPA(nn.Module):
    """iOS-optimized Scaled Dot-Product Attention.

    Unlike PyTorch's fused SDPA, iOS requires each attention head to be computed
    individually to meet hardware constraints and ensure efficient compilation.
    This implementation processes heads sequentially rather than in parallel.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        scale: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        with torch.device("cpu"):
            if scale is None:
                self._scale_factor = nn.Buffer(torch.tensor(head_dim**-0.5), persistent=False)
            else:
                self._scale_factor = (
                    nn.Buffer(scale, persistent=False)
                    if isinstance(scale, torch.Tensor)
                    else nn.Buffer(torch.tensor(scale), persistent=False)
                )

    # Efficient implementation equivalent to the following:
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention for iOS.

        Args:
            query: Query tensor with shape (batch_size, n_heads*head_dim, 1, seq_len)
            key: Key tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            value: Value tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            causal_mask: Causal attention mask with shape (1, max_seq_len, 1, seq_len)

        Returns:
            torch.Tensor: Attention output with shape (batch_size, n_heads*head_dim, 1, seq_len)
        """

        # Apply the scale factor before QK^T for numerical stability
        key = key.transpose(-3, -1) * self._scale_factor
        queries = query.split(self.head_dim, dim=1)
        keys = list(key.split(self.head_dim, dim=-1))

        n_heads = len(queries)

        # permute key heads in advance
        for kv_idx in range(len(keys)):
            keys[kv_idx] = keys[kv_idx].permute(0, 2, 3, 1)

        kv_group_size = len(queries) // len(keys)

        scores = []

        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            q = queries[head_idx].permute(0, 2, 3, 1)
            k = keys[kv_idx]
            attn_score = q @ k
            attn_score = attn_score.permute(0, 3, 1, 2)
            scores.append(attn_score)

        full_scores = torch.cat(scores, dim=2)
        masked_scores = full_scores + torch.cat([causal_mask] * n_heads, dim=2)
        full_scores = masked_scores.softmax(1)

        scores = full_scores.split(1, dim=2)

        values = list(value.split(self.head_dim, dim=1))

        # transpose values in advance
        for kv_idx in range(len(values)):
            values[kv_idx] = values[kv_idx].permute(0, 2, 3, 1).squeeze(1)

        weights = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            s = scores[head_idx].permute(0, 2, 3, 1).squeeze(1)
            v = values[kv_idx]
            weight = (s @ v).unsqueeze(1)
            weight = weight.permute(0, 3, 1, 2)
            weights.append(weight)

        final_score = torch.cat(weights, dim=1)
        return final_score
