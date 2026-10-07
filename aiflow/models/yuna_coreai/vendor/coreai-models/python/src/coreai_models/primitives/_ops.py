# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared custom ops for Core AI model primitives."""

import torch
from torch import Tensor


@torch.library.custom_op("coreai::mutable_slice_update", mutates_args=["x"])
def mutable_slice_update(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    """
    Mutable slice update operation for cache updates.

    Updates a slice of tensor x with the update tensor using dynamic begin/end indices.
    Begin and end indices are passed as tensors for custom op compatibility.

    Args:
        x: The tensor to update
        update: The update values to insert
        begin: Tensor containing start indices for each dimension
        end: Tensor containing end indices for each dimension

    Returns:
        The updated tensor (clone for torch compatibility)
    """
    # Begin and end indices passed in as tensors for custom op compatibility -> split for slicing
    begin = torch.split(begin, 1, dim=0)  # type: ignore
    end = torch.split(end, 1, dim=0)  # type: ignore
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begin, end, strict=False))
    x[slices] = update
    # Note: Not actually in-place for torch
    return x.clone()


@mutable_slice_update.register_fake
def mutable_slice_update_meta(  # type: ignore
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
):
    """Fake implementation for tracing/meta operations."""
    return torch.empty(x.shape, dtype=x.dtype)
