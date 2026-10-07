# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import coreai_torch
import coreai_torch.composite_ops
from typing_extensions import Self


class SDPA(coreai_torch.composite_ops.SDPA):
    """Apply scaled dot product attention to input tensors, with attributes pre-determined."""

    def __init__(
        self: Self,
        scale: float | None = None,
        is_causal: bool = False,
        window_size: int = 0,
    ) -> None:
        _use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        super().__init__(
            scale=scale,
            is_causal=is_causal,
            window_size=window_size,
            _use_hf_impl=_use_hf_impl,
        )
