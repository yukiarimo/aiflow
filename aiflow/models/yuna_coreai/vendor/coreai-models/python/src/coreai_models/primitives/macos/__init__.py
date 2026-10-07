# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""macOS primitives for Core AI model authoring."""

from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm, RMSNormGated, RMSNormPlusOne
from coreai_models.primitives.macos.rope import (
    RoPE,
    YarnRoPE,
    initialize_rope,
)
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwiGLU, SwitchGLU, SwitchLinear

__all__ = [
    "KVCache",
    "MLP",
    "RMSNorm",
    "RMSNormGated",
    "RMSNormPlusOne",
    "RoPE",
    "SDPA",
    "SSMState",
    "SwitchGLU",
    "SwitchLinear",
    "SwiGLU",
    "YarnRoPE",
    "initialize_rope",
]
