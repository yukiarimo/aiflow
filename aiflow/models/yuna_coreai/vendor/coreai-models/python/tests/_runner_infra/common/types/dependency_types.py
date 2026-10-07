# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

from ..._deps import _HAS_COREAI, _HAS_MLX

if _HAS_MLX:
    import mlx
    import mlx.core
    import mlx.nn

if _HAS_COREAI:
    from coreai.authoring import AIProgram
    from coreai.runtime import NDArray

from .source_types import Precision, Source  # noqa: E402

DType: TypeAlias = np.dtype | torch.dtype
if _HAS_MLX:
    DType |= mlx.core.Dtype


Tensor: TypeAlias = npt.NDArray[Any] | torch.Tensor
if _HAS_MLX:
    Tensor |= mlx.core.array
if _HAS_COREAI:
    Tensor |= NDArray


SourceModel: TypeAlias = torch.nn.Module
if _HAS_MLX:
    SourceModel |= mlx.nn.Module


ExportedModel: TypeAlias = torch.nn.Module | torch.export.ExportedProgram
if _HAS_COREAI:
    ExportedModel |= AIProgram


PRECISION_IN_SOURCE: dict[Source, dict[Precision, DType]] = {
    cast("Source", Source.torch): {
        cast("Precision", Precision.f32): torch.float32,
        cast("Precision", Precision.f16): torch.float16,
        cast("Precision", Precision.bf16): torch.bfloat16,
    },
}
if _HAS_MLX:
    PRECISION_IN_SOURCE[cast("Source", Source.mlx)] = {
        cast("Precision", Precision.f32): mlx.core.float32,
        cast("Precision", Precision.f16): mlx.core.float16,
        cast("Precision", Precision.bf16): mlx.core.bfloat16,
    }
