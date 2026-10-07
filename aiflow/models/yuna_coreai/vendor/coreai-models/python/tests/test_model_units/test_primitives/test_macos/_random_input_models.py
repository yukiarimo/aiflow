# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Random-input ``Model`` subclasses used by the parity tests.

These helpers slot above ``Model`` (from
``tests._runner_infra.models.model``) and provide a default
``reference_inputs`` implementation that draws random tensors from
``named_input_shapes`` (and optional integer ``named_index_input_specs``)
once for the f32 / Source.torch ground truth and casts them to other
precisions / sources for downstream comparisons.
"""

import functools
from abc import abstractmethod
from typing import cast

import torch
from typing_extensions import Self, final, override

from tests._runner_infra._deps import _HAS_MLX
from tests._runner_infra.common.types.dependency_types import (
    PRECISION_IN_SOURCE,
    Tensor,
)
from tests._runner_infra.common.types.source_types import (
    Precision,
    Source,
    SourceConfig,
)
from tests._runner_infra.models.model import Model

if _HAS_MLX:
    import mlx
    import mlx.core


class _BaseRandomInputModel(Model):
    """Shared cache + cross-source/precision conversion for random-input models.

    Subclasses provide the canonical torch / f32 tensors via
    ``_initial_inputs``; this base handles caching and casting to every
    other supported ``SourceConfig``. Integer tensors (e.g. index inputs)
    are detected via ``is_floating_point()`` and passed through untouched
    when converting between torch precisions.
    """

    @abstractmethod
    def _initial_inputs(self: Self) -> dict[str, Tensor]:
        """Return the canonical torch / f32 reference tensors."""
        ...

    @final
    @override
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if source_config == SourceConfig():
            assert source_config.source == Source.torch
            assert source_config.precision == Precision.f32
            return self._initial_inputs()

        match source_config.source:
            case Source.torch:
                torch_f32_source_config = SourceConfig(
                    source=cast("Source", Source.torch),
                    precision=cast("Precision", Precision.f32),
                )
                named_inputs_f32 = self.reference_inputs(torch_f32_source_config)
                dtype = PRECISION_IN_SOURCE[cast("Source", Source.torch)][source_config.precision]
                return {
                    name: tensor.to(dtype) if tensor.is_floating_point() else tensor
                    for name, tensor in named_inputs_f32.items()
                }
            case Source.mlx:
                torch_source_config = SourceConfig(
                    source=cast("Source", Source.torch),
                    precision=source_config.precision,
                )
                named_inputs_torch = self.reference_inputs(torch_source_config)
                return {
                    name: mlx.core.array(input_torch)
                    for name, input_torch in named_inputs_torch.items()
                }
            case _:
                msg = f"Source {source_config.source} has no reference inputs"
                raise NotImplementedError(msg)


class RandomInputModel(_BaseRandomInputModel):
    """``Model`` whose reference inputs are drawn from ``named_input_shapes``."""

    @property
    @abstractmethod
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]: ...

    @override
    def _initial_inputs(self: Self) -> dict[str, Tensor]:
        return {
            name: torch.rand(input_shape, dtype=torch.float32)
            for name, input_shape in self.named_input_shapes.items()
        }


class RandomInputWithIndicesModel(_BaseRandomInputModel):
    """``Model`` with both random float inputs and random integer index inputs.

    Float inputs are generated via ``torch.rand()`` and converted across
    precisions/sources. Index inputs are generated via ``torch.randint()``
    and stay as int32 (converted to mlx int32 for MLX source).
    """

    @property
    @abstractmethod
    def named_input_shapes(self: Self) -> dict[str, tuple[int, ...]]:
        """Shapes for float inputs (generated with ``torch.rand``)."""
        ...

    @property
    @abstractmethod
    def named_index_input_specs(
        self: Self,
    ) -> dict[str, tuple[tuple[int, ...], int]]:
        """Specs for integer index inputs: ``{name: (shape, high)}``.

        Values are sampled uniformly from ``[0, high)``.
        """
        ...

    @override
    def _initial_inputs(self: Self) -> dict[str, Tensor]:
        named_inputs: dict[str, Tensor] = {
            name: torch.rand(input_shape, dtype=torch.float32)
            for name, input_shape in self.named_input_shapes.items()
        }
        named_inputs.update(
            {
                name: torch.randint(0, high, shape, dtype=torch.int32)
                for name, (shape, high) in self.named_index_input_specs.items()
            }
        )
        return named_inputs
