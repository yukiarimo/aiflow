# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import mlx
import mlx.core
import mlx.nn
import torch
from typing_extensions import Self, override

from ...common.types.dependency_types import Tensor
from ...common.utils.mlx.tensor import (
    mlx_array_to_torch_tensor,
    torch_tensor_to_mlx_array,
)
from .runner import Runner


def _wrap_outputs_as_list(
    outputs: tuple[mlx.core.array, ...]
    | list[mlx.core.array]
    | dict[str, mlx.core.array]
    | mlx.core.array,
) -> list[mlx.core.array]:
    if isinstance(outputs, (tuple, list)):
        outputs_list = list(outputs)
    elif isinstance(outputs, dict):
        outputs_list = list(outputs.values())
    else:
        assert isinstance(outputs, mlx.core.array)
        outputs_list = [outputs]
    return outputs_list


class MlxRuntime:
    def __init__(
        self: Self,
        mlx_module: mlx.nn.Module,
        output_names: tuple[str] | None = None,
    ) -> None:
        self._mlx_module = mlx_module
        self._output_names = output_names

    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        def _to_mlx(value: Tensor) -> mlx.core.array:
            if isinstance(value, torch.Tensor):
                return torch_tensor_to_mlx_array(value)
            return mlx.core.array(value)

        mlx_inputs = {name: _to_mlx(input_) for name, input_ in named_inputs.items()}
        mlx_outputs = self._mlx_module(**mlx_inputs)

        mlx_outputs_list = _wrap_outputs_as_list(mlx_outputs)
        outputs_list = [mlx_array_to_torch_tensor(mlx_output) for mlx_output in mlx_outputs_list]
        if self._output_names is None:
            outputs_dict = {f"output_{i}": output for i, output in enumerate(outputs_list)}
        else:
            outputs_dict = {
                output_name: output
                for output_name, output in zip(self._output_names, outputs_list, strict=True)
            }
        return outputs_dict


class MlxRunner(Runner):
    def __init__(
        self: Self,
        mlx_module: mlx.nn.Module,
        output_names: tuple[str] | None = None,
    ) -> None:
        super().__init__()
        self._runtime = MlxRuntime(mlx_module, output_names)

    @override
    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        return self._runtime.forward(named_inputs)
