# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from typing_extensions import Self, override

from ...common.types.dependency_types import Tensor
from ...common.utils.torch.graph import (
    extract_outputs_from_torch_exported_program,
)
from .runner import Runner


def _wrap_outputs_as_list(
    outputs: tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, torch.Tensor] | torch.Tensor,
) -> list[torch.Tensor]:
    if isinstance(outputs, (tuple, list)):
        outputs_list = list(outputs)
    elif isinstance(outputs, dict):
        outputs_list = list(outputs.values())
    else:
        assert isinstance(outputs, torch.Tensor)
        outputs_list = [outputs]
    return outputs_list


class TorchEagerRuntime:
    def __init__(
        self: Self,
        torch_module: torch.nn.Module,
        output_names: tuple[str] | None = None,
    ) -> None:
        self._torch_module = torch_module
        self._output_names = output_names

    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        outputs = self._torch_module(**named_inputs)

        outputs_list = _wrap_outputs_as_list(outputs)
        if self._output_names is None:
            outputs_dict = {f"output_{i}": output for i, output in enumerate(outputs_list)}
        else:
            outputs_dict = {
                output_name: output
                for output_name, output in zip(self._output_names, outputs_list, strict=True)
            }
        return outputs_dict


class TorchEagerRunner(Runner):
    def __init__(
        self: Self,
        torch_module: torch.nn.Module,
        output_names: tuple[str] | None = None,
    ) -> None:
        super().__init__()
        self._runtime = TorchEagerRuntime(torch_module, output_names)

    @override
    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        return self._runtime.forward(named_inputs)


class TorchExportRuntime:
    def __init__(
        self: Self,
        exported_program: torch.export.ExportedProgram,
        output_names: tuple[str] | None = None,
    ) -> None:
        self._exported_program = exported_program
        if output_names is None:
            output_names, _ = extract_outputs_from_torch_exported_program(exported_program)
        self._output_names = output_names

    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        outputs = self._exported_program.module()(**named_inputs)

        outputs_list = _wrap_outputs_as_list(outputs)
        return {
            output_name: output
            for output_name, output in zip(self._output_names, outputs_list, strict=True)
        }


class TorchExportRunner(Runner):
    def __init__(
        self: Self,
        exported_program: torch.export.ExportedProgram,
        output_names: tuple[str] | None = None,
    ) -> None:
        super().__init__()
        self._runtime = TorchExportRuntime(exported_program, output_names)

    @override
    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        return self._runtime.forward(named_inputs)
