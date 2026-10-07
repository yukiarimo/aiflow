# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from typing_extensions import Self


class TorchEagerExporter:
    def export(
        self: Self,
        torch_module: torch.nn.Module,
    ) -> torch.nn.Module:
        torch_module.eval()
        torch_module.cpu()
        return torch_module


class TorchExportExporter:
    def export(
        self: Self,
        torch_module: torch.nn.Module,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
    ) -> torch.export.ExportedProgram:
        torch_module.eval()
        torch_module.cpu()
        return torch.export.export(
            torch_module,
            args=(),
            kwargs=reference_inputs,
            dynamic_shapes=dynamic_shapes,
        )
