# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from typing_extensions import Self

from ..._deps import (
    _HAS_COREAI,
    _HAS_MLX,
    _MSG_COREAI_NOT_FOUND,
    _MSG_MLX_NOT_FOUND,
)
from ...common.types.dependency_types import (
    ExportedModel,
    SourceModel,
)
from ...common.types.export_types import (
    Backend,
    ExportConfig,
    Frontend,
)
from ..exporters import (
    TorchEagerExporter,
    TorchExportExporter,
)

if _HAS_MLX:
    from ..exporters import MlxExporter

if _HAS_COREAI:
    from ..exporters import CoreaiExporter


class ExportService:
    def __init__(self: Self, output_names: tuple[str] | None = None) -> None:
        self._output_names = output_names
        self._exported_models: dict[ExportConfig, ExportedModel] = {}

    def _export_to_torch_eager(
        self: Self,
        source_model: SourceModel,
        export_config: ExportConfig,
    ) -> None:
        exporter = TorchEagerExporter()
        exported_model = exporter.export(source_model)
        self._exported_models[export_config] = exported_model

    def _export_to_torch_export(
        self: Self,
        source_model: SourceModel,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None,
        export_config: ExportConfig,
    ) -> None:
        exporter = TorchExportExporter()
        exported_model = exporter.export(
            source_model,
            reference_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        self._exported_models[export_config] = exported_model

    def _export_to_mlx(
        self: Self,
        source_model: SourceModel,
        export_config: ExportConfig,
    ) -> None:
        if not _HAS_MLX:
            raise ModuleNotFoundError(_MSG_MLX_NOT_FOUND)
        exporter = MlxExporter()
        exported_model = exporter.export(source_model)
        self._exported_models[export_config] = exported_model

    def _export_to_coreai(
        self: Self,
        source_model: SourceModel,
        reference_inputs: dict[str, torch.Tensor],
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None,
        export_config: ExportConfig,
    ) -> None:
        if not _HAS_COREAI:
            raise ModuleNotFoundError(_MSG_COREAI_NOT_FOUND)
        if export_config.frontend != Frontend.torch_export:
            msg = (
                f"In principle Core AI supports {export_config.frontend}, "
                f"in practice only torch_export has been implemented for now"
            )
            raise NotImplementedError(msg)

        exporter = CoreaiExporter(
            output_names=self._output_names,
        )
        exported_model = exporter.export(
            source_model,
            reference_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        self._exported_models[export_config] = exported_model

    def export(
        self: Self,
        source_model: SourceModel,
        reference_inputs: dict[str, torch.Tensor] | None = None,
        dynamic_shapes: dict[str, dict[int, torch.export.Dim]] | None = None,
        export_config: ExportConfig = ExportConfig(),  # noqa: B008
    ) -> ExportedModel:
        if export_config in self._exported_models:
            # already exported, quick return
            return self._exported_models[export_config]

        match export_config.backend:
            case Backend.torch_eager:
                self._export_to_torch_eager(source_model, export_config)
            case Backend.torch_export:
                assert reference_inputs is not None
                self._export_to_torch_export(
                    source_model,
                    reference_inputs,
                    dynamic_shapes,
                    export_config,
                )
            case Backend.mlx:
                self._export_to_mlx(source_model, export_config)
            case Backend.coreai:
                assert reference_inputs is not None
                self._export_to_coreai(
                    source_model,
                    reference_inputs,
                    dynamic_shapes,
                    export_config,
                )
            case _:
                msg = f"Backend {export_config.backend} has no export service"
                raise NotImplementedError(msg)
        return self._exported_models[export_config]

    def __getitem__(self: Self, export_config: ExportConfig) -> ExportedModel:
        return self._exported_models[export_config]
