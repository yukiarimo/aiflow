# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

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
    Tensor,
)
from ...common.types.export_types import Backend
from ...common.types.run_types import RunConfig
from ..runners import (
    Runner,
    TorchEagerRunner,
    TorchExportRunner,
)

if _HAS_MLX:
    from ..runners import MlxRunner

if _HAS_COREAI:
    from ..runners import CoreaiRunner


class RunService:
    def __init__(
        self: Self,
        output_names: tuple[str] | None = None,
        root_path: Path = Path("./artifacts/"),
    ) -> None:
        self._output_names = output_names
        self._root_path = root_path
        self._runners: dict[RunConfig, Runner] = {}

    def _load_torch_eager_runner(
        self: Self,
        exported_model: ExportedModel,
        run_config: RunConfig,
    ) -> None:
        self._runners[run_config] = TorchEagerRunner(
            exported_model, output_names=self._output_names
        )

    def _load_torch_export_runner(
        self: Self,
        exported_model: ExportedModel,
        run_config: RunConfig,
    ) -> None:
        self._runners[run_config] = TorchExportRunner(
            exported_model, output_names=self._output_names
        )

    def _load_mlx_runner(
        self: Self,
        exported_model: ExportedModel,
        run_config: RunConfig,
    ) -> None:
        if not _HAS_MLX:
            raise ModuleNotFoundError(_MSG_MLX_NOT_FOUND)
        self._runners[run_config] = MlxRunner(exported_model, output_names=self._output_names)

    def _load_coreai_runner(
        self: Self,
        exported_model: ExportedModel,
        exported_model_path: Path,
        run_config: RunConfig,
    ) -> None:
        if not _HAS_COREAI:
            raise ModuleNotFoundError(_MSG_COREAI_NOT_FOUND)
        self._runners[run_config] = CoreaiRunner(
            exported_model,
            exported_model_path,
            output_names=self._output_names,
        )

    def load_runner(
        self: Self,
        exported_model: ExportedModel,
        run_config: RunConfig = RunConfig(),  # noqa: B008
    ) -> None:
        if run_config in self._runners:
            # already loaded, quick return
            return
        exported_model_path = run_config.exported_model_path(self._root_path)

        match run_config.backend:
            case Backend.torch_eager:
                self._load_torch_eager_runner(exported_model, run_config)
            case Backend.torch_export:
                self._load_torch_export_runner(exported_model, run_config)
            case Backend.mlx:
                self._load_mlx_runner(exported_model, run_config)
            case Backend.coreai:
                self._load_coreai_runner(exported_model, exported_model_path, run_config)
            case _:
                msg = f"Backend {run_config.backend} has no run service"
                raise NotImplementedError(msg)

    def forward(
        self: Self,
        named_inputs: dict[str, torch.Tensor],
        run_config: RunConfig = RunConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        if run_config not in self._runners:
            msg = f"Please call .load_runner for {run_config} before forward"
            raise ValueError(msg)
        runner = self._runners[run_config]
        return runner.forward(named_inputs)
