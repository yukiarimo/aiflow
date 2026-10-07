# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import functools
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from typing_extensions import Self, final

from ..common.types.dependency_types import (
    ExportedModel,
    SourceModel,
    Tensor,
)
from ..common.types.export_types import (
    BACKEND_PREFERRED_FRONTEND,
    SOURCE_PREFERRED_BACKEND,
    ExportConfig,
)
from ..common.types.run_types import RunConfig
from ..common.types.source_types import SourceConfig
from ..common.utils.test import assert_allclose, validate_snr_and_psnr
from ..export.service import ExportService
from ..run.service import RunService


class Model(ABC):
    """
    Model is a very overloaded term, here we care mainly about 3 meaning
    1. the abstract model
    2. the source model, i.e. the instance of the abstract model,
       concretely "write the abstract model in PyTorch / MLX / ..."
    3. the exported model, i.e. the source model after export

    The export process is
        source model -> frontend IR -> backend IR (exported model)
    Controlled by export config, manipulations can be injected, e.g.
    * PyTorch source model can have source transform / compression
    * torch.export.ExportedProgram (torch.export IR) can have quantization
    * coreai.authoring.AIProgram (Core AI IR) can have compression
    * ...
    """

    @final
    @property
    def model_name(self: Self) -> str:
        return self._model_name

    def __init__(self: Self, root_path: Path = Path("./artifacts/")) -> None:
        super().__init__()
        self._root_path = root_path

    # services (lazily initialized) -------------------------------------------

    @property
    def export_service(self: Self) -> ExportService:
        if not hasattr(self, "_export_service"):
            self._export_service = ExportService(self.output_names)
        return self._export_service

    @property
    def run_service(self: Self) -> RunService:
        if not hasattr(self, "_run_service"):
            model_root_path = self._root_path / self.model_name
            self._run_service = RunService(self.output_names, root_path=model_root_path)
        return self._run_service

    # source model definition -------------------------------------------------

    @abstractmethod
    @functools.cache  # noqa: B019
    def source_model(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> SourceModel: ...

    @abstractmethod
    @functools.cache  # noqa: B019
    def reference_inputs(
        self: Self,
        source_config: SourceConfig = SourceConfig(),  # noqa: B008
    ) -> dict[str, Tensor]: ...

    @property
    def dynamic_shapes(
        self: Self,
    ) -> dict[str, dict[int, torch.export.Dim]]:
        msg = f"Model {self.model_name} does not support dynamic shape yet"
        raise NotImplementedError(msg)

    @property
    def output_names(self: Self) -> tuple[str] | None:
        return None

    # export and run ----------------------------------------------------------

    @final
    def export(
        self: Self,
        export_config: ExportConfig = ExportConfig(),  # noqa: B008
    ) -> ExportedModel:
        source_model = self.source_model(export_config)
        inputs = self.reference_inputs(export_config)
        dynamic_shapes = None
        if export_config.dynamic:
            dynamic_shapes = self.dynamic_shapes
        return self.export_service.export(
            source_model,
            inputs,
            dynamic_shapes,
            export_config,
        )

    @final
    def load_runner(
        self: Self,
        run_config: RunConfig = RunConfig(),  # noqa: B008
    ) -> None:
        export_config = run_config.get_underlying_export_config()
        exported_model = self.export_service[export_config]
        self.run_service.load_runner(exported_model, run_config)

    @final
    def forward(
        self: Self,
        named_inputs: dict[str, torch.Tensor],
        run_config: RunConfig = RunConfig(),  # noqa: B008
    ) -> dict[str, Tensor]:
        return self.run_service.forward(named_inputs, run_config)

    # testing -----------------------------------------------------------------

    @functools.cache  # noqa: B019
    def generate_reference_io(
        self: Self,
        run_config: RunConfig = RunConfig(),  # noqa: B008
    ) -> tuple[dict[str, torch.Tensor], dict[str, Tensor]]:
        export_config = run_config.get_underlying_export_config()
        source_config = export_config.get_underlying_source_config()
        self.export(export_config)
        self.load_runner(run_config)
        reference_inputs = self.reference_inputs(source_config)
        outputs = self.forward(reference_inputs, run_config=run_config)
        return reference_inputs, outputs

    def validate(
        self: Self,
        run_config: RunConfig = RunConfig(),  # noqa: B008
        reference_run_config: RunConfig | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        snr_threshold: float = 15.0,
        psnr_threshold: float = 29.5,
    ) -> None:
        """Validate outputs between run config and reference run config."""
        if reference_run_config is None:
            reference_backend = SOURCE_PREFERRED_BACKEND[run_config.source]
            reference_frontend = BACKEND_PREFERRED_FRONTEND[reference_backend]
            reference_run_config = RunConfig(
                author=run_config.author,
                source=run_config.source,
                # default to use f32 precision as golden standard
                frontend=reference_frontend,
                backend=reference_backend,
                dynamic=run_config.dynamic,
            )

        _, outputs = self.generate_reference_io(run_config=run_config)
        _, reference_outputs = self.generate_reference_io(run_config=reference_run_config)
        assert_allclose(
            actual=outputs,
            desired=reference_outputs,
            actual_backend_name=run_config.backend,
            desired_backend_name=reference_run_config.backend,
            rtol=rtol,
            atol=atol,
        )
        validate_snr_and_psnr(
            actual=outputs,
            desired=reference_outputs,
            actual_backend_name=run_config.backend,
            desired_backend_name=reference_run_config.backend,
            snr_threshold=snr_threshold,
            psnr_threshold=psnr_threshold,
        )
