# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import dataclasses
import enum
from pathlib import Path
from typing import cast

import strenum
from typing_extensions import Self

from .source_types import Source, SourceConfig


class Frontend(strenum.StrEnum):
    torch_eager = enum.auto()
    torch_export = enum.auto()
    mlx = enum.auto()


class Backend(strenum.StrEnum):
    torch_eager = enum.auto()
    torch_export = enum.auto()
    mlx = enum.auto()
    coreai = enum.auto()


class CoreaiExportAPI(strenum.StrEnum):
    coreai_torch = enum.auto()


@dataclasses.dataclass(frozen=True)
class ExportConfig(SourceConfig):
    frontend: Frontend | None = None
    backend: Backend = cast("Backend", Backend.coreai)
    dynamic: bool = False

    coreai_export_api: CoreaiExportAPI | None = None

    def __post_init__(self: Self) -> None:
        # validate frontend
        if self.frontend is None:
            object.__setattr__(self, "frontend", BACKEND_PREFERRED_FRONTEND[self.backend])
        elif self.frontend not in BACKEND_SUPPORTED_FRONTENDS[self.backend]:
            msg = f"Backend {self.backend} does not support frontend {self.frontend}"
            raise ValueError(msg)
        # validate coreai export API
        if self.coreai_export_api is None:
            if self.backend == Backend.coreai:
                object.__setattr__(self, "coreai_export_api", CoreaiExportAPI.coreai_torch)
        else:
            if self.backend != Backend.coreai:
                msg = (
                    f"Only Core AI backend requires Core AI export API, not {self.backend} backend"
                )
                raise ValueError(msg)

    def get_underlying_source_config(self: Self) -> SourceConfig:
        source_config_kwargs = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(SourceConfig)
        }
        return SourceConfig(**source_config_kwargs)

    def exported_model_path(self: Self, root: Path) -> Path:
        """Recommend path to serialize exported model under given root."""
        dynamic_string = "dynamic" if self.dynamic else "static"
        directory = root / self.backend / self.frontend / dynamic_string / super().__str__()
        model = {
            Backend.torch_eager: "model.pt",
            Backend.torch_export: "model.pt2",
            Backend.mlx: "model.mlx",
            Backend.coreai: "model.aimodel",
        }[self.backend]
        return directory / model


SOURCE_PREFERRED_BACKEND: dict[Source, Backend] = {
    cast("Source", Source.torch): cast("Backend", Backend.torch_eager),
    cast("Source", Source.mlx): cast("Backend", Backend.mlx),
}

FRONTEND_SOURCE: dict[Frontend, Source] = {
    cast("Frontend", Frontend.torch_eager): cast("Source", Source.torch),
    cast("Frontend", Frontend.torch_export): cast("Source", Source.torch),
    cast("Frontend", Frontend.mlx): cast("Source", Source.mlx),
}

BACKEND_SUPPORTED_FRONTENDS: dict[Backend, tuple[Frontend, ...]] = {
    cast("Backend", Backend.torch_eager): (cast("Frontend", Frontend.torch_eager),),
    cast("Backend", Backend.torch_export): (cast("Frontend", Frontend.torch_export),),
    cast("Backend", Backend.mlx): (cast("Frontend", Frontend.mlx),),
    cast("Backend", Backend.coreai): (cast("Frontend", Frontend.torch_export),),
}

BACKEND_PREFERRED_FRONTEND: dict[Backend, Frontend] = {
    cast("Backend", Backend.torch_eager): cast("Frontend", Frontend.torch_eager),
    cast("Backend", Backend.torch_export): cast("Frontend", Frontend.torch_export),
    cast("Backend", Backend.mlx): cast("Frontend", Frontend.mlx),
    cast("Backend", Backend.coreai): cast("Frontend", Frontend.torch_export),
}
