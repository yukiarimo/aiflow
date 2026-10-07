# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import dataclasses

from typing_extensions import Self, override

from .export_types import ExportConfig


@dataclasses.dataclass(frozen=True)
class RunConfig(ExportConfig):
    @override
    def __post_init__(self: Self) -> None:
        super().__post_init__()

    def get_underlying_export_config(self: Self) -> ExportConfig:
        export_config_kwargs = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(ExportConfig)
        }
        return ExportConfig(**export_config_kwargs)
