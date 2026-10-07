# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import dataclasses
import enum
from typing import cast

import strenum
from typing_extensions import Self, override


class Author(strenum.StrEnum):
    oss = enum.auto()
    coreai = enum.auto()


class Source(strenum.StrEnum):
    torch = enum.auto()
    mlx = enum.auto()


class Precision(strenum.StrEnum):
    f32 = enum.auto()
    f16 = enum.auto()
    bf16 = enum.auto()


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    author: Author = cast("Author", Author.coreai)
    source: Source = cast("Source", Source.torch)
    precision: Precision = cast("Precision", Precision.f32)

    @override
    def __str__(self: Self) -> str:
        field_strs = [
            f"{field.name}-{getattr(self, field.name)}"
            for field in dataclasses.fields(SourceConfig)
        ]
        return ",".join(field_strs)
