# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import mlx
import mlx.nn
from typing_extensions import Self


class MlxExporter:
    def export(
        self: Self,
        mlx_module: mlx.nn.Module,
    ) -> mlx.nn.Module:
        return mlx_module
