# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod

import torch
from typing_extensions import Self

from ...common.types.dependency_types import Tensor


class Runner(ABC):
    """Generic base runner class parameterized by output tensor type."""

    @abstractmethod
    def forward(self: Self, named_inputs: dict[str, Tensor]) -> dict[str, torch.Tensor]:
        """Run inference and return outputs of specific tensor type."""
        ...
