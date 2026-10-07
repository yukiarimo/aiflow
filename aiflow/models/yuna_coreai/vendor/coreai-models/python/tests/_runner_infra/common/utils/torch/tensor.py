# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ml_dtypes
import numpy as np
import torch


def torch_tensor_to_numpy_array(torch_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to numpy.ndarray.

    TODO: Deprecate when all torch.Tensor can be seamlessly convert to numpy.ndarray
    """
    torch_tensor = torch_tensor.detach().cpu()
    if torch_tensor.dtype == torch.bfloat16:
        # torch_tensor.numpy() fails due to not being numpy native dtype
        # so we workaround by torch.bfloat16 -> uint16 bytes -> ml_dtypes.bfloat16
        torch_tensor_bytes = torch_tensor.view(torch.uint16)
        np_array_bytes = torch_tensor_bytes.numpy()
        np_array = np_array_bytes.view(ml_dtypes.bfloat16)
    else:
        np_array = torch_tensor.numpy()
    return np_array
