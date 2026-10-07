# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ml_dtypes
import mlx
import mlx.core
import numpy as np
import torch


def mlx_array_to_numpy_array(mlx_array: mlx.core.array) -> np.ndarray:
    """
    Convert mlx.core.array to numpy.ndarray.

    TODO: Deprecate when all mlx.core.array can be seamlessly convert to numpy.ndarray
    """
    if mlx_array.dtype == mlx.core.bfloat16:
        # numpy does not support bfloat16 yet
        # so we workaround by mlx.core.bfloat16 -> uint16 bytes -> ml_dtypes.bfloat16
        mlx_array_bytes = mlx_array.view(mlx.core.uint16)
        np_array_bytes = np.array(mlx_array_bytes)
        np_array = np_array_bytes.view(ml_dtypes.bfloat16)
    else:
        np_array = np.array(mlx_array)
    return np_array


def mlx_array_to_torch_tensor(mlx_array: mlx.core.array) -> torch.Tensor:
    """
    Convert mlx.core.array to torch.Tensor.

    TODO: Deprecate when all mlx.core.array can be seamlessly convert to torch.Tensor
    """
    if mlx_array.dtype == mlx.core.bfloat16:
        # torch_tensor = torch.tensor(mlx_array) under the hood calls
        #     numpy_array = np.array(mlx_array)
        #     torch_tensor = torch.tensor(numpy_array)
        # this fails for bfloat16 because numpy does not support bfloat16 yet
        # so we workaround by mlx.core.bfloat16 -> uint16 bytes -> torch.bfloat16
        mlx_array_bytes = mlx_array.view(mlx.core.uint16)
        np_array_bytes = np.array(mlx_array_bytes)
        torch_tensor_bytes = torch.from_numpy(np_array_bytes)
        torch_tensor = torch_tensor_bytes.view(torch.bfloat16)
    else:
        torch_tensor = torch.tensor(mlx_array)
    return torch_tensor


def torch_tensor_to_mlx_array(torch_tensor: torch.Tensor) -> mlx.core.array:
    """
    Convert torch.Tensor to mlx.core.array.

    mlx.core.array does not accept torch.Tensor directly; we route through
    numpy. bfloat16 is handled via a uint16 byte view because numpy lacks
    native bfloat16 support.
    """
    detached = torch_tensor.detach().cpu()
    if detached.dtype == torch.bfloat16:
        np_array_bytes = detached.view(torch.uint16).numpy()
        return mlx.core.array(np_array_bytes).view(mlx.core.bfloat16)
    return mlx.core.array(detached.numpy())
