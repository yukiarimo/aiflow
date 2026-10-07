# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ml_dtypes
import numpy as np
import torch
from coreai.runtime import NDArray


def ndarray_to_torch_tensor(ndarray: NDArray) -> torch.Tensor:
    """
    Convert coreai.runtime.NDArray to torch.Tensor.

    TODO: Deprecate when all NDArray can be seamlessly convert to torch.Tensor
    """
    np_array = ndarray.numpy()
    if np_array.dtype == ml_dtypes.bfloat16:
        # torch.from_numpy(np_array) fails due to not being numpy native dtype
        # so we workaround by ml_dtypes.bfloat16 -> uint16 bytes -> torch.bfloat16
        np_array_bytes = np_array.view(np.uint16)
        torch_tensor_bytes = torch.from_numpy(np_array_bytes)
        torch_tensor = torch_tensor_bytes.view(torch.bfloat16)
    else:
        # ``from_numpy`` shares memory with the numpy buffer (no copy);
        # matches the bfloat16 path above and avoids the implicit copy that
        # ``torch.tensor`` would do.
        torch_tensor = torch.from_numpy(np_array)
    return torch_tensor
