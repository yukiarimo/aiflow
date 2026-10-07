# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import torch

from ..types.dependency_types import Tensor


def _to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a ``Tensor`` (torch or castable) to a numpy array.

    bfloat16 is upcast to float32 first because numpy lacks native bfloat16.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    if tensor.dtype == torch.bfloat16:
        return tensor.detach().to(torch.float32).numpy()
    return tensor.detach().numpy()


def assert_allclose(
    actual: dict[str, Tensor],
    desired: dict[str, Tensor],
    actual_backend_name: str = "actual",
    desired_backend_name: str = "desired",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    # Outputs are matched by position, not by key: different backends (eager,
    # torch.export, Core AI) emit different graph-node-derived names for the
    # same logical output, so the dict keys are not expected to align.
    actual_names = list(actual)
    desired_names = list(desired)
    for i, (actual_tensor, desired_tensor) in enumerate(
        zip(actual.values(), desired.values(), strict=True)
    ):
        actual_np_array = _to_numpy(actual_tensor)
        desired_np_array = _to_numpy(desired_tensor)

        allclose_msg = (
            f"Output #{i} (actual={actual_names[i]!r}, desired={desired_names[i]!r}), "
            f"between {actual_backend_name} and {desired_backend_name}, "
            f"does not have all elements close"
        )
        np.testing.assert_allclose(
            actual_np_array,
            desired_np_array,
            rtol=rtol,
            atol=atol,
            err_msg=allclose_msg,
        )


def compute_snr_and_psnr(actual: Tensor, desired: Tensor) -> tuple[float, float]:
    # use torch for computation, so cast to torch if not
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(desired, torch.Tensor):
        desired = torch.tensor(desired)
    # strong signal energy / vanishing noise variance may overflow
    # so always use float32 for computation
    if actual.dtype != torch.float32:
        actual = actual.to(torch.float32)
    if desired.dtype != torch.float32:
        desired = desired.to(torch.float32)

    eps = 1e-5
    eps2 = 0.99e-10
    # any number with abs > inf is considered as infinity, i.e. no longer care
    # about their exact number, simply consider they are equal infinity
    inf = 1e38
    # some pre-defined max value to use in overflow cases
    max_snr, max_psnr = 255, 255

    is_actual_inf = actual.abs().amax().item() > inf
    is_desired_inf = desired.abs().amax().item() > inf
    if is_actual_inf != is_desired_inf:
        # 1 is inf but another is not, so they are not equal, i.e. low snr
        return 0, 0
    elif is_actual_inf and is_desired_inf:
        # both are inf, so simply consider they are equal infinity
        return max_snr, max_psnr

    desired_square = desired * desired
    signal_energy = desired_square.mean().item()
    max_signal_energy = desired_square.amax().item()
    noise = actual - desired
    noise_square = noise * noise
    noise_variance = noise_square.mean().item()
    snr = 10.0 * np.log10((signal_energy + eps) / (noise_variance + eps2))
    psnr = 10.0 * np.log10((max_signal_energy + eps) / (noise_variance + eps2))
    return round(snr, 2), round(psnr, 2)


def validate_snr_and_psnr(
    actual: dict[str, Tensor],
    desired: dict[str, Tensor],
    actual_backend_name: str = "actual",
    desired_backend_name: str = "desired",
    snr_threshold: float = 15.0,
    psnr_threshold: float = 29.5,
) -> None:
    # Outputs are matched by position, not by key (see ``assert_allclose``).
    actual_names = list(actual)
    desired_names = list(desired)
    for i, (actual_tensor, desired_tensor) in enumerate(
        zip(actual.values(), desired.values(), strict=True)
    ):
        snr, psnr = compute_snr_and_psnr(actual_tensor, desired_tensor)
        label = f"Output #{i} (actual={actual_names[i]!r}, desired={desired_names[i]!r})"
        assert snr > snr_threshold, (
            f"{label}, between {actual_backend_name} and {desired_backend_name}, "
            f"SNR {snr} below threshold {snr_threshold}"
        )
        assert psnr > psnr_threshold, (
            f"{label}, between {actual_backend_name} and {desired_backend_name}, "
            f"PSNR {psnr} below threshold {psnr_threshold}"
        )
