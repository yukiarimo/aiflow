# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Compression presets for diffusion model export.

Each preset is a named configuration consumed by the diffusion export pipeline.
Currently the only knob is post-export MLIR weight quantization (applied to
quantizable components — text encoder and UNet). The VAE encoder/decoder is
small and quality-sensitive, so it is never quantized.

Usage::

    from coreai_models.diffusion.presets import get_preset, list_presets

    preset = get_preset("4bit")     # -> {"description": ..., "config": {...}}
    names = list_presets()          # -> ["4bit", "none"]
"""

from typing import Any

DEFAULT_COMPRESSION_PRESET = "none"

PRESETS: dict[str, dict[str, Any]] = {
    "none": {
        "description": "Full precision (no quantization)",
        "config": None,
    },
    "4bit": {
        "description": "INT4 per-block (block_size=32), symmetric",
        "config": {
            "type": "int4",
            "symmetric": True,
            "granularity": "per_block",
            "block_size": 32,
        },
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a compression preset by name.

    Args:
        name: Preset name (e.g., ``"4bit"``, ``"none"``)

    Returns:
        Preset configuration dict

    Raises:
        KeyError: If the preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown diffusion compression preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> list[str]:
    """List all available diffusion compression preset names."""
    return sorted(PRESETS.keys())
