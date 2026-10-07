# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MLIR-level quantization helpers for the export pipeline."""

import logging

from coreai.authoring import AIProgram
from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    quantize_weights,
)
from coreai_opt.coreai_utils.common import QScheme

_GRANULARITY_MAP: dict[str, CompressionGranularity] = {
    "per_tensor": CompressionGranularity.PER_TENSOR,
    "per_channel": CompressionGranularity.PER_CHANNEL,
    "per_block": CompressionGranularity.PER_BLOCK,
    "per_grouped_channel": CompressionGranularity.PER_GROUPED_CHANNEL,
}


logger = logging.getLogger(__name__)


async def apply_mlir_quantization(
    coreai_program: AIProgram,
    quantize_config: dict,
) -> AIProgram:
    """
    Apply post-MLIR INT4 weight quantization to a Core AI program.

    Args:
        coreai_program: The Core AI program to quantize.
        quantize_config: Quantization configuration with keys:
            - type: Quantization type (currently only "int4" supported)
            - symmetric: Whether to use symmetric quantization
            - granularity: Granularity level (e.g., "per_block")
            - block_size: Block size for per-block granularity

    Returns:
        The (potentially modified) Core AI program.
    """
    quant_type = quantize_config.get("type", "int4")
    symmetric = quantize_config.get("symmetric", True)
    granularity = quantize_config.get("granularity", "per_block")
    block_size = quantize_config.get("block_size", 32)

    logger.info(
        f"Applying {quant_type} quantization with {granularity} granularity "
        f"(block_size={block_size})"
    )

    if quant_type == "int4":
        try:
            coreai_program = quantize_weights(
                coreai_program,
                dtype=DType.INT4,
                qscheme=QScheme.SYMMETRIC if symmetric else QScheme.ASYMMETRIC,
                granularity=_GRANULARITY_MAP[granularity],
                block_size=block_size,
                weight_num_threshold=32768,
                in_place=True,
            )
            logger.info("Applied INT4 weight quantization")
        except ImportError:
            logger.warning("Core AI quantization not available, skipping quantization")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    else:
        logger.warning(f"Unsupported quantization type: {quant_type}")

    return coreai_program
