# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Layer count tests for the macOS Qwen2 model.

These verify that this repo's Qwen2 implementation produces the expected
MLIR op counts. The ``EXPECTED_COUNTS`` dict is the parity contract --
divergence here means the implementation has drifted.
"""

import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from coreai_models.models.macos import qwen2
from coreai_models.primitives.macos.cache import KVCache
from tests._layer_count_utils import assert_layer_counts, get_layer_counts

# =============================================================================
# EXPECTED COUNTS
# =============================================================================
#
# These counts verify that critical model optimizations are applied correctly.
# Each test ensures that the PyTorch model compiles to an efficient MLIR<Core AI>
# representation with proper fusion and optimization patterns.
#
# NOTE: When composite_declaration ops are present, they indicate that patterns
# like RMSNorm, RoPE, or SDPA have been fused into single composite operations.
# This is expected and indicates proper fusion.

EXPECTED_COUNTS = {
    # RMSNorm Optimization Pattern:
    # - CRITICAL: composite_declaration indicates the RMSNorm is fused into a single operation
    # - Pattern: x * rsqrt(mean(x^2) + eps) * scale
    # - Verifies normalization is efficient without intermediate materialization
    #
    # GRAPH/INVOKE BREAKDOWN (graph=2, invoke=1):
    #   - Graph 1: @rms_norm composite - Fused RMSNorm implementation
    #   - Graph 2: @main - Entry point
    #   - Invoke 1: main calls rms_norm
    "RMSNorm": {
        "composite_declaration": 1,
        "constant": 3,
        "decomposable.broadcasting_add": 1,
        "decomposable.broadcasting_mul": 3,
        "graph": 2,
        "invoke": 1,
        "name": 5,
        "output": 2,
        "reduce_mean": 1,
        "rsqrt": 1,
    },
    # MLP Optimization Pattern:
    # - CRITICAL: 3 batch_matmul operations (one for each w1, w2, w3 linear layer)
    # - Pattern: w2(silu(w1(x)) * w3(x)) - SwiGLU activation
    # - Verifies linear layers are properly lowered without extra operations
    # - The gate (w1), down (w2), and up (w3) projections should be distinct matmuls
    #
    # GRAPH/INVOKE BREAKDOWN (graph=1, invoke=0):
    #   - Graph 1: @main - Entry point (all operations inline, no composites)
    "MLP": {
        "constant": 4,
        "decomposable.broadcasting_batch_matmul": 3,
        "decomposable.broadcasting_mul": 1,
        "graph": 1,
        "name": 2,
        "output": 1,
        "silu": 1,
        "transpose": 3,
    },
    # Attention Optimization Pattern (Qwen2-specific):
    # - CRITICAL: 4 batch_matmul operations verify Q/K/V projections + attention computation
    # - RoPE (Rotary Position Embedding) pattern: cos/sin operations + gather_along_axis
    # - Softmax should appear exactly once for attention scores normalization
    # - Note: extra broadcasting_add for bias addition specific to Qwen2
    # - Verifies efficient attention without unnecessary intermediate operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=3, invoke=2):
    #   - Graph 1: @rope composite - Rotary Position Embedding
    #   - Graph 2: @scaled_dot_product_attention composite - SDPA
    #   - Graph 3: @main - Entry point
    #   - Invoke 1: main calls rope for position encoding
    #   - Invoke 2: main calls scaled_dot_product_attention
    "Attention": {
        "cast": 1,
        "composite_declaration": 2,
        "concat": 1,
        "constant": 33,
        "cos": 1,
        "decomposable.broadcasting_add": 3,
        "decomposable.broadcasting_batch_matmul": 4,
        "decomposable.broadcasting_mul": 6,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "graph": 3,
        "invoke": 2,
        "name": 10,
        "output": 3,
        "reshape": 8,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 5,
    },
    # TransformerBlock Optimization Pattern:
    # - Combines one Attention + one MLP + two RMSNorm layers
    # - CRITICAL: 7 batch_matmul = 4 (attention) + 3 (MLP)
    # - Verifies the full transformer block composition without redundant operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=4, invoke=4):
    #   - Graph 1: @rms_norm composite
    #   - Graph 2: @rope composite (from attention)
    #   - Graph 3: @scaled_dot_product_attention composite (from attention)
    #   - Graph 4: @main - Entry point
    #   - Invoke 1: main calls rms_norm (pre-attention)
    #   - Invoke 2: main calls rope (via attention)
    #   - Invoke 3: main calls scaled_dot_product_attention (via attention)
    #   - Invoke 4: main calls rms_norm (post-attention)
    "TransformerBlock": {
        "cast": 1,
        "composite_declaration": 3,
        "concat": 1,
        "constant": 35,
        "cos": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 7,
        "decomposable.broadcasting_mul": 10,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "graph": 4,
        "invoke": 4,
        "name": 13,
        "output": 4,
        "reduce_mean": 1,
        "reshape": 4,
        "rsqrt": 1,
        "silu": 1,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 8,
    },
    # ForCausalLM Optimization Pattern:
    # - Complete model: Embedding + 1 TransformerBlock + LM head
    # - CRITICAL: 8 batch_matmul = 7 (TransformerBlock) + 1 (LM head projection)
    # - KV cache operations: create_token, handle, read_handle, write_handle, slice_update
    # - Verifies end-to-end model with stateful KV cache for efficient autoregressive generation
    #
    # GRAPH/INVOKE BREAKDOWN (graph=4, invoke=5):
    #   - Graph 1: @rms_norm composite
    #   - Graph 2: @rope composite (from attention)
    #   - Graph 3: @scaled_dot_product_attention composite (from attention)
    #   - Graph 4: @main - Entry point
    #   - Invoke 1: main calls rms_norm (pre-attention, in block)
    #   - Invoke 2: main calls rope (via attention)
    #   - Invoke 3: main calls scaled_dot_product_attention (via attention)
    #   - Invoke 4: main calls rms_norm (post-attention, in block)
    #   - Invoke 5: main calls rms_norm (final, before LM head)
    "ForCausalLM": {
        "cast": 1,
        "composite_declaration": 3,
        "concat": 1,
        "constant": 44,
        "cos": 1,
        "create_token": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 8,
        "decomposable.broadcasting_mul": 10,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "gather_nd": 1,
        "graph": 4,
        "handle": 2,
        "invoke": 5,
        "name": 17,
        "output": 4,
        "read_handle": 4,
        "reduce_mean": 1,
        "reshape": 9,
        "rsqrt": 1,
        "silu": 1,
        "sin": 1,
        "slice": 8,
        "slice_update": 2,
        "softmax": 1,
        "token": 17,
        "transpose": 9,
        "write_handle": 2,
    },
}


# =============================================================================
# CONFIG FIXTURE
# =============================================================================


@pytest.fixture
def qwen2_config() -> Qwen2Config:
    """Create a small Qwen2Config for testing."""
    return Qwen2Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


# =============================================================================
# TESTS
# =============================================================================


class TestQwen2LayerCounts:
    """Layer count tests for Qwen2 model components."""

    def test_rmsnorm_layer_counts(self) -> None:
        """RMSNorm exports to expected Core AI operations."""
        model = qwen2.RMSNorm(64, eps=1e-5)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["RMSNorm"])

    def test_mlp_layer_counts(self) -> None:
        """MLP exports to expected Core AI operations."""
        model = qwen2.MLP(dim=64, hidden_dim=128)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["MLP"])

    def test_attention_layer_counts(self, qwen2_config: Qwen2Config) -> None:
        """Attention exports to expected Core AI operations."""
        model = qwen2.Attention(config=qwen2_config, layer_idx=0)
        x = torch.randn(2, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0).expand(2, -1)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["Attention"])

    def test_transformer_block_layer_counts(self, qwen2_config: Qwen2Config) -> None:
        """TransformerBlock exports to expected Core AI operations."""
        model = qwen2.TransformerBlock(config=qwen2_config, layer_idx=0)
        x = torch.randn(1, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["TransformerBlock"])

    def test_for_causal_lm_layer_counts(self, qwen2_config: Qwen2Config) -> None:
        """Qwen2ForCausalLM exports to expected Core AI operations."""
        qwen2_config.num_hidden_layers = 1
        qwen2_config.vocab_size = 100

        model = qwen2.Qwen2ForCausalLM(qwen2_config, model_device="cpu")
        input_ids = torch.randint(0, qwen2_config.vocab_size, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(qwen2_config)

        result = get_layer_counts(model=model, inputs=(input_ids, position_ids, k_cache, v_cache))
        assert_layer_counts(result, EXPECTED_COUNTS["ForCausalLM"])
