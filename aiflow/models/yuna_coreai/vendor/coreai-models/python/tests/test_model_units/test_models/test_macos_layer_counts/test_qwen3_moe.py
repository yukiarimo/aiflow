# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Layer count tests for the macOS Qwen3 MoE model.

These verify that this repo's Qwen3 MoE implementation produces the
expected MLIR op counts. The ``EXPECTED_COUNTS`` dict is the parity
contract -- divergence here means the implementation has drifted.
"""

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from coreai_models.models.macos import qwen3_moe
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
    # Attention Optimization Pattern (Qwen3 MoE-specific):
    # - CRITICAL: 4 batch_matmul operations verify Q/K/V projections + attention computation
    # - RoPE (Rotary Position Embedding) pattern: cos/sin operations + gather_along_axis
    # - Softmax should appear exactly once for attention scores normalization
    # - EXTRA RMSNorm ops (rsqrt, reduce_mean) indicate query pre-attention normalization
    # - Verifies efficient attention without unnecessary intermediate operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=4, invoke=3):
    #   - Graph 1: @rope composite - Rotary Position Embedding
    #   - Graph 2: @rms_norm composite - Q-norm (query normalization)
    #   - Graph 3: @scaled_dot_product_attention composite - SDPA
    #   - Graph 4: @main - Entry point
    #   - Invoke 1: main calls rope
    #   - Invoke 2: main calls rms_norm (Q-norm)
    #   - Invoke 3: main calls scaled_dot_product_attention
    "Attention": {
        "cast": 1,
        "composite_declaration": 3,
        "concat": 1,
        "constant": 35,
        "cos": 1,
        "decomposable.broadcasting_add": 3,
        "decomposable.broadcasting_batch_matmul": 4,
        "decomposable.broadcasting_mul": 9,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 2,
        "graph": 4,
        "invoke": 3,
        "name": 13,
        "output": 4,
        "reduce_mean": 1,
        "reshape": 8,
        "rsqrt": 1,
        "sin": 1,
        "slice": 6,
        "softmax": 1,
        "transpose": 5,
    },
    # SparseMoeBlock Optimization Pattern (Qwen3 MoE-specific):
    # - Mixture of Experts routing and expert computation
    # - CRITICAL operations to verify:
    #   * Router softmax for expert selection
    #   * Top-k selection via argsort/sort operations
    #   * gather_along_axis for token routing to experts
    #   * Expert-specific batch_matmul operations (multiple experts)
    #   * reduce_sum for combining expert outputs
    # - This is a placeholder - specific counts depend on expert configuration
    "SparseMoeBlock": {},
    # TransformerBlock Optimization Pattern (Qwen3 MoE-specific):
    # - Combines one Attention + SparseMoeBlock + multiple RMSNorm layers (including query norm)
    # - CRITICAL: MoE-specific operations: argsort, sort, gather_along_axis for expert routing
    # - 2 softmax = 1 (attention) + 1 (MoE router)
    # - reduce_sum for combining expert outputs
    # - Verifies efficient MoE routing and computation without redundant operations
    #
    # GRAPH/INVOKE BREAKDOWN (graph=7, invoke=8):
    #   - Graph 1: @rms_norm composite
    #   - Graph 2: @rope composite (from attention)
    #   - Graph 3: @rms_norm composite (QK-norm, from attention)
    #   - Graph 4: @scaled_dot_product_attention composite (from attention)
    #   - Graph 5: @gather_mm composite (MoE expert-1)
    #   - Graph 6: @gather_mm composite (MoE expert-2)
    #   - Graph 7: @main - Entry point
    #   - Invoke 1: main calls rms_norm (pre-attention)
    #   - Invoke 2: main calls rope
    #   - Invoke 3: main calls rms_norm (QK-norm)
    #   - Invoke 4: main calls scaled_dot_product_attention
    #   - Invoke 5: main calls rms_norm (post-attention)
    #   - Invoke 6: main calls gather_mm (MoE expert-1)
    #   - Invoke 7: main calls gather_mm (MoE expert-2)
    #   - Invoke 8: main calls rms_norm (MoE post-norm or routing)
    "TransformerBlock": {
        "argsort": 1,
        "broadcast_in_dims": 3,
        "cast": 4,
        "composite_declaration": 6,
        "concat": 1,
        "constant": 64,
        "cos": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 7,
        "decomposable.broadcasting_mul": 14,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 4,
        "graph": 7,
        "invoke": 8,
        "name": 24,
        "output": 7,
        "reduce_mean": 2,
        "reduce_sum": 1,
        "reshape": 16,
        "rsqrt": 2,
        "silu": 1,
        "sin": 1,
        "slice": 8,
        "softmax": 2,
        "sort": 1,
        "transpose": 6,
    },
    # ForCausalLM Optimization Pattern (Qwen3 MoE-specific):
    # - Complete model: Embedding + 1 TransformerBlock (with MoE) + LM head
    # - CRITICAL: MoE operations: argsort, sort, gather_along_axis preserved in full model
    # - KV cache operations: create_token, handle, read_handle, write_handle, slice_update
    # - 2 softmax = 1 (attention) + 1 (MoE router)
    # - Verifies end-to-end MoE model with stateful KV cache for efficient autoregressive generation
    #
    # GRAPH/INVOKE BREAKDOWN (graph=7, invoke=9):
    #   - Graph 1-6: Same composites as TransformerBlock
    #   - Graph 7: @main - Entry point
    #   - Invoke 1-8: Same as TransformerBlock
    #   - Invoke 9: main calls rms_norm (final, before LM head)
    "ForCausalLM": {
        "argsort": 1,
        "broadcast_in_dims": 3,
        "cast": 4,
        "composite_declaration": 6,
        "concat": 1,
        "constant": 73,
        "cos": 1,
        "create_token": 1,
        "decomposable.broadcasting_add": 6,
        "decomposable.broadcasting_batch_matmul": 8,
        "decomposable.broadcasting_mul": 14,
        "decomposable.broadcasting_sub": 1,
        "gather_along_axis": 4,
        "gather_nd": 1,
        "graph": 7,
        "handle": 2,
        "invoke": 9,
        "name": 28,
        "output": 7,
        "read_handle": 4,
        "reduce_mean": 2,
        "reduce_sum": 1,
        "reshape": 21,
        "rsqrt": 2,
        "silu": 1,
        "sin": 1,
        "slice": 10,
        "slice_update": 2,
        "softmax": 2,
        "sort": 1,
        "token": 17,
        "transpose": 7,
        "write_handle": 2,
    },
}


# =============================================================================
# CONFIG FIXTURE
# =============================================================================


@pytest.fixture
def qwen3_moe_config() -> Qwen3MoeConfig:
    """Create a small Qwen3MoeConfig for testing."""
    return Qwen3MoeConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )


# =============================================================================
# TESTS
# =============================================================================


class TestQwen3MoeLayerCounts:
    """Layer count tests for Qwen3 MoE model components."""

    def test_rmsnorm_layer_counts(self) -> None:
        """RMSNorm exports to expected Core AI operations."""
        model = qwen3_moe.RMSNorm(64, eps=1e-5)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["RMSNorm"])

    def test_mlp_layer_counts(self) -> None:
        """MLP exports to expected Core AI operations."""
        model = qwen3_moe.MLP(dim=64, hidden_dim=128)
        inputs = torch.randn(2, 4, 64)

        result = get_layer_counts(model=model, inputs=inputs)
        assert_layer_counts(result, EXPECTED_COUNTS["MLP"])

    def test_attention_layer_counts(self, qwen3_moe_config: Qwen3MoeConfig) -> None:
        """Attention exports to expected Core AI operations."""
        model = qwen3_moe.Attention(config=qwen3_moe_config, layer_idx=0)
        x = torch.randn(2, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0).expand(2, -1)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["Attention"])

    def test_transformer_block_layer_counts(self, qwen3_moe_config: Qwen3MoeConfig) -> None:
        """TransformerBlock exports to expected Core AI operations."""
        model = qwen3_moe.TransformerBlock(config=qwen3_moe_config, layer_idx=0)
        x = torch.randn(1, 4, 64)
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)

        result = get_layer_counts(model=model, inputs=(x, position_ids))
        assert_layer_counts(result, EXPECTED_COUNTS["TransformerBlock"])

    def test_for_causal_lm_layer_counts(self, qwen3_moe_config: Qwen3MoeConfig) -> None:
        """Qwen3MoeForCausalLM exports to expected Core AI operations."""
        qwen3_moe_config.num_hidden_layers = 1
        qwen3_moe_config.vocab_size = 100

        model = qwen3_moe.Qwen3MoeForCausalLM(qwen3_moe_config, model_device="cpu")
        input_ids = torch.randint(0, qwen3_moe_config.vocab_size, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(qwen3_moe_config)

        result = get_layer_counts(model=model, inputs=(input_ids, position_ids, k_cache, v_cache))
        assert_layer_counts(result, EXPECTED_COUNTS["ForCausalLM"])
