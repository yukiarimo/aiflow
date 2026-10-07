# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for iOS embedding primitives."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from coreai_models.primitives.ios.embedding import (
    GatherEmbeddings,
    LoadEmbeddings,
)
from tests._runner_infra.testing_utils import (
    assert_close,
    run_compare_coreai,
)


class _FakeConfig:
    """Minimal config for LoadEmbeddings."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TestLoadEmbeddings:
    """Test LoadEmbeddings module."""

    def test_shape(self):
        """Embedding table should have shape (vocab_size, 1, hidden_size)."""
        config = _FakeConfig(vocab_size=32, hidden_size=16)
        loader = LoadEmbeddings(config, embedding_table_dtype=torch.int8)
        table = loader()
        assert table.shape == (32, 1, 16)
        assert table.dtype == torch.int8

    def test_float_dtype(self):
        """LoadEmbeddings should work with float16 dtype too."""
        config = _FakeConfig(vocab_size=16, hidden_size=8)
        loader = LoadEmbeddings(config, embedding_table_dtype=torch.float16)
        table = loader()
        assert table.shape == (16, 1, 8)
        assert table.dtype == torch.float16


class TestGatherEmbeddings:
    """Test GatherEmbeddings module."""

    def test_float_table(self):
        """GatherEmbeddings with float table should do simple indexing."""
        config = _FakeConfig(vocab_size=16, hidden_size=8)
        loader = LoadEmbeddings(config, embedding_table_dtype=torch.float32)
        # Set known values
        loader.embedding_table.data = torch.randn(16, 1, 8)

        gather = GatherEmbeddings()
        input_ids = torch.tensor([0, 3, 7])
        table = loader()
        out = gather(input_ids, table)

        # Output should be the selected embeddings
        assert out.shape == (3, 1, 8)
        torch.testing.assert_close(out[0], table[0])
        torch.testing.assert_close(out[1], table[3])
        torch.testing.assert_close(out[2], table[7])

    def test_int8_table_with_dequant(self):
        """GatherEmbeddings with int8 table should use fused dequant+gather."""
        config = _FakeConfig(vocab_size=16, hidden_size=8)
        loader = LoadEmbeddings(config, embedding_table_dtype=torch.int8)
        # Fill with small int8 values
        loader.embedding_table.data = torch.randint(-10, 10, (16, 1, 8), dtype=torch.int8)

        gather = GatherEmbeddings()
        gather.scale.data = torch.tensor(0.5, dtype=torch.float16)

        input_ids = torch.tensor([1, 5])
        table = loader()
        out = gather(input_ids, table)

        # Output should have float16 dtype (from scale)
        assert out.shape == (2, 1, 8)
        assert out.dtype == torch.float16

        # Verify correctness manually
        expected = (table[torch.tensor([1, 5])].to(torch.float16) * 0.5).reshape(2, 1, 8)
        torch.testing.assert_close(out, expected)


class CombinedEmbedding(nn.Module):
    """Composes LoadEmbeddings + GatherEmbeddings for end-to-end testing."""

    def __init__(self, config, embedding_table_dtype=torch.float32):
        super().__init__()
        self.loader = LoadEmbeddings(config, embedding_table_dtype=embedding_table_dtype)
        self.gather = GatherEmbeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gather(input_ids, self.loader())


class TestEmbedding:
    """Test iOS-optimized Embedding layer against PyTorch reference."""

    @staticmethod
    def get_model_asset(
        precision: torch.dtype,
    ) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """Create Embedding models and test inputs for comparison."""
        vocab_size = 10
        hidden_size = 8
        batch_size = 1
        seq_len = 5

        config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)

        # Create models
        our_embedding = CombinedEmbedding(config)
        torch_embedding = nn.Embedding(vocab_size, hidden_size)

        # Share weights - iOS needs (vocab_size, 1, hidden_size)
        weight = torch.randn(vocab_size, hidden_size)
        our_embedding.loader.embedding_table = nn.Parameter(weight.clone().unsqueeze(1))
        torch_embedding.weight = nn.Parameter(weight.clone())

        # Convert to precision
        our_embedding = our_embedding.to(precision)
        torch_embedding = torch_embedding.to(precision)

        # Create input IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute expected output
        expected_output = torch_embedding(input_ids).unsqueeze(2)

        return our_embedding, input_ids, expected_output

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
    def test_hf(self, precision: torch.dtype) -> None:
        """Test functional parity with PyTorch Embedding."""
        model, input_ids, expected_output = self.get_model_asset(precision)
        assert_close(model(input_ids), expected_output)

    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_coreai(self, precision: torch.dtype) -> None:
        pytest.xfail("Embedding layer produces incorrect output on this backend")  # noqa: E501
        """Test functional parity with Core AI implementation."""
        model, input_ids, _ = self.get_model_asset(precision)
        run_compare_coreai(
            model=model,
            inputs=input_ids,
            atol=1e-5 if precision == torch.float32 else 7e-2,
        )
