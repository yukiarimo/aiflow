# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import nn


@torch.library.custom_op("coreai::fused_dequant_gather_reshape", mutates_args=[])
def fused_dequant_gather_reshape(
    embedding_table: torch.Tensor,
    input_ids: torch.Tensor,
    scale: torch.Tensor,
    final_shape: list[int],
) -> torch.Tensor:
    return (embedding_table[input_ids].to(scale.dtype) * scale).reshape(final_shape)


@fused_dequant_gather_reshape.register_fake
def fused_dequant_gather_reshape_fake(
    embedding_table: torch.Tensor,
    input_ids: torch.Tensor,
    scale: torch.Tensor,
    final_shape: list[int],
) -> torch.Tensor:
    return torch.zeros(final_shape, dtype=scale.dtype)


class LoadEmbeddings(torch.nn.Module):
    def __init__(self, config, embedding_table_dtype=torch.int8):
        super().__init__()
        self.embedding_table = torch.nn.Parameter(
            torch.zeros(config.vocab_size, 1, config.hidden_size, dtype=embedding_table_dtype),
            requires_grad=False,
        )

    def forward(self):
        return self.embedding_table


class GatherEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
        in_id_shape = input_ids.size()
        emb_shp = embedding_table.size()[1:]
        final_shape = in_id_shape + emb_shp
        if not embedding_table.is_floating_point():
            return fused_dequant_gather_reshape(embedding_table, input_ids, self.scale, final_shape)
        return embedding_table[input_ids]
