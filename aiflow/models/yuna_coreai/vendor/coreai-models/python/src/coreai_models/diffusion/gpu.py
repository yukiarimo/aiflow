# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Stateless GPU export for diffusion components.

Simpler than the LLM export path: no KV cache, no dynamic shapes, no
externalized composites.  Each component is a single fixed-shape forward pass.
"""

import logging

import coreai_torch
import torch
from coreai.authoring import AIProgram

logger = logging.getLogger(__name__)


def export_stateless(
    wrapper: torch.nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> AIProgram:
    """Export a stateless model to a Core AI AIProgram.

    Args:
        wrapper: A thin torch.nn.Module that wraps a HF model component.
        dummy_inputs: Reference input tensors (positional) for tracing.
        input_names: Names for the exported model's inputs.
        output_names: Names for the exported model's outputs.

    Returns:
        An optimized AIProgram ready for saving/compilation.
    """
    wrapper.eval()

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            exported = torch.export.export(module, args=dummy_inputs)
        coreai_decomp_table = coreai_torch.get_decomp_table()
        decomposed: torch.export.ExportedProgram = exported.run_decompositions(coreai_decomp_table)
        return decomposed

    converter = coreai_torch.TorchConverter()
    converter.add_pytorch_module(
        wrapper,
        export_fn=export_fn,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    program.optimize()
    return program
