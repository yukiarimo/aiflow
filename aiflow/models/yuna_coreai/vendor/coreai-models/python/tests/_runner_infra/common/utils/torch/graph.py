# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch


def extract_placeholders_from_torch_exported_program(
    exported_program: torch.export.ExportedProgram,
) -> dict[str, torch.fx.Node]:
    """
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        placeholders: dictionary mapping names to placeholder nodes
    """
    placeholders: dict[str, torch.fx.Node] = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op == "placeholder":
            placeholders[node.name] = node
    return placeholders


def extract_inputs_from_torch_exported_program(
    exported_program: torch.export.ExportedProgram,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Extract "inputs" from torch.export.ExportedProgram
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_inputs: dict[str, torch.Tensor]
            dict of user output names to fake tensors
        buffer_mutations: tuple[str]
            dict of persistent buffer names to fake tensors
    """
    placeholders = extract_placeholders_from_torch_exported_program(exported_program)
    user_inputs: dict[str, torch.Tensor] = {}
    persistent_buffers: dict[str, torch.Tensor] = {}
    for input_spec in exported_program.graph_signature.input_specs:
        if input_spec.kind == torch.export.graph_signature.InputKind.USER_INPUT:
            node = placeholders[input_spec.arg.name]
            val = node.meta["val"]
            user_inputs[node.name] = val
        elif (
            input_spec.kind == torch.export.graph_signature.InputKind.BUFFER
            and input_spec.persistent
        ):
            node = placeholders[input_spec.arg.name]
            val = node.meta["val"]
            persistent_buffers[node.name] = val
    return user_inputs, persistent_buffers


def extract_outputs_from_torch_exported_program(
    exported_program: torch.export.ExportedProgram,
) -> tuple[tuple[str], tuple[str]]:
    """
    Extract "outputs" from torch.export.ExportedProgram
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_outputs: tuple[str]
            tuple of user output names
        buffer_mutations: tuple[str, str]
            tuple of buffer mutation names
    """
    user_outputs = []
    buffer_mutations = []
    for output_spec in exported_program.graph_signature.output_specs:
        if output_spec.kind == torch.export.graph_signature.OutputKind.USER_OUTPUT:
            user_outputs.append(output_spec.arg.name)
        elif output_spec.kind == torch.export.graph_signature.OutputKind.BUFFER_MUTATION:
            buffer_mutations.append(output_spec.arg.name)
    return tuple(user_outputs), tuple(buffer_mutations)
