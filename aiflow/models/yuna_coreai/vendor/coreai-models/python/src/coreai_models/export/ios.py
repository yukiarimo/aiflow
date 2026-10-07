# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
iOS model export pipeline.

Exports a PyTorch LLM model to a Core AI AIProgram for iOS.
The iOS export produces 4 entrypoints:
- load_embeddings: returns the embedding table
- gather_embeddings: token IDs -> embedded representations
- extend: single forward pass (decode mode)
- prompt_opt: forward pass in prefill mode
"""

import logging

import torch
from coreai.authoring import AIProgram
from coreai.authoring.types import AllocationType, HardwareConstraints
from coreai_torch import TorchConverter

from coreai_models.export.mlir_ops import (
    register_custom_torch_lowering,
    remove_functionalization,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# iOS graph I/O names (must match what the Swift runner expects)
# ---------------------------------------------------------------------------

KV_CACHE_INTERLEAVE_FACTOR = 8

LOAD_EMBEDDINGS_FUNCTION_NAME = "load_embeddings"
GATHER_EMBEDDINGS_FUNCTION_NAME = "gather_embeddings"
EXTEND_FUNCTION_NAME = "extend"
PROMPT_OPT_FUNCTION_NAME = "prompt_opt"

EMBEDDING_TABLE_INPUT_NAME = "embedding_table"
LOAD_EMBEDDINGS_OUTPUT_NAME = "embedding_table"
TOKEN_IDS_INPUT_NAME = "in_new_token_ids"
GATHERED_EMBEDDINGS_OUTPUT_NAME = "gathered_embeddings"

TRANSFORMER_INPUT_NAME = "transformer_input"
POSITION_IDS_INPUT_NAME = "position_ids"
IN_STEP_INPUT_NAME = "in_step"
CAUSAL_MASK_INPUT_NAME = "causal_mask"
KEY_CACHE_INPUT_NAME = "key_cache"
VALUE_CACHE_INPUT_NAME = "value_cache"
KEY_CACHE_OUTPUT_NAME = "new_k_cache"
VALUE_CACHE_OUTPUT_NAME = "new_v_cache"
OUTPUT_LOGITS_NAME = "out_logits"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ios_reference_inputs(
    model: torch.nn.Module,
    config,
    max_context_length: int,
    vocab_size: int,
) -> dict:
    """Build reference input tensors for iOS model export.

    Returns a dict with all inputs needed by the extend function, plus
    the embed_tokens inputs and dynamic shapes for each entrypoint.
    """
    batch_size = 1
    query_len = 8

    input_ids = torch.randint(1, vocab_size, (batch_size, query_len), dtype=torch.int32)
    position_ids = (
        torch.arange(query_len).to(torch.uint16).unsqueeze(0).expand(batch_size, query_len)
    )
    in_step = torch.zeros((1,), dtype=torch.int32)
    causal_mask = torch.zeros(1, max_context_length, 1, query_len, dtype=torch.float16)

    if hasattr(config, "head_dim") and isinstance(config.head_dim, int):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    key_cache = torch.zeros(
        config.num_hidden_layers,
        1,
        config.num_key_value_heads * head_dim,
        1,
        max_context_length,
        dtype=torch.float16,
    )
    value_cache = key_cache.clone()

    # Generate embeddings from the model
    embedding_table = model.load_embeddings.embedding_table
    transformer_input = model.gather_embeddings(input_ids, embedding_table)

    forward_inputs = {
        TRANSFORMER_INPUT_NAME: transformer_input,
        POSITION_IDS_INPUT_NAME: position_ids,
        IN_STEP_INPUT_NAME: in_step,
        CAUSAL_MASK_INPUT_NAME: causal_mask,
        KEY_CACHE_INPUT_NAME: key_cache,
        VALUE_CACHE_INPUT_NAME: value_cache,
        EMBEDDING_TABLE_INPUT_NAME: embedding_table,
    }

    embed_tokens_inputs = (input_ids, embedding_table)

    seq_len_dim = torch.export.Dim("seq_len", max=max_context_length)
    cache_len_dim = torch.export.Dim("cache_len", max=max_context_length)

    forward_dynamic_shapes = {
        TRANSFORMER_INPUT_NAME: {1: seq_len_dim},
        POSITION_IDS_INPUT_NAME: {1: seq_len_dim},
        IN_STEP_INPUT_NAME: None,
        CAUSAL_MASK_INPUT_NAME: {1: cache_len_dim, 3: seq_len_dim},
        KEY_CACHE_INPUT_NAME: {4: cache_len_dim},
        VALUE_CACHE_INPUT_NAME: {4: cache_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }

    embed_tokens_dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        EMBEDDING_TABLE_INPUT_NAME: None,
    }

    return {
        "forward_inputs": forward_inputs,
        "embed_tokens_inputs": embed_tokens_inputs,
        "forward_dynamic_shapes": forward_dynamic_shapes,
        "embed_tokens_dynamic_shapes": embed_tokens_dynamic_shapes,
    }


def _export_ios_programs(
    model: torch.nn.Module,
    inputs: dict,
) -> tuple:
    """Export the 4 ExportedPrograms for the iOS entrypoints.

    Returns:
        Tuple of (extend_program, prompt_program, gather_program, load_program).
    """
    forward_inputs = inputs["forward_inputs"]
    embed_tokens_inputs = inputs["embed_tokens_inputs"]
    forward_dynamic_shapes = inputs["forward_dynamic_shapes"]
    embed_tokens_dynamic_shapes = inputs["embed_tokens_dynamic_shapes"]

    with torch.no_grad():
        # iOS decomp table: keep silu as-is
        decomp_table = torch.export.default_decompositions()
        decomp_table.pop(torch.ops.aten.silu.default)
        decomp_table.pop(torch.ops.aten.silu.out)

        logger.info("Exporting extend module...")
        extend_exported_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(extend_exported_program)

        model.set_prefill_mode(True)
        logger.info("Exporting extend module (prefill mode)...")
        prompt_exported_program = torch.export.export(
            model.extend,
            args=(),
            kwargs=forward_inputs,
            dynamic_shapes=forward_dynamic_shapes,
        ).run_decompositions(decomp_table)
        remove_functionalization(prompt_exported_program)

        logger.info("Exporting gather_embeddings module...")
        gather_embeddings_exported_program = torch.export.export(
            model.gather_embeddings,
            args=embed_tokens_inputs,
            dynamic_shapes=embed_tokens_dynamic_shapes,
        )

        logger.info("Exporting load_embeddings module...")
        load_embeddings_exported_program = torch.export.export(model.load_embeddings, args=tuple())

    return (
        extend_exported_program,
        prompt_exported_program,
        gather_embeddings_exported_program,
        load_embeddings_exported_program,
    )


async def _convert_to_coreai(
    extend_program: torch.export.ExportedProgram,
    prompt_program: torch.export.ExportedProgram,
    gather_embeddings_program: torch.export.ExportedProgram,
    load_embeddings_program: torch.export.ExportedProgram,
    max_context_length: int,
    kv_cached_embed_size: int,
    hidden_size: int,
    num_layers: int,
) -> AIProgram:
    """Convert exported programs to a single AIProgram with iOS constraints.

    This function:
    1. Adds all 4 exported programs to a TorchConverter
    2. Sets static shape configs for iOS shape specialization
    3. Sets hardware constraints (IOSurface allocations, interleave factors)
    4. Runs optimization and resolve-llo-mapped-composites pass
    """
    converter = TorchConverter()
    register_custom_torch_lowering(converter)

    converter.add_exported_program(
        load_embeddings_program,
        input_names=[],
        output_names=[LOAD_EMBEDDINGS_OUTPUT_NAME],
        entrypoint_name=LOAD_EMBEDDINGS_FUNCTION_NAME,
    )

    converter.add_exported_program(
        gather_embeddings_program,
        input_names=[TOKEN_IDS_INPUT_NAME, EMBEDDING_TABLE_INPUT_NAME],
        output_names=[GATHERED_EMBEDDINGS_OUTPUT_NAME],
        entrypoint_name=GATHER_EMBEDDINGS_FUNCTION_NAME,
    )

    input_names = [
        TRANSFORMER_INPUT_NAME,
        POSITION_IDS_INPUT_NAME,
        IN_STEP_INPUT_NAME,
        CAUSAL_MASK_INPUT_NAME,
        EMBEDDING_TABLE_INPUT_NAME,
    ]
    state_names = [
        KEY_CACHE_INPUT_NAME,
        VALUE_CACHE_INPUT_NAME,
    ]
    output_names = [
        OUTPUT_LOGITS_NAME,
    ]

    converter.add_exported_program(
        extend_program,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
        entrypoint_name=EXTEND_FUNCTION_NAME,
    )
    converter.add_exported_program(
        prompt_program,
        input_names=input_names,
        state_names=state_names,
        output_names=output_names,
        entrypoint_name=PROMPT_OPT_FUNCTION_NAME,
    )

    coreai_program: AIProgram = converter.to_coreai()

    # ----- Static shape configs for iOS specialization -----
    query_lengths = [8, 16, 64]

    gather_static_cfg: dict[str, dict[str, tuple[int, ...]]] = {}
    for q_len in query_lengths:
        gather_static_cfg[f'"{q_len}"'] = {TOKEN_IDS_INPUT_NAME: (1, q_len)}

    forward_static_cfg: dict[str, dict[str, tuple[int, ...]]] = {}
    cache_len = 256
    while cache_len <= max_context_length:
        for q_len in query_lengths:
            forward_static_cfg[f'"{cache_len}_{q_len}"'] = {
                TRANSFORMER_INPUT_NAME: (1, q_len, 1, hidden_size),
                POSITION_IDS_INPUT_NAME: (1, q_len),
                CAUSAL_MASK_INPUT_NAME: (1, cache_len, 1, q_len),
                KEY_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
                VALUE_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
            }
        cache_len *= 2

    coreai_program.set_static_shape_config(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_static_cfg)
    coreai_program.set_static_shape_config(EXTEND_FUNCTION_NAME, forward_static_cfg)
    coreai_program.set_static_shape_config(PROMPT_OPT_FUNCTION_NAME, forward_static_cfg)

    # ----- Hardware constraints -----
    emb_table_constraints = HardwareConstraints(
        AllocationType.IOSurface, interleave=[8, 1, 1], alignments=[1, 1, 1, 1]
    )
    cache_constraints = HardwareConstraints(
        AllocationType.IOSurface,
        interleave=[1, 1, KV_CACHE_INTERLEAVE_FACTOR, 1, 1],
        alignments=[1, 1, 1, 1, KV_CACHE_INTERLEAVE_FACTOR * max_context_length, 1],
    )

    gather_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}
    forward_constraints = {
        EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints,
        KEY_CACHE_INPUT_NAME: cache_constraints,
        KEY_CACHE_OUTPUT_NAME: cache_constraints,
        VALUE_CACHE_INPUT_NAME: cache_constraints,
        VALUE_CACHE_OUTPUT_NAME: cache_constraints,
    }
    load_constraints = {EMBEDDING_TABLE_INPUT_NAME: emb_table_constraints}

    coreai_program.set_hardware_constraints(LOAD_EMBEDDINGS_FUNCTION_NAME, load_constraints)
    coreai_program.set_hardware_constraints(GATHER_EMBEDDINGS_FUNCTION_NAME, gather_constraints)
    coreai_program.set_hardware_constraints(EXTEND_FUNCTION_NAME, forward_constraints)
    coreai_program.set_hardware_constraints(PROMPT_OPT_FUNCTION_NAME, forward_constraints)

    logger.info("Applying optimization passes...")
    coreai_program.optimize()

    return coreai_program


async def export_ios_model(
    model: torch.nn.Module,
    config,
    export_config,
) -> AIProgram:
    """Export an iOS model to a AIProgram.

    This is the main entry point for iOS model export. It:
    1. Builds reference inputs for all 4 entrypoints
    2. Exports each entrypoint through torch.export
    3. Converts to a single multi-function AIProgram with iOS constraints

    Args:
        model: A loaded PyTorch iOS model (already in the correct dtype).
            Must have ``extend``, ``gather_embeddings``, ``load_embeddings``
            submodules and a ``set_prefill_mode`` method.
        config: HuggingFace model config.
        export_config: An ExportConfig instance.

    Returns:
        An optimized AIProgram with 4 entrypoints, static shape
        configs, and hardware constraints set for iOS.
    """
    max_context_length = getattr(export_config, "max_context_length", None)
    if max_context_length is None:
        max_context_length = getattr(config, "max_position_embeddings", 2048)

    vocab_size = config.vocab_size

    logger.info(
        f"Exporting iOS model (max_context_length={max_context_length}, vocab_size={vocab_size})"
    )

    # 1. Build reference inputs
    inputs = _build_ios_reference_inputs(model, config, max_context_length, vocab_size)

    # 2. Export 4 programs
    (
        extend_program,
        prompt_program,
        gather_program,
        load_program,
    ) = _export_ios_programs(model, inputs)

    # 3. Convert to Core AI with iOS constraints
    if hasattr(config, "head_dim") and isinstance(config.head_dim, int):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads
    coreai_program = await _convert_to_coreai(
        extend_program=extend_program,
        prompt_program=prompt_program,
        gather_embeddings_program=gather_program,
        load_embeddings_program=load_program,
        max_context_length=max_context_length,
        kv_cached_embed_size=config.num_key_value_heads * head_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
    )

    return coreai_program
