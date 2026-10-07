# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import re

from coreai.authoring import AIProgram
from coreai.compiler.ir import Type as MLIRType


def parse_coreai_type(
    mlir_type: MLIRType,
) -> tuple[tuple[int, ...], str]:
    # temporarily parse shape from str
    # TODO: migrate once Core AI adds APIs to retrieve shape info from Type object
    mlir_type_str = str(mlir_type)
    # Type string is something like "!coreai.handle<tensor<18x1x4x2048x256xf16>>", so our steps are
    # 1. extract "tensor< dim0 x dim1 x ...x dtype >"
    match = re.search(r"tensor<[a-zA-Z0-9\?]*>", mlir_type_str)
    assert match
    # 2. further extract "dim0 x dim1 x ...x dtype"
    shape_dtype_str = match.group(0)[7:-1]
    # 3. split by 'x' to get [dim0, dim1, ..., dtype]
    shape_dtype_str_list = shape_dtype_str.split("x")
    shape = tuple([-1 if dim == "?" else int(dim) for dim in shape_dtype_str_list[:-1]])
    dtype = shape_dtype_str_list[-1]
    return shape, dtype


def stringify_mlir_graph(
    coreai_program: AIProgram,
) -> str:
    full_mlir_text = str(coreai_program)
    mlir_text_lines = full_mlir_text.split("\n")
    result_text = ""
    for line in mlir_text_lines:
        if "dialect_resources" in line:
            break
        result_text += line + "\n"
    return result_text
