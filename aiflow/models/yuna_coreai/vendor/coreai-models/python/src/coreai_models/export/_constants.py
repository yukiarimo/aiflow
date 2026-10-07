# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constants for the export pipeline."""

# KV cache names used by the Swift runner
KEY_CACHE_NAME = "keyCache"
VALUE_CACHE_NAME = "valueCache"

# Trace-time KV cache sequence length. Used only for export/quantization tracing
# to bound peak memory; at inference the actual cache size is determined
# dynamically.
TRACE_KV_CACHE_SEQ_LEN = 2048

# Trace-time `input_ids` length and `position_ids` offset for export/quantization
QUANT_TRACE_QUERY_LEN = 16
QUANT_TRACE_OFFSET = 8
