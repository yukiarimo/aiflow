# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end iOS model conversion tests.

The autouse ``use_hf_impl`` fixture comes from the directory ``conftest.py``.
"""

import pytest
import torch
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM as HFMistralForCausalLM,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
)

from coreai_models.models.ios.mistral import (
    MistralForCausalLMForiOS as MistralForCausalLM,
)
from coreai_models.models.ios.qwen2 import (
    Qwen2ForCausalLMForiOS as Qwen2ForCausalLM,
)
from coreai_models.models.ios.qwen3 import (
    Qwen3ForCausalLMForiOS as Qwen3ForCausalLM,
)
from tests._runner_infra.testing_utils import (
    load_state_dict_from_ref_model,
    run_torch_prompt_extend_test_ios,
)


class TestQwen2EndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen2.5-tiny-random",
            pytest.param("Qwen/Qwen2.5-0.5B", marks=pytest.mark.slow),
            pytest.param("Qwen/Qwen2.5-1.5B-Instruct", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_hf(hf_model_id: str, precision: torch.dtype) -> None:
        """Test model comparison with prompt / extend."""
        ref_model = HFQwen2ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = 1000
        model = Qwen2ForCausalLM(config, "cpu", disable_embedding_quantization=True).eval()
        load_state_dict_from_ref_model(model, ref_model)
        run_torch_prompt_extend_test_ios(
            model,
            ref_model,
            precision=precision,
            rtol=1e2,
            strict_compare_numerical=(precision == torch.float32),
            use_additional_transpose=True,
        )


class TestQwen3EndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/qwen3-tiny-random",
            pytest.param("Qwen/Qwen3-0.6B", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_hf(hf_model_id: str, precision: torch.dtype) -> None:
        """Test model comparison with prompt / extend."""
        ref_model = HFQwen3ForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = 1000
        model = Qwen3ForCausalLM(config, "cpu", disable_embedding_quantization=True).eval()
        load_state_dict_from_ref_model(model, ref_model)
        run_torch_prompt_extend_test_ios(
            model,
            ref_model,
            precision=precision,
            rtol=1e2,
            strict_compare_numerical=(precision == torch.float32),
            use_additional_transpose=True,
        )


class TestMistralEndtoEnd:
    @staticmethod
    @pytest.mark.parametrize(
        "hf_model_id",
        [
            "yujiepan/mistral-v0.3-tiny-random",
        ],
    )
    @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
    def test_hf(hf_model_id: str, precision: torch.dtype) -> None:
        """Test model comparison with prompt / extend."""
        ref_model = HFMistralForCausalLM.from_pretrained(hf_model_id).eval()
        config = ref_model.config
        config.max_position_embeddings = 1000
        model = MistralForCausalLM(config, "cpu", disable_embedding_quantization=True).eval()
        load_state_dict_from_ref_model(model, ref_model)
        run_torch_prompt_extend_test_ios(
            model,
            ref_model,
            precision=precision,
            rtol=1e2,
            strict_compare_numerical=(precision == torch.float32),
            use_additional_transpose=True,
        )
