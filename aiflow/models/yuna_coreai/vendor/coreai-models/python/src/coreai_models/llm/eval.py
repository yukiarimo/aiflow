# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Evaluate LLM models using lm-eval-harness.

CLI: coreai.llm.eval
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.llm.eval",
        description="Evaluate a Core AI LLM against standard benchmarks",
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or path to model bundle"
    )
    parser.add_argument("--tasks", nargs="+", help="Evaluation tasks (e.g. tinyMMLU tinyGSM8k)")
    return parser


def main() -> None:
    parser = build_parser()
    _args = parser.parse_args()
    parser.error(
        "Evaluation support is coming soon. See models/README.md for current capabilities."
    )


if __name__ == "__main__":
    main()
