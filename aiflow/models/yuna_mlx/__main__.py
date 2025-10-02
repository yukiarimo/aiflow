#!/usr/bin/env python
import importlib
import sys

def main():
    subcommands = {
        "generate",
        "convert",
        "train",
    }

    if len(sys.argv) < 2:
        print(f"Usage: python -m yuna_mlx <subcommand>")
        print(f"Available subcommands: {list(subcommands)}")
        sys.exit(1)

    subcommand = sys.argv.pop(1)
    if subcommand not in subcommands:
        raise ValueError(f"Invalid subcommand. Available: {list(subcommands)}")

    submodule = importlib.import_module(f"yuna_mlx.{subcommand}")
    submodule.main()

if __name__ == "__main__":
    main()