#!/usr/bin/env python3
"""Run the mandatory baseline and solver-family validation gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.outputs import ExperimentPaths, read_config  # noqa: E402
from modular_dp.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "validation.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config.resolve())
    paths = ExperimentPaths(EXPERIMENT_ROOT, "validation")
    manifest = validate_all(config, paths, verbosity=0 if args.quiet else 1)
    print(
        f"Validation PASS: {manifest['unit_tests']} unit tests and "
        f"{manifest['checks']} numerical checks."
    )
    print(paths.results / "VALIDATION.md")


if __name__ == "__main__":
    main()
