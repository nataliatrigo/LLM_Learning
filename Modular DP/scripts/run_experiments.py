#!/usr/bin/env python3
"""Run a validated modular discounted-DP experiment profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.outputs import ExperimentPaths, read_config  # noqa: E402
from modular_dp.pipeline import run_experiment  # noqa: E402
from modular_dp.dynamic_panels import run_dynamic_panel_experiment  # noqa: E402
from modular_dp.reporting import write_report  # noqa: E402
from modular_dp.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "quick.json",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config.resolve())
    # The validation suite is intentionally rerun before every experiment.
    # It is fast relative to the sweep and prevents a stale passing manifest
    # from surviving changes to solvers, tests, or validation configuration.
    validation_config = read_config(args.validation_config.resolve())
    validation_manifest = validate_all(
        validation_config,
        ExperimentPaths(EXPERIMENT_ROOT, "validation"),
    )
    profile = str(config["profile"])
    paths = ExperimentPaths(EXPERIMENT_ROOT, profile)
    tables = run_experiment(
        config,
        paths,
        validation_manifest=validation_manifest,
    )
    tables.update(run_dynamic_panel_experiment(config, paths))
    write_report(EXPERIMENT_ROOT, config, tables)
    print(f"Completed profile {profile!r}.")
    print(paths.results / "run_manifest.json")
    print(EXPERIMENT_ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
