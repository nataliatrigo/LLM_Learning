#!/usr/bin/env python3
"""Regenerate REPORT.md from saved result tables without solving the DP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.outputs import read_config  # noqa: E402
from modular_dp.reporting import write_report  # noqa: E402


TABLE_NAMES = [
    "summary",
    "sensitivity_regressions",
    "discrete_convergence",
    "continuous_convergence",
    "continuous_sensitivity",
    "simulation_summary",
    "dynamic_methods_p0_timeseries",
    "dynamic_panel_summary",
    "dynamic_solver_checks",
    "observation_forgetting_convergence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "full.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config.resolve())
    results = EXPERIMENT_ROOT / "results" / str(config["profile"])
    tables = {}
    optional = {
        "dynamic_methods_p0_timeseries",
        "dynamic_panel_summary",
        "dynamic_solver_checks",
        "observation_forgetting_convergence",
    }
    for name in TABLE_NAMES:
        path = results / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        elif name not in optional:
            raise FileNotFoundError(path)
    write_report(EXPERIMENT_ROOT, config, tables)
    print(EXPERIMENT_ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
