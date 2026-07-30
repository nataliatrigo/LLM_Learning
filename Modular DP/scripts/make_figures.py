#!/usr/bin/env python3
"""Regenerate figures from saved tidy result tables without solving the DP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.plotting import make_all_figures  # noqa: E402
from modular_dp.dynamic_panels import make_dynamic_panel_figures  # noqa: E402


TABLE_NAMES = [
    "summary",
    "demand_sensitivity",
    "sensitivity_regressions",
    "discrete_policy_points",
    "continuous_policy_points",
    "simulation_summary",
    "representative_trajectories",
    "dynamic_methods_p0_timeseries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = EXPERIMENT_ROOT / "results" / args.profile
    tables = {}
    for name in TABLE_NAMES:
        candidates = [results / f"{name}.csv", results / f"{name}.csv.gz"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is not None:
            tables[name] = pd.read_csv(path)
    make_all_figures(tables, EXPERIMENT_ROOT / "figures" / args.profile)
    if "dynamic_methods_p0_timeseries" in tables:
        make_dynamic_panel_figures(
            tables["dynamic_methods_p0_timeseries"],
            EXPERIMENT_ROOT / "figures" / args.profile,
            preliminary=args.profile != "full",
        )
    print(EXPERIMENT_ROOT / "figures" / args.profile)


if __name__ == "__main__":
    main()
