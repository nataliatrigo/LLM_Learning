#!/usr/bin/env python3
"""Generate the dynamic Monte Carlo panels and tidy confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.dynamic_panels import run_dynamic_panel_experiment  # noqa: E402
from modular_dp.outputs import ExperimentPaths, read_config  # noqa: E402
from modular_dp.reporting import write_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "quick.json",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="regenerate REPORT.md by loading the profile's saved tables",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "reuse compatible saved method-p0 series and solve only newly "
            "configured specifications"
        ),
    )
    return parser.parse_args()


def _saved_report_tables(results: Path) -> dict:
    import pandas as pd

    names = [
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
    return {
        name: pd.read_csv(results / f"{name}.csv")
        for name in names
        if (results / f"{name}.csv").exists()
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config.resolve())
    profile = str(config["profile"])
    paths = ExperimentPaths(EXPERIMENT_ROOT, profile)
    tables = run_dynamic_panel_experiment(
        config,
        paths,
        reuse_existing=args.reuse_existing,
    )
    if not tables:
        print(f"Dynamic panels are disabled for profile {profile!r}.")
        return
    if args.update_report:
        write_report(
            EXPERIMENT_ROOT,
            config,
            _saved_report_tables(paths.results),
        )
    print(paths.results / "dynamic_panels_manifest.json")
    first_p0 = min(float(value) for value in config["dynamic_panels"]["p0_values"])
    p0_tag = f"{first_p0:g}".replace("-", "m").replace(".", "p")
    print(paths.figures / "dynamic" / f"13_dynamic_p0_{p0_tag}_across_methods.png")
    print(
        paths.figures
        / "dynamic"
        / "focused"
        / f"18_dynamic_p0_{p0_tag}_focused_methods.png"
    )


if __name__ == "__main__":
    main()
