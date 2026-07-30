#!/usr/bin/env python3
"""Plot late product-2 cycles for two saved representative sample paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/modular_dp_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--paths", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--start", type=int, default=4000)
    parser.add_argument("--end", type=int, default=5000)
    return parser.parse_args()


def cycle_summary(path: pd.DataFrame) -> dict[str, float | int]:
    prescribed = path["product2_prescribed"].astype(bool).to_numpy()
    used_periods = path.loc[path["product2_used"].astype(bool), "period"].to_numpy()
    gaps = np.diff(used_periods)
    return {
        "path": int(path["representative_path"].iloc[0]),
        "window_start": int(path["period"].min()),
        "window_end": int(path["period"].max()),
        "product2_prescribed_periods": int(prescribed.sum()),
        "product2_uses": int(len(used_periods)),
        "prescription_switches": int(np.count_nonzero(np.diff(prescribed.astype(int)))),
        "mean_gap_between_uses": float(gaps.mean()) if len(gaps) else np.nan,
        "max_gap_between_uses": int(gaps.max()) if len(gaps) else np.nan,
    }


def _method_specs() -> list[tuple[str, float | None, str]]:
    """One representative specification from each learning-rule family."""
    return [
        ("standard_ts", None, "Standard TS"),
        ("scaled_updates", 0.5, "Scaled, eta=0.5"),
        (
            "mean_preserving_temperature",
            2.0,
            "Temperature, T=2",
        ),
        ("epsilon_greedy", 0.1, "Epsilon-greedy, eps=0.1"),
        (
            "observation_forgetting_ts",
            0.95,
            "Observation forgetting, rho=0.95",
        ),
        (
            "calendar_forgetting_ts",
            0.95,
            "Calendar forgetting, rho=0.95",
        ),
    ]


def plot_method_comparison(
    trajectories: pd.DataFrame,
    *,
    paths: tuple[int, int] | list[int],
    start: int,
    end: int,
    p0: float,
    output: Path,
    stem_name: str,
) -> pd.DataFrame:
    """Plot prescription bands and realized uses for all method families."""
    specs = _method_specs()
    fig, axes = plt.subplots(
        len(specs), 2, figsize=(13.2, 12.8), sharex=True, sharey=True
    )
    summaries: list[dict[str, float | int | str]] = []
    for row, (method, parameter, label) in enumerate(specs):
        method_data = trajectories[trajectories["method"] == method]
        if parameter is not None:
            method_data = method_data[
                np.isclose(method_data["parameter_value"], parameter)
            ]
        method_data = method_data[
            method_data["representative_path"].isin(paths)
            & method_data["period"].between(start, end)
        ]
        for column, path_id in enumerate(paths):
            ax = axes[row, column]
            path = method_data[
                method_data["representative_path"] == path_id
            ].sort_values("period")
            if path.empty:
                raise ValueError(f"missing {method}, parameter={parameter}, path={path_id}")
            summary = cycle_summary(path)
            summary.update(
                {
                    "method": method,
                    "parameter_value": parameter,
                    "p0": p0,
                }
            )
            summaries.append(summary)
            periods = path["period"].to_numpy()
            prescribed = path["product2_prescribed"].astype(bool).to_numpy()
            ax.fill_between(
                periods,
                0,
                prescribed.astype(float),
                step="mid",
                color="#99f6e4",
                alpha=0.9,
            )
            used = path[path["product2_used"].astype(bool)]
            ax.scatter(
                used["period"],
                np.full(len(used), 0.5),
                s=6,
                color="#be123c",
                marker="|",
                linewidths=0.7,
                zorder=3,
            )
            if column == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == 0:
                ax.set_title(f"Sample path {path_id}")
            if row == len(specs) - 1:
                ax.set_xlabel("Calendar period")
            ax.set(ylim=(-0.03, 1.03), yticks=[0, 1])

    window_label = "Full-horizon" if start == 1 else "Late"
    fig.suptitle(
        rf"{window_label} product-2 use across methods at $p_0={p0:g}$",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.985,
        "Aqua: prescribed   |   Red: actually used",
        ha="right",
        va="top",
        color="#475569",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    stem = output / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    return summary_frame


def main() -> None:
    args = parse_args()
    config_path = EXPERIMENT_ROOT / "configs" / f"{args.profile}.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    saved_p0 = float(config.get("seller", {}).get("p0", 0.5))
    if not np.isclose(saved_p0, args.p0, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"profile {args.profile!r} was generated at p0={saved_p0:g}, not {args.p0:g}"
        )

    source = EXPERIMENT_ROOT / "results" / args.profile / "representative_trajectories.csv.gz"
    if not source.exists():
        source = source.with_suffix("")
    trajectories = pd.read_csv(source)
    selected = trajectories[
        (trajectories["method"] == "observation_forgetting_ts")
        & np.isclose(trajectories["parameter_value"], args.rho)
        & trajectories["representative_path"].isin(args.paths)
        & trajectories["period"].between(args.start, args.end)
    ].copy()
    missing = set(args.paths) - set(selected["representative_path"].unique())
    if missing:
        raise ValueError(f"sample paths not found: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 6.6), sharex=True)
    summaries: list[dict[str, float | int]] = []
    for column, path_id in enumerate(args.paths):
        path = selected[selected["representative_path"] == path_id].sort_values("period")
        summaries.append(cycle_summary(path))
        top, bottom = axes[:, column]

        top.plot(path["period"], path["posterior_mean"], color="#3b1b73", lw=1.0)
        top.axhline(args.p0, color="#64748b", ls="--", lw=1.0, label=rf"$p_0={args.p0:g}$")
        top.set(title=f"Sample path {path_id}", ylabel="Posterior mean", ylim=(0, 1))
        top.legend(loc="upper right")

        prescribed = path["product2_prescribed"].astype(bool).to_numpy()
        periods = path["period"].to_numpy()
        bottom.fill_between(
            periods,
            0,
            prescribed.astype(float),
            step="mid",
            color="#99f6e4",
            alpha=0.9,
            label="Product 2 prescribed",
        )
        used = path[path["product2_used"].astype(bool)]
        bottom.scatter(
            used["period"],
            np.full(len(used), 0.55),
            s=7,
            color="#be123c",
            marker="|",
            linewidths=0.8,
            label="Product 2 actually used",
            zorder=3,
        )
        bottom.set(xlabel="Calendar period", ylabel="Product 2", ylim=(-0.03, 1.03), yticks=[0, 1])
        if column == 0:
            bottom.legend(loc="upper left", ncol=2, fontsize=8)

    fig.suptitle(
        rf"Late product-2 cycles at $p_0={args.p0:g}$: observation-time forgetting ($\rho={args.rho:g}$)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = EXPERIMENT_ROOT / "figures" / args.profile / "simulation"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "26_product2_sample_paths_p0_0p5"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    plot_method_comparison(
        trajectories,
        paths=args.paths,
        start=args.start,
        end=args.end,
        p0=args.p0,
        output=output,
        stem_name="27_product2_sample_paths_all_methods_p0_0p5",
    )
    plot_method_comparison(
        trajectories,
        paths=args.paths,
        start=1,
        end=int(trajectories["period"].max()),
        p0=args.p0,
        output=output,
        stem_name="28_product2_sample_paths_full_horizon_p0_0p5",
    )
    print(stem)


if __name__ == "__main__":
    main()
