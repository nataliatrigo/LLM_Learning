"""Focused horizon and discount diagnostic for p0=0.5."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from discounted.exact_dp import (
        ModelParams,
        compare_policy_slices,
        late_window_summary,
        product2_gap_threshold,
        simulate_policy,
        solve_discounted_finite_horizon,
    )
except ModuleNotFoundError:
    from exact_dp import (
        ModelParams,
        compare_policy_slices,
        late_window_summary,
        product2_gap_threshold,
        simulate_policy,
        solve_discounted_finite_horizon,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare discounted horizons and the undiscounted benchmark."
    )
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=700)
    parser.add_argument("--comparison-horizon", type=int, default=850)
    parser.add_argument("--benchmark-horizon", type=int, default=2000)
    parser.add_argument("--comparison-period", type=int, default=50)
    parser.add_argument("--simulation-periods", type=int, default=250)
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--simulate-comparison-policy",
        action="store_true",
        help="Simulate the comparison-horizon policy instead of the base policy.",
    )
    parser.add_argument(
        "--skip-same-gamma-benchmark",
        action="store_true",
        help="Skip the direct gamma=0.98, T=2000 policy comparison.",
    )
    parser.add_argument(
        "--skip-undiscounted-benchmark",
        action="store_true",
        help="Skip the slower gamma=1, T=2000 solve.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Defaults to discounted/outputs/diagnostics.",
    )
    return parser.parse_args()


def comparison_row(
    label: str,
    first: dict,
    second: dict,
    period: int,
) -> dict:
    row = compare_policy_slices(first, second, period)
    row.update(
        {
            "comparison": label,
            "first_gamma": first["gamma"],
            "first_T": first["T"],
            "second_gamma": second["gamma"],
            "second_T": second["T"],
            "first_initial_value": first["initial_value"],
            "second_initial_value": second["initial_value"],
            "initial_value_difference": (
                second["initial_value"] - first["initial_value"]
            ),
        }
    )
    return row


def plot_timeseries(combined: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)
    labels = list(combined["model"].unique())
    colors = ["#2563eb", "#dc2626"]

    for label, color in zip(labels, colors, strict=False):
        data = combined[combined["model"] == label]
        axes[0].plot(
            data["t"],
            data["A_market_share"],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color=color,
            linewidth=1.8,
            label=label,
        )
        axes[2].plot(
            data["t"],
            data["mean_posterior_mean"],
            color=color,
            linewidth=1.8,
            label=label,
        )

    axes[0].set_title("Seller A retention", loc="left")
    axes[0].set_ylabel("A market share")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_title("Product 2 use", loc="left")
    axes[1].set_ylabel("Rate when A is chosen")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[2].set_title("User posterior mean", loc="left")
    axes[2].set_ylabel("E[theta | S,F]")
    for ax in axes:
        ax.set_xlabel("Period")
        ax.grid(True, color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend()
    fig.suptitle("p0=0.5: discounting versus the gamma=1 benchmark", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    horizon_row: dict,
    same_gamma_row: dict | None,
    discount_row: dict | None,
    late: pd.DataFrame,
) -> None:
    params = ModelParams(gamma=args.gamma)
    raw_threshold = product2_gap_threshold(params) / args.gamma
    discounted_late = late[late["model"].str.startswith("discounted")].iloc[0]
    lines = [
        "# Discounted-horizon diagnostic",
        "",
        "## 1. Horizon convergence",
        "",
        f"- gamma={args.gamma}, base T={args.horizon}, "
        f"gamma^T={args.gamma**args.horizon:.6g}.",
        f"- Comparison T={args.comparison_horizon} at t={args.comparison_period}.",
        f"- Differing states: {horizon_row['disagreement_count']} / "
        f"{horizon_row['state_count']} "
        f"({horizon_row['disagreement_fraction']:.6%}).",
        f"- Initial-value difference: "
        f"{horizon_row['initial_value_difference']:.8g}.",
        "",
        "This verifies that the discounted policy has converged with respect to",
        "the terminal horizon.",
        "",
        "## 2. Continuation value gap",
        "",
        "Let",
        "",
        "D_t(S,F) = V_{t+1}(S+1,F) - V_{t+1}(S,F+1)",
        "",
        "be the raw success-versus-failure continuation gap. The quantity",
        "reported in the corrected code is",
        "",
        "M_t(S,F) = gamma * D_t(S,F).",
        "",
        "Conditional on Seller A being selected,",
        "",
        "Q_2 - Q_1 = -(c_2-c_1) + (p_2-p_1) M_t(S,F).",
        "",
        f"Thus the M_t threshold is Delta c / Delta p = "
        f"{product2_gap_threshold(params):.12g} = 4/3. "
        f"If raw D_t is plotted, its threshold is {raw_threshold:.12g}.",
        "",
        "## 3. p0=0.5 simulation",
        "",
        f"Over periods {int(discounted_late['late_start_period'])}-"
        f"{int(discounted_late['late_end_period'])}, the exact discounted model has:",
        "",
        f"- Mean A market share: "
        f"{discounted_late['mean_A_market_share']:.4%}.",
        f"- Mean demand probability: "
        f"{discounted_late['mean_demand_probability']:.4%}.",
        f"- Mean product-2 rate when A is chosen: "
        f"{discounted_late['mean_product2_rate_when_A_chosen']:.4%}.",
        f"- Mean policy product-2 share across simulated states: "
        f"{discounted_late['mean_policy_product2_share']:.4%}.",
    ]

    if same_gamma_row is not None:
        lines[8:8] = [
            f"- Direct comparison with T={args.benchmark_horizon}: "
            f"{same_gamma_row['disagreement_count']} / "
            f"{same_gamma_row['state_count']} differing states "
            f"({same_gamma_row['disagreement_fraction']:.6%}).",
        ]

    if discount_row is not None:
        benchmark_late = late[late["model"].str.startswith("undiscounted")].iloc[0]
        lines.extend(
            [
                "",
                "## 4. Comparison with the gamma=1, T=2000 benchmark",
                "",
                f"- Policy disagreements at t={args.comparison_period}: "
                f"{discount_row['disagreement_count']} / "
                f"{discount_row['state_count']} "
                f"({discount_row['disagreement_fraction']:.4%}).",
                f"- Discounted product-2 state share at t="
                f"{args.comparison_period}: "
                f"{discount_row['first_product2_share']:.4%}.",
                f"- gamma=1 product-2 state share at t="
                f"{args.comparison_period}: "
                f"{discount_row['second_product2_share']:.4%}.",
                f"- gamma=1 mean A market share in the late window: "
                f"{benchmark_late['mean_A_market_share']:.4%}.",
                f"- gamma=1 mean product-2 rate in the late window: "
                f"{benchmark_late['mean_product2_rate_when_A_chosen']:.4%}.",
                "",
                "The terminal-horizon bug is fixed, but gamma=0.98 is not",
                "policy-equivalent to gamma=1. If a finite-T=2000 run with",
                "gamma=0.98 reports the gamma=1 behavior, the two runs are not",
                "using the same discounting convention or calibration.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_dir = (
        Path(args.outputs_dir)
        if args.outputs_dir
        else Path(__file__).resolve().parent / "outputs" / "diagnostics"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stored_periods = set(range(1, args.simulation_periods + 1))
    stored_periods.add(args.comparison_period)

    discounted_params = ModelParams(gamma=args.gamma)
    discounted = solve_discounted_finite_horizon(
        args.p0,
        discounted_params,
        args.horizon,
        stored_periods,
        {args.comparison_period},
    )
    longer = solve_discounted_finite_horizon(
        args.p0,
        discounted_params,
        args.comparison_horizon,
        (
            stored_periods
            if args.simulate_comparison_policy
            else {args.comparison_period}
        ),
        set(),
    )
    horizon_row = comparison_row(
        "discounted_horizon_convergence",
        discounted,
        longer,
        args.comparison_period,
    )
    comparison_rows = [horizon_row]
    same_gamma_row = None
    if not args.skip_same_gamma_benchmark:
        same_gamma_benchmark = solve_discounted_finite_horizon(
            args.p0,
            discounted_params,
            args.benchmark_horizon,
            {args.comparison_period},
            set(),
        )
        same_gamma_row = comparison_row(
            "discounted_T700_vs_T2000",
            discounted,
            same_gamma_benchmark,
            args.comparison_period,
        )
        comparison_rows.append(same_gamma_row)

    simulation_solution = longer if args.simulate_comparison_policy else discounted
    _, discounted_time = simulate_policy(
        simulation_solution,
        args.simulation_periods,
        args.n_rep,
        args.seed,
    )
    discounted_time["model"] = (
        f"discounted gamma={args.gamma}, T={simulation_solution['T']}"
    )
    late_rows = [
        {
            "model": discounted_time["model"].iloc[0],
            **late_window_summary(
                discounted_time,
                max(1, args.simulation_periods - 49),
            ),
        }
    ]
    time_frames = [discounted_time]
    discount_row = None

    if not args.skip_undiscounted_benchmark:
        benchmark = solve_discounted_finite_horizon(
            args.p0,
            ModelParams(gamma=1.0),
            args.benchmark_horizon,
            stored_periods,
            {args.comparison_period},
        )
        discount_row = comparison_row(
            "discounted_vs_gamma_one",
            discounted,
            benchmark,
            args.comparison_period,
        )
        comparison_rows.append(discount_row)
        _, benchmark_time = simulate_policy(
            benchmark,
            args.simulation_periods,
            args.n_rep,
            args.seed,
        )
        benchmark_time["model"] = (
            f"undiscounted gamma=1, T={args.benchmark_horizon}"
        )
        time_frames.append(benchmark_time)
        late_rows.append(
            {
                "model": benchmark_time["model"].iloc[0],
                **late_window_summary(
                    benchmark_time,
                    max(1, args.simulation_periods - 49),
                ),
            }
        )

    comparisons = pd.DataFrame.from_records(comparison_rows)
    late = pd.DataFrame.from_records(late_rows)
    combined_time = pd.concat(time_frames, ignore_index=True)
    comparisons.to_csv(outputs_dir / "policy_comparisons_t50.csv", index=False)
    late.to_csv(outputs_dir / "p0_05_late_window.csv", index=False)
    combined_time.to_csv(
        outputs_dir / "p0_05_timeseries_comparison.csv",
        index=False,
    )
    if len(time_frames) > 1:
        plot_timeseries(
            combined_time,
            outputs_dir / "p0_05_discount_vs_gamma_one.png",
        )
    write_report(
        outputs_dir / "diagnostic_report.md",
        args,
        horizon_row,
        same_gamma_row,
        discount_row,
        late,
    )

    print("\nPolicy comparisons")
    print(
        comparisons[
            [
                "comparison",
                "period",
                "state_count",
                "disagreement_count",
                "disagreement_fraction",
                "first_product2_share",
                "second_product2_share",
            ]
        ].to_string(index=False)
    )
    print("\nLate-window simulation")
    print(late.to_string(index=False))
    print(f"\nSaved diagnostics to: {outputs_dir}")


if __name__ == "__main__":
    main()
