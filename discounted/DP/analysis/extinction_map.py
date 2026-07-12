"""Verify empirical investment extinction for the exact discounted DP.

This script is intentionally analysis-only: it imports the existing DP solver,
solves policy slices with the solver's default economic parameters, and writes
figures/tables under ``discounted/DP/analysis/outputs``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/tmp") / "llm_learning_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discounted.DP.exact_dp import (  # noqa: E402
    ModelParams,
    solve_discounted_finite_horizon,
    state_count_for_period,
)


DEFAULT_BASE_HORIZON = 700
DEFAULT_BASE_POLICY_PERIOD = 50
DEFAULT_DIAGNOSTIC_P0 = 0.5
DEFAULT_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_TAIL_TOLERANCE = 1e-6

PRODUCT2_COLOR = "#0f766e"
P0_COLOR = "#475569"
C1_COLOR = "#2563eb"
C2_COLOR = "#dc6b19"
INK = "#111827"


@dataclass(frozen=True)
class Attempt:
    factor: int
    n_max: int
    policy_period: int
    horizon: int
    n_star_max: int
    product2_states: int
    interior: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build investment-extinction verification figures."
    )
    parser.add_argument("--p0", type=float, default=DEFAULT_DIAGNOSTIC_P0)
    parser.add_argument(
        "--base-policy-period",
        type=int,
        default=DEFAULT_BASE_POLICY_PERIOD,
        help="Starting policy slice; n_max is policy_period - 1.",
    )
    parser.add_argument("--base-horizon", type=int, default=DEFAULT_BASE_HORIZON)
    parser.add_argument(
        "--max-factor",
        type=int,
        default=64,
        help="Largest power-of-two expansion factor for n_max.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def solve_slice(
    p0: float,
    params: ModelParams,
    n_max: int,
    remaining_horizon: int,
) -> tuple[dict, pd.DataFrame, Attempt]:
    policy_period = n_max + 1
    horizon = policy_period + remaining_horizon
    solution = solve_discounted_finite_horizon(
        p0=p0,
        params=params,
        horizon=horizon,
        stored_policy_periods={policy_period},
        snapshot_periods=set(),
    )
    policy = solution["policy_by_period"][policy_period]
    discounted_gap = solution["discounted_gap_by_period"][policy_period]
    count = len(policy)
    state_space = solution["state_space"]
    successes = state_space.S[:count]
    failures = state_space.F[:count]
    total = successes + failures
    frame = pd.DataFrame(
        {
            "S": successes,
            "F": failures,
            "n": total,
            "m": (successes + 1.0) / (total + 2.0),
            "action": policy,
            "invest": policy == 2,
            "discounted_continuation_gap": discounted_gap,
        }
    )
    invested = frame["invest"].to_numpy()
    n_star_max = int(frame.loc[invested, "n"].max()) if invested.any() else -1
    attempt = Attempt(
        factor=-1,
        n_max=n_max,
        policy_period=policy_period,
        horizon=horizon,
        n_star_max=n_star_max,
        product2_states=int(invested.sum()),
        interior=n_star_max <= 0.8 * n_max,
    )
    return solution, frame, attempt


def solve_until_interior(
    p0: float,
    params: ModelParams,
    base_n_max: int,
    remaining_horizon: int,
    max_factor: int,
) -> tuple[dict, pd.DataFrame, list[Attempt]]:
    attempts: list[Attempt] = []
    factor = 1
    final_solution: dict | None = None
    final_frame: pd.DataFrame | None = None
    while factor <= max_factor:
        n_max = base_n_max * factor
        print(
            f"Solving p0={p0:.3f}, n_max={n_max}, factor={factor}...",
            flush=True,
        )
        solution, frame, attempt = solve_slice(
            p0,
            params,
            n_max,
            remaining_horizon,
        )
        attempt = Attempt(
            factor=factor,
            n_max=attempt.n_max,
            policy_period=attempt.policy_period,
            horizon=attempt.horizon,
            n_star_max=attempt.n_star_max,
            product2_states=attempt.product2_states,
            interior=attempt.interior,
        )
        attempts.append(attempt)
        final_solution = solution
        final_frame = frame
        if attempt.interior:
            break
        factor *= 2
    if final_solution is None or final_frame is None:
        raise RuntimeError("No grid was solved.")
    return final_solution, final_frame, attempts


def band_summary(frame: pd.DataFrame, p0: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for n_value, layer in frame.groupby("n", sort=True):
        invested = layer[layer["invest"]].sort_values("S")
        if invested.empty:
            continue
        successes = invested["S"].to_numpy(dtype=int)
        product2_runs = int(1 + np.sum(np.diff(successes) > 1))
        m_lower = float(invested["m"].min())
        m_upper = float(invested["m"].max())
        sd = math.sqrt(p0 * (1.0 - p0) / (float(n_value) + 2.0))
        c_required = max(abs(m_lower - p0), abs(m_upper - p0)) / sd
        rows.append(
            {
                "n": int(n_value),
                "product2_nodes": int(len(invested)),
                "product2_runs": product2_runs,
                "is_band": product2_runs <= 1,
                "m_lower": m_lower,
                "m_upper": m_upper,
                "m_width": m_upper - m_lower,
                "c_required": c_required,
                "contained_in_c2_collar": c_required <= 2.0 + 1e-12,
            }
        )
    return pd.DataFrame(rows)


def premium_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("n", as_index=False)["discounted_continuation_gap"]
        .max()
        .rename(columns={"discounted_continuation_gap": "G"})
    )


def theoretical_bound(p0: float, params: ModelParams) -> tuple[float, float]:
    dc = params.c2 - params.c1
    dp = params.p2 - params.p1
    kappa = 1.0 / math.sqrt(2.0 * math.pi * p0 * (1.0 - p0))
    n_bar = (
        params.gamma
        * kappa
        * (params.revenue - params.c1)
        * dp
        / (((1.0 - params.gamma) ** 3) * dc)
    ) ** 2 - 3.0
    return kappa, n_bar


def plot_investment_map(
    frame: pd.DataFrame,
    p0: float,
    n_star_max: int,
    n_max: int,
    path: Path,
) -> None:
    configure_plot_style()
    invested = frame[frame["invest"]]
    x_max = n_max if n_star_max < 0 else max(10.0, 1.3 * n_star_max)
    x_max = min(float(n_max), x_max)
    n_grid = np.linspace(0.0, max(x_max, 1.0), 600)
    sd = np.sqrt(p0 * (1.0 - p0) / (n_grid + 2.0))

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.scatter(
        invested["n"],
        invested["m"],
        s=8,
        color=PRODUCT2_COLOR,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        label="Investment states, x*=2",
    )
    ax.axhline(p0, color=P0_COLOR, linestyle="--", linewidth=1.3, label="m=p0")
    for c_value, color in ((1, C1_COLOR), (2, C2_COLOR)):
        lower = np.clip(p0 - c_value * sd, 0.0, 1.0)
        upper = np.clip(p0 + c_value * sd, 0.0, 1.0)
        label = f"c={c_value} collar"
        ax.plot(n_grid, lower, color=color, linewidth=1.2, label=label)
        ax.plot(n_grid, upper, color=color, linewidth=1.2)
    if n_star_max >= 0:
        ax.axvline(
            n_star_max,
            color=INK,
            linestyle="--",
            linewidth=1.2,
            label="empirical extinction",
        )
        ax.annotate(
            "empirical extinction",
            xy=(n_star_max, 0.96),
            xytext=(5, -8),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Observations, n=S+F")
    ax.set_ylabel("Posterior mean, m=(S+1)/(n+2)")
    ax.set_title("Empirical investment region in fluid coordinates", loc="left")
    ax.grid(True, alpha=0.45)
    ax.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_premium_decay(
    premium: pd.DataFrame,
    threshold: float,
    n_star_max: int,
    path: Path,
) -> float:
    configure_plot_style()
    usable = premium[(premium["n"] >= 1) & (premium["G"] > 0.0)].copy()
    slope = float("nan")
    if len(usable) >= 2:
        slope, _ = np.polyfit(
            np.log(usable["n"].to_numpy(dtype=float)),
            np.log(usable["G"].to_numpy(dtype=float)),
            1,
        )

    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    ax.loglog(
        usable["n"],
        usable["G"],
        color=PRODUCT2_COLOR,
        linewidth=2.0,
        label="G(n)=max diagonal premium",
    )
    if len(usable):
        target = max(1, n_star_max // 2) if n_star_max > 0 else 1
        anchor_idx = int((usable["n"] - target).abs().idxmin())
        anchor_n = float(usable.loc[anchor_idx, "n"])
        anchor_g = float(usable.loc[anchor_idx, "G"])
        ref_n = usable["n"].to_numpy(dtype=float)
        ref_g = anchor_g * (ref_n / anchor_n) ** (-0.5)
        ax.loglog(
            ref_n,
            ref_g,
            color="#64748b",
            linestyle=":",
            linewidth=1.8,
            label="reference slope -1/2",
        )
    ax.axhline(
        threshold,
        color=INK,
        linestyle="--",
        linewidth=1.2,
        label="dc/dp threshold",
    )
    ax.set_xlabel("Observations, n")
    ax.set_ylabel("Discounted continuation premium G(n)")
    ax.set_title("Premium decay versus investment threshold", loc="left")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return slope


def format_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def write_summary(
    path: Path,
    p0: float,
    params: ModelParams,
    attempts: list[Attempt],
    bands: pd.DataFrame,
    premium_slope: float,
    kappa: float,
    n_bar: float,
    map_path: Path,
    premium_path: Path,
) -> None:
    final = attempts[-1]
    dc = params.c2 - params.c1
    dp = params.p2 - params.p1
    threshold = dc / dp
    ratio = final.n_star_max / n_bar if n_bar > 0 and final.n_star_max >= 0 else float("nan")
    containment_fraction = (
        float(bands["contained_in_c2_collar"].mean()) if len(bands) else float("nan")
    )
    min_c_all = float(bands["c_required"].max()) if len(bands) else float("nan")
    non_band = bands[~bands["is_band"]] if len(bands) else pd.DataFrame()

    attempt_rows = [
        "| factor | n_max | policy_period | horizon | n_star_max | x*=2 states | interior? |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for attempt in attempts:
        attempt_rows.append(
            "| "
            f"{attempt.factor} | {attempt.n_max} | {attempt.policy_period} | "
            f"{attempt.horizon} | {attempt.n_star_max} | "
            f"{attempt.product2_states} | {'yes' if attempt.interior else 'NO'} |"
        )

    warning_lines: list[str] = []
    if not final.interior:
        warning_lines.append(
            "**WARNING: the investment set is still not interior on the largest grid.**"
        )
    if len(non_band):
        sample = ", ".join(str(int(n)) for n in non_band["n"].head(12))
        warning_lines.append(
            "**WARNING: investment set is NOT A BAND on "
            f"{len(non_band)} diagonals. First affected n values: {sample}.**"
        )
    else:
        warning_lines.append(
            "Band check: every diagonal with investment is a single interval."
        )

    lines = [
        "# Investment Extinction Verification",
        "",
        "## 1. Solver Inspection Reported First",
        "",
        "Source inspected: `discounted/DP/exact_dp.py` and `discounted/DP/main.py`.",
        "",
        "Solver parameter values found and used:",
        "",
        "```text",
        (
            "ModelParams("
            f"p1={params.p1}, p2={params.p2}, c1={params.c1}, "
            f"c2={params.c2}, revenue={params.revenue}, "
            f"gamma={params.gamma}, tol={params.tol}, "
            f"demand_floor={params.demand_floor})"
        ),
        f"p0 = {p0}",
        f"DEFAULT_P0_GRID = {DEFAULT_P0_GRID}",
        f"main.py default T / horizon = {DEFAULT_BASE_HORIZON}",
        f"main.py default policy_period = {DEFAULT_BASE_POLICY_PERIOD}",
        f"main.py default diagnostic_p0 = {DEFAULT_DIAGNOSTIC_P0}",
        f"main.py default tail_tolerance = {DEFAULT_TAIL_TOLERANCE}",
        "```",
        "",
        "Policy/value storage found:",
        "",
        "- The solver builds `StateSpace(horizon, S, F, total, success_index, failure_index)`.",
        "- It uses rolling arrays `next_value` and `current_value`; the full value table is not returned.",
        "- Requested cuts are returned in `policy_by_period`, `raw_gap_by_period`, and `discounted_gap_by_period`.",
        "- `discounted_gap_by_period[t]` stores `gamma * [V(S+1,F)-V(S,F+1)]` for the requested period.",
        "- For a stored period `t`, `state_count_for_period(t)=t*(t+1)//2`, so the slice covers `n=S+F=0,...,t-1`.",
        (
            f"- Therefore the inspected default policy slice has "
            f"`n_max={DEFAULT_BASE_POLICY_PERIOD - 1}`."
        ),
        "",
        "## 2. Extinction Check",
        "",
        *attempt_rows,
        "",
        f"Grid used for figures: `n_max={final.n_max}`, `policy_period={final.policy_period}`, `horizon={final.horizon}`.",
        f"`n_star_max = {final.n_star_max}`.",
        (
            "Interior check: "
            f"{final.n_star_max} <= 0.8 * {final.n_max} = {0.8 * final.n_max:.1f} "
            f"-> {'PASS' if final.interior else 'FAIL'}."
        ),
        "",
        "## 3. Band And Collar Containment",
        "",
        *warning_lines,
        "",
        f"Diagonals with investment: {len(bands)}.",
        f"Fraction contained in the c=2 collar: {format_float(containment_fraction)}.",
        f"Smallest c containing all observed investment bands: {format_float(min_c_all)}.",
        "",
        "## 4. Bound Comparison",
        "",
        f"`dc = c2-c1 = {format_float(dc)}`.",
        f"`dp = p2-p1 = {format_float(dp)}`.",
        f"`dc/dp = {format_float(threshold)}`.",
        f"`kappa = 1/sqrt(2*pi*p0*(1-p0)) = {format_float(kappa)}`.",
        f"`n_bar = {format_float(n_bar)}`.",
        f"`n_star_max / n_bar = {format_float(ratio)}`.",
        "",
        "## 5. Premium Decay",
        "",
        f"Log-log fitted slope for positive `G(n)` values: {format_float(premium_slope)}.",
        "",
        "## 6. Figures And Tables",
        "",
        f"- Investment map: `{map_path.name}`",
        f"- Premium decay: `{premium_path.name}`",
        "- Band endpoints: `investment_band_endpoints.csv`",
        "- Premium series: `premium_decay.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.base_policy_period <= 1:
        raise ValueError("base-policy-period must exceed 1.")
    if args.base_horizon <= args.base_policy_period:
        raise ValueError("base-horizon must exceed base-policy-period.")
    if args.max_factor < 1:
        raise ValueError("max-factor must be positive.")

    outputs_dir = args.outputs_dir.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    params = ModelParams()
    base_n_max = args.base_policy_period - 1
    remaining_horizon = args.base_horizon - args.base_policy_period
    solution, frame, attempts = solve_until_interior(
        p0=args.p0,
        params=params,
        base_n_max=base_n_max,
        remaining_horizon=remaining_horizon,
        max_factor=args.max_factor,
    )
    del solution

    final = attempts[-1]
    bands = band_summary(frame, args.p0)
    premium = premium_summary(frame)
    dc = params.c2 - params.c1
    dp = params.p2 - params.p1
    threshold = dc / dp
    kappa, n_bar = theoretical_bound(args.p0, params)

    bands.to_csv(outputs_dir / "investment_band_endpoints.csv", index=False)
    premium.to_csv(outputs_dir / "premium_decay.csv", index=False)

    map_path = outputs_dir / "investment_extinction_map.png"
    premium_path = outputs_dir / "premium_decay.png"
    plot_investment_map(
        frame,
        args.p0,
        final.n_star_max,
        final.n_max,
        map_path,
    )
    slope = plot_premium_decay(premium, threshold, final.n_star_max, premium_path)

    summary_path = outputs_dir / "extinction_summary.md"
    write_summary(
        summary_path,
        args.p0,
        params,
        attempts,
        bands,
        slope,
        kappa,
        n_bar,
        map_path,
        premium_path,
    )
    print(f"Saved extinction analysis under: {outputs_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
