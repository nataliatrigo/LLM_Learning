"""Sweep gamma and measure the empirical investment-extinction diagonal.

The analysis reuses the exact discounted DP solver and the band-containment
helpers from ``extinction_map.py``.  It does not modify the solver or the
single-gamma extinction analysis.
"""

from __future__ import annotations

import argparse
import json
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
ANALYSIS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from discounted.DP.exact_dp import (  # noqa: E402
    ModelParams,
    required_horizon,
)
from extinction_map import (  # noqa: E402
    band_summary,
    configure_plot_style,
    format_float,
    solve_slice,
)


GAMMAS = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)
P0 = 0.5
BASE_SCALE = 500.0
REFERENCE_ONE_MINUS_GAMMA = 0.02
MIN_N_MAX = 200
MAX_N_MAX = 6000
MAX_DOUBLINGS = 2
TAIL_TOLERANCE = 1e-6

PRODUCT2_COLOR = "#0f766e"
N_BAR_COLOR = "#9333ea"
REFERENCE_COLOR = "#64748b"
INK = "#111827"


@dataclass(frozen=True)
class OLSFit:
    intercept: float
    beta: float
    beta_se: float
    r2: float
    n_obs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep gamma for empirical investment extinction."
    )
    parser.add_argument(
        "--gammas",
        default=",".join(f"{g:g}" for g in GAMMAS),
        help="Comma-separated gamma values.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=ANALYSIS_DIR / "outputs_gamma_sweep")
    parser.add_argument("--max-n-max", type=int, default=MAX_N_MAX)
    return parser.parse_args()


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one gamma.")
    return values


def starting_n_max(gamma: float) -> int:
    proposed = math.ceil(
        BASE_SCALE * (REFERENCE_ONE_MINUS_GAMMA / (1.0 - gamma)) ** 2
    )
    return max(MIN_N_MAX, proposed)


def theoretical_bound(gamma: float, params: ModelParams) -> tuple[float, float]:
    dc = params.c2 - params.c1
    dp = params.p2 - params.p1
    kappa0 = (
        3.0
        * math.exp(1.0 / 12.0)
        / math.sqrt(2.0 * math.pi * P0 * (1.0 - P0))
    )
    n_bar = (
        gamma
        * kappa0
        * (params.revenue - params.c1)
        * dp
        / (((1.0 - gamma) ** 3) * dc)
    ) ** 2 - 3.0
    return kappa0, n_bar


def attempt_summary(attempts: list[dict[str, object]]) -> str:
    return "; ".join(
        (
            f"n_max={a['n_max']}, n_star={a['n_star_max']}, "
            f"interior={a['interior']}"
        )
        for a in attempts
    )


def solve_gamma(gamma: float, max_n_max: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    params = ModelParams(gamma=gamma)
    remaining_horizon = required_horizon(gamma, TAIL_TOLERANCE)
    n_max = starting_n_max(gamma)
    attempts: list[dict[str, object]] = []
    final_frame: pd.DataFrame | None = None
    final_attempt = None
    complete = False
    stopped_reason = ""

    for doubling in range(MAX_DOUBLINGS + 1):
        if n_max > max_n_max:
            stopped_reason = f"next n_max={n_max} exceeds cap {max_n_max}"
            break
        print(
            f"Solving gamma={gamma:.2f}, n_max={n_max}, "
            f"remaining_horizon={remaining_horizon}...",
            flush=True,
        )
        _, frame, attempt = solve_slice(
            p0=P0,
            params=params,
            n_max=n_max,
            remaining_horizon=remaining_horizon,
        )
        record = {
            "gamma": gamma,
            "doubling": doubling,
            "n_max": attempt.n_max,
            "policy_period": attempt.policy_period,
            "horizon": attempt.horizon,
            "remaining_horizon": remaining_horizon,
            "n_star_max": attempt.n_star_max,
            "investment_states": attempt.product2_states,
            "interior": bool(attempt.interior),
        }
        attempts.append(record)
        final_frame = frame
        final_attempt = attempt
        if attempt.interior:
            complete = True
            break
        if doubling == MAX_DOUBLINGS:
            stopped_reason = "failed interiority after the allowed doublings"
            break
        n_max *= 2

    kappa0, n_bar = theoretical_bound(gamma, params)
    if final_frame is not None and final_attempt is not None:
        bands = band_summary(final_frame, P0)
        has_bands = len(bands) > 0
        non_band = bands[~bands["is_band"]] if has_bands else pd.DataFrame()
        min_c_all = float(bands["c_required"].max()) if has_bands else float("nan")
        containment_c2 = (
            float(bands["contained_in_c2_collar"].mean())
            if has_bands
            else float("nan")
        )
        n_star_max = final_attempt.n_star_max
        investment_states = final_attempt.product2_states
        n_max_used = final_attempt.n_max
        policy_period = final_attempt.policy_period
        horizon = final_attempt.horizon
        band_diagonals = int(len(bands))
        band_violations = int(len(non_band))
        first_band_violations = (
            ",".join(str(int(n)) for n in non_band["n"].head(12))
            if band_violations
            else ""
        )
    else:
        min_c_all = float("nan")
        containment_c2 = float("nan")
        n_star_max = -1
        investment_states = 0
        n_max_used = n_max
        policy_period = n_max + 1
        horizon = policy_period + remaining_horizon
        band_diagonals = 0
        band_violations = 0
        first_band_violations = ""

    row: dict[str, object] = {
        "gamma": gamma,
        "one_minus_gamma": 1.0 - gamma,
        "inv_one_minus_gamma": 1.0 / (1.0 - gamma),
        "start_n_max": starting_n_max(gamma),
        "n_max_used": n_max_used,
        "policy_period": policy_period,
        "remaining_horizon": remaining_horizon,
        "horizon": horizon,
        "complete": complete,
        "stopped_reason": stopped_reason,
        "n_star_max": n_star_max,
        "investment_states": investment_states,
        "band_diagonals": band_diagonals,
        "band_violations": band_violations,
        "first_band_violations": first_band_violations,
        "min_c_all": min_c_all,
        "c2_containment_fraction": containment_c2,
        "kappa0": kappa0,
        "n_bar": n_bar,
        "n_bar_over_n_star": (
            n_bar / n_star_max
            if n_star_max and n_star_max > 0 and np.isfinite(n_bar)
            else float("nan")
        ),
        "attempts": json.dumps(attempts),
        "attempts_summary": attempt_summary(attempts),
        "solver_method": "exact finite-horizon backward induction",
        "iterations_used": "not_applicable",
        "residual": "not_applicable",
        "residual_flag": "not_applicable_no_value_iteration",
        "tol": params.tol,
        "p0": P0,
        "p1": params.p1,
        "p2": params.p2,
        "c1": params.c1,
        "c2": params.c2,
        "revenue": params.revenue,
    }
    return row, attempts


def ols_fit(frame: pd.DataFrame) -> OLSFit:
    if len(frame) < 3:
        return OLSFit(float("nan"), float("nan"), float("nan"), float("nan"), len(frame))
    x = np.log(frame["inv_one_minus_gamma"].to_numpy(dtype=float))
    y = np.log(frame["n_star_max"].to_numpy(dtype=float))
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - beta * x_mean
    fitted = intercept + beta * x
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y_mean) ** 2))
    sigma2 = sse / (len(frame) - 2)
    beta_se = math.sqrt(sigma2 / sxx)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return OLSFit(intercept, beta, beta_se, r2, len(frame))


def valid_fit_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        summary["complete"].eq(True)
        & summary["n_star_max"].gt(0)
        & np.isfinite(summary["n_star_max"])
    ].copy()


def plot_scaling(summary: pd.DataFrame, fit: OLSFit, path: Path) -> None:
    configure_plot_style()
    valid = valid_fit_rows(summary)
    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    x = valid["inv_one_minus_gamma"].to_numpy(dtype=float)
    y = valid["n_star_max"].to_numpy(dtype=float)
    ax.loglog(
        x,
        y,
        marker="o",
        linewidth=0,
        color=PRODUCT2_COLOR,
        label="empirical n_star_max",
    )
    for _, row in valid.iterrows():
        ax.annotate(
            f"{row['gamma']:.2f}",
            (row["inv_one_minus_gamma"], row["n_star_max"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    if len(valid) >= 2 and np.isfinite(fit.beta):
        x_grid = np.linspace(float(x.min()), float(x.max()), 200)
        y_fit = np.exp(fit.intercept) * x_grid**fit.beta
        ax.loglog(
            x_grid,
            y_fit,
            color=INK,
            linewidth=1.7,
            label=f"OLS slope={fit.beta:.3g}",
        )
        anchor_x = float(np.median(x))
        anchor_y = float(np.exp(fit.intercept) * anchor_x**fit.beta)
        y_ref = anchor_y * (x_grid / anchor_x) ** 2
        ax.loglog(
            x_grid,
            y_ref,
            color=REFERENCE_COLOR,
            linestyle=":",
            linewidth=1.8,
            label="reference slope 2",
        )
    ok_nbar = summary[
        summary["complete"].eq(True)
        & summary["n_bar"].gt(0)
        & np.isfinite(summary["n_bar"])
    ]
    ax.loglog(
        ok_nbar["inv_one_minus_gamma"],
        ok_nbar["n_bar"],
        marker="^",
        linewidth=0,
        markerfacecolor="none",
        markeredgecolor=N_BAR_COLOR,
        markersize=7,
        label="theoretical n_bar",
    )
    ax.set_xlabel("1 / (1-gamma)")
    ax.set_ylabel("extinction diagonal")
    ax.set_title("Scaling of empirical extinction with patience", loc="left")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap(summary: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    data = summary[
        summary["complete"].eq(True)
        & summary["n_bar_over_n_star"].gt(0)
        & np.isfinite(summary["n_bar_over_n_star"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.plot(
        data["gamma"],
        data["n_bar_over_n_star"],
        marker="o",
        color=N_BAR_COLOR,
        linewidth=1.8,
    )
    ax.set_yscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("n_bar / n_star_max")
    ax.set_title("Looseness of theoretical extinction bound", loc="left")
    ax.grid(True, which="both", alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row[column]) for column in headers)
            + " |"
        )
    return lines


def write_summary(
    path: Path,
    summary: pd.DataFrame,
    attempts: pd.DataFrame,
    fit_all: OLSFit,
    fit_excl_090: OLSFit,
) -> None:
    valid = valid_fit_rows(summary)
    beta_p1_pass = np.isfinite(fit_all.beta) and 1.6 <= fit_all.beta <= 2.4
    gamma_090 = summary[np.isclose(summary["gamma"], 0.90)]
    if len(gamma_090) and bool(gamma_090.iloc[0]["complete"]):
        n090 = float(gamma_090.iloc[0]["n_star_max"])
        p2_pass = 30.0 <= n090 <= 40.0
        p2_text = f"n_star_max(0.90)={n090:g}"
    else:
        p2_pass = False
        p2_text = "gamma=0.90 incomplete"
    c_values = valid["min_c_all"].dropna().to_numpy(dtype=float)
    if len(c_values):
        c_cv = float(np.std(c_values) / np.mean(c_values))
        p3_pass = c_cv <= 0.25
        p3_text = (
            f"min_c_all range [{c_values.min():.3g}, {c_values.max():.3g}], "
            f"CV={c_cv:.3g}"
        )
    else:
        p3_pass = False
        p3_text = "no completed c values"

    flags: list[str] = []
    incomplete = summary[~summary["complete"]]
    if len(incomplete):
        flags.append(
            "Incomplete gammas: "
            + ", ".join(f"{g:.2f}" for g in incomplete["gamma"])
            + "."
        )
    non_band = summary[summary["band_violations"].gt(0)]
    if len(non_band):
        flags.append(
            "SINGLE-INTERVAL VIOLATIONS: "
            + "; ".join(
                (
                    f"gamma={row.gamma:.2f}: {int(row.band_violations)} "
                    f"diagonals ({row.first_band_violations})"
                )
                for row in non_band.itertuples()
            )
            + "."
        )
    else:
        flags.append("Single-interval check: no violations in completed solves.")
    flags.append(
        "Tolerance/residual: exact_dp.py uses exact finite-horizon backward "
        "induction, not value iteration; iterations and residual are not applicable."
    )

    table_cols = [
        "gamma",
        "n_max_used",
        "horizon",
        "complete",
        "n_star_max",
        "investment_states",
        "min_c_all",
        "n_bar_over_n_star",
        "band_violations",
    ]
    table = summary[table_cols].copy()
    table["min_c_all"] = table["min_c_all"].map(lambda x: format_float(float(x)))
    table["n_bar_over_n_star"] = table["n_bar_over_n_star"].map(
        lambda x: format_float(float(x))
    )

    attempt_lines = [
        "| gamma | doubling | n_max | horizon | n_star_max | investment states | interior? |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in attempts.itertuples():
        attempt_lines.append(
            f"| {row.gamma:.2f} | {row.doubling} | {row.n_max} | "
            f"{row.horizon} | {row.n_star_max} | {row.investment_states} | "
            f"{'yes' if row.interior else 'NO'} |"
        )

    lines = [
        "# Gamma Sweep: Investment Extinction Scaling",
        "",
        "## Fits",
        "",
        (
            "All completed gammas: "
            f"beta={format_float(fit_all.beta)}, "
            f"SE={format_float(fit_all.beta_se)}, "
            f"R^2={format_float(fit_all.r2)}, n={fit_all.n_obs}."
        ),
        (
            "Excluding gamma=0.90: "
            f"beta={format_float(fit_excl_090.beta)}, "
            f"SE={format_float(fit_excl_090.beta_se)}, "
            f"R^2={format_float(fit_excl_090.r2)}, n={fit_excl_090.n_obs}."
        ),
        "",
        "## Predictions",
        "",
        f"- P1 beta in [1.6, 2.4]: **{pass_fail(beta_p1_pass)}**.",
        f"- P2 n_star_max at gamma=0.90 approximately 30-40: **{pass_fail(p2_pass)}** ({p2_text}).",
        f"- P3 containment constant c roughly stable across gamma: **{pass_fail(p3_pass)}** ({p3_text}).",
        "",
        "## Per-Gamma Table",
        "",
        *markdown_table(table),
        "",
        "## Attempt Log",
        "",
        *attempt_lines,
        "",
        "## Flags",
        "",
        *[f"- {flag}" for flag in flags],
        "",
        "## Takeaway",
        "",
        f"1. The empirical extinction exponent is about {format_float(fit_all.beta)} on the completed sweep.",
        f"2. The theoretical bound remains very loose: n_bar/n_star ranges from {format_float(float(valid['n_bar_over_n_star'].min()))} to {format_float(float(valid['n_bar_over_n_star'].max()))} on completed runs.",
        f"3. Collar localization is fairly stable only if the reported c-range/CV is accepted: {p3_text}.",
        "",
        "## Outputs",
        "",
        "- `gamma_sweep.csv`",
        "- `gamma_sweep_attempts.csv`",
        "- `scaling_plot.png`",
        "- `gap_plot.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    gammas = parse_float_list(args.gammas)
    outputs_dir = args.outputs_dir.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_attempts: list[dict[str, object]] = []
    for gamma in gammas:
        row, attempts = solve_gamma(gamma, args.max_n_max)
        rows.append(row)
        all_attempts.extend(attempts)

    summary = pd.DataFrame(rows)
    attempts = pd.DataFrame(all_attempts)
    summary.to_csv(outputs_dir / "gamma_sweep.csv", index=False)
    attempts.to_csv(outputs_dir / "gamma_sweep_attempts.csv", index=False)

    fit_data = valid_fit_rows(summary)
    fit_all = ols_fit(fit_data)
    fit_excl_090 = ols_fit(fit_data[~np.isclose(fit_data["gamma"], 0.90)])

    plot_scaling(summary, fit_all, outputs_dir / "scaling_plot.png")
    plot_gap(summary, outputs_dir / "gap_plot.png")
    write_summary(
        outputs_dir / "SUMMARY.md",
        summary,
        attempts,
        fit_all,
        fit_excl_090,
    )
    print(f"Saved gamma sweep under: {outputs_dir}", flush=True)


if __name__ == "__main__":
    main()
