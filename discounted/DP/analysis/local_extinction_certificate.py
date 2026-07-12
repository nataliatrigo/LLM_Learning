"""Compute the local extinction certificate on truncated triangular grids.

This script implements the certificate from the "Local extinction certificate"
proposition without editing the paper. It writes plots, CSV diagnostics, and a
short report under ``discounted/DP/analysis/outputs_local_certificate``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc, betaincc, gammaln

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discounted.DP.exact_dp import (  # noqa: E402
    ModelParams,
    solve_discounted_finite_horizon,
)


VALID_TERMINALS = ("crude", "lemma2", "unrolled", "selected")
ALL_TERMINALS = VALID_TERMINALS + ("optimistic_zero",)

CERTIFIED_COLOR = "#2563eb"
NONCERT_COLOR = "#e5e7eb"
INVEST_COLOR = "#b91c1c"
INK = "#111827"


@dataclass(frozen=True)
class Params:
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.98
    p0: float = 0.50

    @property
    def delta_p(self) -> float:
        return self.p2 - self.p1

    @property
    def delta_c(self) -> float:
        return self.c2 - self.c1

    @property
    def threshold(self) -> float:
        return self.delta_c / self.delta_p

    @property
    def cheap_margin(self) -> float:
        return self.revenue - self.c1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local extinction certificates on triangular grids."
    )
    parser.add_argument(
        "--n-grid",
        default="800,1200,1600,2400",
        help="Comma-separated list of N values for S+F <= N.",
    )
    parser.add_argument(
        "--diagnostic-n-grid",
        default="800,1200,1600,2400",
        help="Comma-separated list of N values for state-level diagnostic plots.",
    )
    parser.add_argument(
        "--plot-n",
        type=int,
        default=800,
        help="Grid size used for state-level plots and empirical comparison.",
    )
    parser.add_argument(
        "--empirical-n",
        type=int,
        default=800,
        help="Grid size for the empirical Bellman slice comparison.",
    )
    parser.add_argument(
        "--remaining-horizon",
        type=int,
        default=650,
        help="Remaining finite-horizon tail used by the existing DP solver.",
    )
    parser.add_argument(
        "--identity-max-n",
        type=int,
        default=1000,
        help="Largest diagonal n used to verify the Lemma 1 identity.",
    )
    parser.add_argument(
        "--tail-tol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for the unrolled terminal tail contribution.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_local_certificate",
    )
    return parser.parse_args()


def parse_n_grid(value: str) -> list[int]:
    grid = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not grid or min(grid) < 1:
        raise ValueError("--n-grid must contain positive integers.")
    return grid


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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def demand_diag(n: int, params: Params) -> np.ndarray:
    """Return D(S,n-S) for S=0,...,n."""
    successes = np.arange(n + 1, dtype=np.float64)
    failures = n - successes
    return np.clip(
        betaincc(successes + 1.0, failures + 1.0, params.p0),
        0.0,
        1.0,
    )


def ell(demand: np.ndarray | float, params: Params) -> np.ndarray | float:
    return 1.0 - params.gamma + params.gamma * demand


def binom_pmf(k: np.ndarray | int, n: int, p: float) -> np.ndarray | float:
    k_arr = np.asarray(k)
    log_pmf = (
        gammaln(n + 1.0)
        - gammaln(k_arr + 1.0)
        - gammaln(n - k_arr + 1.0)
        + k_arr * math.log(p)
        + (n - k_arr) * math.log1p(-p)
    )
    values = np.exp(log_pmf)
    if np.isscalar(k):
        return float(values)
    return values


def binom_max_pmf(record_diagonal: int, params: Params) -> float:
    trials = record_diagonal + 2
    mode = int(math.floor((trials + 1) * params.p0))
    candidates = np.array(
        [max(0, min(trials, mode)), max(0, min(trials, mode - 1))],
        dtype=np.int64,
    )
    return float(np.max(binom_pmf(candidates, trials, params.p0)))


def kappa0(params: Params) -> float:
    return 3.0 * math.exp(1.0 / 12.0) / math.sqrt(
        2.0 * math.pi * params.p0 * (1.0 - params.p0)
    )


def theorem1_bound(params: Params) -> float:
    return (
        params.gamma
        * kappa0(params)
        * params.cheap_margin
        * params.delta_p
        / (((1.0 - params.gamma) ** 3) * params.delta_c)
    ) ** 2


def verify_demand_identity(max_n: int, params: Params) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n in range(max_n + 1):
        d_next = demand_diag(n + 1, params)
        demand_diff = d_next[1:] - d_next[:-1]
        successes = np.arange(n + 1, dtype=np.int64)
        pmf = binom_pmf(successes + 1, n + 2, params.p0)
        abs_err = np.abs(demand_diff - pmf)
        rel_err = abs_err / np.maximum(pmf, 1e-300)
        significant = pmf >= 1e-12
        rows.append(
            {
                "n": n,
                "states": n + 1,
                "max_abs_error": float(abs_err.max()),
                "max_rel_error": float(rel_err.max()),
                "max_rel_error_pmf_ge_1e_minus_12": (
                    float(rel_err[significant].max()) if significant.any() else np.nan
                ),
                "min_demand_diff": float(demand_diff.min()),
                "max_demand_diff": float(demand_diff.max()),
            }
        )
    return pd.DataFrame(rows)


def verify_demand_survival(max_n: int, params: Params) -> pd.DataFrame:
    max_abs_survival_error = 0.0
    max_abs_distance_from_cdf = 0.0
    min_demand = 1.0
    max_demand = 0.0
    checked_states = 0
    for n in range(max_n + 1):
        successes = np.arange(n + 1, dtype=np.float64)
        failures = n - successes
        demand = demand_diag(n, params)
        survival = betaincc(successes + 1.0, failures + 1.0, params.p0)
        cdf_below = betainc(successes + 1.0, failures + 1.0, params.p0)
        max_abs_survival_error = max(
            max_abs_survival_error,
            float(np.max(np.abs(demand - survival))),
        )
        max_abs_distance_from_cdf = max(
            max_abs_distance_from_cdf,
            float(np.max(np.abs(demand - cdf_below))),
        )
        min_demand = min(min_demand, float(demand.min()))
        max_demand = max(max_demand, float(demand.max()))
        checked_states += n + 1

    if max_abs_survival_error > 1e-12:
        raise AssertionError("Demand is not equal to the Beta survival function.")
    if max_abs_distance_from_cdf < 1e-8:
        raise AssertionError("Demand is indistinguishable from the lower-tail CDF.")

    return pd.DataFrame(
        [
            {
                "checked_diagonals": max_n + 1,
                "checked_states": checked_states,
                "max_abs_survival_error": max_abs_survival_error,
                "max_abs_distance_from_cdf_below_p0": max_abs_distance_from_cdf,
                "min_D": min_demand,
                "max_D": max_demand,
            }
        ]
    )


class TerminalBounds:
    def __init__(self, params: Params, tail_tol: float) -> None:
        self.params = params
        self.tail_tol = tail_tol

    @lru_cache(maxsize=None)
    def g(self, record_diagonal: int) -> float:
        return binom_max_pmf(record_diagonal, self.params)

    def crude(self, record_diagonal: int) -> float:
        del record_diagonal
        return self.params.cheap_margin / (1.0 - self.params.gamma)

    def lemma2(self, record_diagonal: int) -> float:
        return (
            self.params.cheap_margin
            * kappa0(self.params)
            / (((1.0 - self.params.gamma) ** 3) * math.sqrt(record_diagonal + 2.0))
        )

    def unrolled(self, record_diagonal: int) -> tuple[float, int, float]:
        total = 0.0
        gamma_power = 1.0
        t = 0
        scale = self.params.cheap_margin / ((1.0 - self.params.gamma) ** 2)
        while True:
            total += gamma_power * self.g(record_diagonal + t)
            t += 1
            gamma_power *= self.params.gamma
            tail_sum_bound = (
                kappa0(self.params)
                * gamma_power
                / ((1.0 - self.params.gamma) * math.sqrt(record_diagonal + t + 2.0))
            )
            tail_value_bound = scale * tail_sum_bound
            if tail_value_bound <= self.tail_tol:
                break
            if t > 250_000:
                raise RuntimeError("Unrolled terminal tail did not converge.")
        value = min(self.crude(record_diagonal), scale * total)
        return value, t, tail_value_bound

    def all_bounds(self, record_diagonal: int) -> dict[str, float]:
        crude = self.crude(record_diagonal)
        lemma2 = self.lemma2(record_diagonal)
        unrolled, _, _ = self.unrolled(record_diagonal)
        selected = min(crude, lemma2, unrolled)
        return {
            "crude": crude,
            "lemma2": lemma2,
            "unrolled": unrolled,
            "selected": selected,
            "optimistic_zero": 0.0,
        }


def summarize_certified_diagonals(
    all_certified_by_n: list[bool],
    any_noncert_by_n: list[bool],
) -> tuple[int, int | None]:
    noncert = [n for n, has_noncert in enumerate(any_noncert_by_n) if has_noncert]
    largest_noncert = max(noncert) if noncert else -1
    extinction_diagonal = None if largest_noncert >= len(all_certified_by_n) - 1 else largest_noncert + 1
    return largest_noncert, extinction_diagonal


def compute_certificate_grid(
    N: int,
    params: Params,
    bounds: TerminalBounds,
    store_states: bool = False,
    store_kind: str = "selected",
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    terminal_diagonal = N + 1
    terminal_info = bounds.all_bounds(terminal_diagonal)
    terminal_rows: list[dict[str, float | int | str]] = []
    unrolled_value, unrolled_terms, unrolled_tail = bounds.unrolled(terminal_diagonal)
    for kind in ALL_TERMINALS:
        terminal_rows.append(
            {
                "N": N,
                "terminal_diagonal": terminal_diagonal,
                "terminal_kind": kind,
                "terminal_bound": terminal_info[kind],
                "unrolled_terms": unrolled_terms if kind == "unrolled" else np.nan,
                "unrolled_tail_bound": unrolled_tail if kind == "unrolled" else np.nan,
                "is_valid_certificate": kind != "optimistic_zero",
            }
        )

    if any(terminal_info[kind] < 0.0 for kind in VALID_TERMINALS):
        raise AssertionError("A valid terminal bound is negative.")

    U_next = {
        kind: np.full(N + 2, terminal_info[kind], dtype=np.float64)
        for kind in ALL_TERMINALS
    }
    any_noncert_by_kind = {kind: [False] * (N + 1) for kind in ALL_TERMINALS}
    all_cert_by_kind = {kind: [False] * (N + 1) for kind in ALL_TERMINALS}
    certified_count_by_kind = {kind: 0 for kind in ALL_TERMINALS}
    max_margin_by_kind = {kind: -np.inf for kind in ALL_TERMINALS}
    min_margin_by_kind = {kind: np.inf for kind in ALL_TERMINALS}
    max_u_by_kind = {kind: -np.inf for kind in ALL_TERMINALS}
    min_D = 1.0
    max_D = 0.0
    min_ell = np.inf
    min_A = np.inf
    max_A = -np.inf
    min_B = np.inf
    max_B = -np.inf
    B_lower_violations = 0
    B_upper_violations = 0
    state_chunks: list[pd.DataFrame] = []

    for n in range(N, -1, -1):
        d_next = demand_diag(n + 1, params)
        if np.any((d_next < -1e-14) | (d_next > 1.0 + 1e-14)):
            raise AssertionError(f"Demand outside [0,1] on diagonal {n+1}.")
        min_D = min(min_D, float(d_next.min()))
        max_D = max(max_D, float(d_next.max()))
        d_success = d_next[1:]
        d_failure = d_next[:-1]
        ell_success = ell(d_success, params)
        ell_failure = ell(d_failure, params)
        if np.any(ell_success <= 0.0) or np.any(ell_failure <= 0.0):
            raise AssertionError(f"Nonpositive ell(D) on diagonal {n}.")
        min_ell = min(min_ell, float(ell_success.min()), float(ell_failure.min()))
        # Evaluate the beta-survival increment directly.  Subtracting the two
        # survival probabilities loses all significant digits in the tails.
        demand_diff = np.asarray(
            binom_pmf(np.arange(n + 1, dtype=np.int64) + 1, n + 2, params.p0)
        )
        if np.any(demand_diff <= 0.0) and n <= 1000:
            raise AssertionError(f"Nonpositive demand increment on diagonal {n}.")
        A = params.cheap_margin * demand_diff / (ell_success * ell_failure)
        B = params.gamma * d_failure / ell_failure
        B_lower_violations += int(np.sum(B < -1e-12))
        B_upper_violations += int(np.sum(B > params.gamma + 1e-12))
        if B_lower_violations or B_upper_violations:
            raise AssertionError(f"B outside [0,gamma] on diagonal {n}.")
        min_A = min(min_A, float(A.min()))
        max_A = max(max_A, float(A.max()))
        min_B = min(min_B, float(B.min()))
        max_B = max(max_B, float(B.max()))

        current_by_kind: dict[str, np.ndarray] = {}
        for kind in ALL_TERMINALS:
            next_values = U_next[kind]
            cont_1 = params.p1 * next_values[1:] + (1.0 - params.p1) * next_values[:-1]
            cont_2 = params.p2 * next_values[1:] + (1.0 - params.p2) * next_values[:-1]
            U = A + B * np.maximum(cont_1, cont_2)
            if np.min(U) < -1e-10:
                raise AssertionError(f"Negative U for {kind} on diagonal {n}.")
            U = np.maximum(U, 0.0)
            margin = params.gamma * U - params.threshold
            certified = margin < 0.0
            any_noncert_by_kind[kind][n] = bool((~certified).any())
            all_cert_by_kind[kind][n] = bool(certified.all())
            certified_count_by_kind[kind] += int(certified.sum())
            max_margin_by_kind[kind] = max(max_margin_by_kind[kind], float(margin.max()))
            min_margin_by_kind[kind] = min(min_margin_by_kind[kind], float(margin.min()))
            max_u_by_kind[kind] = max(max_u_by_kind[kind], float(U.max()))
            current_by_kind[kind] = U

            if store_states and kind == store_kind:
                S = np.arange(n + 1, dtype=np.int32)
                F = n - S
                state_chunks.append(
                    pd.DataFrame(
                        {
                            "S": S,
                            "F": F,
                            "n": n,
                            "m": (S + 1.0) / (n + 2.0),
                            "U": U,
                            "gamma_U": params.gamma * U,
                            "threshold": params.threshold,
                            "gamma_U_minus_threshold": margin,
                            "certified_product1": certified,
                            "noncertified": ~certified,
                        }
                    )
                )
        U_next = current_by_kind

    state_count = (N + 1) * (N + 2) // 2
    summary_rows: list[dict[str, float | int | str | bool | None]] = []
    for kind in ALL_TERMINALS:
        largest_noncert, extinction_diagonal = summarize_certified_diagonals(
            all_cert_by_kind[kind],
            any_noncert_by_kind[kind],
        )
        summary_rows.append(
            {
                "N": N,
                "terminal_kind": kind,
                "terminal_diagonal": terminal_diagonal,
                "terminal_bound": terminal_info[kind],
                "is_valid_certificate": kind != "optimistic_zero",
                "state_count": state_count,
                "certified_states": certified_count_by_kind[kind],
                "certified_share": certified_count_by_kind[kind] / state_count,
                "largest_noncertified_diagonal": largest_noncert,
                "certified_extinction_diagonal": extinction_diagonal,
                "max_gamma_U_minus_threshold": max_margin_by_kind[kind],
                "min_gamma_U_minus_threshold": min_margin_by_kind[kind],
                "max_U": max_u_by_kind[kind],
            }
        )

    state_frame = pd.concat(state_chunks, ignore_index=True) if state_chunks else None
    terminal_frame = pd.DataFrame(terminal_rows)
    check_frame = pd.DataFrame(
        [
            {
                "N": N,
                "A_formula": (
                    "(R-c1)*(D(S+1,F)-D(S,F+1))/"
                    "(ell(D(S+1,F))*ell(D(S,F+1)))"
                ),
                "B_formula": "gamma*D(S,F+1)/ell(D(S,F+1))",
                "min_D": min_D,
                "max_D": max_D,
                "min_ell_D": min_ell,
                "min_A": min_A,
                "max_A": max_A,
                "min_B": min_B,
                "max_B": max_B,
                "B_lower_violations": B_lower_violations,
                "B_upper_violations": B_upper_violations,
                "B_range_pass": B_lower_violations == 0
                and B_upper_violations == 0
                and min_B >= -1e-12
                and max_B <= params.gamma + 1e-12,
            }
        ]
    )
    return pd.DataFrame(summary_rows), state_frame, terminal_frame, check_frame


def solve_empirical_slice(
    N: int,
    params: Params,
    remaining_horizon: int,
) -> pd.DataFrame:
    policy_period = N + 1
    horizon = policy_period + remaining_horizon
    model_params = ModelParams(
        p1=params.p1,
        p2=params.p2,
        c1=params.c1,
        c2=params.c2,
        revenue=params.revenue,
        gamma=params.gamma,
    )
    solution = solve_discounted_finite_horizon(
        p0=params.p0,
        params=model_params,
        horizon=horizon,
        stored_policy_periods={policy_period},
        snapshot_periods={policy_period},
    )
    snapshot = solution["snapshots"][policy_period]
    S = snapshot["S"].astype(np.int32)
    F = snapshot["F"].astype(np.int32)
    n = S + F
    discounted_gap = snapshot["discounted_continuation_gap"]
    empirical_product2 = discounted_gap >= params.threshold
    return pd.DataFrame(
        {
            "S": S,
            "F": F,
            "n": n,
            "m": (S + 1.0) / (n + 2.0),
            "discounted_continuation_gap": discounted_gap,
            "empirical_product2": empirical_product2,
            "empirical_margin": discounted_gap - params.threshold,
        }
    )


def validate_against_empirical(
    empirical: pd.DataFrame,
    certificate_states: pd.DataFrame,
    terminal_kind: str,
) -> dict[str, int | str | float]:
    merged = empirical.merge(
        certificate_states[
            ["S", "F", "certified_product1", "gamma_U_minus_threshold"]
        ],
        on=["S", "F"],
        how="inner",
        validate="one_to_one",
    )
    empirical_product2 = merged["empirical_product2"]
    violations = empirical_product2 & (merged["gamma_U_minus_threshold"] < -1e-10)
    count = int(violations.sum())
    if count:
        bad = merged.loc[
            violations,
            ["S", "F", "n", "m", "gamma_U_minus_threshold"],
        ].head(10)
        raise AssertionError(
            f"{count} empirical product-2 states are certified product-1 for "
            f"{terminal_kind}. Examples:\n{bad}"
        )
    empirical_margins = merged.loc[empirical_product2, "gamma_U_minus_threshold"]
    return {
        "terminal_kind": terminal_kind,
        "compared_states": int(len(merged)),
        "empirical_product2_states": int(empirical_product2.sum()),
        "violations": count,
        "min_gamma_U_minus_threshold_on_empirical_product2": (
            float(empirical_margins.min()) if not empirical_margins.empty else np.nan
        ),
        "max_gamma_U_minus_threshold_on_empirical_product2": (
            float(empirical_margins.max()) if not empirical_margins.empty else np.nan
        ),
    }


def plot_empirical_investment(
    empirical: pd.DataFrame,
    path: Path,
    grid_limit: int | None = None,
) -> None:
    configure_plot_style()
    data = empirical if grid_limit is None else empirical[empirical["n"] <= grid_limit]
    invested = data[data["empirical_product2"]]
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    ax.scatter(
        invested["n"],
        invested["m"],
        s=7,
        color=INVEST_COLOR,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        label="Empirical product 2",
    )
    ax.axhline(0.5, color="#475569", linestyle="--", linewidth=1.2, label="p0")
    ax.set_title("Empirical investment region", loc="left")
    ax.set_xlabel("Record size n=S+F")
    ax.set_ylabel("Posterior mean m=(S+1)/(S+F+2)")
    if grid_limit is not None:
        ax.set_xlim(0, grid_limit)
    ax.legend()
    ax.grid(True, axis="y")
    save_figure(fig, path)


def plot_noncertified_region(states: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    noncert = states[states["noncertified"]]
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    ax.scatter(
        noncert["n"],
        noncert["m"],
        s=5,
        color="#dc6b19",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        label=r"Non-certified: $\gamma U\geq\Delta c/\Delta p$",
    )
    ax.set_title("Non-certified region", loc="left")
    ax.set_xlabel("Record size n=S+F")
    ax.set_ylabel("Posterior mean m=(S+1)/(S+F+2)")
    ax.set_xlim(0, int(states["n"].max()))
    ax.legend(markerscale=4)
    ax.grid(True, axis="y")
    save_figure(fig, path)


def plot_margin_heatmap(states: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    margin = states["gamma_U_minus_threshold"].to_numpy()
    finite_margin = margin[np.isfinite(margin)]
    vmax = float(np.nanpercentile(np.abs(finite_margin), 98))
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10.4, 6.1), constrained_layout=True)
    scatter = ax.scatter(
        states["n"],
        states["m"],
        c=states["gamma_U_minus_threshold"],
        s=4,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(r"Certificate margin $\gamma U-\Delta c/\Delta p$", loc="left")
    ax.set_xlabel("Record size n=S+F")
    ax.set_ylabel("Posterior mean m=(S+1)/(S+F+2)")
    fig.colorbar(scatter, ax=ax, label="gamma U - threshold")
    save_figure(fig, path)


def plot_empirical_vs_noncertified(
    empirical: pd.DataFrame,
    states: pd.DataFrame,
    path: Path,
) -> None:
    configure_plot_style()
    merged = empirical.merge(
        states[["S", "F", "noncertified"]],
        on=["S", "F"],
        how="inner",
        validate="one_to_one",
    )
    noncert = merged[merged["noncertified"]]
    invested = merged[merged["empirical_product2"]]
    fig, ax = plt.subplots(figsize=(10.4, 6.1), constrained_layout=True)
    ax.scatter(
        noncert["n"],
        noncert["m"],
        s=5,
        color="#f59e0b",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        label="Non-certified region",
    )
    ax.scatter(
        invested["n"],
        invested["m"],
        s=8,
        color=INVEST_COLOR,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        label="Empirical product 2",
    )
    ax.set_title("Empirical product 2 inside non-certified region", loc="left")
    ax.set_xlabel("Record size n=S+F")
    ax.set_ylabel("Posterior mean m=(S+1)/(S+F+2)")
    ax.set_xlim(0, int(states["n"].max()))
    ax.legend(markerscale=4)
    ax.grid(True, axis="y")
    save_figure(fig, path)


def noncertified_boundaries(states: pd.DataFrame) -> pd.DataFrame:
    noncert = states[states["noncertified"]]
    if noncert.empty:
        return pd.DataFrame(columns=["n", "m_low", "m_high", "noncertified_states"])
    return (
        noncert.groupby("n", as_index=False)
        .agg(
            m_low=("m", "min"),
            m_high=("m", "max"),
            noncertified_states=("m", "size"),
        )
        .sort_values("n")
    )


def plot_noncertified_boundaries(boundaries: pd.DataFrame, path: Path, N: int) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.7), constrained_layout=True)
    if boundaries.empty:
        ax.text(
            0.5,
            0.5,
            "No non-certified states on this grid.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#475569",
        )
    else:
        ax.plot(
            boundaries["n"],
            boundaries["m_low"],
            color="#2563eb",
            linewidth=2.0,
            label=r"$m_{\mathrm{low}}(n)$",
        )
        ax.plot(
            boundaries["n"],
            boundaries["m_high"],
            color="#b91c1c",
            linewidth=2.0,
            label=r"$m_{\mathrm{high}}(n)$",
        )
        ax.fill_between(
            boundaries["n"].to_numpy(),
            boundaries["m_low"].to_numpy(),
            boundaries["m_high"].to_numpy(),
            color="#f59e0b",
            alpha=0.12,
            linewidth=0,
        )
    ax.set_title("Non-certified boundary by diagonal", loc="left")
    ax.set_xlabel("Record size n=S+F")
    ax.set_ylabel("Posterior mean m")
    ax.set_xlim(0, N)
    ax.set_ylim(0.0, 1.0)
    if not boundaries.empty:
        ax.legend()
    ax.grid(True)
    save_figure(fig, path)


def write_state_diagnostics(
    N: int,
    states: pd.DataFrame,
    empirical: pd.DataFrame,
    outputs_dir: Path,
) -> pd.DataFrame:
    grid_dir = outputs_dir / f"diagnostics_N{N}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    empirical_for_grid = empirical[empirical["n"] <= N].copy()

    plot_empirical_investment(
        empirical_for_grid,
        grid_dir / f"plot1_empirical_product2_region_N{N}.png",
        grid_limit=N,
    )
    plot_noncertified_region(
        states,
        grid_dir / f"plot2_noncertified_region_N{N}.png",
    )
    plot_empirical_vs_noncertified(
        empirical_for_grid,
        states,
        grid_dir / f"plot3_empirical_vs_noncertified_N{N}.png",
    )
    boundaries = noncertified_boundaries(states)
    boundaries.to_csv(grid_dir / f"plot4_noncertified_boundaries_N{N}.csv", index=False)
    plot_noncertified_boundaries(
        boundaries,
        grid_dir / f"plot4_noncertified_boundaries_N{N}.png",
        N=N,
    )
    plot_margin_heatmap(
        states,
        grid_dir / f"plot5_gamma_U_margin_heatmap_N{N}.png",
    )
    return boundaries.assign(N=N)


def boundary_sensitivity(boundaries: pd.DataFrame) -> pd.DataFrame:
    if boundaries.empty:
        return pd.DataFrame()
    reference_N = int(boundaries["N"].max())
    reference = boundaries[boundaries["N"] == reference_N][
        ["n", "m_low", "m_high"]
    ].rename(
        columns={
            "m_low": "m_low_reference",
            "m_high": "m_high_reference",
        }
    )
    rows: list[dict[str, float | int | bool]] = []
    for N, layer in boundaries.groupby("N", sort=True):
        merged = layer.merge(reference, on="n", how="inner")
        if merged.empty:
            continue
        low_diff = np.abs(merged["m_low"] - merged["m_low_reference"])
        high_diff = np.abs(merged["m_high"] - merged["m_high_reference"])
        rows.append(
            {
                "N": int(N),
                "reference_N": reference_N,
                "common_diagonals": int(len(merged)),
                "max_abs_m_low_change": float(low_diff.max()),
                "mean_abs_m_low_change": float(low_diff.mean()),
                "max_abs_m_high_change": float(high_diff.max()),
                "mean_abs_m_high_change": float(high_diff.mean()),
                "substantial_upper_change": bool(high_diff.max() > 0.02),
            }
        )
    return pd.DataFrame(rows)


def collapse_duplicate_series(
    frame: pd.DataFrame,
    value_col: str,
    kinds: tuple[str, ...] = ("crude", "lemma2", "unrolled"),
) -> list[tuple[str, pd.DataFrame]]:
    """Return one plotted series for terminal kinds with identical y-values."""
    series: list[tuple[list[str], pd.DataFrame, np.ndarray]] = []
    for kind in kinds:
        layer = frame[frame["terminal_kind"] == kind].sort_values("N")
        if layer.empty:
            continue
        values = layer[value_col].astype(float).to_numpy()
        for labels, _, existing in series:
            if np.allclose(values, existing, equal_nan=True):
                labels.append(kind)
                break
        else:
            series.append(([kind], layer, values))
    return [(" / ".join(labels), layer) for labels, layer, _ in series]


def collapse_duplicate_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical terminal-kind rows for report display only."""
    comparable_cols = [
        "terminal_bound",
        "certified_share",
        "largest_noncertified_diagonal",
        "certified_extinction_diagonal",
    ]
    collapsed: list[tuple[list[str], pd.DataFrame, np.ndarray]] = []
    for kind in ("crude", "lemma2", "unrolled"):
        layer = frame[frame["terminal_kind"] == kind].sort_values("N").copy()
        if layer.empty:
            continue
        signature = layer[comparable_cols].astype(float).to_numpy().ravel()
        for labels, _, existing in collapsed:
            if np.allclose(signature, existing, equal_nan=True):
                labels.append(kind)
                break
        else:
            collapsed.append(([kind], layer, signature))

    pieces: list[pd.DataFrame] = []
    for labels, layer, _ in collapsed:
        display = layer.copy()
        display["terminal_kind"] = " / ".join(labels)
        pieces.append(display)
    if not pieces:
        return frame.copy()
    return pd.concat(pieces, ignore_index=True).sort_values(["N", "terminal_kind"])


def collapse_duplicate_terminal_columns(terminal_wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical terminal-bound columns for report display only."""
    result = terminal_wide.copy()
    if {"crude", "unrolled"}.issubset(result.columns):
        crude = result["crude"].astype(float).to_numpy()
        unrolled = result["unrolled"].astype(float).to_numpy()
        if np.allclose(crude, unrolled, equal_nan=True):
            insert_at = list(result.columns).index("crude")
            result = result.drop(columns=["crude", "unrolled"])
            result.insert(insert_at, "crude / unrolled", crude)
    return result


def plot_grid_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    valid = summary[summary["terminal_kind"].isin(("crude", "lemma2", "unrolled"))]
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    plotted_any = False
    for label, layer in collapse_duplicate_series(valid, "certified_extinction_diagonal"):
        y = layer["certified_extinction_diagonal"].astype(float)
        y = y.where(layer["certified_extinction_diagonal"].notna(), np.nan)
        if np.isfinite(y).any():
            plotted_any = True
            ax.plot(layer["N"], y, marker="o", linewidth=2.0, label=label)
    ax.set_title("Certified extinction diagonal vs grid size", loc="left")
    ax.set_xlabel("Grid limit N")
    ax.set_ylabel("Extinction diagonal within grid")
    if plotted_any:
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No valid terminal condition certifies a full tail\ninside these grids.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#475569",
        )
    ax.grid(True)
    save_figure(fig, path)


def plot_terminal_conditions(summary: pd.DataFrame, terminal: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    valid = terminal[terminal["terminal_kind"].isin(("crude", "lemma2", "unrolled"))]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    ax0, ax1 = axes
    for label, layer in collapse_duplicate_series(valid, "terminal_bound"):
        ax0.plot(layer["N"], layer["terminal_bound"], marker="o", linewidth=2.0, label=label)
    ax0.set_yscale("log")
    ax0.set_title("Terminal upper bounds", loc="left")
    ax0.set_xlabel("Grid limit N")
    ax0.set_ylabel("Terminal bound on diagonal N+1")
    ax0.legend()
    ax0.grid(True, which="both")

    compare = summary[summary["terminal_kind"].isin(("crude", "lemma2", "unrolled"))]
    for label, layer in collapse_duplicate_series(compare, "certified_share"):
        ax1.plot(
            layer["N"],
            layer["certified_share"],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    ax1.set_title("Certified share of states", loc="left")
    ax1.set_xlabel("Grid limit N")
    ax1.set_ylabel("Share certified product 1")
    ax1.set_ylim(0.0, 1.02)
    ax1.legend()
    ax1.grid(True)
    save_figure(fig, path)


def write_report(
    path: Path,
    params: Params,
    identity: pd.DataFrame,
    demand_survival: pd.DataFrame,
    recursion_checks: pd.DataFrame,
    summary: pd.DataFrame,
    terminal: pd.DataFrame,
    empirical: pd.DataFrame,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    plot_N: int,
    empirical_N: int,
    diagnostic_grid: list[int],
) -> None:
    n_emp = int(empirical.loc[empirical["empirical_product2"], "n"].max())
    empirical_count = int(empirical["empirical_product2"].sum())
    old_bound = theorem1_bound(params)
    old_extinction_diagonal = old_bound - 2.0
    identity_max_abs = float(identity["max_abs_error"].max())
    identity_max_rel = float(identity["max_rel_error"].max())
    identity_max_rel_significant = float(
        identity["max_rel_error_pmf_ge_1e_minus_12"].max()
    )
    survival_row = demand_survival.iloc[0]
    recursion_display = recursion_checks[
        [
            "N",
            "min_B",
            "max_B",
            "B_lower_violations",
            "B_upper_violations",
            "B_range_pass",
        ]
    ].copy()

    selected = summary[summary["terminal_kind"] == "selected"].copy()
    valid = summary[summary["terminal_kind"].isin(("crude", "lemma2", "unrolled"))].copy()
    valid_display = collapse_duplicate_summary_rows(valid)
    selected_max_grid = selected.loc[selected["N"].idxmax()]
    finite_extinction = valid["certified_extinction_diagonal"].dropna()
    terminal_wide = terminal.pivot_table(
        index="N",
        columns="terminal_kind",
        values="terminal_bound",
        aggfunc="first",
    ).reset_index()
    terminal_wide_display = collapse_duplicate_terminal_columns(terminal_wide)

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        table = frame.copy()
        for column in table.columns:
            if pd.api.types.is_float_dtype(table[column]):
                table[column] = table[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.6g}"
                )
            else:
                table[column] = table[column].map(
                    lambda value: "" if pd.isna(value) else str(value)
                )
        headers = [str(column) for column in table.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in table.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("# Local Extinction Certificate Report")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(
        "```text\n"
        f"p1={params.p1}, p2={params.p2}, c1={params.c1}, c2={params.c2}, "
        f"R={params.revenue}, gamma={params.gamma}, p0={params.p0}\n"
        f"Delta_p={params.delta_p}, Delta_c={params.delta_c}, "
        f"Delta_c/Delta_p={params.threshold:.12g}\n"
        "```"
    )
    lines.append("")
    lines.append("## Demand Identity Check")
    lines.append("")
    lines.append(
        f"Verified `D(S+1,F)-D(S,F+1)=P(Bin(S+F+2,p0)=S+1)` "
        f"through diagonal n={int(identity['n'].max())}."
    )
    lines.append("")
    lines.append(f"- Maximum absolute error: `{identity_max_abs:.3e}`")
    lines.append(f"- Maximum relative error: `{identity_max_rel:.3e}`")
    lines.append(
        f"- Maximum relative error where the binomial mass is at least `1e-12`: "
        f"`{identity_max_rel_significant:.3e}`"
    )
    lines.append("")
    lines.append("## Correctness Checks")
    lines.append("")
    lines.append(
        "Demand is computed with the Beta survival function "
        "`betaincc(S+1,F+1,p0) = P(Beta(S+1,F+1) >= p0)`."
    )
    lines.append("")
    lines.append(
        f"- Checked demand states: `{int(survival_row['checked_states'])}`"
    )
    lines.append(
        f"- Max absolute error versus Beta survival: "
        f"`{float(survival_row['max_abs_survival_error']):.3e}`"
    )
    lines.append(
        f"- Max distance from lower-tail CDF: "
        f"`{float(survival_row['max_abs_distance_from_cdf_below_p0']):.3e}`"
    )
    lines.append("")
    lines.append(
        "The local recursion uses "
        "`A=(R-c1)*(D(S+1,F)-D(S,F+1))/(ell(D(S+1,F))*ell(D(S,F+1)))` "
        "and `B=gamma*D(S,F+1)/ell(D(S,F+1))`."
    )
    lines.append("")
    lines.append(markdown_table(recursion_display))
    lines.append("")
    lines.append("## Empirical Bellman Comparison")
    lines.append("")
    lines.append(
        f"The empirical comparison used the existing finite-horizon DP solver on "
        f"`S+F <= {empirical_N}`."
    )
    lines.append("")
    lines.append(f"- Empirical product-2 states: `{empirical_count}`")
    lines.append(f"- Empirical largest investment diagonal `n_emp`: `{n_emp}`")
    lines.append("")
    lines.append("Validation against the certificate:")
    lines.append("")
    lines.append(markdown_table(validation))
    lines.append("")
    lines.append(
        "The validation checks the requested inequality directly: every empirical "
        "product-2 state must have "
        "`gamma*U(S,F)-Delta_c/Delta_p >= 0`. If violations were positive, the "
        "script would stop."
    )
    lines.append("")
    lines.append("## Certified Extinction")
    lines.append("")
    lines.append(
        "The table below uses only valid terminal upper bounds. "
        "`certified_extinction_diagonal` is blank when the artificial terminal "
        "boundary leaves at least one non-certified state on the last grid "
        "diagonal."
    )
    lines.append("")
    cols = [
        "N",
        "terminal_kind",
        "terminal_bound",
        "certified_share",
        "largest_noncertified_diagonal",
        "certified_extinction_diagonal",
    ]
    lines.append(markdown_table(valid_display[cols]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if finite_extinction.empty:
        lines.append(
            "No valid terminal condition produced a full certified extinction "
            "tail inside the computed grids. In every valid run, the last "
            "artificial grid diagonal still contains at least one non-certified "
            "state, so the certified extinction diagonal is blank."
        )
    else:
        best_extinction = float(finite_extinction.min())
        lines.append(
            f"The smallest certified extinction diagonal observed across valid "
            f"runs is `{best_extinction:.6g}`."
        )
    lines.append("")
    lines.append(
        f"At the largest grid, `N={int(selected_max_grid['N'])}`, the selected "
        f"valid certificate rules out product 2 on "
        f"`{float(selected_max_grid['certified_share']):.3%}` of states, but its "
        f"largest non-certified diagonal is still "
        f"`{int(selected_max_grid['largest_noncertified_diagonal'])}` because of "
        "the terminal-bound boundary layer."
    )
    lines.append("")
    lines.append(
        f"Thus this truncated-grid implementation improves the old theorem in a "
        f"state-by-state sense inside the grid, and it passes the empirical "
        f"no-overlap check against `n_emp={n_emp}`. It does not yet provide a "
        "valid finite-grid extinction cutoff close to the empirical diagonal "
        "`435` under these diagonal-wide terminal bounds."
    )
    lines.append("")
    lines.append("## Old Theorem 1 Bound")
    lines.append("")
    lines.append(f"- `kappa0(p0) = {kappa0(params):.12g}`")
    lines.append(f"- `bar_n = {old_bound:.6e}`")
    lines.append(
        f"- The theorem certifies product 1 once `S+F+2 > bar_n`, "
        f"i.e. beyond approximately diagonal `{old_extinction_diagonal:.6e}`."
    )
    lines.append("")
    lines.append("## Terminal-Bound Sensitivity")
    lines.append("")
    lines.append(markdown_table(terminal_wide_display))
    lines.append("")
    lines.append("Boundary sensitivity across diagnostic grids:")
    lines.append("")
    lines.append(markdown_table(sensitivity))
    lines.append("")
    lines.append(
        "Identical valid terminal choices are combined in the displayed tables "
        "and plots; the raw CSV files still keep each terminal condition "
        "separately."
    )
    lines.append("")
    lines.append(
        "The `optimistic_zero` terminal is included in the CSV outputs only as a "
        "non-certified diagnostic. It is not used as a certificate."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `demand_identity_check.csv`")
    lines.append("- `grid_sensitivity.csv`")
    lines.append("- `terminal_bounds.csv`")
    lines.append("- `correctness_checks.csv`")
    lines.append("- `boundary_sensitivity.csv`")
    lines.append("- `empirical_comparison_N{}.csv.gz`".format(empirical_N))
    lines.append("- `certificate_states_N{}.csv.gz`".format(plot_N))
    lines.append(
        "- Per-grid diagnostics: "
        + ", ".join(f"`diagnostics_N{N}/`" for N in diagnostic_grid)
    )
    lines.append("- Root copies for `N={}`:".format(plot_N))
    lines.append("  - `plot1_empirical_product2_region_N{}.png`".format(plot_N))
    lines.append("  - `plot2_noncertified_region_N{}.png`".format(plot_N))
    lines.append("  - `plot3_empirical_vs_noncertified_N{}.png`".format(plot_N))
    lines.append("  - `plot4_noncertified_boundaries_N{}.png`".format(plot_N))
    lines.append("  - `plot5_gamma_U_margin_heatmap_N{}.png`".format(plot_N))
    lines.append("- `grid_sensitivity_extinction_diagonal.png`")
    lines.append("- `terminal_condition_comparison.png`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    params = Params()
    n_grid = parse_n_grid(args.n_grid)
    diagnostic_n_grid = parse_n_grid(args.diagnostic_n_grid)
    n_grid = sorted(set(n_grid + diagnostic_n_grid))
    if args.plot_n not in n_grid:
        n_grid = sorted(set(n_grid + [args.plot_n]))
    if args.empirical_n != args.plot_n:
        raise ValueError("This script compares empirical and certificate plots on plot_n.")

    outputs_dir = args.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    bounds = TerminalBounds(params, args.tail_tol)

    print("Verifying demand identity...", flush=True)
    identity = verify_demand_identity(args.identity_max_n, params)
    identity.to_csv(outputs_dir / "demand_identity_check.csv", index=False)
    demand_survival = verify_demand_survival(args.identity_max_n, params)
    demand_survival.to_csv(outputs_dir / "demand_survival_check.csv", index=False)

    print(f"Solving empirical DP slice N={args.empirical_n}...", flush=True)
    empirical = solve_empirical_slice(args.empirical_n, params, args.remaining_horizon)
    empirical.to_csv(outputs_dir / f"empirical_comparison_N{args.empirical_n}.csv.gz", index=False)

    summary_frames: list[pd.DataFrame] = []
    terminal_frames: list[pd.DataFrame] = []
    recursion_check_frames: list[pd.DataFrame] = []
    validation_rows: list[dict[str, int | str | float]] = []
    boundary_frames: list[pd.DataFrame] = []
    plot_states: pd.DataFrame | None = None
    for N in n_grid:
        print(f"Computing certificate grid N={N}...", flush=True)
        summary, states, terminal, recursion_checks = compute_certificate_grid(
            N=N,
            params=params,
            bounds=bounds,
            store_states=N in diagnostic_n_grid,
            store_kind="selected",
        )
        summary_frames.append(summary)
        terminal_frames.append(terminal)
        recursion_check_frames.append(recursion_checks)
        if states is not None:
            validation_rows.append(
                validate_against_empirical(empirical[empirical["n"] <= N], states, f"selected_N{N}")
            )
            boundaries = write_state_diagnostics(N, states, empirical, outputs_dir)
            boundary_frames.append(boundaries)
            if N == args.plot_n:
                plot_states = states.copy()
                plot_states.to_csv(
                    outputs_dir / f"certificate_states_N{args.plot_n}.csv.gz",
                    index=False,
                )
                empirical_for_plot = empirical[empirical["n"] <= N].copy()
                plot_empirical_investment(
                    empirical_for_plot,
                    outputs_dir / f"plot1_empirical_product2_region_N{N}.png",
                    grid_limit=N,
                )
                plot_noncertified_region(
                    states,
                    outputs_dir / f"plot2_noncertified_region_N{N}.png",
                )
                plot_empirical_vs_noncertified(
                    empirical_for_plot,
                    states,
                    outputs_dir / f"plot3_empirical_vs_noncertified_N{N}.png",
                )
                plot_noncertified_boundaries(
                    boundaries,
                    outputs_dir / f"plot4_noncertified_boundaries_N{N}.png",
                    N=N,
                )
                plot_margin_heatmap(
                    states,
                    outputs_dir / f"plot5_gamma_U_margin_heatmap_N{N}.png",
                )

    if plot_states is None:
        raise RuntimeError("No state-level certificate frame was produced.")

    summary_all = pd.concat(summary_frames, ignore_index=True)
    terminal_all = pd.concat(terminal_frames, ignore_index=True)
    recursion_checks_all = pd.concat(recursion_check_frames, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    boundary_all = (
        pd.concat(boundary_frames, ignore_index=True)
        if boundary_frames
        else pd.DataFrame(columns=["n", "m_low", "m_high", "noncertified_states", "N"])
    )
    sensitivity = boundary_sensitivity(boundary_all)
    summary_all.to_csv(outputs_dir / "grid_sensitivity.csv", index=False)
    terminal_all.to_csv(outputs_dir / "terminal_bounds.csv", index=False)
    recursion_checks_all.to_csv(outputs_dir / "recursion_checks.csv", index=False)
    validation.to_csv(outputs_dir / "empirical_certificate_validation.csv", index=False)
    boundary_all.to_csv(outputs_dir / "noncertified_boundaries_all.csv", index=False)
    sensitivity.to_csv(outputs_dir / "boundary_sensitivity.csv", index=False)
    correctness_checks = pd.DataFrame(
        [
            {
                "check": "empirical_product2_not_certified_product1",
                "violations": int(validation["violations"].sum()),
                "pass": bool((validation["violations"] == 0).all()),
            },
            {
                "check": "demand_is_beta_survival",
                "violations": int(demand_survival["max_abs_survival_error"].iloc[0] > 1e-12),
                "pass": bool(demand_survival["max_abs_survival_error"].iloc[0] <= 1e-12),
            },
            {
                "check": "B_in_0_gamma",
                "violations": int(
                    recursion_checks_all["B_lower_violations"].sum()
                    + recursion_checks_all["B_upper_violations"].sum()
                ),
                "pass": bool(recursion_checks_all["B_range_pass"].all()),
            },
        ]
    )
    correctness_checks.to_csv(outputs_dir / "correctness_checks.csv", index=False)
    plot_grid_sensitivity(
        summary_all,
        outputs_dir / "grid_sensitivity_extinction_diagonal.png",
    )
    plot_terminal_conditions(
        summary_all,
        terminal_all,
        outputs_dir / "terminal_condition_comparison.png",
    )

    write_report(
        outputs_dir / "REPORT.md",
        params,
        identity,
        demand_survival,
        recursion_checks_all,
        summary_all,
        terminal_all,
        empirical,
        validation,
        sensitivity,
        args.plot_n,
        args.empirical_n,
        diagnostic_n_grid,
    )

    print(f"Wrote outputs to {outputs_dir}", flush=True)


if __name__ == "__main__":
    main()
