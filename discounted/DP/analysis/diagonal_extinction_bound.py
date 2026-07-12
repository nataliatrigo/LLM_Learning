"""Compute the one-dimensional uniform diagonal extinction bound H_n.

For every diagonal n, H_n bounds all local value gaps rooted on that
diagonal.  The recursion retains the dependence between the source A and the
continuation coefficient B instead of combining their unrelated worst cases.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discounted.DP.analysis.local_extinction_certificate import (  # noqa: E402
    Params, binom_pmf, configure_plot_style, demand_diag, ell, save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-grid", default="800,1200,1600,2400,3200,5000")
    parser.add_argument("--plot-n", type=int, default=5000)
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=Path(__file__).resolve().parent / "outputs_diagonal_bound",
    )
    return parser.parse_args()


def solve_diagonal_bound(N_outer: int, params: Params) -> tuple[np.ndarray, np.ndarray, float]:
    """Return H_n and its maximizing S using a valid terminal bound at N+1."""
    start = time.perf_counter()
    h_next = params.cheap_margin / (1.0 - params.gamma)
    H = np.empty(N_outer + 1)
    maximizing_S = np.empty(N_outer + 1, dtype=np.int32)
    for n in range(N_outer, -1, -1):
        d_next = demand_diag(n + 1, params)
        d_a, d_b = d_next[1:], d_next[:-1]
        pmf = np.asarray(
            binom_pmf(np.arange(n + 1, dtype=np.int64) + 1, n + 2, params.p0)
        )
        A = params.cheap_margin * pmf / (ell(d_a, params) * ell(d_b, params))
        B = params.gamma * d_b / ell(d_b, params)
        candidates = A + B * h_next
        argmax = int(np.argmax(candidates))
        H[n] = candidates[argmax]
        maximizing_S[n] = argmax
        h_next = H[n]
    return H, maximizing_S, time.perf_counter() - start


def below_threshold_runs(values: np.ndarray, threshold: float) -> str:
    idx = np.flatnonzero(values < threshold)
    if not len(idx):
        return ""
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(idx) - 1]
    return ";".join(f"{idx[a]}:{idx[b]}" for a, b in zip(starts, ends))


def main() -> None:
    args = parse_args()
    params = Params()
    grids = sorted({int(item) for item in args.outer_grid.split(",")})
    out = args.outputs_dir
    out.mkdir(parents=True, exist_ok=True)
    threshold = params.delta_c / (params.gamma * params.delta_p)
    solutions: dict[int, np.ndarray] = {}
    maximizers: dict[int, np.ndarray] = {}
    summary_rows = []
    for N in grids:
        print(f"Computing uniform diagonal bound through N_outer={N}...", flush=True)
        H, argmax, runtime = solve_diagonal_bound(N, params)
        solutions[N], maximizers[N] = H, argmax
        summary_rows.append({
            "N_outer": N,
            "terminal_diagonal": N + 1,
            "terminal_bound": params.cheap_margin / (1.0 - params.gamma),
            "minimum_H": float(H.min()),
            "minimum_H_diagonal": int(H.argmin()),
            "below_threshold_runs": below_threshold_runs(H, threshold),
            "H_435": H[435] if N >= 435 else np.nan,
            "H_800": H[800] if N >= 800 else np.nan,
            "runtime_seconds": runtime,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "diagonal_bound_outer_grid_summary.csv", index=False)

    largest = grids[-1]
    H, argmax = solutions[largest], maximizers[largest]
    n = np.arange(min(args.plot_n, largest) + 1)
    states = pd.DataFrame({
        "n": n, "H": H[n], "gamma_H": params.gamma * H[n],
        "U_crit": threshold, "certifies_full_diagonal": H[n] < threshold,
        "maximizing_S": argmax[n], "maximizing_m": (argmax[n] + 1) / (n + 2),
    })
    states.to_csv(out / "diagonal_bound_by_n.csv", index=False)

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for N in grids:
        use = np.arange(min(args.plot_n, N) + 1)
        ax.plot(use, solutions[N][use], lw=1.15, label=f"N_outer={N}")
    ax.axhline(threshold, color="#b91c1c", ls="--", lw=1.5, label="U_crit")
    ax.set(xlabel="n", ylabel="Uniform diagonal bound H_n",
           title="Explicit uniform diagonal extinction bound")
    ax.legend(ncol=2)
    save_figure(fig, out / "diagonal_extinction_bound.png")

    # The valid terminal bound produces a boundary layer of roughly 176
    # diagonals at gamma=.98; keep a conservative 200-diagonal buffer when
    # assessing convergence of consecutive outer grids.
    comparison_n = min(grids[-2] - 200, args.plot_n)
    max_difference = float(np.max(np.abs(solutions[grids[-1]][:comparison_n+1] - solutions[grids[-2]][:comparison_n+1])))
    report = f"""# Uniform Diagonal Extinction Bound

The recursion is

`H_n = max_(S+F=n) [A(S,F) + B(S,F) H_(n+1)]`,

with the valid terminal value `{params.cheap_margin/(1-params.gamma):.12g}` imposed on diagonal `N_outer+1`. It implies `delta(S,F) <= H_n` for every state on diagonal `n`. A complete diagonal is certified product 1 when `H_n < U_crit`, where `U_crit={threshold:.12g}`.

The largest exact computation used `N_outer={largest}`. It gives `H_435={H[435]:.12g}`, `H_800={H[800]:.12g}`, and minimum `H_n={H.min():.12g}` at `n={H.argmin()}`. Therefore no complete diagonal through `n={largest}` is certified by this bound. This means the bound has not crossed yet, not that a finite crossing cannot exist.

The maximum difference between `N_outer={grids[-2]}` and `{largest}` on their common displayed region through `n={comparison_n}` is `{max_difference:.6g}`. Values near the larger grid's terminal boundary remain contaminated and must not be interpreted as an economic increase.

This bound is far tighter than the closed-form Theorem 1 cutoff but remains conservative relative to the empirical Bellman extinction diagonal `435`.
"""
    (out / "REPORT_diagonal_extinction_bound.md").write_text(report)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
