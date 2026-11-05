"""Independent order-sensitive refinement of the large-window approximation.

Run after solving final policies:
    .venv/bin/python -B -m forgettin_fixed.code.refinement

State comparisons are diagnostic samples, not confidence bounds for the
optimal policy.  Certificates remain the exhaustive FIFO Bellman residuals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .large_window import LargeWindowPolicy, solve_large_window


ROOT = Path(__file__).resolve().parents[1]
CASES = [(.1, 33, 8, 10), *[(p, 100, 10, 12) for p in (.1, .3, .5, .7, .9)], (.5, 1000, 4, 6)]


def update_initial_certificate(policy):
    """Add starting-empty bounds to checkpoints created by an earlier import."""
    d, p, n = policy.diagnostics, policy.primitives, policy.n
    factor = p.gamma ** n / (1 - p.gamma)
    lower, upper = d["fifo_bellman_residual_min"], d["fifo_bellman_residual_max"]
    initial = d["initial_value_projected_terminal"]
    d.update(empty_initial_value_error_bound=factor * d["fifo_bellman_residual_sup"],
             empty_initial_greedy_policy_loss_bound=min(p.net_rewards[0] / (1 - p.gamma), factor * (upper - lower) + (d["transient_advantage_float32_max_error"] + policy.action_tolerance) / (1 - p.gamma)),
             empty_initial_optimal_value_lower_bound=initial + factor * lower,
             empty_initial_optimal_value_upper_bound=initial + factor * upper)


def _state(policy, queue, m):
    prefix = np.zeros(len(queue), dtype=np.int64)
    for age in range(min(policy.prefix_bits, m)):
        prefix = 2 * prefix + queue[:, age]
    count = queue[:, :m].sum(axis=1)
    b = queue[:, policy.prefix_bits] if m > policy.prefix_bits else 0
    return policy.evaluate_state(m, count, prefix, b)


def _residuals(policy, queue):
    k, n, p = policy.prefix_bits, policy.n, policy.primitives
    count = queue.sum(axis=1, dtype=np.int64)
    prefix = np.zeros(len(queue), dtype=np.int64)
    for age in range(k):
        prefix = 2 * prefix + queue[:, age]
    pc = np.array([int(x).bit_count() for x in prefix])
    j = count - pc
    value = policy.mature_value[prefix, j]
    b = queue[:, k]
    nextprefix = ((prefix << 1) & ((1 << k) - 1)) | b
    vf = policy.mature_value[nextprefix, j - b]
    vs = policy.mature_value[nextprefix, j - b + 1]
    q1 = p.net_rewards[0] + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
    q2 = p.net_rewards[1] + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
    demand = policy.demand_table[n * (n + 1) // 2 + count]
    return demand / (1 - p.gamma * (1 - demand)) * np.maximum(q1, q2) - value


def histories(n, m, rng, samples):
    """Uniform conditional ordering, plus deliberately nonexchangeable orders."""
    for probability in (.35, .5, .8):
        queue = np.zeros((samples, n), dtype=np.uint8)
        queue[:, :m] = rng.random((samples, m)) < probability
        yield f"iid_success_probability_{probability:g}", queue
    queue = np.zeros((samples, n), dtype=np.uint8)
    probability = rng.random(samples)
    queue[:, :m] = rng.random((samples, m)) < probability[:, None]
    yield "uniform_success_count_random_order", queue
    for oldest_first in (True, False):
        queue = np.zeros((m + 1, n), dtype=np.uint8)
        for s in range(m + 1):
            if s:
                queue[s, :s] = 1 if oldest_first else 0
                if not oldest_first:
                    queue[s, m - s:m] = 1
        yield "successes_oldest_first" if oldest_first else "successes_newest_first", queue
    queue = np.zeros((2, n), dtype=np.uint8)
    queue[0, :m] = np.arange(m) % 2
    queue[1, :m] = 1 - queue[0, :m]
    yield "alternating", queue


def paired_calendar(coarse, fine, *, paths=2000, periods=1000, seed=20261103):
    """Common-uniform comparison on actual FIFO queues from the empty prior."""
    from .simulation import _FIFOState
    summaries = []
    for policy in (coarse, fine):
        state = _FIFOState(policy, paths)
        rng = np.random.default_rng(seed)
        revenue = np.zeros(paths)
        a_counts = np.zeros(paths, dtype=np.int64)
        p2_counts = np.zeros(paths, dtype=np.int64)
        discount = 1.
        p = policy.primitives
        for _ in range(periods):
            action, demand = state.evaluate()
            buyers = rng.random(paths) < demand
            success = rng.random(paths) < np.where(action, p.p2, p.p1)
            revenue += discount * buyers * np.where(action, p.net_rewards[1], p.net_rewards[0])
            a_counts += buyers
            p2_counts += buyers & action
            state.advance(buyers, success)
            discount *= p.gamma
        summaries.append((revenue, a_counts / periods, p2_counts / periods))
    frame = pd.DataFrame({"path": np.arange(paths)})
    result = {"paths": paths, "calendar_periods": periods, "seed": seed,
              "coarse_prefix_bits": coarse.prefix_bits, "fine_prefix_bits": fine.prefix_bits,
              "comparison": "fine minus coarse, common uniforms, true FIFO execution"}
    for col, a, b in zip(("discounted_profit", "a_market_share", "product2_unconditional_rate"), summaries[0], summaries[1]):
        difference = b - a
        frame[f"coarse_{col}"] = a
        frame[f"fine_{col}"] = b
        frame[f"difference_{col}"] = difference
        se = float(difference.std(ddof=1) / np.sqrt(paths))
        result[col] = {"coarse_mean": float(a.mean()), "fine_mean": float(b.mean()),
                       "paired_difference": float(difference.mean()), "paired_standard_error": se,
                       "paired_ci95_lower": float(difference.mean() - 1.959963984540054 * se),
                       "paired_ci95_upper": float(difference.mean() + 1.959963984540054 * se)}
    return frame, result


def run(samples=2048, seed=20261103, run_paired=True):
    directory = ROOT / "data" / "refinement"
    directory.mkdir(parents=True, exist_ok=True)
    rows, examples, diagnostics = [], [], []
    started = time.monotonic()
    for case_index, (p0, n, klo, khi) in enumerate(CASES):
        label = f"p0_{p0:g}_n_{n}"
        final_path = ROOT / "data" / "policies" / f"{label}.npz"
        if not final_path.exists():
            raise FileNotFoundError(f"solve final policies first: {final_path}")
        fine = LargeWindowPolicy.load(final_path)
        if fine.prefix_bits != khi:
            raise ValueError(f"expected k={khi} for {label}; found {fine.prefix_bits}")
        if "empty_initial_value_error_bound" not in fine.diagnostics:
            update_initial_certificate(fine)
            fine.save(final_path)
            final_path.with_suffix(".json").write_text(json.dumps(fine.diagnostics, indent=2) + "\n")
        coarse_path = directory / f"{label}_k_{klo}.npz"
        if coarse_path.exists():
            coarse = LargeWindowPolicy.load(coarse_path)
        else:
            coarse = solve_large_window(n, fine.primitives, prefix_bits=klo,
                                         tolerance=1e-10, action_tolerance=fine.action_tolerance)
            coarse.save(coarse_path)
        diagnostics.append({"p0": p0, "n": n, "coarse": coarse.diagnostics, "fine": fine.diagnostics})
        rng = np.random.default_rng(seed + case_index)
        lengths = sorted(set([min(16, n), n // 2, n - 1, n]))
        for m in lengths:
            for schema, queue in histories(n, m, rng, samples):
                action0, demand0, gap0 = _state(coarse, queue, m)
                action1, demand1, gap1 = _state(fine, queue, m)
                np.testing.assert_array_equal(demand0, demand1)
                difference = np.abs(gap1 - gap0)
                row = {"p0": p0, "n": n, "remembered_length": m, "schema": schema,
                       "histories": len(queue), "coarse_prefix_bits": klo, "fine_prefix_bits": khi,
                       "action_disagreement_share": float(np.mean(action0 != action1)),
                       "coarse_product2_share": float(action0.mean()), "fine_product2_share": float(action1.mean()),
                       "absolute_advantage_difference_median": float(np.median(difference)),
                       "absolute_advantage_difference_q95": float(np.quantile(difference, .95)),
                       "absolute_advantage_difference_max": float(difference.max())}
                if m == n:
                    for name, policy in (("coarse", coarse), ("fine", fine)):
                        residual = np.abs(_residuals(policy, queue))
                        row.update({f"{name}_absolute_fifo_residual_mean": float(residual.mean()),
                                    f"{name}_absolute_fifo_residual_q95": float(np.quantile(residual, .95)),
                                    f"{name}_absolute_fifo_residual_max": float(residual.max())})
                rows.append(row)
                for idx in np.flatnonzero(action0 != action1)[:8]:
                    examples.append({"p0": p0, "n": n, "remembered_length": m, "schema": schema,
                                     "oldest_to_newest_outcomes": "".join(map(str, queue[idx, :m])),
                                     "coarse_advantage": float(gap0[idx]), "fine_advantage": float(gap1[idx])})
        print(f"refinement {label}: k={klo} versus {khi}", flush=True)
    pd.DataFrame(rows).to_csv(directory / "state_comparison.csv", index=False)
    pd.DataFrame(examples).to_csv(directory / "disagreement_examples.csv", index=False)
    if run_paired:
        fine = LargeWindowPolicy.load(ROOT / "data" / "policies" / "p0_0.5_n_100.npz")
        coarse = LargeWindowPolicy.load(directory / "p0_0.5_n_100_k_10.npz")
        frame, result = paired_calendar(coarse, fine)
        frame.to_csv(directory / "paired_calendar_paths_p0_0.5_n100.csv", index=False)
        (directory / "paired_calendar_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {"seed": seed, "random_histories_per_schema": samples, "cases": CASES,
                "lengths": "16, floor(n/2), n-1, n (duplicates removed)",
                "schemas": "iid Bernoulli(.35,.5,.8), uniform count/random order, both sorted orders, both alternating phases",
                "empty_initial_certificates": "mature residual is reached only after n A purchases; discount factor at most gamma**n",
                "interpretation": "state-sample disagreements describe sensitivity to prefix resolution; they do not certify proximity to the globally optimal mature FIFO policy",
                "seconds": time.monotonic() - started, "diagnostics": diagnostics}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--skip-paired", action="store_true")
    args = parser.parse_args()
    run(args.samples, args.seed, not args.skip_paired)
