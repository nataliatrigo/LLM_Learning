"""Compare saved fluid competition policies from different mature states.

No equilibrium or HJB is solved again. Both sellers follow the same saved
policy and start at the same reputation. Demand uses their actual states.
Long-run destinations follow the autonomous phase line, separately from the
finite-horizon terminal states.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.trajectories import _policy, simulate_pair


ROOT = Path(__file__).resolve().parents[1]
STEM = "15_initial_conditions_p0_0p8_rho_0p9"
DEFAULT_INITIALS = [.1, .3, .5, .55, .7, .9]


def phase_limit(policy, x0):
    """Follow drift directions until the first stationary state is reached.

    Beta/Thompson demand is strictly positive for the present p0<1. It scales
    speed without changing the phase line. This classifies asymptotic limits
    without mistaking a nearly flat finite-time path for a stationary state.
    """
    x = float(x0)
    region, holding = policy.initial(x)
    if holding:
        return x
    for _ in range(len(policy.roots)+2):
        probability = policy.probability(policy.products[region])
        if probability == x:
            return float(x)
        direction = 1 if probability > x else -1
        index = region if direction > 0 else region-1
        if not 0 <= index < len(policy.roots):
            return float(probability)
        root = policy.roots[index]
        if direction*(probability-root) <= 0:
            return float(probability)
        if policy.kind(index) == "attractive":
            return float(root)
        x = root
        region += direction
    raise AssertionError("Unexpected cycle in the scalar phase line")


def plot_comparison(params, p0, equilibria, initials, frames, output, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {x0: color for x0, color in zip(
        initials, ["#0072B2", "#009E73", "#E69F00", "#6F42C1", "#CC79A7", "#D55E00"],
        strict=True,
    )}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0),
                             gridspec_kw={"height_ratios": [3, 1.45]})
    for column, equilibrium in enumerate(equilibria):
        z = equilibrium["x_star"]
        policy = _policy(params, equilibrium["policy"])
        ax, basin = axes[0, column], axes[1, column]
        for endpoint in (params.p1, params.p2):
            ax.axhline(endpoint, color="#737373", ls="--", lw=.8, alpha=.6)
        repelling = [root for index, root in enumerate(policy.roots)
                     if policy.kind(index) == "repelling"]
        for root in repelling:
            ax.axhline(root, color="#555555", ls=":", lw=1.25)
            ax.text(.975*horizon, root+.012, f"threshold {root:.4f}",
                    ha="right", fontsize=9, color="#555555")
        for x0 in initials:
            frame = frames[(z, x0)]
            ax.plot(frame.time, frame.x_a, color=colors[x0],
                    lw=2.6 if x0 == .55 else 1.8, zorder=4 if x0 == .55 else 3)
        ax.set(title=rf"Policy optimized against $\bar x={z:g}$",
               xlabel="Continuous calendar time, t", ylabel="Reputation x(t)",
               xlim=(0, horizon), ylim=(.05, .95))
        ax.grid(axis="y", alpha=.15)

        sample = sorted(set([*np.linspace(0, 1, 1001), *policy.roots,
                             *[float(np.nextafter(root, 1.)) for root in policy.roots]]))
        limits = [phase_limit(policy, x) for x in sample]
        basin.plot(sample, limits, color="#7b848d", lw=1.5)
        for root in repelling:
            basin.axvline(root, color="#555555", ls=":", lw=1.1)
        for x0 in initials:
            basin.scatter(x0, phase_limit(policy, x0), s=75 if x0 == .55 else 48,
                          facecolor=colors[x0], edgecolor="white", linewidth=.6, zorder=5)
        basin.set(title="Long-run outcome under the frozen policy",
                  xlabel="Common initial reputation, x(0)", ylabel="Limit of x(t)",
                  xlim=(0, 1), ylim=(.25, .9), yticks=[params.p1, params.p2])
        basin.grid(axis="y", alpha=.15)
    handles = [Line2D([], [], color=colors[x0], lw=2.6 if x0 == .55 else 1.8,
                      label=f"x(0) = {x0:g}") for x0 in initials]
    handles.extend([
        Line2D([], [], color="#555555", ls=":", lw=1.25, label="Repelling policy threshold"),
        Line2D([], [], color="#737373", ls="--", lw=.8, label="Stationary reputation levels"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(.5, .055), frameon=False)
    fig.suptitle(rf"Fluid competition: initial conditions under two saved policies | $p_0={p0:g}$, $\rho={params.rho:g}$",
                 fontsize=16, x=.075, ha="left", y=.98)
    fig.text(.075, .929,
             "Symmetric mature starts: x_A(0) = x_B(0). Both sellers use the same policy; demand uses both actual reputations.",
             fontsize=10)
    fig.text(.075, .021,
             f"Top: trajectories through T = {horizon:g}; slow paths may remain far from their limits. "
             "Bottom: phase-line limits, which need not match the policy's rival conjecture.", fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, top=.865, bottom=.18, hspace=.44, wspace=.20)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"{STEM}.{extension}", dpi=180)
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=3000.)
    parser.add_argument("--samples", type=int, default=3001)
    args = parser.parse_args()
    if not math.isfinite(args.horizon) or args.horizon <= 0 or args.samples < 2:
        parser.error("Use a positive finite horizon and at least two samples")
    config = json.loads((ROOT/"config.json").read_text())
    checkpoint = ROOT/"data/checkpoints/p0_0p8_rho_0p9.json.gz"
    with gzip.open(checkpoint, "rt") as handle:
        saved = json.load(handle)
    params = Parameters(**config["parameters"], rho=.9)
    p0 = .8
    grid = saved["grids"][-1]
    equilibria = sorted(grid["result"]["equilibria"], key=lambda row: row["x_star"])
    assert len(equilibria) == 2 and all(row["verified"] for row in equilibria)
    assert saved["inputs"]["parameters"] == config["parameters"]
    initials = DEFAULT_INITIALS
    times = np.linspace(0, args.horizon, args.samples)
    settings = config["trajectories"]
    data_dir = ROOT/"data/initial_conditions_p0_0p8_rho_0p9"
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries, diagnostics, frames = [], [], {}
    maximum_refinement_change = 0.
    for equilibrium in equilibria:
        z = equilibrium["x_star"]
        policy = _policy(params, equilibrium["policy"])
        for x0 in initials:
            result = simulate_pair(params, p0, equilibrium["policy"], initial=(x0, x0),
                                   times=times, bin_width=settings["bin_width"],
                                   rtol=settings["rtol"], atol=settings["atol"],
                                   max_step=settings["max_step"])
            checked = simulate_pair(params, p0, equilibrium["policy"], initial=(x0, x0),
                                    times=times, bin_width=settings["bin_width"],
                                    rtol=1e-11, atol=1e-13, max_step=5.)
            frame = result["by_time"].copy()
            refinement_change = float(np.max(np.abs(
                frame[["x_a", "x_b"]].to_numpy()-checked["by_time"][["x_a", "x_b"]].to_numpy())))
            if refinement_change > 2e-8:
                raise AssertionError("Reputation paths changed under ODE refinement")
            maximum_refinement_change = max(maximum_refinement_change, refinement_change)
            np.testing.assert_allclose(frame.x_a, frame.x_b, atol=1e-12, rtol=0)
            motion = np.diff(frame.x_a)
            if np.any(motion > 2e-10) and np.any(motion < -2e-10):
                raise AssertionError("An autonomous symmetric path reversed direction")
            limit = phase_limit(policy, x0)
            if np.any((frame.x_a-limit)*(x0-limit) < -2e-10):
                raise AssertionError("A reputation path crossed its limiting state")
            frames[(z, x0)] = frame
            target_product = 1 if limit == params.p1 else 2
            region, holding = policy.initial(limit)
            if not holding and policy.products[region] != target_product:
                raise AssertionError("Phase-line destination is not stationary under the policy")
            final = float(frame.x_a.iloc[-1])
            summaries.append(dict(
                p0=p0, rho=params.rho, rival_conjecture=z, x0=x0,
                initial_product2_fraction=float(frame.alpha_a.iloc[0]),
                horizon=args.horizon, terminal_x=final, long_run_limit=limit,
                terminal_distance_to_limit=abs(final-limit),
                limit_matches_conjecture=bool(abs(limit-z)<1e-10),
                repelling_threshold=next(root for i, root in enumerate(policy.roots)
                                         if policy.kind(i) == "repelling"),
                maximum_ode_refinement_change=refinement_change,
            ))
            diagnostics.append(dict(rival_conjecture=z, x0=x0,
                                    diagnostics=result["diagnostics"], events=result["events"]))
            print(f"z={z:g}, x0={x0:g}: x(T)={final:.9f}, phase-line limit={limit:g}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(data_dir/"summary.csv", index=False)
    samples = []
    for (z, x0), frame in frames.items():
        subset = frame[["time", "x_a", "x_b", "alpha_a", "alpha_b",
                        "demand_a", "demand_b", "outside_rate"]].copy()
        subset.insert(0, "x0", x0)
        subset.insert(0, "rival_conjecture", z)
        samples.append(subset)
    pd.concat(samples, ignore_index=True).to_csv(data_dir/"trajectories.csv", index=False)
    # The old common-start experiment is an independent saved baseline.
    if args.horizon == 3000.:
        old = pd.read_csv(ROOT/"data/fluid_summary.csv")
        for equilibrium in equilibria:
            z = equilibrium["x_star"]
            row = old[(old.p0 == p0)&(old.rho == params.rho)&np.isclose(old.x_star, z)].iloc[0]
            current = summary[(summary.rival_conjecture == z)&(summary.x0 == .5)].iloc[0]
            if abs(current.terminal_x-row.trajectory_terminal_x) > 2e-9:
                raise AssertionError("Common-start result differs from saved experiment")
    plot_comparison(params, p0, equilibria, initials, frames, ROOT/"figures/overview", args.horizon)
    sources = [Path(__file__), ROOT/"code/trajectories.py", checkpoint]
    manifest = dict(
        p0=p0, parameters=config["parameters"], rho=params.rho,
        initial_states=initials, horizon=args.horizon, samples=args.samples,
        path_count=len(summary), state="Mature normalized reputations, not empty histories",
        policy_source="Saved finest-grid verified equilibrium policies; no new HJB solves",
        demand="Actual joint-state Thompson demand, not the frozen rival conjecture",
        maximum_ode_refinement_change=maximum_refinement_change,
        source_sha256={str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                       for source in sources},
        policies=[dict(rival_conjecture=eq["x_star"], policy=eq["policy"]) for eq in equilibria],
        runs=diagnostics,
        interpretation="Long-run limits are phase-line predictions. A limit different from the "
                       "optimization conjecture is not an additional verified equilibrium.",
    )
    (data_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(f"Saved comparison figure and {len(summary)} verified paths", flush=True)


if __name__ == "__main__":
    run()
