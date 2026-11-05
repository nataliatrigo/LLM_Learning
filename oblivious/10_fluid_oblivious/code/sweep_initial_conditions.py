"""Plot symmetric mature starts for every saved fluid competition case.

Use the verified finest-grid policies, actual joint-state demand, and the
existing event-driven integrator. Outputs are separate from the original
experiment and the earlier two-policy illustration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.compare_initial_conditions import DEFAULT_INITIALS, phase_limit
from oblivious.fluid_oblivious.code.trajectories import _policy, simulate_pair


ROOT = Path(__file__).resolve().parents[1]


def tag(value):
    return str(float(value)).replace(".", "p")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policies(p0, rho, config):
    path = ROOT/f"data/checkpoints/p0_{tag(p0)}_rho_{tag(rho)}.json.gz"
    with gzip.open(path, "rt") as handle:
        saved = json.load(handle)
    if saved["inputs"]["parameters"] != config["parameters"]:
        raise ValueError(f"Economic parameters differ from checkpoint {path}")
    finest = max(saved["grids"], key=lambda row: row["result"]["search"]["grid_nodes"])
    equilibria = sorted(finest["result"]["equilibria"], key=lambda row: row["x_star"])
    if not equilibria or not all(row["verified"] for row in equilibria):
        raise ValueError(f"Checkpoint has no fully verified policies: {path}")
    return path, equilibria, finest["result"]["search"]["grid_nodes"]


def calculate_case(task):
    p0, rho, config, data_root, horizon, samples, figures_only, baseline = task
    params = Parameters(**config["parameters"], rho=rho)
    checkpoint, equilibria, policy_nodes = load_policies(p0, rho, config)
    folder = Path(data_root)/f"p0_{tag(p0)}"/f"rho_{tag(rho)}"
    settings = config["trajectories"]
    inputs = dict(
        p0=p0, rho=rho, parameters=config["parameters"],
        initial_states=DEFAULT_INITIALS, horizon=horizon, samples=samples,
        bin_width=settings["bin_width"], rtol=settings["rtol"], atol=settings["atol"],
        max_step=settings["max_step"], check_rtol=1e-11, check_atol=1e-13,
        check_max_step=5., demand_order=256,
    )
    sources = [Path(__file__), ROOT/"code/trajectories.py",
               ROOT/"code/compare_initial_conditions.py", checkpoint]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    manifest_path = folder/"manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get("inputs") == inputs and previous.get("source_sha256") == hashes
                and (folder/"summary.csv").exists() and (folder/"trajectories.csv").exists()):
            return previous
    if figures_only:
        raise ValueError(f"Missing or stale verified trajectory data: {folder}")

    times = np.linspace(0, horizon, samples)
    summaries, trajectories, runs = [], [], []
    maximum_refinement_change = 0.
    maximum_baseline_change = 0.
    for equilibrium in equilibria:
        z = equilibrium["x_star"]
        policy = _policy(params, equilibrium["policy"])
        repelling = [root for i, root in enumerate(policy.roots) if policy.kind(i) == "repelling"]
        for x0 in DEFAULT_INITIALS:
            result = simulate_pair(
                params, p0, equilibrium["policy"], initial=(x0, x0), times=times,
                bin_width=settings["bin_width"], rtol=settings["rtol"],
                atol=settings["atol"], max_step=settings["max_step"],
            )
            checked = simulate_pair(
                params, p0, equilibrium["policy"], initial=(x0, x0), times=times,
                bin_width=settings["bin_width"], rtol=1e-11, atol=1e-13, max_step=5.,
            )
            frame = result["by_time"]
            refinement_change = float(np.max(np.abs(
                frame[["x_a", "x_b"]].to_numpy()-checked["by_time"][["x_a", "x_b"]].to_numpy())))
            if refinement_change > 2e-8:
                raise AssertionError(f"ODE refinement failed: p0={p0}, rho={rho}, z={z}, x0={x0}")
            maximum_refinement_change = max(maximum_refinement_change, refinement_change)
            np.testing.assert_allclose(frame.x_a, frame.x_b, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame[["x_a", "x_b"]].iloc[0], x0, atol=1e-12, rtol=0)
            motion = np.diff(frame.x_a)
            if np.any(motion > 2e-10) and np.any(motion < -2e-10):
                raise AssertionError("An autonomous symmetric path reversed direction")
            limit = phase_limit(policy, x0)
            if np.any((frame.x_a-limit)*(x0-limit) < -2e-10):
                raise AssertionError("Reputation crossed its phase-line destination")
            region, holding = policy.initial(limit)
            limit_alpha = policy.holding_alpha(limit) if holding else float(policy.products[region] == 2)
            if abs(params.p1+(params.p2-params.p1)*limit_alpha-limit) > 1e-11:
                raise AssertionError("Phase-line destination is not stationary")
            final = float(frame.x_a.iloc[-1])
            if abs(final-limit) > abs(x0-limit)+2e-9:
                raise AssertionError("Reputation moved away from its phase-line destination")
            if horizon == 3000. and x0 == .5:
                old = next(row for row in baseline if abs(row["x_star"]-z) < 1e-10)
                change = abs(final-old["trajectory_terminal_x"])
                maximum_baseline_change = max(maximum_baseline_change, change)
                if change > 2e-9:
                    raise AssertionError("Common-start trajectory differs from original saved study")
            summaries.append(dict(
                p0=p0, rho=rho, rival_conjecture=z, x0=x0,
                equilibrium_policy_count=len(equilibria), horizon=horizon,
                initial_product2_fraction=float(frame.alpha_a.iloc[0]),
                terminal_x=final, long_run_limit=limit,
                limit_product2_fraction=limit_alpha,
                limit_kind="mixed_holding" if holding else "fixed_product",
                terminal_distance_to_limit=abs(final-limit),
                limit_matches_conjecture=bool(abs(limit-z)<1e-10),
                repelling_thresholds=";".join(format(root, ".15g") for root in repelling),
                maximum_ode_refinement_change=refinement_change,
            ))
            subset = frame[["time", "x_a", "x_b", "alpha_a", "alpha_b",
                            "demand_a", "demand_b", "outside_rate"]].copy()
            subset.insert(0, "x0", x0)
            subset.insert(0, "rival_conjecture", z)
            trajectories.append(subset)
            runs.append(dict(rival_conjecture=z, x0=x0,
                             diagnostics=result["diagnostics"], events=result["events"]))

    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(folder/"summary.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(folder/"trajectories.csv", index=False)
    manifest = dict(
        inputs=inputs, source_sha256=hashes, policy_nodes=policy_nodes,
        policy_count=len(equilibria), path_count=len(summaries),
        maximum_ode_refinement_change=maximum_refinement_change,
        maximum_common_start_baseline_change=maximum_baseline_change,
        maximum_symmetry_error=max(run["diagnostics"]["symmetry_error"] for run in runs),
        policies=[dict(rival_conjecture=eq["x_star"], policy=eq["policy"]) for eq in equilibria],
        runs=runs,
        interpretation="Symmetric mature starts under saved optimal policies and actual joint-state demand. "
                       "Phase-line limits are distinct from finite-horizon terminal states. "
                       "A destination different from the optimization conjecture is not a new verified equilibrium.",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0s", type=float, nargs="+")
    parser.add_argument("--rhos", type=float, nargs="+")
    parser.add_argument("--horizon", type=float, default=3000.)
    parser.add_argument("--samples", type=int, default=3001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.horizon) or args.horizon <= 0 or args.samples < 2 or args.workers < 1:
        parser.error("Use a positive horizon, at least two samples, and at least one worker")
    config = json.loads((ROOT/"config.json").read_text())
    p0s = args.p0s if args.p0s is not None else config["p0s"]
    rhos = args.rhos if args.rhos is not None else config["rhos"]
    if len(set(p0s)) != len(p0s) or len(set(rhos)) != len(rhos):
        parser.error("Parameter lists must contain distinct values")
    data_root = ROOT/"data/initial_conditions_sweep"
    figure_root = ROOT/"figures/15_initial_conditions"
    original = pd.read_csv(ROOT/"data/fluid_summary.csv")
    tasks = []
    for p0 in p0s:
        for rho in rhos:
            load_policies(p0, rho, config)
            baseline = original[(original.p0 == p0)&(original.rho == rho)].to_dict("records")
            tasks.append((p0, rho, config, str(data_root), args.horizon, args.samples,
                          args.figures_only, baseline))
    manifests = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(calculate_case, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            p0, rho = futures[future]
            manifests[(p0, rho)] = future.result()
            print(f"Verified p0={p0:g}, rho={rho:g}: "
                  f"{manifests[(p0, rho)]['path_count']} paths ({len(manifests)}/{len(tasks)} cases)", flush=True)

    from matplotlib.backends.backend_pdf import PdfPages
    from oblivious.fluid_oblivious.code.initial_condition_figures import plot_case

    figure_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    with PdfPages(figure_root/"initial_conditions_all_p0.pdf") as all_pages:
        all_pages.infodict()["Title"] = "Fluid competition: initial conditions across outside options"
        for p0 in p0s:
            figure_dir = figure_root/f"p0_{tag(p0)}"
            figure_dir.mkdir(parents=True, exist_ok=True)
            with PdfPages(figure_dir/"initial_conditions_all_rhos.pdf") as p0_pages:
                p0_pages.infodict()["Title"] = f"Fluid competition: initial conditions, p0={p0:g}"
                for rho in rhos:
                    folder = data_root/f"p0_{tag(p0)}"/f"rho_{tag(rho)}"
                    manifest = manifests[(p0, rho)]
                    full = pd.read_csv(folder/"trajectories.csv", float_precision="round_trip")
                    frames = {(float(z), float(x0)): frame for (z, x0), frame
                              in full.groupby(["rival_conjecture", "x0"], sort=False)}
                    equilibria = [dict(x_star=row["rival_conjecture"], policy=row["policy"])
                                  for row in manifest["policies"]]
                    params = Parameters(**config["parameters"], rho=rho)
                    plot_case(params, p0, equilibria, DEFAULT_INITIALS, frames, figure_dir,
                              args.horizon, pdf_pages=[all_pages, p0_pages])
                    summaries.append(pd.read_csv(folder/"summary.csv", float_precision="round_trip"))
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(data_root/"summary.csv", index=False)
    overview = dict(
        p0s=p0s, rhos=rhos, initial_states=DEFAULT_INITIALS,
        horizon=args.horizon, samples=args.samples, case_count=len(manifests),
        policy_count=sum(row["policy_count"] for row in manifests.values()),
        path_count=len(combined),
        maximum_ode_refinement_change=max(row["maximum_ode_refinement_change"] for row in manifests.values()),
        maximum_common_start_baseline_change=max(row["maximum_common_start_baseline_change"] for row in manifests.values()),
        maximum_symmetry_error=max(row["maximum_symmetry_error"] for row in manifests.values()),
        interpretation="Limits describe saved-policy phase lines, not additional equilibrium verification.",
        cases=[dict(p0=p0, rho=rho, policy_count=manifests[(p0, rho)]["policy_count"],
                    path_count=manifests[(p0, rho)]["path_count"])
               for p0 in p0s for rho in rhos],
    )
    (data_root/"manifest.json").write_text(json.dumps(overview, indent=2, allow_nan=False)+"\n")
    lines = ["# Fluid competition: initial-condition plots", "",
             "[All outside options, combined PDF](initial_conditions_all_p0.pdf).", "",
             "Each case applies every saved verified optimal policy to the same six symmetric mature starts.",
             "Top panels show actual-state trajectories; bottom panels show their phase-line limits.",
             "Mixed holding states are included. A nearly flat trajectory need not be stationary.", ""]
    for p0 in p0s:
        folder = f"p0_{tag(p0)}"
        lines.extend([f"## Outside option p0 = {p0:g}", "",
                      f"[All retention values, PDF]({folder}/initial_conditions_all_rhos.pdf).", ""])
        for rho in rhos:
            stem = f"{folder}/initial_conditions_rho_{tag(rho)}"
            lines.append(f"- rho = {rho:g}: [PNG]({stem}.png) | [PDF]({stem}.pdf)")
        lines.append("")
    (figure_root/"README.md").write_text("\n".join(lines)+"\n")
    print(f"Saved {len(manifests)} case figures and {len(combined)} paths in {figure_root}", flush=True)


if __name__ == "__main__":
    main()
