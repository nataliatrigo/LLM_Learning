"""Plot the optimal monopoly feedback rules used by the saved fluid paths.

Read the verified piecewise policies, rather than estimating a policy from
trajectories or solving a different HJB. Keep isolated holding mixtures
separate from the pure-product intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from simulate_trajectories import PiecewisePolicy, Primitives


ROOT = Path(__file__).resolve().parents[1]


def token(value):
    return str(float(value)).replace(".", "p")


def inspect_dataset(dataset, config):
    parameters = dataset["parameters"]
    expected = dict(config["parameters"], p0=parameters["p0"])
    if parameters != expected:
        raise ValueError("Saved trajectory economics differ from the current configuration")
    if [case["rho"] for case in dataset["cases"]] != config["rhos"]:
        raise ValueError("Saved retention cases differ from the trajectory configuration")
    primitives = Primitives.from_mapping(parameters)
    interval_rows, threshold_rows, state_rows = [], [], []
    compact_cases = []
    maximum_field_alpha_error = 0.
    for case in dataset["cases"]:
        rho = case["rho"]
        base = dict(p0=parameters["p0"], rho=rho)
        policy = PiecewisePolicy(primitives, rho, case["roots"],
                                 [row["product"] for row in case["intervals"]])
        edges = [0., *policy.roots, 1.]
        for index, interval in enumerate(case["intervals"]):
            if abs(interval["lo"]-edges[index]) > 1e-11 or abs(interval["hi"]-edges[index+1]) > 1e-11:
                raise AssertionError("Saved policy intervals do not meet at their thresholds")
            product, alpha = policy.action_at((interval["lo"]+interval["hi"])/2)
            if product != interval["product"]:
                raise AssertionError("Policy interval action does not match the saved rule")
            interval_rows.append(dict(base, lo=interval["lo"], hi=interval["hi"],
                                      product=product, product2_fraction=alpha))
        kinds = [policy.root_kind(i) for i in range(len(policy.roots))]
        if kinds != case["root_kinds"]:
            raise AssertionError("Saved threshold classification changed")
        for index, root in enumerate(policy.roots):
            product, alpha = policy.action_at(root)
            threshold_rows.append(dict(base, x=root, kind=kinds[index],
                                       product_at_threshold=product,
                                       product2_fraction_at_threshold=alpha))
        actual_holding = policy.holding_points()
        saved_holding = case["holding_points"]
        if len(actual_holding) != len(saved_holding):
            raise AssertionError("Holding-mixture count changed")
        for actual, saved in zip(actual_holding, saved_holding):
            if abs(actual["x"]-saved["x"]) > 1e-10 or abs(actual["alpha"]-saved["alpha"]) > 1e-10:
                raise AssertionError("Holding mixture differs from the trajectory policy")
            if not 0 <= actual["alpha"] <= 1:
                raise AssertionError("Invalid holding fraction")
            if abs(primitives.p1+primitives.delta_p*actual["alpha"]-actual["x"]) > 1e-11:
                raise AssertionError("Holding mixture does not have zero drift")
            state_rows.append(dict(base, x=actual["x"], product2_fraction=actual["alpha"],
                                   kind="mixed_holding"))
        for x, product in [(primitives.p1, 1), (primitives.p2, 2)]:
            selected, alpha = policy.action_at(x)
            if selected == product:
                state_rows.append(dict(base, x=x, product2_fraction=alpha,
                                       kind=f"pure_product{product}"))
        for field in case["field"]:
            product, alpha = policy.action_at(field["x"])
            error = abs(alpha-field["alpha"])
            maximum_field_alpha_error = max(maximum_field_alpha_error, error)
            if product != field["product"] or error > 1e-9:
                raise AssertionError("Displayed policy differs from the saved trajectory field")
        diagnostics = case["diagnostics"]
        if max(diagnostics["hjb_residual"], diagnostics["refined_hjb_residual"]) > 1e-8:
            raise AssertionError("The saved numerical HJB did not meet its residual criterion")
        if len(diagnostics["refined_roots"]) != len(policy.roots):
            raise AssertionError("Policy topology changed under the saved grid refinement")
        if diagnostics["maximum_root_change"] > 5/(diagnostics["policy_nodes"]-1):
            raise AssertionError("Saved threshold refinement exceeded its criterion")
        compact_cases.append(dict(
            rho=rho, intervals=case["intervals"], roots=policy.roots,
            root_kinds=kinds, holding_points=actual_holding,
            stationary_states=[row for row in state_rows if row["rho"] == rho],
            diagnostics=diagnostics,
        ))
    compact = dict(parameters=parameters, cases=compact_cases,
                   policy="Product-2 fraction alpha*(x), conditional on an own sale",
                   state="Mature normalized reputation x=(1-rho)S",
                   maximum_saved_field_alpha_difference=maximum_field_alpha_error)
    return interval_rows, threshold_rows, state_rows, compact


def write_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"outputs/p0_sweep")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/optimal_policies")
    parser.add_argument("--p0s", type=float, nargs="+", default=[.1, .3, .5, .7, .8, .9])
    args = parser.parse_args()
    if len(set(args.p0s)) != len(args.p0s):
        parser.error("Outside-option values must be distinct")
    config_path = ROOT/"config.json"
    config = json.loads(config_path.read_text())
    datasets = []
    intervals, thresholds, states, compact_data = [], [], [], []
    sources = {}
    for p0 in args.p0s:
        source = args.input_dir/f"p0_{token(p0)}"/"trajectory-data.json"
        dataset = json.loads(source.read_text())
        if dataset["parameters"]["p0"] != p0:
            raise ValueError(f"Outside-option label differs from saved data: {source}")
        interval_rows, threshold_rows, state_rows, compact = inspect_dataset(dataset, config)
        datasets.append(dataset)
        intervals.extend(interval_rows)
        thresholds.extend(threshold_rows)
        states.extend(state_rows)
        compact_data.append(compact)
        sources[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()

    from matplotlib.backends.backend_pdf import PdfPages
    from policy_figures import make_policy_figure

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output_dir/"optimal-policies-all-p0.pdf") as pages:
        pages.infodict()["Title"] = "Optimal fluid monopoly policies with forgetting"
        pages.infodict()["Subject"] = "Feedback rules across outside options and retention values"
        for dataset, compact in zip(datasets, compact_data):
            p0 = dataset["parameters"]["p0"]
            folder = args.output_dir/f"p0_{token(p0)}"
            folder.mkdir(parents=True, exist_ok=True)
            make_policy_figure(dataset, folder, pdf_pages=pages)
            (folder/"policy-data.json").write_text(json.dumps(compact, indent=2, allow_nan=False)+"\n")
            print(f"Saved p0={p0:g}: {len(dataset['cases'])} optimal policies", flush=True)
    write_table(args.output_dir/"policy-intervals.csv", intervals)
    write_table(args.output_dir/"policy-thresholds.csv", thresholds)
    write_table(args.output_dir/"stationary-states.csv", states)
    code_sources = [Path(__file__), ROOT/"code/policy_figures.py", ROOT/"code/simulate_trajectories.py"]
    manifest = dict(
        p0s=args.p0s, rhos=config["rhos"], case_count=sum(len(d["cases"]) for d in datasets),
        policy_nodes=config["policy_nodes"], refinement_nodes=config["refinement_nodes"],
        source_datasets_sha256=sources,
        source_code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in code_sources},
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        threshold_count=len(thresholds),
        holding_state_count=sum(row["kind"] == "mixed_holding" for row in states),
        pure_stationary_state_count=sum(row["kind"] != "mixed_holding" for row in states),
        maximum_saved_field_alpha_difference=max(d["maximum_saved_field_alpha_difference"] for d in compact_data),
        interpretation="One saved numerical HJB feedback policy per parameter case. "
                       "Pure intervals are 0 or 1; holding mixtures are isolated states, not ramps. "
                       "Initial reputations are not inputs to these feedback-rule plots.",
    )
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    lines = ["# Optimal fluid monopoly policies with forgetting", "",
             "[All outside options, combined PDF](optimal-policies-all-p0.pdf).", "",
             "Each page shows six retention values and the optimal feedback rule alpha*(x).",
             "0% means product 1; 100% means product 2. Orange diamonds mark isolated holding mixtures.",
             "Open black circles identify pure stationary states. Dashed vertical lines are policy thresholds.",
             "Policy thresholds are not automatically stationary states. Initial conditions do not change the rule.", ""]
    for p0 in args.p0s:
        folder = f"p0_{token(p0)}"
        lines.append(f"- p0 = {p0:g}: [PNG]({folder}/optimal-policies.png) | [PDF]({folder}/optimal-policies.pdf)")
    lines.extend(["", "Numerical policies are the exact piecewise rules used in the saved trajectory study.",
                  "HJB residual and grid-refinement checks are retained from those datasets.",
                  "CSV files provide every pure interval, threshold action, and stationary state.", ""])
    (args.output_dir/"README.md").write_text("\n".join(lines))
    print(f"Verified {manifest['case_count']} policies, {len(thresholds)} thresholds, "
          f"and {len(states)} stationary states", flush=True)


if __name__ == "__main__":
    main()
