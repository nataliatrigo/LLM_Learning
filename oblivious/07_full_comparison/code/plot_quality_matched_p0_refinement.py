"""Compare monopoly with endogenous p0 at the rival's mean success per sale.

For each saved endogenous-p0 equilibrium, calibrate the monopoly outside option
to the rival's true quality conditional on its sales. This reference is distinct
from the equilibrium's scalar p0 (the calendar-time mean Thompson score).
"""

import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "quality_matched_distribution_helpers",
    Path(__file__).with_name("plot_quality_matched_refinement.py"))
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)
BASE, ROOT, RUN, COMMON = QUALITY.BASE, QUALITY.ROOT, QUALITY.RUN, QUALITY.COMMON

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oblivious.oe_p0.code.core import fixed_demand, posterior_mean
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.primitives import Parameters, demand_matrix
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from quality_matched_p0_figures import make_figures

FAMILY = "quality_matched_endogenous_p0"


def collect_reference(campaign, n, rhos):
    """Read certified p0 cases and reconstruct both scalar and sales metrics."""
    path = campaign/"manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    BASE.require("p0" in config["models"] and n in config["grids"]
                 and set(rhos) <= set(config["rhos"]),
                 "Grid/rhos/endogenous p0 absent from campaign")
    inputs = {str(path.relative_to(ROOT)): COMMON.digest(path)}
    rows = []
    for rho in rhos:
        base = RUN.case_dir(campaign, "p0", rho, n)/"solution"
        case = RUN.load_checked(base)
        BASE.require(case is not None and RUN.verified(case.get("selected"))
                     and case.get("demand_audit", {}).get("verified")
                     and case.get("campaign_fingerprint") == manifest["fingerprint"]
                     and case.get("rho") == rho and case.get("n") == n
                     and case["configuration"]["parameters"] == dict(config["parameters"], rho=rho)
                     and case["configuration"]["order"] == config["order"],
                     f"Missing or incompatible certified endogenous p0: rho={rho}, n={n}")
        candidate = case["selected"]
        grid, alpha, nu = (np.asarray(candidate[k]) for k in ("grid", "policy", "nu"))
        BASE.require(np.array_equal(grid, np.linspace(0, 1, n)) and alpha.shape == (n,)
                     and np.isfinite(alpha).all() and np.all((alpha >= 0) & (alpha <= 1))
                     and nu.shape == (n, n) and np.isfinite(nu).all() and np.all(nu >= 0)
                     and abs(nu.sum()-1) < 1e-10 and np.max(abs(nu-nu.T)) < 1e-9,
                     "Expected a feasible policy and symmetric normalized p0 law on the common grid")
        params = Parameters(**case["configuration"]["parameters"])
        equilibrium_p0 = float(candidate["p0"])
        score_mean = float(nu.sum(axis=0) @ posterior_mean(grid, params))
        closure_error = abs(score_mean-equilibrium_p0)
        BASE.require(0 < equilibrium_p0 < 1 and closure_error <= config["tol"]
                     and abs(score_mean-candidate["F"]) < 1e-10
                     and abs(score_mean-equilibrium_p0-candidate["G"]) < 1e-10,
                     "The saved endogenous p0 does not match its calendar-time score closure")
        fixed = fixed_demand(grid, equilibrium_p0, params)
        evaluation = evaluate_policy(grid, fixed, alpha, params)
        regret = float(np.max(np.maximum(evaluation["advantage"], 0)
                              - alpha*evaluation["advantage"]))
        BASE.require(np.max(abs(fixed-candidate["d_fixed"])) < 1e-12
                     and evaluation["residual"] < 1e-10 and regret < 1e-9
                     and np.max(abs(evaluation["value"]-candidate["value"])) < 1e-8,
                     "The p0 reference policy does not verify in its fixed-benchmark Bellman")
        demand, _ = demand_matrix(grid, params, config["order"])
        sold = (nu*demand).sum(axis=1)
        sale_rate = float(sold.sum())
        usage = float(sold @ alpha / sale_rate)
        embedded = purchase_chain_statistics(grid, alpha, params)
        sale_l1 = float(abs(sold/sale_rate-embedded["sale_stationary"]).sum())
        BASE.require(abs(sale_rate-.5) < 1e-10
                     and abs(usage-case["predicted"]["product2_market"]) < 1e-10
                     and sale_l1 < 1e-9,
                     "Endogenous p0 own-sales usage disagrees with the actual game or sale-time chain")
        COMMON.record(base, inputs)
        hashes = base.with_name(base.name+"_sha256.json")
        inputs[str(hashes.relative_to(ROOT))] = COMMON.digest(hashes)
        rows.append(dict(rho=rho, n=n, model="p0", equilibrium_p0=equilibrium_p0,
                         equilibrium_score_mean=score_mean, scalar_consistency_error=closure_error,
                         product2_conditional=usage, sale_rate=sale_rate,
                         product2_calendar=sale_rate*usage,
                         reference_conditional_regret=regret,
                         reference_value_residual=evaluation["residual"],
                         reference_sale_stationary_l1=sale_l1,
                         mixed_nodes=int(np.sum((alpha > 1e-9) & (alpha < 1-1e-9))),
                         verified=True, source=str(base.relative_to(ROOT))))
    return rows, inputs, config


def build_pairs(references, n, parameters, order, action_tolerance):
    """Reuse the same quality matching and original monopoly solver as before."""
    pairs, data, policies = QUALITY.build_pairs(
        references, n, parameters, order, action_tolerance)
    data = data.rename(columns={
        "oe_product2_conditional": "p0_product2_conditional",
        "monopoly_minus_oe_pp": "monopoly_minus_p0_pp",
        "oe_sale_rate": "p0_sale_rate", "oe_source": "p0_source"})
    reference_by_rho = {row["rho"]: row for row in references}
    for column in ("equilibrium_p0", "equilibrium_score_mean", "scalar_consistency_error",
                   "reference_conditional_regret", "reference_value_residual",
                   "reference_sale_stationary_l1", "mixed_nodes"):
        data[column] = data.rho.map(lambda rho: reference_by_rho[rho][column])
    data["reference_model"] = "p0"
    policies["model"] = policies.model.replace({"oe": "p0"})
    policies["equilibrium_p0"] = policies.rho.map(
        lambda rho: reference_by_rho[rho]["equilibrium_p0"])
    for pair in pairs:
        for previous, current in (("oe_grid", "p0_grid"), ("oe_policy", "p0_policy"),
                                  ("oe_product2_conditional", "p0_product2_conditional"),
                                  ("oe_source", "p0_source")):
            pair[current] = pair.pop(previous)
        pair["equilibrium_p0"] = reference_by_rho[pair["rho"]]["equilibrium_p0"]
    return pairs, data, policies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="refinement_101_201")
    parser.add_argument("--grid", type=int, default=101)
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name or args.name in ("", ".", "..") or args.grid < 3:
        parser.error("Require a campaign folder name and grid>=3")
    campaign = RUN.OUTPUT/args.name
    manifest = json.loads((campaign/"manifest.json").read_text())
    rhos = sorted(rho for rho in manifest["configuration"]["rhos"] if rho > .5)
    BASE.require(bool(rhos), "Need at least one rho > 0.5")
    references, inputs, config = collect_reference(campaign, args.grid, rhos)
    paper_path = ROOT/"forgetting/Code/experiments/config_paper.json"
    action_tolerance = json.loads(paper_path.read_text())["solver"]["action_tolerance"]
    inputs[str(paper_path.relative_to(ROOT))] = COMMON.digest(paper_path)
    pairs, data, policies = build_pairs(
        references, args.grid, config["parameters"], config["order"], action_tolerance)
    data_dir = campaign/FAMILY/f"n{args.grid:03d}"
    directory = ROOT/"oblivious/07_full_comparison/Figures"/args.name/f"n{args.grid:03d}"/FAMILY
    data_dir.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = dict(campaign=args.name, grid=args.grid, rhos=rhos,
                         parameters=config["parameters"], reference_model="p0",
                         rho_lower_bound_exclusive=.5, p0_filter=None,
                         reference_rule="p0_ref = p1 + (p2-p1) * rival_product2_conditional_endogenous_p0",
                         action_tolerance=action_tolerance)
    source_paths = [Path(__file__), Path(__file__).with_name("quality_matched_p0_figures.py"),
                    Path(QUALITY.__file__), Path(BASE.__file__), Path(COMMON.__file__),
                    Path(COMMON.PLOT.__file__), Path(RUN.__file__),
                    ROOT/"forgetting/Code/model.py", ROOT/"oblivious/03_oe_following_monopoly/code/event_chain.py"]
    source_paths += [ROOT/f"oblivious/02_oe_p0/code/{name}" for name in
                     ("core.py", "policy.py", "primitives.py", "positive_stationary.py",
                      "positive_stationary.cpp", "run_experiment.py")]
    sources = {str(p.relative_to(ROOT)): COMMON.digest(p) for p in source_paths}
    fingerprint = RUN.fingerprint(dict(configuration, input_sha256=inputs), sources)
    base = data_dir/"results"
    RUN.save_checked(base, dict(configuration=configuration, fingerprint=fingerprint, pairs=pairs))
    data.to_csv(data_dir/"comparison.csv", index=False)
    policies.to_csv(data_dir/"policies.csv", index=False)
    pd.DataFrame(references).to_csv(data_dir/"endogenous_p0_reference.csv", index=False)
    with plt.rc_context(BASE.STYLE):
        figures = make_figures(pairs, data, args.grid, directory, BASE.save_figure)
    generated = [base.with_suffix(s) for s in (".json", ".npz")]
    generated += [base.with_name(base.name+"_sha256.json"), data_dir/"comparison.csv",
                  data_dir/"policies.csv", data_dir/"endogenous_p0_reference.csv"]
    report = dict(configuration=configuration, fingerprint=fingerprint, input_sha256=inputs,
                  source_sha256=sources,
                  generated_sha256={str(p.relative_to(ROOT)): COMMON.digest(p) for p in generated},
                  figures={str(p.relative_to(ROOT)): COMMON.digest(p) for p in figures},
                  verified_cases=len(pairs), maximum_calibration_error=float(data.calibration_error.max()),
                  maximum_scalar_consistency_error=float(data.scalar_consistency_error.max()),
                  maximum_conditional_regret=float(data.conditional_regret.max()),
                  definitions=dict(
                      usage="Fraction of each seller's OWN SALES using product 2, in both models",
                      reference="Rival's true success probability conditional on its sales in the saved endogenous-p0 equilibrium",
                      equilibrium_p0="Saved endogenous scalar: rival's calendar-time average Thompson posterior mean; reported separately and not used as monopoly outside option",
                      matching="Monopoly outside option equals rival quality; monopoly's resulting quality may differ",
                      scope="Same-grid stationary comparison; original monopoly solved at exact unrounded p0_ref; saved endogenous-p0 equilibria read-only; no new consistency closure or Monte Carlo",
                      interpretation="Endogenous p0 already uses a fixed-benchmark Bellman; this comparison changes that benchmark to rival quality, not a full rival distribution"))
    RUN.atomic_json(data_dir/"manifest.json", report)
    RUN.atomic_json(directory/"figure_manifest.json", report)
    print(f"Saved {len(figures)//2} comparisons, PNG and PDF: {directory}")
    print(data[["rho", "equilibrium_p0", "p0_ref", "monopoly_product2_conditional",
                "p0_product2_conditional", "monopoly_minus_p0_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
