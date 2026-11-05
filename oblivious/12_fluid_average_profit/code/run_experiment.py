"""Run the entire configured grid, refinement, audits, and English reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/average-profit-mpl")
import numpy as np
import pandas as pd
import scipy

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability
from oblivious.fluid_oblivious.code.model import FixedRivalDemand
from .model import AverageProfitModel, PreparedCompetition, global_maxima, solve_case

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    request = json.loads(path.read_text())
    source_path = (path.parent/request["source_config"]).resolve()
    source = json.loads(source_path.read_text())
    # Economic primitives and the active parameter grids are read at runtime.
    economic = {key: source["parameters"][key] for key in ("p1", "p2", "c1", "c2", "R")}
    config = dict(**request, parameters=economic, p0s=source["p0s"], rhos=source["rhos"],
                  source_configuration=source,
                  source_config_path=str(source_path.relative_to(PROJECT)),
                  source_config_sha256=digest(source_path),
                  source_discount_ignored=source["parameters"].get("gamma"),
                  discount_used_in_objective=False)
    return config


def refined_settings(config):
    settings = dict(config["numerics"])
    settings.update(own_points=settings["refined_own_points"],
                    search_points=settings["refined_search_points"],
                    quadrature_order=settings["refined_quadrature_order"])
    return settings


def refinement_rows(coarse, fine, limit):
    rows = []
    for model in ("monopoly", "competition"):
        a = sorted((r for r in coarse["solutions"] if r["model"] == model), key=lambda r: r["u"])
        b = sorted((r for r in fine["solutions"] if r["model"] == model), key=lambda r: r["u"])
        matched = len(a) == len(b)
        # Ordered matching here is only a refinement diagnostic at fixed p0,rho;
        # no across-parameter branch identity or selection is imposed.
        pairs = list(zip(a, b)) if matched else []
        rows.append(dict(p0=fine["p0"], rho=fine["rho"], model=model,
                         coarse_solution_count=len(a), fine_solution_count=len(b),
                         solution_count_matches=matched,
                         max_u_change=max((abs(x["u"]-y["u"]) for x,y in pairs), default=0.) if matched else None,
                         max_demand_change=max((abs(x["demand"]-y["demand"]) for x,y in pairs), default=0.) if matched else None,
                         max_profit_change=max((abs(x["average_profit"]-y["average_profit"]) for x,y in pairs), default=0.) if matched else None,
                         u_change_tolerance=limit,
                         passed=matched and all(abs(x["u"]-y["u"]) <= limit for x,y in pairs)))
    return rows


def independent_audits(case, params, config):
    """Reevaluate every competitive finalist/BR peak by adaptive integration."""
    p0 = case["p0"]
    model = AverageProfitModel(params, p0, case["settings"]["quadrature_order"])
    rows = []
    for candidate in case["competition"]["candidates"]:
        rival = candidate["u"]
        own_choices = {0., 1., rival, *(r["u"] for r in candidate["best_responses"])}
        independent_current = outside_demand_probability(float(model.q(rival)), float(model.q(rival)), params, p0)*float(model.margin(rival))
        for own in sorted(own_choices):
            independent = outside_demand_probability(float(model.q(own)), float(model.q(rival)), params, p0)
            production = float(model.competition_demand(own, rival))
            error = abs(independent-production)
            profit_error = error*float(model.margin(own))
            gain = max(0., independent*float(model.margin(own))-independent_current)
            accepted = candidate["verified"]
            tolerance = candidate["profit_tolerance"]
            rows.append(dict(p0=p0, rho=params.rho, rival_u=rival, own_u=own,
                             candidate_verified=accepted, production_demand=production,
                             independent_demand=independent, absolute_error=error,
                             profit_error=profit_error, independent_deviation_gain=gain,
                             profit_tolerance=tolerance,
                             passed=error <= config["numerics"]["demand_audit_atol"]
                                    and profit_error <= .1*tolerance
                                    and (not accepted or gain <= tolerance)))
    return rows


def scan_quadrature_audits(case, params, config):
    """Check accelerated scan integrals away from just the finalized peaks."""
    model = AverageProfitModel(params, case["p0"], case["settings"]["quadrature_order"])
    prepared = PreparedCompetition(model, case["settings"]["own_points"])
    indices = np.unique(np.r_[np.arange(0, len(prepared.grid), 16), len(prepared.grid)-1])
    own = prepared.grid[indices]
    exact = FixedRivalDemand(params, case["p0"], model.q(own), model.order)
    rivals = sorted({0., .25, .5, .75, 1., *(row["u"] for row in case["competition"]["candidates"])})
    rows = []
    for rival in rivals:
        scan = prepared.profits(rival)[indices]
        reference = exact(float(model.q(rival)))*model.margin(own)
        error = float(np.max(abs(scan-reference)))
        tolerance = config["numerics"]["profit_atol"]+config["numerics"]["profit_rtol"]*float(np.max(abs(reference)))
        rows.append(dict(p0=case["p0"], rho=params.rho, rival_u=rival, own_probe_count=len(own),
                         max_profit_error=error, profit_tolerance=tolerance,
                         passed=error <= .1*tolerance))
    return rows


def comparisons(solutions):
    rows = []
    for (p0, rho), case in solutions.groupby(["p0", "rho"]):
        m = case[case.model.eq("monopoly")]
        c = case[case.model.eq("competition")]
        for monopoly in m.to_dict("records"):
            for competition in c.to_dict("records"):
                row = dict(p0=p0, rho=rho, monopoly_solution_id=monopoly["solution_id"],
                           competition_solution_id=competition["solution_id"])
                for metric in ("u", "reputation", "demand", "average_profit", "product2_per_seller", "product2_market"):
                    row["monopoly_"+metric] = monopoly[metric]
                    row["competition_"+metric] = competition[metric]
                    row["competition_minus_monopoly_"+metric] = competition[metric]-monopoly[metric]
                # Exact symmetric decomposition: change in per-seller use =
                # average demand * change in choice + average choice * change in demand.
                row["choice_contribution_to_seller_use_change"] = .5*(monopoly["demand"]+competition["demand"])*(competition["u"]-monopoly["u"])
                row["demand_contribution_to_seller_use_change"] = .5*(monopoly["u"]+competition["u"])*(competition["demand"]-monopoly["demand"])
                rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["competition_minus_monopoly_"+metric for metric in
                                ("u", "demand", "product2_per_seller", "product2_market")])


def write_diagnostic(output, config, cases):
    target = config["diagnostic"]
    case = next((case for case in cases if np.isclose(case["p0"], target["p0"])
                 and np.isclose(case["rho"], target["rho"])), None)
    if case is None:
        dump(output/"data/diagnostic.json", dict(status="target_not_in_active_grid", **target))
        return
    params = Parameters(**config["parameters"], rho=case["rho"])
    model = AverageProfitModel(params, case["p0"], case["settings"]["quadrature_order"])
    settings = config["numerics"]
    prepared = PreparedCompetition(model, settings["refined_own_points"])
    kwargs = {key: settings[key] for key in ("profit_atol", "profit_rtol", "u_tolerance", "optimizer_xatol")}
    grid = np.linspace(0., 1., target["points"])
    curves, best_responses, payoffs = [], [], []
    for rival in grid:
        br = global_maxima(lambda own: model.competition_profit(own, rival),
                           points=len(prepared.grid), sampled_values=prepared.profits(rival), **kwargs)
        current = float(model.competition_profit(rival, rival))
        curves.append(dict(u=float(rival), monopoly_profit=float(model.monopoly_profit(rival)),
                           symmetric_competition_profit=current,
                           best_response_profit=max(br["max_value"], current),
                           deviation_gain=max(0., br["max_value"]-current),
                           best_response_u_low=min(row["u"] for row in br["maximizers"]),
                           best_response_u_high=max(row["u"] for row in br["maximizers"])))
        best_responses.extend(dict(rival_u=float(rival), best_response_u=row["u"], profit=row["value"])
                              for row in br["maximizers"])
    fixed_rivals = sorted({0., 1., *(row["u"] for row in case["competition"]["candidates"])})
    for rival in fixed_rivals:
        values = model.competition_profit(grid, rival)
        payoffs.extend(dict(u=float(own), rival_u=float(rival), profit=float(profit))
                       for own, profit in zip(grid, values))
    pd.DataFrame(curves).to_csv(output/"data/diagnostic_curves.csv", index=False)
    pd.DataFrame(best_responses).to_csv(output/"data/diagnostic_best_responses.csv", index=False)
    pd.DataFrame(payoffs).to_csv(output/"data/diagnostic_payoffs.csv", index=False)
    # Deliberately calculate the wrong diagonal optimum as a labelled diagnostic,
    # never as a competition solution.
    diagonal = global_maxima(lambda u: model.competition_profit(u,u), points=len(prepared.grid), **kwargs)
    diagonal_rows = []
    for row in diagonal["maximizers"]:
        u = row["u"]
        br = global_maxima(lambda v: model.competition_profit(v,u), points=len(prepared.grid), **kwargs)
        diagonal_rows.append(dict(u=u, diagonal_profit=row["value"],
                                  unilateral_deviation_gain=max(0., br["max_value"]-row["value"])))
    dump(output/"data/diagnostic.json", dict(status="completed", p0=case["p0"], rho=case["rho"],
         monopoly=[row for row in case["solutions"] if row["model"] == "monopoly"],
         equilibria=[row for row in case["solutions"] if row["model"] == "competition"],
         candidates=case["competition"]["candidates"],
         deviation_tolerance=max(row["profit_tolerance"] for row in case["competition"]["candidates"]),
         diagonal_objective_optima_for_diagnostic_only=diagonal_rows))


def write_results(output, config, frame, paired, validation):
    fmt = lambda x: f"{x:.6f}"
    lines = ["# Experiment 12 results: long-run average profit", "",
             f"Completed {len(config['p0s'])*len(config['rhos'])} parameter combinations. "
             f"The active grids were read from `{config['source_config_path']}`: "
             f"p0 = {config['p0s']}; rho = {config['rhos']}.", "",
             "Economic parameters: "+", ".join(f"{key}={value:g}" for key,value in config["parameters"].items())+". "
             f"The source gamma={config['source_discount_ignored']} is recorded for provenance and excluded from the objective.", "",
             "Both markets are reoptimized over all constant proportions u in [0,1]. "
             "The objective is long-run sales per unit time times margin: D(q(u))*m(u). "
             "Demand retains the rho-dependent Beta posterior and the outside option. "
             "Initial conditions, transient earnings and discounting do not enter this criterion.", ""]
    for model in ("monopoly", "competition"):
        part = frame[frame.model.eq(model)]
        counts = part.groupby(["p0", "rho"]).size()
        lines.append(f"- {model.capitalize()}: {len(part)} detected verified solutions; "
                     f"{int((counts>1).sum())} cases with multiple solutions; "
                     f"{len(config['p0s'])*len(config['rhos'])-len(counts)} cases without a verified solution; "
                     f"{int(((part.u>1e-6)&(part.u<1-1e-6)).sum())} interior mixtures.")
    eps = 2e-6
    lines += ["", "## Product choice, seller demand, and total use", "",
              "Every detected monopoly/competition pair is retained in `data/comparisons.csv`; no solutions are averaged.", ""]
    for metric, description in (("u", "product-2 fraction among sales"), ("demand", "demand per seller"),
                                 ("product2_per_seller", "product-2 use per seller per unit time"),
                                 ("product2_market", "total market product-2 use per unit time")):
        diff = paired["competition_minus_monopoly_"+metric]
        lines.append(f"- Competition {description}: higher in {int((diff>eps).sum())} pairs, "
                     f"lower in {int((diff < -eps).sum())}, equal within {eps:g} in {int((abs(diff)<=eps).sum())}.")
    lines += ["", "A higher fraction u is a change in product choice. The rate D*u also changes when demand changes. "
              "The table records an exact decomposition of the per-seller rate difference into choice and demand contributions. "
              "The market measure uses D_M*u_M for monopoly and 2*D_C*u_C for competition.", "",
              "## All parameter cases", "",
              "Each row is one verified solution; multiple rows would display multiplicity explicitly.", "",
              "| p0 | rho | Market | u | q(u) | Demand / seller | Average profit / seller | Product 2 / seller / time | Product 2 / market / time |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in frame.sort_values(["p0", "rho", "model", "u"]).to_dict("records"):
        lines.append("| "+" | ".join([f"{row['p0']:g}", f"{row['rho']:g}",row["model"],
                     *(fmt(row[key]) for key in ("u", "reputation", "demand", "average_profit", "product2_per_seller", "product2_market"))])+" |")
    target = frame[np.isclose(frame.p0,.8)&np.isclose(frame.rho,.9)]
    if len(target):
        lines += ["", "## Focal diagnostic: p0=0.8, rho=0.9", ""]
        for row in target.to_dict("records"):
            lines.append(f"- {row['model'].capitalize()}: u={row['u']:.8f}, q={row['reputation']:.8f}, "
                         f"D={row['demand']:.10f}, average profit={row['average_profit']:.10f}, "
                         f"per-seller use={row['product2_per_seller']:.10f}, market use={row['product2_market']:.10f}, "
                         f"maximum deviation gain={row['max_deviation_gain']:.3g} (tolerance {row['profit_tolerance']:.3g}).")
        lines += ["", "The fixed-rival payoff curves and the full best-response correspondence appear in the diagnostic figure. "
                  "At this case both markets choose product 2 exclusively, so the per-seller use difference comes from demand. "
                  "The two competing sellers together serve more product 2 than the monopolist.", "",
                  "Experiment 10 reported two discounted stationary states at this parameter case. Here the average-profit "
                  "objectives are optimized anew: u=0 is rejected by a profitable constant-mixture deviation. "
                  "The criteria and solution concepts differ, so the earlier stationary branches are not imported."]
    lines += ["", "## Numerical verification", "",
              "Each global maximum search scans the whole interval, refines every sampled local peak, and compares both endpoints. "
              "The competition search scans and refines minima of the nonnegative unilateral deviation gain, including zeros "
              "without sign changes, and supplements them with partial-own-derivative candidates. Rival actions remain fixed in every best response.", "",
              "Acceptance tolerance is `1e-12 + 1e-9 * profit_scale`, with profit_scale the maximum absolute payoff "
              "in the relevant full best-response search. Distinct candidates are deduplicated within 1e-6 in u; "
              "no candidate averaging is performed. The saved solution rows include actual scales and tolerances.", "",
              f"The coarse/fine runs use {config['numerics']['own_points']}/{config['numerics']['refined_own_points']} own-action nodes, "
              f"{config['numerics']['search_points']}/{config['numerics']['refined_search_points']} rival scan nodes, and "
              f"quadrature orders {config['numerics']['quadrature_order']}/{config['numerics']['refined_quadrature_order']}. "
              f"Maximum solution change in u: {validation['max_refinement_u_change']:.3g}. "
              f"Maximum independent adaptive-demand error: {validation['max_independent_demand_error']:.3g}. "
              f"Maximum accelerated-scan profit error against shared Jacobi quadrature: {validation['max_scan_quadrature_profit_error']:.3g}. "
              f"All required numerical checks passed: {validation['passed']}.", "",
              "These are global exploration, refinement and independent quadrature checks, not a mathematical proof "
              "that an arbitrarily narrow additional optimum cannot exist. All detected verified solutions are preserved. "
              "If no equilibrium is detected, the best residual and its location remain in `data/case_summary.csv` and `data/search_results.json`.", "",
              "## Files", "",
              "- [Three comparisons and diagnostic, combined PDF](figures/overview/average_profit_monopoly_vs_competition.pdf)",
              "- [All solutions](data/solutions.csv), [all branch pairs](data/comparisons.csv), [case diagnostics](data/case_summary.csv)",
              "- [Configuration used](data/configuration_used.json), [validation](data/validation.json), [manifest](data/manifest.json)",
              "- [Independent demand audits](data/independent_validation.csv), [refinement](data/refinement.csv)"]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.figures_only:
        config = json.loads((output/"data/configuration_used.json").read_text())
        from .reporting import make_figures
        make_figures(output, config)
        return
    started = time.monotonic()
    config = load_config(args.config)
    settings = refined_settings(config)
    (output/"data").mkdir(parents=True, exist_ok=True)
    dump(output/"data/configuration_used.json", config)
    cases, coarse_cases, refinement, audits, scan_audits = [], [], [], [], []
    for p0 in config["p0s"]:
        for rho in config["rhos"]:
            params = Parameters(**config["parameters"], rho=rho)
            coarse = solve_case(params, p0, config["numerics"])
            fine = solve_case(params, p0, settings)
            coarse_cases.append(coarse)
            cases.append(fine)
            refinement.extend(refinement_rows(coarse, fine, config["numerics"]["refinement_u_tolerance"]))
            audits.extend(independent_audits(fine, params, config))
            scan_audits.extend(scan_quadrature_audits(fine, params, config))
            print(f"p0={p0:g}, rho={rho:g}: monopoly={len(fine['monopoly']['maximizers'])}, "
                  f"competition={len(fine['competition']['equilibria'])}, "
                  f"min gain={fine['competition']['minimum_residual']:.3g}", flush=True)
    dump(output/"data/search_results.json", cases)
    dump(output/"data/coarse_search_results.json", coarse_cases)
    solutions = pd.DataFrame([row for case in cases for row in case["solutions"]])
    solutions.to_csv(output/"data/solutions.csv", index=False)
    for market in ("monopoly", "competition"):
        solutions[solutions.model.eq(market)].to_csv(output/f"data/{market}_solutions.csv", index=False)
    pd.DataFrame(refinement).to_csv(output/"data/refinement.csv", index=False)
    pd.DataFrame(audits).to_csv(output/"data/independent_validation.csv", index=False)
    pd.DataFrame(scan_audits).to_csv(output/"data/scan_quadrature_validation.csv", index=False)
    summaries = []
    for case in cases:
        summaries.append(dict(p0=case["p0"], rho=case["rho"],
                              monopoly_solution_count=len(case["monopoly"]["maximizers"]),
                              competition_solution_count=len(case["competition"]["equilibria"]),
                              status=case["competition"]["status"],
                              minimum_deviation_residual=case["competition"]["minimum_residual"],
                              minimum_residual_u=case["competition"]["minimum_residual_u"],
                              best_response_searches=case["competition"]["best_response_searches"]))
    pd.DataFrame(summaries).to_csv(output/"data/case_summary.csv", index=False)
    paired = comparisons(solutions)
    paired.to_csv(output/"data/comparisons.csv", index=False)
    write_diagnostic(output, config, cases)
    accounting = float(np.max(abs(solutions.product2_market-solutions.seller_count*solutions.product2_per_seller)))
    gains_passed = bool((solutions.max_deviation_gain <= solutions.profit_tolerance).all())
    validation = dict(passed=all(row["passed"] for row in refinement+audits+scan_audits) and gains_passed and accounting < 1e-12,
                      refinement_passed=all(row["passed"] for row in refinement),
                      independent_demand_passed=all(row["passed"] for row in audits),
                      scan_quadrature_passed=all(row["passed"] for row in scan_audits),
                      deviation_gains_passed=gains_passed,
                      max_refinement_u_change=max(row["max_u_change"] or 0. for row in refinement),
                      max_independent_demand_error=max(row["absolute_error"] for row in audits),
                      max_scan_quadrature_profit_error=max(row["max_profit_error"] for row in scan_audits),
                      max_market_accounting_error=accounting, independent_audit_rows=len(audits),
                      scope="Numerical exploration and checks, not an exhaustive analytic existence certificate")
    dump(output/"data/validation.json", validation)
    write_results(output, config, solutions, paired, validation)
    from .reporting import make_figures
    paths = make_figures(output, config)
    sources = [Path(__file__), ROOT/"code/model.py", ROOT/"code/reporting.py", args.config.resolve(),
               PROJECT/config["source_config_path"],
               Path(importlib.import_module("oblivious.fluid_oblivious.code.model").__file__),
               Path(importlib.import_module("oblivious.oe_p0.code.primitives").__file__),
               Path(importlib.import_module("oblivious.oe_distribution_fixed_p0.code.model").__file__)]
    dump(output/"data/manifest.json", dict(experiment_id=12, completed_at_utc=datetime.now(timezone.utc).isoformat(),
         elapsed_seconds=time.monotonic()-started, configuration=config,
         software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
         source_sha256={str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path):digest(path) for path in sources},
         output_sha256={str(path.relative_to(output)):digest(path) for path in sorted((output/"data").glob("*")) if path.is_file() and path.name != "manifest.json"},
         validation=validation, figure_count=len(paths)))
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["passed"]:
        raise SystemExit("Required numerical validation failed; inspect saved diagnostics")


if __name__ == "__main__":
    main()
