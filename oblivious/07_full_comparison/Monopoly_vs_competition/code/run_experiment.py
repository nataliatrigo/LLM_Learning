"""Solve the original forgetting monopoly at each saved OE-p0 benchmark.

Run from the project root:
  .venv/bin/python -B 'oblivious/07_full_comparison/Monopoly_vs_competition/code/run_experiment.py'

Competition policies, stationary laws and simulations are reused read-only.
The fixed benchmark is never closed again. The original forgetting pure action
rule is the main comparison. The OE-p0 mixed selector is saved as sensitivity.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monopoly-vs-competition-mpl")
OUTPUT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import diags
from oblivious.oe_p0.code.core import fixed_demand, posterior_mean, _jump_system, _positive_invariant
from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.policy import evaluate_policy, transition_matrix, stationary_distribution
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, save_bundle, write_json, write_csv
from oblivious.equilibrium_comparison.code.analysis import interval
from forgetting.Code.model import Primitives, BoundaryGrid, solve_boundary
from simulation import simulate, finite_window, ratio_interval

METRICS = ("product2_conditional", "product2_calendar", "seller_profit", "sale_rate")
MODELS = {
    "p0": dict(label="Competition (OE p0)", color="#D55E00", linestyle="--"),
    "monopoly": dict(label="Monopoly (forgetting)", color="#0072B2", linestyle="-"),
}
SOURCE = ROOT / "oblivious/02_oe_p0/data/rho_forgetting_mixed"
COMPARISON = ROOT / "oblivious/06_equilibrium_comparison/data/cross_rho_n51"
SIMULATION = dict(periods=100000, burn_in=10000, replications=20, seed=20261017)
ACTION_TOLERANCE = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stationary_laws(grid, demand, alpha, params):
    event = transition_matrix(grid, np.ones(len(grid)), alpha, params)
    jump, rates = _jump_system(event)
    eta = _positive_invariant(jump, rates)
    eta_reverse = _positive_invariant(jump, rates, np.arange(len(grid))[::-1])
    logs = np.log(eta)-np.log(demand)
    mu = np.exp(logs-logs.max())
    mu /= mu.sum()
    calendar = transition_matrix(grid, demand, alpha, params)
    cj, cr = _jump_system(calendar)
    direct = _positive_invariant(cj, cr)
    reverse = _positive_invariant(cj, cr, np.arange(len(grid))[::-1])
    existing = stationary_distribution(calendar)["mu"]
    discrepancy = max(float(abs(mu-direct).sum()), float(abs(mu-reverse).sum()),
                      float(abs(mu-existing).sum()), float(abs(eta-eta_reverse).sum()))
    sold = mu*demand
    sale_rate = float(sold.sum())
    conditional_l1 = float(abs(sold/sale_rate-eta).sum())
    residual = float(abs(mu @ calendar-mu).sum())
    event_residual = float(abs(eta @ event-eta).sum())
    require(discrepancy < 1e-10 and residual < 1e-11 and event_residual < 1e-11
            and conditional_l1 < 1e-10, "Monopoly invariant-law verification failed")
    return calendar, mu, eta, dict(verified=True, stationary_residual_l1=residual,
        event_stationary_residual_l1=event_residual, independent_order_l1=discrepancy,
        sale_weighted_law_l1=conditional_l1, minimum_demand=float(demand.min()),
        methods=["Own-sale event chain positive GTH; log-space eta/d normalization",
                 "Calendar jump chain positive GTH in forward and reversed state order",
                 "Existing oblivious.oe_p0.policy.stationary_distribution"])


def monopoly_predictions(mu, eta, demand, alpha, params):
    sales = float(mu @ demand)
    usage = float(eta @ alpha)
    count2 = sales*usage
    profit = float(mu @ (demand*(params.R-params.c1-(params.c2-params.c1)*alpha)))
    require(abs(profit-((params.R-params.c1)*sales-(params.c2-params.c1)*count2)) < 1e-12,
            "Monopoly profit accounting failed")
    return dict(product2_conditional=usage, product2_calendar=count2,
                seller_profit=profit, sale_rate=sales)


def main():
    data = OUTPUT / "data"
    arrays = data / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    files = {}
    forgetting_config = json.loads((ROOT/"forgetting/Code/experiments/config_paper.json").read_text())
    require(forgetting_config["solver"]["action_tolerance"] == ACTION_TOLERANCE,
            "Existing forgetting action tolerance changed")

    def record(path):
        files[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()

    def bundle(base):
        for suffix in (".json", ".npz"):
            record(base.with_suffix(suffix))
        return load_bundle(base)

    manifests = []
    for directory in (SOURCE, COMPARISON):
        path = directory/"manifest.json"
        record(path)
        manifest = json.loads(path.read_text())
        require(manifest["status"] == "complete", f"Incomplete source {directory}")
        manifests.append(manifest)
    cfg = manifests[1]["configuration"]
    n, rhos = cfg["grid"], sorted(cfg["rhos"])
    require(n == 51 and cfg["mode"] == "grid", "Expected common n51 grid")
    require(all(cfg[k] == v for k, v in SIMULATION.items()), "Simulation design mismatch")
    require(rhos == sorted(manifests[0]["configuration"]["rhos"]), "Rho mismatch")
    policies, outcomes, replications, verification, sensitivity = [], [], [], [], []
    p0s, warnings, params_common = {}, [], None
    for rho in rhos:
        tag = case_tag(rho, n)
        case = bundle(SOURCE/"arrays"/tag)
        comparison = bundle(COMPARISON/"checkpoints"/tag)
        source = case["selected"]
        params = Parameters(**case["configuration"]["parameters"])
        p0 = float(source["p0"])
        p0s[str(rho)] = p0
        grid, alpha = source["grid"], source["policy"]
        nu = source["nu"]
        profile = comparison["profiles"]["p0"]
        require(case["rho"] == rho and case["n"] == n and params.rho == rho, "Source case mismatch")
        require(source["verified"] and source["candidate_verified"] and
                source["dp_diagnostics"]["verified"] and source["stationary_diagnostics"]["verified"]
                and case["demand_audit"]["verified"], "Unverified OE-p0 source")
        require(comparison["verified"] and comparison["audit"]["verified"], "Unverified comparison")
        require(comparison["parameters"] == asdict(params), "Competition primitives changed")
        require(np.array_equal(grid, comparison["grid"]) and np.array_equal(alpha, profile["policy"])
                and np.array_equal(nu, profile["stationary_joint"]), "Cached competition mismatch")
        provenance = comparison["simulation_provenance"]["metadata"]
        metadata = provenance.get("metadata", provenance)
        expected = dict(params=asdict(params), mode="grid", initial=[.5, .5], **SIMULATION)
        require(all(metadata.get(k) == v for k,v in expected.items()), "Competition simulation metadata mismatch")
        common = {k:v for k,v in asdict(params).items() if k != "rho"}
        if params_common is None:
            params_common = common
        require(common == params_common, "Primitives differ across rho")
        consistency = float(abs(p0-posterior_mean(grid, params) @ nu.sum(axis=0)))
        require(consistency < 1e-7, "Source p0 equilibrium consistency failed")

        # Actual forgetting solver, independently converged on the same 51 nodes.
        primitives = Primitives(p0=p0, p1=params.p1, p2=params.p2, c1=params.c1,
                                c2=params.c2, revenue=params.R, gamma=params.gamma)
        solution = solve_boundary(primitives, rho, grid=BoundaryGrid(grid.copy(), n, 0, 9),
                                  tolerance=1e-13)
        demand = fixed_demand(grid, p0, params)
        require(np.max(abs(demand-solution.demand)) < 1e-12, "Beta survival functions disagree")
        evaluated = evaluate_policy(grid, demand, alpha, params)
        regret = np.maximum(evaluated["advantage"], 0)-alpha*evaluated["advantage"]
        mixed = (alpha > 1e-9) & (alpha < 1-1e-9)
        mix_residual = float(np.max(abs(evaluated["advantage"][mixed]), initial=0))
        gap = float(np.max(abs(evaluated["value"]-solution.value)))
        require(float(regret.max()) < 1e-10 and mix_residual < 1e-10 and gap < 1e-9,
                "Saved policy is not a verified optimum of the independent forgetting solve")
        # Existing forgetting simulator chooses product 1 at numerical ties.
        pure_alpha = (solution.advantage > ACTION_TOLERANCE).astype(float)
        pure_eval = evaluate_policy(grid, demand, pure_alpha, params)
        pure_regret = float(np.max(np.maximum(pure_eval["advantage"], 0)-pure_alpha*pure_eval["advantage"]))
        require(pure_regret < 1e-10, "Independently solved pure alternative is not optimal")
        transition, mu, eta, stationary_audit = stationary_laws(grid, demand, pure_alpha, params)
        mono = monopoly_predictions(mu, eta, demand, pure_alpha, params)
        matched_transition, matched_mu, matched_eta, matched_audit = stationary_laws(grid, demand, alpha, params)
        matched = monopoly_predictions(matched_mu, matched_eta, demand, alpha, params)
        for metric in METRICS:
            sensitivity.append(dict(rho=rho, n=n, p0=p0, metric=metric,
                                    monopoly_pure=mono[metric], monopoly_matched_mixture=matched[metric]))

        game = Game(grid, params, order=case["configuration"]["order"])
        sigma = np.broadcast_to(alpha[:, None], (n,n))
        joint = game.transition(sigma, sigma.T)
        joint_residual = float(np.max(abs(nu.ravel() @ joint-nu.ravel())))
        require(joint_residual < 1e-10, "Competition stationary verification failed")
        usage = float(np.sum(nu*(game.D*sigma+game.D.T*sigma.T)))
        profit = float(np.sum(game.rewards(sigma, sigma.T)*nu[None,:,:])/2)
        comp = dict(product2_conditional=usage, product2_calendar=.5*usage,
                    seller_profit=profit, sale_rate=.5)
        require(abs(profit-.5*(params.R-params.c1-(params.c2-params.c1)*usage)) < 1e-12,
                "Competition profit accounting failed")
        require(abs(usage-profile["predicted"]["product2_market"]) < 1e-10
                and abs(profit-profile["predicted"]["seller_profit"]) < 1e-10,
                "Cached competition outcomes changed")
        require(abs(matched["product2_conditional"]-usage) < 1e-9,
                "Shared own-sale kernel conditional-usage identity failed")
        comp_reps = []
        require(len(profile["replications"]) == SIMULATION["replications"], "Wrong source replication count")
        for row in profile["replications"]:
            q = row["product2_market"]
            require(abs(row["seller_profit"]-.5*(params.R-params.c1-(params.c2-params.c1)*q)) < 1e-12,
                    "Saved simulation profit accounting failed")
            comp_reps.append(dict(replication=row["replication"], sales=SIMULATION["periods"]*.5,
                                  product2_sales=q*SIMULATION["periods"]*.5,
                                  product2_conditional=q, product2_calendar=.5*q,
                                  seller_profit=row["seller_profit"], sale_rate=.5))
        for metric, original in (("product2_conditional", "product2_market"), ("seller_profit", "seller_profit")):
            estimate = interval([row[metric] for row in comp_reps])
            require(all(np.isclose(estimate[k], profile["summary"][original][k], rtol=0, atol=1e-12)
                        for k in estimate), "Reused simulation summary mismatch")
        print(f"rho={rho:g}, p0={p0:.9f}: independent forgetting optimum verified; simulating monopoly", flush=True)
        simulation = simulate(grid, pure_alpha, demand, params, **SIMULATION)
        mono_reps = simulation["replications"]
        finite = finite_window(transition, grid, demand, pure_alpha, params,
                               periods=SIMULATION["periods"], burn_in=SIMULATION["burn_in"])
        require(finite["probability_sum_error"] < 1e-8, "Finite-window kernel normalization failed")
        cfinite = profile["finite_window_expected"]
        finite_comp = dict(product2_conditional=cfinite["product2_market"],
                           product2_calendar=.5*cfinite["product2_market"],
                           seller_profit=cfinite["seller_profit"], sale_rate=.5)
        for key, predicted, reps, fexpected in (("p0", comp, comp_reps, finite_comp),
                                               ("monopoly", mono, mono_reps, finite["outcomes"])):
            for row in reps:
                replications.append(dict(rho=rho, n=n, model=key, **row))
            for metric in METRICS:
                estimate = ratio_interval(reps) if metric == "product2_conditional" else interval([row[metric] for row in reps])
                outcomes.append(dict(rho=rho, n=n, model=key, label=MODELS[key]["label"], metric=metric,
                                     stationary=predicted[metric], finite_window_expected=fexpected[metric],
                                     **{f"simulation_{k}":v for k,v in estimate.items()}))
            chosen_alpha = alpha if key == "p0" else pure_alpha
            chosen_mixed = (chosen_alpha > 1e-9) & (chosen_alpha < 1-1e-9)
            for x, a, mix in zip(grid, chosen_alpha, chosen_mixed):
                policies.append(dict(rho=rho, n=n, model=key, label=MODELS[key]["label"],
                                     reputation=float(x), product2_probability=float(a), mixed=bool(mix)))
        occupation_difference = float(abs(finite["occupation"]-mu).sum())
        warning = None
        if occupation_difference > .01:
            warning = dict(rho=rho, model="monopoly", kind="initialization_and_slow_mixing",
                           occupation_l1_from_stationary=occupation_difference,
                           stationary_sale_rate=mono["sale_rate"], finite_window_sale_rate=finite["outcomes"]["sale_rate"],
                           message="The fixed initial-state finite window does not approximate the stationary law; Monte Carlo intervals exclude initialization bias.")
            warnings.append(warning)
        verification.append(dict(rho=rho, n=n, p0=p0, verified=True, source_consistency_residual=consistency,
            policy_max_difference=float(abs(alpha-pure_alpha).max()), mixed_nodes=int(mixed.sum()), mixed_indifference_residual=mix_residual,
            conditional_regret=float(regret.max()), value_gap_from_independent_forgetting=gap,
            independent_bellman_residual=solution.bellman_residual, pure_alternative_regret=pure_regret,
            stationary_residual_max=stationary_audit["stationary_residual_l1"],
            independent_stationary_l1=stationary_audit["independent_order_l1"],
            competition_stationary_residual_max=joint_residual,
            matched_mixture_conditional_usage_identity_error=abs(matched["product2_conditional"]-usage),
            finite_window_probability_sum_error=finite["probability_sum_error"],
            finite_window_stationary_l1=occupation_difference,
            minimum_monopoly_demand=float(demand.min()), monopoly_stationary_sale_rate=mono["sale_rate"],
            monopoly_finite_window_sale_rate=finite["outcomes"]["sale_rate"],
            zero_sales_replications=sum(row["sales"] == 0 for row in mono_reps)))
        save_bundle(arrays/tag, dict(rho=rho, n=n, p0=p0, parameters=asdict(params), verified=True,
            grid=grid, policy=pure_alpha, pure_solver_policy=pure_alpha,
            monopoly=dict(demand=demand, stationary=mu, own_sale_stationary=eta,
                value=pure_eval["value"], advantage=pure_eval["advantage"],
                independent_solver_value=solution.value, independent_solver_advantage=solution.advantage,
                pure_alternative_value=pure_eval["value"], pure_alternative_advantage=pure_eval["advantage"],
                predicted=mono, stationary_audit=stationary_audit, finite_window=finite,
                simulation=simulation, warning=warning),
            matched_mixture_sensitivity=dict(policy=alpha, value=evaluated["value"],
                advantage=evaluated["advantage"], stationary=matched_mu,
                own_sale_stationary=matched_eta, predicted=matched, stationary_audit=matched_audit,
                conditional_regret=float(regret.max()), mixed_indifference_residual=mix_residual),
            competition=dict(policy=alpha, stationary_joint=nu, predicted=comp,
                             finite_window_expected=finite_comp, replications=comp_reps,
                             simulation_provenance=comparison["simulation_provenance"])))
        print(f"  stationary sale={mono['sale_rate']:.6g}, finite-window sale={finite['outcomes']['sale_rate']:.6g}, "
              f"MC sale={np.mean([r['sale_rate'] for r in mono_reps]):.6g}; saved", flush=True)

    for name, rows in (("policies",policies),("outcomes",outcomes),("replications",replications),("verification",verification),
                       ("tie_sensitivity",sensitivity)):
        write_csv(data/f"{name}.csv", rows)
    write_json(data/"comparison.json", dict(rhos=rhos, grid=n, models=MODELS, parameters=params_common,
        p0_by_rho=p0s, simulation={**SIMULATION, "mode":"grid"}, initial_monopoly=.5,
        initial_competition=[.5,.5], no_new_competition_simulation=True,
        monopoly_solver="forgetting.Code.model.solve_boundary; original forgetting problem at fixed saved OE-p0 benchmark",
        equilibrium_closed_again=False, action_tolerance=ACTION_TOLERANCE,
        policy_tie_convention="Existing forgetting rule: product 2 iff advantage > 1e-6; product 1 at numerical ties. Saved OE-p0 mixtures remain the competition policy. Matched-mixture monopoly is sensitivity only, without extra simulations.",
        warnings=warnings, definitions=dict(
            policy="P(product 2 | own seller selected, own reputation x).",
            product2_conditional="Stationary: product 2 fraction of own sales. Simulation: total product 2 sales / total own sales across independent paths (pooled ratio); competition pools both symmetric sellers.",
            product2_calendar="Product 2 sales per calendar period for one seller; competition averages A and B.",
            seller_profit="Profit per calendar period for one seller, including zero-profit periods when the outside option or rival sells; competition averages A and B.",
            sale_rate="Probability the seller makes a sale in a calendar period; competition pools symmetric sellers and equals 1/2.",
            conditional_uncertainty="95% Student-t interval using independent-path cluster delta-method SE for the pooled ratio; zero-sale paths remain in the cluster sample; entire zero-sale samples are unavailable.",
            other_uncertainty="95% Student-t intervals across 20 independent path means; all MC intervals exclude initialization bias.",
            finite_window_expected="Exact finite-grid expected outcomes over retained periods from initial reputation .5 after burn-in; conditional usage is ratio of expected sale counts, the pooled estimator target.",
            stationary_usage_identity="Only the matched-mixture sensitivity shares the competition own-sale event chain; its stationary conditional product-2 share equals competition. The main forgetting curve uses its original pure tie rule.",
            scope="Finite common 51-point grid, stationary and finite-window outcomes; saved OE-p0 competition is not asserted to be a full-state Markov-perfect equilibrium.")))
    for path in (ROOT/"forgetting/Code/model.py", ROOT/"forgetting/Code/simulation.py",
                 ROOT/"forgetting/Code/experiments/config_paper.json", ROOT/"oblivious/02_oe_p0/code/policy.py",
                 ROOT/"oblivious/02_oe_p0/code/core.py", ROOT/"oblivious/02_oe_p0/code/game.py",
                 ROOT/"oblivious/02_oe_p0/code/primitives.py", ROOT/"oblivious/02_oe_p0/code/run_experiment.py",
                 ROOT/"oblivious/06_equilibrium_comparison/code/analysis.py", ROOT/"oblivious/02_oe_p0/code/diagnostics.py"):
        record(path)
    write_json(data/"manifest.json", dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        all_16_model_rho_cases_verified=True, reused_competition=True, new_monopoly_simulations=True,
        source_files_sha256=files, data_files_sha256={str(path.relative_to(data)):sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.rglob("*")) if path.is_file() and
            (path.parent == arrays or path.name in {"policies.csv", "outcomes.csv", "replications.csv",
             "verification.csv", "tie_sensitivity.csv", "comparison.json"})},
        code_files_sha256={path.name:sha256(path.read_bytes()).hexdigest()
                           for path in sorted(Path(__file__).parent.glob("*.py"))}))
    print(f"Completed eight forgetting-monopoly solves and comparisons in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
