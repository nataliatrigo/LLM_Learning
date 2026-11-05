"""Compare the original monopoly family with saved OE distribution equilibria.

Both outcomes are stationary fractions of the seller's OWN SALES using product
2. Only the one-dimensional monopoly problems are solved; competitive equilibria
are read from the refinement campaign, with the certified .99/n101 retry.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monopoly-p0-refinement-mpl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "monopoly_p0_saved_inputs", Path(__file__).with_name("plot_refinement_comparison.py"))
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
RUN = COMMON.RUN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_p0.code.core import fixed_demand, _jump_system, _positive_invariant
from oblivious.oe_p0.code.policy import evaluate_policy, transition_matrix
from oblivious.oe_p0.code.primitives import Parameters, demand_matrix
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from forgetting.Code.model import Primitives, BoundaryGrid, solve_boundary

MONOPOLY_COLOR, OE_COLOR = "#D55E00", "#009E73"
STYLE = dict(COMMON.PLOT.STYLE)
FAMILY = "monopoly_p0_comparison"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def monopoly_case(rho, p0, n, parameters, action_tolerance=1e-6):
    """Original forgetting DP, with independent Bellman and invariant checks."""
    require(n >= 3 and 0 < p0 < 1 and action_tolerance >= 0,
            "Require n>=3, 0<p0<1 and a nonnegative action tolerance")
    params = Parameters(**parameters, rho=rho)
    grid = np.linspace(0, 1, n)
    primitives = Primitives(p0=p0, p1=params.p1, p2=params.p2, c1=params.c1,
                            c2=params.c2, revenue=params.R, gamma=params.gamma)
    solution = solve_boundary(primitives, rho, grid=BoundaryGrid(grid, n, 0, 9),
                              tolerance=1e-13)
    # Preserve the pure selector used in the original forgetting experiments.
    policy = (solution.advantage > action_tolerance).astype(float)
    demand = fixed_demand(grid, p0, params)
    require(np.max(abs(demand-solution.demand)) < 1e-12,
            "Independent monopoly demand implementations disagree")
    evaluation = evaluate_policy(grid, demand, policy, params)
    regret = float(np.max(np.maximum(evaluation["advantage"], 0)
                          - policy*evaluation["advantage"]))
    value_gap = float(np.max(abs(evaluation["value"]-solution.value)))
    sale = purchase_chain_statistics(grid, policy, params)
    eta = sale["sale_stationary"]
    require(np.all(demand > 0) and np.all(eta > 0),
            "Need positive demand and sale-time invariant mass; underflow is not a zero-sale equilibrium")
    # Log-space holding-time correction remains stable even for tiny demand.
    logs = np.log(eta)-np.log(demand)
    mu = np.exp(logs-logs.max())
    mu /= mu.sum()
    kernel = transition_matrix(grid, demand, policy, params)
    jump, rates = _jump_system(kernel)
    independent_mu = _positive_invariant(jump, rates)
    calendar_l1 = float(abs(mu-independent_mu).sum())
    calendar_residual = float(abs(kernel.T @ mu-mu).sum())
    sold = mu*demand
    sale_rate = float(sold.sum())
    weighted_l1 = float(abs(sold/sale_rate-eta).sum())
    usage = float(eta @ policy)
    calendar_usage = float(sold @ policy)
    diagnostics = dict(
        verified=True, conditional_regret=regret,
        strict_optimality_verified=bool(regret <= 1e-10),
        policy_bellman_residual=evaluation["residual"],
        original_bellman_residual=solution.bellman_residual,
        value_gap=value_gap, event_stationary_residual=sale["stationary_residual"],
        event_reverse_order_l1=sale["stability_l1"],
        calendar_stationary_residual=calendar_residual,
        calendar_independent_l1=calendar_l1, sale_weighted_l1=weighted_l1,
        quality_reputation_difference=sale["quality_reputation_difference"],
        minimum_demand=float(demand.min()))
    require(regret <= max(1e-10, action_tolerance+1e-10)
            and evaluation["residual"] < 1e-10
            and value_gap <= max(1e-9, action_tolerance/(1-params.gamma)+1e-9)
            and calendar_l1 < 1e-9 and calendar_residual < 1e-11
            and weighted_l1 < 1e-10
            and abs(sale["quality_reputation_difference"]) < 1e-10
            and abs(calendar_usage-sale_rate*usage) < 1e-12,
            f"Monopoly verification failed: rho={rho}, p0={p0}: {diagnostics}")
    return dict(rho=rho, p0=p0, n=n, grid=grid, policy=policy, demand=demand,
                value=evaluation["value"], sale_stationary=eta, calendar_stationary=mu,
                outcomes=dict(product2_conditional=usage, sale_rate=sale_rate,
                              product2_calendar=calendar_usage), diagnostics=diagnostics)


def collect_reference(campaign, n, rhos, retry_name):
    """Check provenance and independently reconstruct the own-sales denominator."""
    path = campaign/"manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    require("oe" in config["models"] and n in config["grids"]
            and set(rhos) <= set(config["rhos"]), "Grid/rhos/OE absent from campaign")
    inputs = {str(path.relative_to(ROOT)): COMMON.digest(path)}
    retry_manifest = retry_case = None
    failed = RUN.load_checked(RUN.case_dir(campaign, "oe", .99, n)/"solution") if .99 in rhos else None
    if n == 101 and .99 in rhos and not RUN.verified((failed or {}).get("selected")):
        require(bool(retry_name), "The failed .99/n101 case needs its certified OE retry")
        retry_manifest, retry_case = COMMON.load_oe_retry(
            RUN.OUTPUT/retry_name, campaign, manifest, inputs)
    rows = []
    for rho in rhos:
        base = RUN.case_dir(campaign, "oe", rho, n)/"solution"
        fingerprint = manifest["fingerprint"]
        if rho == .99 and retry_case is not None:
            base = RUN.OUTPUT/retry_name/"solution"
            case, fingerprint = retry_case, retry_manifest["fingerprint"]
        else:
            case = RUN.load_checked(base)
        require(case is not None and RUN.verified(case.get("selected"))
                and case.get("demand_audit", {}).get("verified")
                and case.get("campaign_fingerprint") == fingerprint
                and case.get("rho") == rho and case.get("n") == n
                and case["configuration"]["parameters"] == dict(config["parameters"], rho=rho)
                and case["configuration"]["order"] == config["order"],
                f"Missing or incompatible certified OE distribution: rho={rho}, n={n}")
        candidate = case["selected"]
        grid, alpha, nu = (np.asarray(candidate[k]) for k in ("grid", "policy", "nu"))
        require(np.array_equal(grid, np.linspace(0, 1, n)) and alpha.shape == (n,)
                and nu.shape == (n, n) and np.isfinite(nu).all() and np.all(nu >= 0)
                and abs(nu.sum()-1) < 1e-10 and np.max(abs(nu-nu.T)) < 1e-9,
                "Expected symmetric normalized OE law on the common grid")
        params = Parameters(**case["configuration"]["parameters"])
        demand, _ = demand_matrix(grid, params, config["order"])
        sold = (nu*demand).sum(axis=1)
        sale_rate = float(sold.sum())
        usage = float(sold @ alpha / sale_rate)
        embedded = purchase_chain_statistics(grid, alpha, params)
        require(abs(sale_rate-.5) < 1e-10
                and abs(usage-case["predicted"]["product2_market"]) < 1e-10
                and abs(usage-embedded["sale_stationary"] @ alpha) < 1e-9,
                "OE predicted market usage does not match its own-sales fraction")
        COMMON.record(base, inputs)
        hash_path = base.with_name(base.name+"_sha256.json")
        inputs[str(hash_path.relative_to(ROOT))] = COMMON.digest(hash_path)
        rows.append(dict(rho=rho, n=n, model="oe", product2_conditional=usage,
                         sale_rate=sale_rate, product2_calendar=sale_rate*usage,
                         verified=True, source=str(base.relative_to(ROOT))))
    return rows, inputs, config


def save_figure(fig, directory, name):
    paths = []
    for extension in ("png", "pdf"):
        path = directory/f"{name}.{extension}"
        fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def make_figures(data, references, fixed_p0, n, directory):
    rhos = sorted(references["rho"])
    rows = (len(rhos)+1)//2
    fig, axes = plt.subplots(rows, 2, figsize=(11.8, 3.0*rows+1), squeeze=False)
    fig.subplots_adjust(left=.08, right=.98, bottom=.065, top=.89, hspace=.44, wspace=.23)
    for rho, ax in zip(rhos, axes.flat):
        series = data[data.rho == rho].sort_values("p0")
        oe = float(references.loc[references.rho == rho, "product2_conditional"].iloc[0])
        ax.plot(series.p0, series.product2_conditional, color=MONOPOLY_COLOR, lw=1.8)
        selected = series[np.isclose(series.p0.to_numpy()[:, None], fixed_p0).any(axis=1)]
        ax.scatter(selected.p0, selected.product2_conditional, color=MONOPOLY_COLOR,
                   s=21, edgecolors="white", linewidths=.5, zorder=4)
        ax.axhline(oe, color=OE_COLOR, lw=1.8, ls="--")
        ax.text(.98, .94, f"OE = {100*oe:.1f}%", transform=ax.transAxes,
                ha="right", va="top", color=OE_COLOR, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
        ax.set(title=rf"$\rho={rho:g}$", xlabel=r"Outside option fija $p_0$",
               ylabel="Producto 2 / ventas propias", xlim=(series.p0.min(), series.p0.max()),
               ylim=(-.025, 1.025), xticks=[.1, .3, .5, .7, .9])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(rhos):]:
        ax.set_visible(False)
    fig.suptitle("Uso de producto 2: monopolio frente al equilibrio OE · ρ > 0.5", y=.99, fontsize=15)
    fig.legend(handles=[
        Line2D([], [], color=MONOPOLY_COLOR, lw=2, label=r"Monopolio $u^M(\rho,p_0)$"),
        Line2D([], [], color=OE_COLOR, lw=2, ls="--", label=r"OE distribution $u^{OE}(\rho)$"),
        Line2D([], [], color=MONOPOLY_COLOR, marker="o", ls="none", ms=5,
               label=r"$p_0$ usados previamente")],
        loc="upper center", bbox_to_anchor=(.53, .965), ncol=3, frameon=False)
    fig.text(.53, .022, f"Predicciones estacionarias · grilla común n={n} · "
             "monopolio: selector puro original · líneas unen evaluaciones de p₀.",
             ha="center", fontsize=8.5)
    paths = save_figure(fig, directory, "01_product2_usage_vs_p0")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.77)
    colors = plt.get_cmap("plasma")(np.linspace(.12, .82, len(fixed_p0)))
    for p0, color in zip(fixed_p0, colors):
        series = data[np.isclose(data.p0, p0)].set_index("rho").loc[rhos]
        ax.plot(range(len(rhos)), series.product2_conditional, color=color,
                marker="o", ms=4, lw=1.6, label=rf"Monopolio $p_0={p0:g}$")
    series = references.set_index("rho").loc[rhos]
    ax.plot(range(len(rhos)), series.product2_conditional, color=OE_COLOR,
            marker="D", mfc="white", ms=6, lw=2.5, label="OE distribution", zorder=5)
    ax.set_xticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    ax.set(xlabel=r"Factor de olvido $\rho$ (valores discretos)",
           ylabel="Producto 2 / ventas propias", ylim=(-.025, 1.025),
           xlim=(-.15, len(rhos)-.85))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Monopolio para distintas outside options y equilibrio OE", fontsize=14, y=.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .92),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.53, .09, f"Predicciones estacionarias · grilla común n={n} · "
             "misma fracción de ventas propias en ambos modelos.", ha="center", fontsize=9)
    fig.text(.53, .045, "Los valores de ρ están equiespaciados; las líneas orientan la lectura.",
             ha="center", fontsize=8.5)
    paths += save_figure(fig, directory, "02_product2_usage_vs_rho")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="refinement_101_201")
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--oe-retry", default="oe099_retry_n101")
    parser.add_argument("--plots-only", action="store_true", help="Require the compatible saved monopoly sweep")
    parser.add_argument("--recompute", action="store_true", help="Recalculate the generated monopoly sweep")
    args = parser.parse_args(argv)
    for value in (args.name, args.oe_retry):
        if Path(value).name != value or value in ("", ".", ".."):
            parser.error("Require campaign/retry folder names")
    if args.grid < 3 or (args.plots_only and args.recompute):
        parser.error("Require grid>=3 and choose at most one of --plots-only/--recompute")
    campaign = RUN.OUTPUT/args.name
    source_config_path = ROOT/"forgetting/Code/experiments/config_paper.json"
    source_config = json.loads(source_config_path.read_text())
    manifest = json.loads((campaign/"manifest.json").read_text())
    rhos = sorted(rho for rho in manifest["configuration"]["rhos"] if rho > .5)
    require(bool(rhos), "No campaign rho values above .5")
    references, inputs, campaign_config = collect_reference(campaign, args.grid, rhos, args.oe_retry)
    inputs[str(source_config_path.relative_to(ROOT))] = COMMON.digest(source_config_path)
    fixed_p0 = source_config["p0_comparison"]["values"]
    parameters = campaign_config["parameters"]
    seller = source_config["seller"]
    original_parameters = {k: seller[k] for k in ("p1", "p2", "c1", "c2")}
    original_parameters.update(R=seller["revenue"], gamma=source_config["rho_p0_comparative_statics"]["gamma"])
    require(parameters == original_parameters, "Monopoly and refinement primitives differ")
    scan = source_config["rho_p0_comparative_statics"]
    p0s = sorted(set(np.round(np.linspace(scan["p0_start"], scan["p0_stop"], 181), 12))
                 | set(fixed_p0))
    config = dict(campaign=args.name, grid=args.grid, rhos=rhos, p0_values=p0s,
                  fixed_p0=fixed_p0, rho_lower_bound_exclusive=.5, parameters=parameters,
                  action_tolerance=source_config["solver"]["action_tolerance"])
    code_paths = [Path(__file__), Path(COMMON.__file__), Path(RUN.__file__),
                  ROOT/"forgetting/Code/model.py", ROOT/"oblivious/03_oe_following_monopoly/code/event_chain.py"]
    code_paths += [ROOT/f"oblivious/02_oe_p0/code/{name}" for name in
                   ("core.py", "policy.py", "primitives.py", "positive_stationary.py",
                    "positive_stationary.cpp", "run_experiment.py")]
    sources = {str(p.relative_to(ROOT)): COMMON.digest(p) for p in code_paths}
    fingerprint = RUN.fingerprint(dict(config, input_sha256=inputs), sources)
    data_dir = campaign/FAMILY/f"n{args.grid:03d}"
    directory = ROOT/"oblivious/07_full_comparison/Figures"/args.name/f"n{args.grid:03d}"/FAMILY
    data_dir.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    base = data_dir/"results"
    saved = None if args.recompute else RUN.load_checked(base)
    if saved is not None:
        require(saved["fingerprint"] == fingerprint,
                "Sweep inputs/code changed; use --recompute to regenerate these derived results")
        cases = saved["cases"]
    else:
        require(not args.plots_only, "No compatible saved monopoly sweep")
        cases = []
        for rho in rhos:
            cases.extend(monopoly_case(rho, float(p0), args.grid, parameters, config["action_tolerance"])
                         for p0 in p0s)
            print(f"Verified monopoly family: rho={rho:g}, n={args.grid}, {len(p0s)} p0 values", flush=True)
        RUN.save_checked(base, dict(fingerprint=fingerprint, configuration=config, cases=cases))
    rows = []
    oe_by_rho = {row["rho"]: row["product2_conditional"] for row in references}
    for case in cases:
        rows.append(dict(rho=case["rho"], p0=case["p0"], n=case["n"], **case["outcomes"],
                         oe_product2_conditional=oe_by_rho[case["rho"]],
                         monopoly_minus_oe_pp=100*(case["outcomes"]["product2_conditional"]-oe_by_rho[case["rho"]]),
                         **case["diagnostics"]))
    data, reference_data = pd.DataFrame(rows), pd.DataFrame(references)
    data.to_csv(data_dir/"comparison.csv", index=False)
    reference_data.to_csv(data_dir/"oe_reference.csv", index=False)
    data[np.isclose(data.p0.to_numpy()[:, None], fixed_p0).any(axis=1)].to_csv(
        data_dir/"fixed_p0_comparison.csv", index=False)
    with plt.rc_context(STYLE):
        figures = make_figures(data, reference_data, fixed_p0, args.grid, directory)
    generated = [base.with_suffix(s) for s in (".json", ".npz")]
    generated += [base.with_name(base.name+"_sha256.json")]
    generated += [data_dir/name for name in ("comparison.csv", "oe_reference.csv", "fixed_p0_comparison.csv")]
    report = dict(configuration=config, fingerprint=fingerprint, input_sha256=inputs,
                  source_sha256=sources, generated_sha256={str(p.relative_to(ROOT)): COMMON.digest(p) for p in generated},
                  figures={str(p.relative_to(ROOT)): COMMON.digest(p) for p in figures},
                  verified_cases=len(cases), strict_optimality_cases=int(data.strict_optimality_verified.sum()),
                  maximum_conditional_regret=float(data.conditional_regret.max()),
                  definitions=dict(monopoly_usage="eta @ alpha: fraction of the monopolist's own sales using product 2",
                                   oe_usage="sum(nu * D_A * alpha_A) / sum(nu * D_A); equals market share by seller symmetry",
                                   calendar_usage="sale_rate * product2_conditional; not the plotted quantity",
                                   selector="Product 2 iff original forgetting conditional advantage > action_tolerance; product 1 otherwise",
                                   scope="Stationary finite-grid predictions; original monopoly DP newly evaluated, saved OE equilibria read-only; no new competitive solves or Monte Carlo",
                                   sampling="Lines connect p0 evaluations; policy switches and equality thresholds are not localized; no uniqueness or continuous-grid convergence claim"))
    RUN.atomic_json(data_dir/"manifest.json", report)
    RUN.atomic_json(directory/"figure_manifest.json", report)
    print(f"Saved two comparisons, PNG and PDF: {directory}")
    print(f"Data and verification: {data_dir}; max conditional regret={data.conditional_regret.max():.3g}")


if __name__ == "__main__":
    main()
