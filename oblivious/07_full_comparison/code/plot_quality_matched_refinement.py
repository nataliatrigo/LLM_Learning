"""Compare monopoly at p0_ref = the saved OE rival's mean success per sale.

The benchmark is calibrated once from OE distribution, without a new fixed-point
condition. Competitive equilibria remain read-only. Monopoly is solved at the
exact reference for each rho, on the same reputation grid.
"""

import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "quality_matched_monopoly_helpers", Path(__file__).with_name("plot_monopoly_p0_refinement.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
ROOT, RUN, COMMON = BASE.ROOT, BASE.RUN, BASE.COMMON

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters, demand_matrix
from oblivious.layout import resolve_saved_path

FAMILY = "quality_matched_comparison"
MONOPOLY_COLOR, OE_COLOR = BASE.MONOPOLY_COLOR, BASE.OE_COLOR


def rival_sales_quality(candidate, parameters, order):
    """Weight rival B's true quality by its sales, with rows=A, columns=B."""
    grid, policy, nu = (np.asarray(candidate[k]) for k in ("grid", "policy", "nu"))
    params = Parameters(**parameters)
    demand, _ = demand_matrix(grid, params, order)
    rival_sold = (nu*demand.T).sum(axis=0)
    sale_rate = float(rival_sold.sum())
    BASE.require(sale_rate > 0, "Cannot define quality without rival sales")
    usage = float(rival_sold @ policy / sale_rate)
    quality = params.p1 + (params.p2-params.p1)*policy
    success = float(rival_sold @ quality / sale_rate)
    return dict(rival_sale_rate=sale_rate, rival_product2_conditional=usage,
                rival_success_probability=success)


def build_pairs(references, n, parameters, order, action_tolerance):
    pairs, rows, policy_rows = [], [], []
    for reference in references:
        rho = reference["rho"]
        case = RUN.load_checked(resolve_saved_path(reference["source"], project_root=ROOT))
        BASE.require(case is not None and RUN.verified(case.get("selected")),
                     "The certified OE reference changed while building the comparison")
        candidate = case["selected"]
        rival = rival_sales_quality(candidate, dict(parameters, rho=rho), order)
        u_oe = reference["product2_conditional"]
        p0_ref = parameters["p1"] + (parameters["p2"]-parameters["p1"])*u_oe
        calibration_error = abs(p0_ref-rival["rival_success_probability"])
        BASE.require(abs(rival["rival_product2_conditional"]-u_oe) < 1e-10
                     and abs(rival["rival_sale_rate"]-.5) < 1e-10
                     and calibration_error < 1e-12
                     and abs(p0_ref-case["predicted"]["success_rate"]) < 1e-10,
                     "Reference must match the rival's true success conditional on its own sales")
        # Never round to a p0 sample in the previously plotted monopoly family.
        monopoly = BASE.monopoly_case(rho, p0_ref, n, parameters, action_tolerance)
        u_monopoly = monopoly["outcomes"]["product2_conditional"]
        rows.append(dict(rho=rho, n=n, p0_ref=p0_ref,
                         oe_product2_conditional=u_oe,
                         monopoly_product2_conditional=u_monopoly,
                         monopoly_minus_oe_pp=100*(u_monopoly-u_oe),
                         oe_sale_rate=reference["sale_rate"],
                         monopoly_sale_rate=monopoly["outcomes"]["sale_rate"],
                         **rival, calibration_error=calibration_error,
                         **monopoly["diagnostics"], oe_source=reference["source"]))
        for model, alpha in (("oe", candidate["policy"]), ("monopoly", monopoly["policy"])):
            policy_rows.extend(dict(rho=rho, n=n, p0_ref=p0_ref, model=model,
                                    reputation=float(x), product2_probability=float(a))
                               for x, a in zip(monopoly["grid"], alpha))
        pairs.append(dict(rho=rho, p0_ref=p0_ref, monopoly=monopoly,
                          oe_grid=candidate["grid"], oe_policy=candidate["policy"],
                          oe_product2_conditional=u_oe, oe_source=reference["source"]))
    return pairs, pd.DataFrame(rows), pd.DataFrame(policy_rows)


def make_figures(pairs, data, n, directory):
    data = data.sort_values("rho")
    rhos = data.rho.to_numpy()
    x = np.arange(len(rhos))
    fig, (ax, ref_ax) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[2.5, 1]))
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.16, hspace=.25)
    ax.plot(x, data.monopoly_product2_conditional, color=MONOPOLY_COLOR,
            marker="o", lw=2, ms=5,
            label=r"Monopolio con $p_0=p_0^{ref}(\rho)$")
    ax.plot(x, data.oe_product2_conditional, color=OE_COLOR,
            marker="D", mfc="white", lw=2, ms=5, label="OE distribution")
    for xi, row in zip(x, data.itertuples()):
        ax.annotate(f"{100*row.monopoly_product2_conditional:.1f}%",
                    (xi, row.monopoly_product2_conditional), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=MONOPOLY_COLOR)
        ax.annotate(f"{100*row.oe_product2_conditional:.1f}%",
                    (xi, row.oe_product2_conditional), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=9, color=OE_COLOR)
    ax.set(ylabel="Producto 2 / ventas propias", ylim=(-.025, 1.025))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ref_ax.plot(x, data.p0_ref, color=".3", marker="o", lw=1.6, ms=4)
    for xi, p0 in zip(x, data.p0_ref):
        ref_ax.annotate(f"{p0:.4f}", (xi, p0), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
    ref_ax.set(ylabel=r"Outside option $p_0^{ref}$",
               xlabel=r"Factor de olvido $\rho$ (valores discretos)",
               xlim=(-.2, len(rhos)-.8),
               ylim=(data.p0_ref.min()-.045, data.p0_ref.max()+.065))
    ref_ax.set_xticks(x, [f"{rho:g}" for rho in rhos])
    for panel in (ax, ref_ax):
        panel.grid(axis="y", alpha=.2)
        panel.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Outside option igual a la calidad media del rival OE", fontsize=15, y=.985)
    fig.text(.53, .925, r"$p_0^{ref}(\rho)=p_1+(p_2-p_1)\,u_j^{OE}(\rho)$"
             + "     ·     calidad entre las ventas del rival", ha="center", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .89),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.53, .075, f"Predicciones estacionarias · grilla común n={n} · "
             "todos los ρ > 0.5 disponibles · sin filtro sobre p₀.", ha="center", fontsize=9)
    fig.text(.53, .035, "Se iguala la calidad media de la alternativa. "
             "Los valores de ρ están equiespaciados; las líneas orientan la lectura.",
             ha="center", fontsize=8.5)
    paths = BASE.save_figure(fig, directory, "01_product2_usage_quality_matched")

    count = len(pairs)
    fig, axes = plt.subplots((count+1)//2, 2, figsize=(11.8, 3.1*((count+1)//2)+.8), squeeze=False)
    fig.subplots_adjust(left=.08, right=.98, bottom=.065, top=.90, hspace=.43, wspace=.23)
    for pair, panel in zip(pairs, axes.flat):
        monopoly, oe_policy = pair["monopoly"], pair["oe_policy"]
        panel.plot(monopoly["grid"], monopoly["policy"], color=MONOPOLY_COLOR, lw=1.8)
        panel.plot(pair["oe_grid"], oe_policy, color=OE_COLOR, lw=1.7, ls="--")
        mixed = (oe_policy > 1e-9) & (oe_policy < 1-1e-9)
        panel.scatter(pair["oe_grid"][mixed], oe_policy[mixed], color=OE_COLOR,
                      marker="D", s=27, edgecolors="white", linewidths=.5, zorder=4)
        panel.text(.02, .91, rf"$p_0^{{ref}}={pair['p0_ref']:.4f}$", transform=panel.transAxes,
                   va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
        panel.set(title=rf"$\rho={pair['rho']:g}$", xlabel="Reputación propia",
                  ylabel="Probabilidad de producto 2", xlim=(-.02, 1.02), ylim=(-.04, 1.04),
                  xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        panel.grid(alpha=.2)
        panel.spines[["top", "right"]].set_visible(False)
    for panel in list(axes.flat)[count:]:
        panel.set_visible(False)
    fig.suptitle("Políticas con outside option igual a la calidad media del rival OE", fontsize=14, y=.99)
    fig.legend(handles=[
        Line2D([], [], color=MONOPOLY_COLOR, lw=2, label=r"Monopolio con $p_0=p_0^{ref}(\rho)$"),
        Line2D([], [], color=OE_COLOR, lw=2, ls="--", label="OE distribution"),
        Line2D([], [], color=OE_COLOR, marker="D", ls="none", ms=5, label="Mezcla en un nodo OE")],
        loc="upper center", bbox_to_anchor=(.53, .965), ncol=3, frameon=False)
    fig.text(.53, .02, f"Grilla común n={n} · selector puro original del monopolio · "
             "líneas unen nodos resueltos · referencia calibrada desde OE, sin nuevo cierre.",
             ha="center", fontsize=8.3)
    return paths + BASE.save_figure(fig, directory, "02_policies_quality_matched")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="refinement_101_201")
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--oe-retry", default="oe099_retry_n101")
    args = parser.parse_args(argv)
    for value in (args.name, args.oe_retry):
        if Path(value).name != value or value in ("", ".", ".."):
            parser.error("Require campaign/retry folder names")
    if args.grid < 3:
        parser.error("Require grid>=3")
    campaign = RUN.OUTPUT/args.name
    manifest = json.loads((campaign/"manifest.json").read_text())
    rhos = sorted(rho for rho in manifest["configuration"]["rhos"] if rho > .5)
    references, inputs, config = BASE.collect_reference(campaign, args.grid, rhos, args.oe_retry)
    params = config["parameters"]
    paper_path = ROOT/"forgetting/Code/experiments/config_paper.json"
    paper = json.loads(paper_path.read_text())
    action_tolerance = paper["solver"]["action_tolerance"]
    inputs[str(paper_path.relative_to(ROOT))] = COMMON.digest(paper_path)
    pairs, data, policies = build_pairs(references, args.grid, params, config["order"], action_tolerance)
    data_dir = campaign/FAMILY/f"n{args.grid:03d}"
    directory = ROOT/"oblivious/07_full_comparison/Figures"/args.name/f"n{args.grid:03d}"/FAMILY
    data_dir.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = dict(campaign=args.name, grid=args.grid, rhos=rhos, parameters=params,
                         rho_lower_bound_exclusive=.5, p0_filter=None,
                         reference_rule="p0_ref = p1 + (p2-p1) * rival_product2_conditional_OE",
                         action_tolerance=action_tolerance)
    source_paths = [Path(__file__), Path(BASE.__file__), Path(COMMON.__file__),
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
    with plt.rc_context(BASE.STYLE):
        figures = make_figures(pairs, data, args.grid, directory)
    generated = [base.with_suffix(s) for s in (".json", ".npz")]
    generated += [base.with_name(base.name+"_sha256.json"), data_dir/"comparison.csv", data_dir/"policies.csv"]
    report = dict(configuration=configuration, fingerprint=fingerprint, input_sha256=inputs,
                  source_sha256=sources,
                  generated_sha256={str(p.relative_to(ROOT)): COMMON.digest(p) for p in generated},
                  figures={str(p.relative_to(ROOT)): COMMON.digest(p) for p in figures},
                  verified_cases=len(pairs), maximum_calibration_error=float(data.calibration_error.max()),
                  maximum_conditional_regret=float(data.conditional_regret.max()),
                  definitions=dict(usage="Fraction of each seller's OWN SALES using product 2, in both models",
                                   reference="Rival's true stationary success probability conditional on its sales, taken from the saved OE distribution equilibrium",
                                   matching="Only the outside option is calibrated to rival quality; the monopolist's resulting quality may differ",
                                   scope="Same-grid stationary comparison; original monopoly solved at exact unrounded p0_ref; saved competitive equilibria read-only; no new consistency closure or Monte Carlo",
                                   limitation="Matching mean success does not match the rival's entire distribution of Thompson scores"))
    RUN.atomic_json(data_dir/"manifest.json", report)
    RUN.atomic_json(directory/"figure_manifest.json", report)
    print(f"Saved {len(figures)//2} comparisons, PNG and PDF: {directory}")
    print(data[["rho", "p0_ref", "monopoly_product2_conditional", "oe_product2_conditional",
                "monopoly_minus_oe_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
