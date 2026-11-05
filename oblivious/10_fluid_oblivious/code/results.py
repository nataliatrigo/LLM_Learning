"""Readable findings for the matched fluid monopoly/competition campaign."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _table(frame, columns, percentages=()):
    lines = ["| " + " | ".join(columns.values()) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for key in columns:
            value = row.get(key)
            if value is None or (isinstance(value,float) and not np.isfinite(value)):
                values.append("—")
            elif key in percentages:
                values.append(f"{100*value:.3f}%")
            elif isinstance(value,float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_results(summary, data, output, config):
    data,output = Path(data),Path(output)
    verified = summary[summary.verified]
    models = [("fluid_monopoly","Fluid monopoly"),("fluid_competition","Fluid competition OE")]
    parts = ["# Results: fluid monopoly versus fluid competition", "",
        f"The matched study uses {len(config['p0s'])*len(config['rhos'])} parameter combinations "
        f"per model, with rho = {', '.join(f'{r:g}' for r in config['rhos'])} and "
        f"p0 = {', '.join(f'{p:g}' for p in config['p0s'])}. Both models use identical "
        "economic parameters, Beta posteriors, continuous-time calibration, state grids, "
        "and mature initial reputation 0.5.", ""]
    for model,label in models:
        selected = summary[summary.model.eq(model)]
        accepted = selected[selected.verified]
        counts = accepted.groupby(["p0","rho"]).size()
        parts.append(f"- **{label}:** {len(accepted)} verified numerical stationary states "
                     f"across {len(counts)} cases; {int((counts>1).sum())} cases have multiple states; "
                     f"{len(selected[~selected.verified])} unresolved rows.")
    parts.extend(["", "The monopoly states are stationary states of its optimal policy. "
        "Competition states additionally satisfy the symmetric fixed-rival consistency equation. "
        "All branches are retained; no arbitrary pure branch is imposed.", "",
        "## Matched example: p0=0.5, rho=0.7", ""])
    example = verified[np.isclose(verified.p0,.5)&np.isclose(verified.rho,.7)]
    if len(example):
        parts.extend([_table(example,{"model":"Fluid model","x_star":"Stationary reputation",
            "product2_conditional":"Product 2 / own sales","sale_rate":"Own market share",
            "product2_calendar":"Own seller & product 2 / period",
            "customer_product2_calendar":"Customer receives product 2 / period"},
            ("product2_conditional","sale_rate","product2_calendar","customer_product2_calendar")),""])
    parts.extend(["The primary measure is product 2 among own sales. The own-seller calendar "
        "measure is demand times that fraction. The secondary customer measure sums "
        "all sellers: one in monopoly, two in symmetric competition.", "",
        "## Multiple stationary states", ""])
    groups = verified.groupby(["model","p0","rho"]).equilibrium_id.transform("size")
    multiple = verified[groups>1]
    if len(multiple):
        parts.extend([_table(multiple,{"model":"Model","p0":"p0","rho":"rho", "equilibrium_id":"Branch",
            "x_star":"Stationary reputation","product2_conditional":"Product 2 / own sales"},
            ("product2_conditional",)),""])
    else:
        parts.extend(["One stationary state was found in every verified case.",""])
    parts.extend(["## Deterministic trajectories", "",
        f"Both models start at mature reputation 0.5 and run for {config['trajectories']['horizon']} "
        "continuous calendar units. Competition starts at (0.5,0.5); realized demand uses "
        "both actual reputations while its policy is optimized against the stationary conjecture. "
        "Monopoly uses the same outside option with one seller. Threshold events and holding "
        "mixtures avoid artificial oscillations.", "",
        "Each monopoly case has one optimal policy and one common-start path even if that "
        "policy has several stationary states. Competition has one path for each equilibrium "
        "policy. A stationary state need not be reached from the common initial condition; "
        "terminal reputations and distances are saved separately.", "",
        "Product-2 use in each time block is the ratio of integrated product-2 sales to "
        "integrated own sales. Calendar rates divide the integrated sales by block duration. "
        "There are no Monte Carlo confidence bands in this deterministic comparison.", "",
        "## Verification and numerical sensitivity", "",
        f"Both models use state grids {config['grids']}. Competition retains its full HJB "
        "optimality, holding-value, tight root-consistency, quadrature-doubling and independent "
        "demand audits. Monopoly checks the same HJB and stationary holding optimality, "
        "and refines its stationary states and policies across the same grids.", ""])
    for filename,label in [("grid_refinement.csv","Competition")]:
        frame = pd.read_csv(data/filename)
        fine = frame[frame.fine_n.eq(config['grids'][-1])]
        parts.append(f"{label} maximum final-grid changes: reputation "
                     f"{fine.reputation_change.max():.3g}, own-sale product-2 fraction "
                     f"{fine.alpha_change.max():.3g}, own share "
                     f"{fine.market_share_change.max():.3g}.")
    parts.extend(["", "These are numerical optimality and sensitivity checks, not rigorous "
        "continuous-state error bounds or an exhaustive existence proof. The additional "
        "[near-boundary check](data/boundary_refinement.json) documents weaker relative "
        "precision for the small competition mixture at p0=.1, rho=.6.", "",
        "## Files", "",
        "- [All comparison figures](figures/overview/fluid_monopoly_vs_competition.pdf).",
        "- [Primary product-2 comparison](figures/overview/02_primary_product2_given_sale_by_rho.png).",
        "- [Secondary customer comparison](figures/overview/04_secondary_customer_product2_by_rho.png).",
        "- [Policies, plot 11](figures/11_policies/), "
        "[reputation paths, plot 13](figures/13_reputation_trajectories/), and "
        "[transient metrics, plot 14](figures/14_transient_metrics/).",
        "- [Matched summaries](data/comparison_summary.csv) and "
        "[all branch-pair differences](data/fluid_vs_fluid_differences.csv).",
        "- [Competition diagnostics](data/manifest.json) and [monopoly diagnostics](data/monopoly_diagnostics.json).",
        "", "Previous campaigns are preserved in data/archive. The active comparison uses "
        "only fluid monopoly and fluid competition at the four configured rho values.", ""])
    (output/"RESULTS.md").write_text("\n".join(parts))
