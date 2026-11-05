"""Experiment-10 comparison panels with experiment-11 initial-state consistency.

All rendering primitives come from experiment 10. The comparison uses the same
common mature start x0=.5 for metrics and trajectories. Policy figures retain
every verified competition equilibrium, with its explicit conjecture.
The source tables are read only; this module never rewrites numerical results.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from oblivious.fluid_oblivious.code.reporting import (
    _legend, _metric_figure, _models, _normalise_fluid, _raw_policy,
    _status_figure, _style, _subplots, _trajectory_figures,
)
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import contains


SOURCE = Path(__file__).resolve().parents[2]/"10_fluid_oblivious"
COMMON_START = .5
POLICY_COLORS = ("#D55E00", "#009E73")
POLICY_LINEWIDTH = 1.8


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _requested(frame, p0s, rhos):
    return frame[frame.p0.isin(p0s) & frame.rho.isin(rhos)].copy()


def _filter_competition(frame, selected):
    mask = frame.model.eq("fluid_monopoly")
    for p0, rho, identity in selected:
        mask |= (frame.model.eq("fluid_competition") & np.isclose(frame.p0, p0)
                 & np.isclose(frame.rho, rho) & frame.equilibrium_id.eq(identity))
    return frame.loc[mask].copy()


def _solid_series(figure):
    """Apply the requested solid-line presentation to curves and their legends.

    The original experiment-10 renderers already distinguish the two models
    with blue and orange. Preserve those colors and marker-only stationary
    points while removing dashed/dotted line coding.
    """
    marker_only = {None, "None", "none", "", " "}
    for axis in figure.axes:
        for line in axis.lines:
            if line.get_linestyle() not in marker_only:
                line.set_linestyle("-")
    legends = list(figure.legends)
    legends.extend(axis.get_legend() for axis in figure.axes if axis.get_legend() is not None)
    for legend in legends:
        for handle in legend.legend_handles:
            if hasattr(handle, "get_linestyle") and handle.get_linestyle() not in marker_only:
                handle.set_linestyle("-")


def _comparison_tables(cases, p0s, rhos, parameters):
    configuration_path = SOURCE/"config.json"
    configuration = json.loads(configuration_path.read_text())
    if configuration["parameters"] != parameters:
        raise ValueError("The experiment-10 reference uses different economic parameters")
    if (not set(p0s).issubset(configuration["p0s"])
            or not set(rhos).issubset(configuration["rhos"])):
        raise ValueError("The experiment-10 reference does not cover the requested parameter grid")
    if configuration["trajectories"]["initial"] != [COMMON_START, COMMON_START]:
        raise ValueError("The experiment-10 reference does not start from common reputation .5")
    paths = {name: SOURCE/"data"/name for name in (
        "comparison_summary.csv", "comparison_policies.csv", "trajectories.csv",
        "trajectory_bins.csv", "monopoly_trajectories.csv", "monopoly_trajectory_bins.csv",
    )}
    summary = _requested(_normalise_fluid(pd.read_csv(paths["comparison_summary.csv"])), p0s, rhos)
    policies = _requested(_models(pd.read_csv(paths["comparison_policies.csv"])), p0s, rhos)
    selected = set()
    eligible, excluded = [], []
    for case in cases:
        p0, rho = float(case["p0"]), float(case["rho"])
        if p0 not in p0s or rho not in rhos:
            continue
        reference = summary[(summary.model == "fluid_competition")
                            & np.isclose(summary.p0, p0) & np.isclose(summary.rho, rho)]
        for equilibrium in case["equilibria"]:
            x_bar = float(equilibrium["x_bar"])
            matching = reference[np.isclose(reference.x_star, x_bar, atol=1e-9, rtol=0)]
            if len(matching) != 1:
                raise ValueError(f"No unique experiment-10 branch for p0={p0}, rho={rho}, x_bar={x_bar}")
            row = matching.iloc[0]
            if not bool(row.verified) or int(row.n) != int(case["grid_nodes"]):
                raise ValueError("Reference branch verification or numerical grid differs")
            record = dict(p0=p0, rho=rho, x_bar=x_bar,
                          experiment11_equilibrium_id=int(equilibrium["equilibrium_id"]),
                          experiment10_equilibrium_id=int(row.equilibrium_id))
            initial_set = equilibrium.get("initial_state_set", equilibrium.get("initial_conditions", []))
            if equilibrium["verified"] and contains(initial_set, COMMON_START):
                selected.add((p0, rho, int(row.equilibrium_id)))
                eligible.append(record)
            else:
                excluded.append(record)
    filtered = _filter_competition(summary, selected)
    # Empty competition cases remain explicit outcomes in the status panel.
    missing = []
    for p0 in p0s:
        for rho in rhos:
            if not any(a == p0 and b == rho for a, b, _ in selected):
                missing.append(dict(p0=p0, rho=rho, model="fluid_competition",
                                    equilibrium_id=-1, verified=False,
                                    status="no_equilibrium_found", seller_count=2))
    if missing:
        filtered = pd.concat([filtered, pd.DataFrame(missing)], ignore_index=True)
    policies = _filter_competition(policies, selected)
    competition_paths = _requested(_models(pd.read_csv(paths["trajectories.csv"])), p0s, rhos)
    competition_bins = _requested(_models(pd.read_csv(paths["trajectory_bins.csv"])), p0s, rhos)
    monopoly_paths = _requested(_models(pd.read_csv(paths["monopoly_trajectories.csv"])), p0s, rhos)
    monopoly_bins = _requested(_models(pd.read_csv(paths["monopoly_trajectory_bins.csv"])), p0s, rhos)
    trajectories = pd.concat([monopoly_paths, _filter_competition(competition_paths, selected)], ignore_index=True)
    bins = pd.concat([monopoly_bins, _filter_competition(competition_bins, selected)], ignore_index=True)
    metadata = dict(
        comparison_initial_state=COMMON_START,
        eligible_competition_branches=eligible,
        excluded_competition_branches=excluded,
        monopoly_stationary_branch_count=int(filtered.model.eq("fluid_monopoly").sum()),
        source_sha256={str(path.relative_to(SOURCE.parent.parent)): _digest(path)
                       for path in [configuration_path, SOURCE/"code/reporting.py", *paths.values()]},
        interpretation="Metric, status and trajectory comparisons use competition equilibria whose initial sets contain x0=.5. "
                       "Policy figures retain every verified competition equilibrium. All verified monopoly stationary "
                       "reference states are retained as in experiment 10.",
    )
    return filtered, policies, trajectories, bins, metadata


def _all_equilibrium_policy_figure(cases, summary, policies, *, p0, rhos):
    """Show the monopoly benchmark and every verified competition policy."""
    lookup = {(float(case["p0"]), float(case["rho"])):case for case in cases}
    figure, axes = _subplots(len(rhos))
    for axis, rho in zip(axes, rhos, strict=True):
        monopoly = summary[(summary.model == "fluid_monopoly") & summary.verified
                           & np.isclose(summary.p0, p0) & np.isclose(summary.rho, rho)]
        states = monopoly.to_dict("records")
        if states:
            samples = _raw_policy(policies, states[0])
            if not samples.empty:
                axis.step(samples.reputation, samples.product2_probability, where="mid",
                          color="#0072B2", ls="-", lw=POLICY_LINEWIDTH, label="Fluid monopoly", zorder=2)
            for state in states:
                axis.scatter(state["x_star"], state["product2_conditional"], marker="D", s=30,
                             facecolors="none", edgecolors="#0072B2", linewidths=1.4, zorder=8)
        equilibria = sorted((eq for eq in lookup[(p0,rho)]["equilibria"] if eq["verified"]),
                            key=lambda eq:eq["x_bar"])
        for branch, equilibrium in enumerate(equilibria):
            color = POLICY_COLORS[branch]
            policy = equilibrium["policy"]
            edges = [0., *policy["roots"], 1.]
            xx = np.repeat(edges, 2)[1:-1]
            yy = np.repeat([float(product == 2) for product in policy["products"]], 2)
            label = rf"Competition: $\rho={rho:g}$, $\bar x={equilibrium['x_bar']:.6f}$"
            axis.plot(xx, yy, color=color, ls="-", lw=POLICY_LINEWIDTH,
                      label=label, zorder=3+branch)
            axis.scatter(equilibrium["x_bar"], equilibrium["alpha"], marker="D", s=30,
                         facecolors="none", edgecolors=color, linewidths=1.4, zorder=9+branch)
        axis.set(title=rf"$\rho={rho:g}$", xlabel="Own normalized reputation x",
                 ylabel=r"$\sigma(x)$: product 2 probability", xlim=(0,1), ylim=(-.03,1.04))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        _style(axis)
    _legend(figure, axes)
    figure.suptitle(rf"Fluid optimal policies | outside option $p_0={p0:g}$"+
                    "\nAll verified competition equilibria; diamonds show the stationary product mix.", fontsize=12)
    return figure


def comparison_figures(cases, figure_root, *, p0s, rhos, parameters):
    """Return ([(folder, stem, Figure), ...], source/selection metadata).

    ``figure_root`` is the caller's intended output location; saving remains
    with the caller so the full suite can share one combined PDF and manifest.
    """
    p0s, rhos = list(map(float, p0s)), list(map(float, rhos))
    data, policies, trajectories, bins, metadata = _comparison_tables(cases, p0s, rhos, parameters)
    figures = []
    for number, (metric, stem, title, ylabel, limits) in enumerate((
        ("product2_conditional", "primary_product2_given_sale", "Primary: product 2 among own sales", "Product 2 (% of own sales)", True),
        ("customer_product2_calendar", "secondary_customer_product2", "Secondary: product 2 received by the customer", "Any seller & product 2\n(% of calendar time)", False),
        ("product2_calendar", "seller_product2_per_period", "Own seller selected and uses product 2", "One seller & product 2\n(% of calendar time)", False),
    )):
        for offset, horizontal in enumerate(("p0", "rho"), 1):
            if metric == "product2_calendar" and horizontal == "p0":
                continue
            figure = _metric_figure(data, p0s, rhos, horizontal=horizontal, metric=metric,
                                    title=title, ylabel=ylabel, fixed_limits=limits)
            figure.suptitle(title+"\nBoth models fluid; competition consistency from common x0=0.5.", fontsize=12)
            figures.append(("overview", f"{2*number+offset:02d}_{stem}_by_{horizontal}", figure))
    for metric, stem, title, ylabel in (
        ("x_star", "07_equilibrium_reputations", "Fluid stationary reputations", "Normalized stationary reputation x*"),
        ("sale_rate", "08_market_shares", "Own-seller market share", "One seller selected\n(% of calendar time)"),
        ("outside_rate", "09_outside_option_share", "Outside-option market share", "Outside selected\n(% of calendar time)"),
    ):
        figure = _metric_figure(data, p0s, rhos, horizontal="rho", metric=metric,
                                title=title, ylabel=ylabel)
        figure.suptitle(title+"\nBoth models fluid; competition consistency from common x0=0.5.", fontsize=12)
        figures.append(("overview", stem, figure))
    status = _status_figure(data, p0s, rhos)
    status.suptitle("Verified optimal stationary states\nMonopoly reference; competition consistency from common x0=0.5.", fontsize=12)
    figures.append(("overview", "10_equilibrium_status", status))
    for p0 in p0s:
        figure = _all_equilibrium_policy_figure(cases, data, policies, p0=p0, rhos=rhos)
        figures.append(("11_policies", "11_policies_p0_"+str(p0).replace(".", "p"), figure))
    for stem, figure in _trajectory_figures(trajectories, bins, p0s, rhos):
        folder = "13_reputation_trajectories" if stem.startswith("13_") else "14_transient_metrics"
        figures.append((folder, stem, figure))
    for _, _, figure in figures:
        _solid_series(figure)
    metadata["figure_count"] = len(figures)
    metadata["output_root"] = str(figure_root)
    metadata["line_presentation"] = "Solid curves; blue monopoly, distinct colors for competition equilibrium policies"
    metadata["policy_scope"] = "All verified competition equilibria; policy figures do not filter on x0=.5"
    metadata["policy_colors"] = dict(monopoly="#0072B2", single_or_lower_equilibrium=POLICY_COLORS[0],
                                     higher_equilibrium=POLICY_COLORS[1])
    metadata["policy_competition_branch_count"] = sum(
        eq["verified"] for case in cases if case["p0"] in p0s and case["rho"] in rhos
        for eq in case["equilibria"])
    return figures, metadata
