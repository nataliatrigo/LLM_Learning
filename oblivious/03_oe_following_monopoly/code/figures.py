"""Figures for the true-quality scalar closure, using saved results only.

The policy and all stationary predictions are inputs.  This module never
re-solves an optimization problem or changes a policy to make a plot smoother.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np


COLOR = "#7B3294"
STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "axes.linewidth": .8,
    "axes.spines.top": True, "axes.spines.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _save(fig, directory, name):
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panels(rhos):
    columns = min(2, len(rhos))
    rows = (len(rhos) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.7 * rows),
                             squeeze=False, layout="constrained")
    flat = list(axes.flat)
    for ax in flat[len(rhos):]:
        ax.set_visible(False)
    return fig, flat[:len(rhos)]


def _verified(e):
    return bool(e and e.get("verified") and e.get("candidate_verified")
                and e.get("stationary_diagnostics", {}).get("verified"))


def _title(case):
    e = case.get("selected")
    status = "verified on grid" if _verified(e) else "UNVERIFIED diagnostic"
    if e is None:
        status = "UNAVAILABLE"
    return f"ρ={case['rho']:g}, n={case['n']} | {status}"


def _missing(ax, text="No saved result available"):
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center",
            color=".35", fontsize=10)


def _choose(cases):
    chosen = {}
    for case in cases:
        rho = float(case["rho"])
        old = chosen.get(rho)
        score = (_verified(case.get("selected")), case["n"])
        if old is None or score > (_verified(old.get("selected")), old["n"]):
            chosen[rho] = case
    return {rho: chosen[rho] for rho in sorted(chosen)}


def _reports(reports, chosen):
    if isinstance(reports, dict):
        reports = list(reports.values())
    result = {}
    for rho, case in chosen.items():
        matching = [report for report in reports
                    if np.isclose(report["rho"], rho, rtol=0, atol=1e-12)
                    and report["n"] == case["n"]]
        if matching:
            e = case.get("selected")
            result[rho] = min(matching, key=lambda r: abs(
                r.get("p0", e["p0"] if e else 0) - (e["p0"] if e else 0)))
    return result


def _distribution(report):
    return report.get("simulation", {}).get("distribution", {})


def _cdf(values):
    array = np.asarray(values, dtype=float)
    return array if array.ndim == 1 else array.mean(axis=0)


def _config_note(reports):
    configurations = [r.get("simulation", {}).get("config", {}) for r in reports.values()]
    if not configurations:
        return "Stationary predictions from the saved finite reputation grid."
    config = configurations[0]
    if not config:
        return "Actual two-seller Thompson Sampling demand in all simulations."
    pieces = []
    for key, label in (("replications", "independent paths"), ("periods", "retained periods"),
                       ("burn_in", "burn-in periods")):
        if key in config:
            pieces.append(f"{config[key]:,} {label}")
    return ("; ".join(pieces) + ".\nActual two-seller TS; start (0.5, 0.5). "
            "MC intervals exclude initialization bias.")


def _policy_and_advantage(chosen, directory, label):
    advantages = [np.asarray(c["selected"]["advantage"], dtype=float)
                  for c in chosen.values() if c.get("selected") is not None]
    lo = min([0.] + [float(a.min()) for a in advantages])
    hi = max([0.] + [float(a.max()) for a in advantages])
    pad = .05 * max(hi - lo, 1.)
    for kind, name, title in (
        ("policy", "01_policy", f"Product choice: {label}"),
        ("advantage", "06_advantage", f"Product 2 advantage: {label}"),
    ):
        fig, axes = _panels(chosen)
        for (rho, case), ax in zip(chosen.items(), axes):
            e = case.get("selected")
            ax.set_title(_title(case))
            ax.set(xlabel="Own reputation", xlim=(-.04, 1.04),
                   xticks=np.linspace(0, 1, 6))
            ax.grid(alpha=.25)
            if kind == "policy":
                ax.set(ylabel="Probability of product 2", ylim=(-.04, 1.04),
                       yticks=np.linspace(0, 1, 6))
            else:
                ax.set(ylabel="Product 2 advantage\nwhen selected", ylim=(lo-pad, hi+pad))
                ax.axhline(0, color="black", ls="--", lw=.8, label="Indifference")
            if e is None:
                _missing(ax)
                continue
            grid, alpha = np.asarray(e["grid"]), np.asarray(e["policy"])
            values = alpha if kind == "policy" else np.asarray(e["advantage"])
            curve_label = label if kind == "policy" else "Product 2 − product 1"
            if not _verified(e):
                curve_label += " (unverified)"
            ax.plot(grid, values, ".-" if _verified(e) else ".:", color=COLOR,
                    ms=3, lw=1.3, label=curve_label, zorder=3)
            mixed = (alpha > 1e-9) & (alpha < 1-1e-9)
            if mixed.any():
                ax.scatter(grid[mixed], values[mixed], marker="D", color=COLOR,
                           s=26, label="Mixing at a grid point", zorder=4)
            ax.legend(loc="best")
        fig.suptitle(title, fontsize=13)
        _save(fig, directory, name)


def _reputation_cdfs(chosen, reports, directory, label):
    fig, axes = _panels(chosen)
    for (rho, case), ax in zip(chosen.items(), axes):
        e = case.get("selected")
        ax.set(title=_title(case), xlabel="Reputation", ylabel="Cumulative probability",
               xlim=(0, 1), ylim=(-.02, 1.02), xticks=np.linspace(0, 1, 6))
        ax.grid(alpha=.25)
        if e is None:
            _missing(ax)
            continue
        grid, nu = np.asarray(e["grid"]), np.asarray(e["nu"])
        stationary_cdf = np.cumsum(.5 * (nu.sum(axis=0) + nu.sum(axis=1)))
        dist = _distribution(reports.get(rho, {}))
        if "cdf" in dist:
            ax.step(grid, _cdf(dist["cdf"]), where="post", color=COLOR, lw=1.5,
                    label="Simulation")
            if "cdf_low" in dist and "cdf_high" in dist:
                ax.fill_between(grid, _cdf(dist["cdf_low"]), _cdf(dist["cdf_high"]),
                                color=COLOR, alpha=.16, linewidth=0, step="post", label="Pointwise 95% MC CI")
        suffix = "" if _verified(e) else " (unverified)"
        ax.step(grid, stationary_cdf, where="post", color=COLOR,
                ls="--" if _verified(e) else ":", lw=1.5,
                label="Saved stationary prediction" + suffix, zorder=4)
        ax.legend(loc="best")
    fig.suptitle(f"{label}: reputation distributions\n" + _config_note(reports), fontsize=12)
    _save(fig, directory, "02_reputation_cdfs")


def _joint_arrays(chosen, reports):
    arrays = []
    for rho, case in chosen.items():
        e = case.get("selected")
        if e is not None:
            arrays.append(np.asarray(e["nu"], dtype=float))
        dist = _distribution(reports.get(rho, {}))
        if "joint" in dist:
            arrays.append(np.asarray(dist["joint"], dtype=float))
    return arrays


def _log_norm(chosen, reports):
    arrays = _joint_arrays(chosen, reports)
    positives = [array[array > 0] for array in arrays if np.any(array > 0)]
    if not positives:
        return LogNorm(vmin=1e-6, vmax=1)
    maximum = max(float(a.max()) for a in positives)
    # Keep the informative low-probability region visible, without allowing
    # numerical stationary tails near machine underflow to flatten the scale.
    minimum = min(1e-6, maximum/1000)
    return LogNorm(vmin=minimum, vmax=max(maximum, minimum*10))


def _edges(grid):
    grid = np.asarray(grid, dtype=float)
    middle = .5 * (grid[:-1] + grid[1:])
    return np.r_[grid[0] - .5*(grid[1]-grid[0]), middle,
                 grid[-1] + .5*(grid[-1]-grid[-2])]


def _joint_figures(chosen, reports, directory, label):
    norm = _log_norm(chosen, reports)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.))
    cmap.set_under(cmap(0.))
    half_width = max((.5 * np.max(np.diff(c["selected"]["grid"]))
                      for c in chosen.values() if c.get("selected") is not None), default=.01)
    limits = (-half_width, 1+half_width)
    for simulated, name in ((False, "03_joint_stationary"),
                             (True, "S01_simulated_joint")):
        fig, axes = _panels(chosen)
        im = None
        for (rho, case), ax in zip(chosen.items(), axes):
            e = case.get("selected")
            ax.set(title=_title(case), xlabel="Seller A's reputation",
                   ylabel="Seller B's reputation", xlim=limits, ylim=limits,
                   xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
            ax.set_aspect("equal", adjustable="box")
            dist = _distribution(reports.get(rho, {}))
            if e is None or (simulated and "joint" not in dist):
                _missing(ax)
                continue
            joint = np.asarray(dist["joint"] if simulated else e["nu"], dtype=float)
            im = ax.pcolormesh(_edges(e["grid"]), _edges(e["grid"]),
                               np.ma.masked_less_equal(joint.T, 0), cmap=cmap,
                               norm=norm, shading="flat", rasterized=True)
        if im is not None:
            fig.colorbar(im, ax=axes, label="Joint probability mass (log scale)",
                         shrink=.85, pad=.025, extend="min")
        method = "Simulated" if simulated else "Saved stationary"
        note = _config_note(reports) if simulated else "Actual two-seller TS game; saved finite-grid predictions."
        fig.suptitle(f"{label}: {method.lower()} joint reputation distributions\n"
                     f"{note}\nCommon color scale in this figure and "
                     + ("03_joint_stationary." if simulated else "S01_simulated_joint."), fontsize=12)
        _save(fig, directory, name)


def _demand(chosen, reports, directory, label):
    fig, axes = _panels(chosen)
    curves = (("d_fixed", "Used for product choice:\nrival's mean success probability p₀", "-", COLOR),
              ("d_marginal", "Real game: rival marginal", "--", "#4D4D4D"),
              ("d_conditional", "Real game: conditional on own reputation", ":", "#0072B2"))
    for (rho, case), ax in zip(chosen.items(), axes):
        e = case.get("selected")
        ax.set(title=_title(case), xlabel="Own reputation",
               ylabel="Probability seller A is selected", xlim=(-.02, 1.02), ylim=(-.03, 1.03))
        ax.grid(alpha=.25)
        arrays = reports.get(rho, {}).get("demand", {}).get("arrays", {})
        if e is None:
            _missing(ax)
            continue
        for key, name, style, color in curves:
            values = arrays.get(key, e.get(key) if key == "d_fixed" else None)
            if values is not None:
                ax.plot(e["grid"], values, ls=style, color=color, lw=1.6, label=name)
        if ax.lines:
            ax.legend(loc="best", fontsize=7)
        else:
            _missing(ax, "No saved demand diagnostic")
    fig.suptitle(f"Demand comparison: {label}\n"
                 "Fixed benchmark calibrated to success probability among the rival's sales", fontsize=12)
    _save(fig, directory, "04_demand")


def _fixed_point_map(cases, chosen, directory, label):
    fig, axes = _panels(chosen)
    for (rho, chosen_case), ax in zip(chosen.items(), axes):
        ax.set(title=_title(chosen_case), xlabel="Assumed true-quality p₀",
               ylabel="Generated success probability\namong rival's sales")
        ax.grid(alpha=.25)
        group = sorted((c for c in cases if c["rho"] == rho), key=lambda c: c["n"])
        pvalues = []
        for index, case in enumerate(group):
            rows = [r for r in case.get("map_rows", []) if r.get("F") is not None
                    and str(r.get("strategy_mode", "pure")).startswith("pure")]
            rows.sort(key=lambda row: row["p0"])
            if rows:
                ax.plot([r["p0"] for r in rows], [r["F"] for r in rows], ".-",
                        color=COLOR, alpha=.45+.55*(index+1)/len(group), lw=1, ms=2,
                        label=f"Pure optimal selector, n={case['n']}")
                pvalues += [float(r["p0"]) for r in rows]
            for branch in case.get("mixing_branches", []):
                trials = [t for t in branch.get("trials", []) if t.get("F") is not None]
                if trials and min(t["G"] for t in trials) <= 0 <= max(t["G"] for t in trials):
                    ax.vlines(branch["p0"], min(t["F"] for t in trials),
                              max(t["F"] for t in trials), color=COLOR, alpha=.3, lw=2)
            for candidate in case.get("candidates", []):
                if not _verified(candidate):
                    continue
                alpha = np.asarray(candidate["policy"])
                mixed = np.any((alpha > 1e-9) & (alpha < 1-1e-9))
                ax.plot(candidate["p0"], candidate["F"], marker="D" if mixed else "o",
                        color=COLOR, mec="black", ms=6, linestyle="none", zorder=5)
                pvalues.append(float(candidate["p0"]))
        parameters = chosen_case.get("configuration", {}).get("parameters", {})
        lo = parameters.get("p1", min(pvalues, default=0))
        hi = parameters.get("p2", max(pvalues, default=1))
        if hi <= lo:
            lo, hi = 0, 1
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="F(p₀) = p₀")
        pad = .04*(hi-lo)
        ax.set(xlim=(lo-pad, hi+pad))
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"{label}: true-quality fixed-point map\n"
                 "Filled diamonds: verified mixed candidates; circles: verified pure candidates\n"
                 "Lines connect sampled pure-selector values; detected candidates need not be exhaustive", fontsize=12)
    _save(fig, directory, "07_quality_fixed_point_map")


def _outcomes(chosen, reports, directory, label):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout="constrained")
    rhos = np.array(list(chosen))
    positions = np.arange(len(rhos))
    metrics = (("product2_market", "Product 2 purchase frequency"),
               ("seller_profit", "Mean seller profit / calendar period"),
               ("success_rate", "Customer success rate"))
    for ax, (metric, ylabel) in zip(axes, metrics):
        predictions, means, lows, highs = [], [], [], []
        for rho, case in chosen.items():
            report = reports.get(rho, {})
            predictions.append(report.get("predicted", {}).get(metric, np.nan)
                               if _verified(case.get("selected")) else np.nan)
            estimate = report.get("simulation", {}).get("summary", {}).get(metric, {})
            means.append(estimate.get("mean", np.nan))
            lows.append(estimate.get("ci_low", np.nan))
            highs.append(estimate.get("ci_high", np.nan))
        predictions, means, lows, highs = map(lambda x: np.asarray(x, dtype=float),
                                             (predictions, means, lows, highs))
        ax.plot(positions, predictions, "D--", color=COLOR, ms=4.5, mfc="white", lw=1.25,
                label="Saved stationary prediction")
        ax.plot(positions, means, "o-", color=COLOR, ms=4, lw=1.4, label="Simulation")
        valid = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
        if valid.any():
            # Draw endpoints directly; a finite-sample interval need not be
            # symmetric around its estimate for every supported MC method.
            ax.vlines(positions[valid], lows[valid], highs[valid], color=COLOR, lw=1.1)
            ax.plot(positions[valid], lows[valid], "_", color=COLOR, ms=5)
            ax.plot(positions[valid], highs[valid], "_", color=COLOR, ms=5)
        ax.set(xlabel="Forgetting parameter ρ", ylabel=ylabel,
               xlim=(-.3, len(rhos)-.7))
        ticklabels = [f"{rho:g}" + ("*" if not _verified(chosen[rho].get("selected")) else "")
                      for rho in rhos]
        ax.set_xticks(positions, ticklabels, rotation=45)
        if metric != "seller_profit":
            ax.set_ylim(-.025, 1.025)
        ax.grid(alpha=.25)
        ax.legend(loc="best", fontsize=8)
    note = "Vertical bars: 95% MC confidence intervals. Rates equally spaced; lines guide the eye."
    if any(not _verified(case.get("selected")) for case in chosen.values()):
        note += " *Unverified: stationary prediction omitted."
    fig.suptitle(f"{label}: outcomes across forgetting rates", fontsize=13)
    fig.get_layout_engine().set(rect=(0, .20, 1, .80))
    fig.text(.5, .02, _config_note(reports) + "\n" + note,
             ha="center", va="bottom", fontsize=8, linespacing=1.35)
    _save(fig, directory, "08_usage_profit_success")


def make_figures(cases, reports, directory, label="OE quality"):
    """Save the cross-rho plot suite without altering equilibrium inputs.

    ``cases`` use the scalar experiment checkpoint schema. ``reports`` contain
    rho/n, simulation summary (mean/ci_low/ci_high per metric), simulation
    distribution (cdf/cdf_low/cdf_high/joint), predicted outcome metrics, and
    the existing demand diagnostic arrays. Missing diagnostics remain visibly
    unavailable. Only fully verified candidates receive equilibrium labels.
    """
    cases = list(cases)
    chosen = _choose(cases)
    if not chosen:
        raise ValueError("At least one saved rho/grid case is required for figures.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = _reports(reports, chosen)
    with plt.rc_context(STYLE):
        _policy_and_advantage(chosen, directory, label)
        _reputation_cdfs(chosen, reports, directory, label)
        _joint_figures(chosen, reports, directory, label)
        _demand(chosen, reports, directory, label)
        _fixed_point_map(cases, chosen, directory, label)
        _outcomes(chosen, reports, directory, label)
