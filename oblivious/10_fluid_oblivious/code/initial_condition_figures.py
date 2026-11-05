"""Plot actual-state paths and phase-line limits for saved fluid policies.

``plot_case`` expects ``frames[(z, x0)]`` to contain a DataFrame with ``time``
and ``x_a`` columns, where ``z`` is the saved equilibrium's ``x_star``. Both
sellers have the same policy and initial state, so a single reputation curve
represents both. ``pdf_pages`` may be an iterable of open PdfPages writers
(for example, one global report and one report for the current p0). A single
PdfPages writer is also accepted. Writers remain open for the caller.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np

from oblivious.fluid_oblivious.code.compare_initial_conditions import phase_limit
from oblivious.fluid_oblivious.code.trajectories import _policy


COLORS = ("#0072B2", "#009E73", "#E69F00", "#6F42C1", "#CC79A7", "#D55E00")


def _distinct(values, tolerance=1e-9):
    """Merge roundoff-equivalent stationary levels without rounding the data."""
    result = []
    for value in sorted(map(float, values)):
        if not result or abs(value-result[-1]) > tolerance:
            result.append(value)
    return result


def _label(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _phase_samples(policy):
    # Include both sides of every root beyond _Policy.initial's tie tolerance.
    # The tight bracketing makes repelling basin boundaries look like steps,
    # rather than broad sloping regions caused by the plotting sample grid.
    sample = [*np.linspace(0, 1, 1501), *policy.roots]
    for root in policy.roots:
        sample.extend((max(0., root-1e-9), min(1., root+1e-9)))
    sample = np.asarray(sorted(set(sample)))
    limits = np.asarray([phase_limit(policy, x) for x in sample])
    return sample, limits


def plot_case(params, p0, equilibria, initials, frames, figure_dir, horizon,
              *, pdf_pages=None):
    """Save one PNG/PDF figure and append its page to optional PDF reports.

    There is one column per saved equilibrium policy, normally one or two.
    The top row shows realized trajectories; the bottom row maps initial
    reputation to its asymptotic destination under that same fixed policy.
    Returns ``{"png": Path, "pdf": Path}`` for the two saved files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    equilibria, initials = list(equilibria), list(map(float, initials))
    if not equilibria or not initials:
        raise ValueError("At least one saved policy and one initial state are required")
    if len(initials) > len(COLORS):
        raise ValueError("The comparison palette supports up to six initial states")
    columns = len(equilibria)
    colors = dict(zip(initials, COLORS))
    policies = [_policy(params, eq["policy"]) for eq in equilibria]
    samples = [_phase_samples(policy) for policy in policies]
    levels = [_distinct([*limits, *[p["x"] for p in policy.holding_points]])
              for policy, (_, limits) in zip(policies, samples)]
    stationary_levels = _distinct(level for values in levels for level in values)
    repelling_roots = [[root for i, root in enumerate(policy.roots)
                        if policy.kind(i) == "repelling"] for policy in policies]
    has_repelling = any(repelling_roots)
    has_holding = any(policy.holding_points for policy in policies)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rho_token = f"{params.rho:g}".replace(".", "p")
    stem = f"initial_conditions_rho_{rho_token}"
    paths = {extension: figure_dir/f"{stem}.{extension}" for extension in ("png", "pdf")}

    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, columns, squeeze=False,
                                 figsize=(9.8 if columns == 1 else 13.8, 9.7),
                                 gridspec_kw={"height_ratios": [3, 1.45]})
        try:
            for column, (equilibrium, policy, (sample, limits)) in enumerate(
                    zip(equilibria, policies, samples)):
                z = float(equilibrium["x_star"])
                ax, basin = axes[0, column], axes[1, column]
                for level in levels[column]:
                    ax.axhline(level, color="#737373", ls="--", lw=.9, alpha=.75)
                for root in repelling_roots[column]:
                    ax.axhline(root, color="#555555", ls=":", lw=1.25)
                    ax.annotate(f"repelling threshold {_label(root)}",
                                xy=(.98*horizon, root), xytext=(0, 5),
                                textcoords="offset points", ha="right", va="bottom",
                                fontsize=8.5, color="#555555",
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
                for point in policy.holding_points:
                    ax.annotate(f"holding state {_label(point['x'])}",
                                xy=(.98*horizon, point["x"]), xytext=(0, 5),
                                textcoords="offset points", ha="right", va="bottom",
                                fontsize=8.5, color="#555555",
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
                for x0 in initials:
                    frame = frames[(z, x0)]
                    emphasized = abs(x0-.55) < 1e-12
                    ax.plot(frame["time"], frame["x_a"], color=colors[x0],
                            lw=2.7 if emphasized else 1.8,
                            zorder=4 if emphasized else 3)
                visible_states = [*initials, *levels[column], *repelling_roots[column]]
                low, high = min(visible_states), max(visible_states)
                margin = max(.04, .06*(high-low))
                ax.set(title=f"Policy optimized against z = {z:.6f}",
                       xlabel="Continuous calendar time, t", ylabel="Reputation x(t)",
                       xlim=(0, horizon), ylim=(max(0., low-margin), min(1., high+margin)))
                ax.grid(axis="y", alpha=.15)

                basin.plot(sample, limits, color="#7b848d", lw=1.5)
                for root in repelling_roots[column]:
                    basin.axvline(root, color="#555555", ls=":", lw=1.1)
                for point in policy.holding_points:
                    basin.scatter(point["x"], point["x"], marker="D", s=52,
                                  facecolor="white", edgecolor="#444444", linewidth=1.2,
                                  zorder=4)
                for x0 in initials:
                    basin.scatter(x0, phase_limit(policy, x0),
                                  s=78 if abs(x0-.55) < 1e-12 else 48,
                                  facecolor=colors[x0], edgecolor="white", linewidth=.6,
                                  zorder=5)
                low, high = min(stationary_levels), max(stationary_levels)
                margin = max(.055, .11*(high-low))
                basin.set(title="Long-run outcome under this frozen policy",
                          xlabel="Common initial reputation, x(0)",
                          ylabel="Long-run reputation", xlim=(0, 1),
                          ylim=(max(0., low-margin), min(1., high+margin)),
                          yticks=stationary_levels,
                          yticklabels=[_label(level) for level in stationary_levels])
                basin.grid(axis="y", alpha=.15)

            handles = [Line2D([], [], color=colors[x0],
                              lw=2.7 if abs(x0-.55) < 1e-12 else 1.8,
                              label=f"x(0) = {x0:g}") for x0 in initials]
            if has_repelling:
                handles.append(Line2D([], [], color="#555555", ls=":", lw=1.25,
                                      label="Repelling policy threshold"))
            handles.append(Line2D([], [], color="#737373", ls="--", lw=.9,
                                  label="Stationary state under the policy"))
            if has_holding:
                handles.append(Line2D([], [], color="#444444", marker="D", ls="none",
                                      markerfacecolor="white", label="Mixed holding state"))
            fig.legend(handles=handles, loc="lower center",
                       ncol=3 if columns == 1 else 4,
                       bbox_to_anchor=(.5, .065), frameon=False, fontsize=9)
            policy_word = "policy" if columns == 1 else "policies"
            fig.suptitle(
                f"Fluid competition: initial conditions under saved optimal {policy_word}",
                fontsize=15, x=.085, ha="left", y=.98)
            fig.text(.085, .94, rf"$p_0={p0:g}$, $\rho={params.rho:g}$"
                     "  |  Symmetric mature starts: x_A(0) = x_B(0).", fontsize=10)
            fig.text(.085, .915,
                     "Both sellers use the same frozen policy; demand uses both actual reputations.",
                     fontsize=10)
            footer = (f"Top: realized paths through T = {horizon:g}. Bottom: phase-line limits; "
                      "slow paths may remain far from their limits. A path's limit need not "
                      "equal the policy's rival conjecture.")
            fig.text(.085, .018,
                     textwrap.fill(footer, width=110 if columns == 1 else 152), fontsize=9)
            # Six-decimal mixed-state ticks need extra room beside the
            # vertical label in the narrower single-policy figure.
            fig.subplots_adjust(left=.135 if columns == 1 else .085, right=.98, top=.855,
                                bottom=.235, hspace=.49, wspace=.25)
            for path in paths.values():
                fig.savefig(path, dpi=180, facecolor="white")
            if pdf_pages is not None:
                writers = [pdf_pages] if hasattr(pdf_pages, "savefig") else pdf_pages
                for writer in writers:
                    writer.savefig(fig, facecolor="white")
        finally:
            plt.close(fig)
    return paths
