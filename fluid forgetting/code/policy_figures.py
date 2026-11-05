"""Plot saved fluid monopoly feedback rules without interpolating actions.

Each panel contains one policy over mature reputation x. Horizontal segments
are pure choices; markers define the action at a switching state. The input
is the saved trajectory dataset, so no HJB problem is solved here.
"""
from __future__ import annotations

import math
from pathlib import Path

from forgetting.Code.model import Primitives
from simulate_trajectories import PiecewisePolicy


POLICY_COLOR = "#0072B2"
HOLDING_COLOR = "#D55E00"
THRESHOLD_COLOR = "#8A9098"


def make_policy_figure(dataset, output_dir, *, pdf_pages=None):
    """Save ``optimal-policies.png/pdf`` and optionally append to PdfPages.

    ``pdf_pages`` is one open matplotlib PdfPages writer, owned by the caller.
    Returns a mapping from ``"png"``/``"pdf"`` to the corresponding Path.
    The panel grid matches the saved trajectory overview: up to three columns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    cases = list(dataset["cases"])
    if not cases:
        raise ValueError("At least one saved policy is required")
    primitives = Primitives.from_mapping(dataset["parameters"])
    columns = min(3, len(cases))
    rows = math.ceil(len(cases)/columns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {extension: output_dir/f"optimal-policies.{extension}"
             for extension in ("png", "pdf")}
    has_holding, has_stationary, has_threshold = False, False, False
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(4.35*columns, 3.45*rows+2.1),
                                 sharex=True, sharey=True)
        try:
            for ax, case in zip(axes.flat, cases):
                roots = list(map(float, case["roots"]))
                products = [int(interval["product"]) for interval in case["intervals"]]
                policy = PiecewisePolicy(primitives, case["rho"], roots, products)
                for interval in case["intervals"]:
                    alpha = float(interval["product"] == 2)
                    # Separate segments preserve discontinuities: no ramps or
                    # blue vertical connectors are introduced at the roots.
                    ax.plot([interval["lo"], interval["hi"]], [alpha, alpha],
                            color=POLICY_COLOR, lw=2.3, solid_capstyle="butt", zorder=3)
                for index, root in enumerate(roots):
                    has_threshold = True
                    ax.axvline(root, color=THRESHOLD_COLOR, ls=":", lw=1.0, zorder=1)
                    adjacent = {float(products[index] == 2), float(products[index+1] == 2)}
                    for alpha in adjacent:
                        ax.plot(root, alpha, marker="o", markersize=5.2,
                                markerfacecolor="white", markeredgecolor=POLICY_COLOR,
                                markeredgewidth=1.3, ls="none", zorder=5)
                    product, alpha = policy.action_at(root)
                    if product == 0:
                        has_holding = True
                        ax.plot(root, alpha, marker="D", markersize=6.3,
                                markerfacecolor=HOLDING_COLOR,
                                markeredgecolor="white", markeredgewidth=.7,
                                ls="none", zorder=7)
                    else:
                        # The simulator selects product 1 at a repelling tie;
                        # crossing ties select the outgoing pure action.
                        ax.plot(root, alpha, marker="o", markersize=3.0,
                                markerfacecolor=POLICY_COLOR,
                                markeredgecolor=POLICY_COLOR,
                                ls="none", zorder=6)
                for probability in (primitives.p1, primitives.p2):
                    product, alpha = policy.action_at(probability)
                    if product and abs(policy.success_probability(product)-probability) < 1e-12:
                        has_stationary = True
                        ax.plot(probability, alpha, marker="o", markersize=7.0,
                                markerfacecolor="white", markeredgecolor="#222222",
                                markeredgewidth=1.3, ls="none", zorder=8)

                # List nearby roots together instead of putting numeric labels
                # beside individual vertical lines, where they can overlap.
                threshold_text = ("Switching x: " + ", ".join(f"{root:.4f}" for root in roots)
                                  if roots else "No switching threshold")
                ax.text(.025, .965, threshold_text, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.5, color="#525B66",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
                ax.set_title(rf"$\rho={case['rho']:g}$", loc="left", fontsize=12)
                ax.set(xlim=(-.015, 1.015), ylim=(-.12, 1.25),
                       xticks=[0, .2, .4, .6, .8, 1], yticks=[0, .5, 1],
                       xlabel="Normalized reputation, x",
                       ylabel=r"Product 2 among sales, $\alpha^*(x)$")
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.grid(axis="y", alpha=.16)
            for ax in list(axes.flat)[len(cases):]:
                ax.set_visible(False)

            handles = [Line2D([], [], color=POLICY_COLOR, lw=2.3,
                              label="Optimal pure-action intervals")]
            if has_threshold:
                handles.extend([
                    Line2D([], [], color=THRESHOLD_COLOR, ls=":", lw=1,
                           label="Policy switching threshold"),
                    Line2D([], [], color=POLICY_COLOR, marker="o", ls="none",
                           markersize=5, markerfacecolor="white",
                           label="Open endpoint at a jump"),
                    Line2D([], [], color=POLICY_COLOR, marker="o", ls="none",
                           markersize=3, label="Selected pure action at the threshold"),
                ])
            if has_holding:
                handles.append(Line2D([], [], color=HOLDING_COLOR, marker="D", ls="none",
                                      markersize=6, label="Isolated holding mixture"))
            if has_stationary:
                handles.append(Line2D([], [], color="#222222", marker="o", ls="none",
                                      markersize=7, markerfacecolor="white",
                                      label="Pure stationary state"))
            fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                       bbox_to_anchor=(.5, .075), fontsize=9)
            fig.suptitle("Optimal fluid monopoly policies with forgetting",
                         fontsize=17, x=.085, ha="left", y=.98)
            caption = (rf"$p_0={primitives.p0:g}$, $p_1={primitives.p1:g}$, "
                       rf"$p_2={primitives.p2:g}$, $c_1={primitives.c1:g}$, "
                       rf"$c_2={primitives.c2:g}$, $R={primitives.revenue:g}$, "
                       rf"$\gamma={primitives.gamma:g}$; mature boundary.")
            fig.text(.085, .935, caption, fontsize=10)
            fig.text(.085, .905,
                     "One optimal feedback rule per panel. 0% means product 1; 100% means product 2.",
                     fontsize=10)
            fig.text(.085, .032,
                     "At an attractive threshold, the orange diamond is the holding fraction "
                     r"$\alpha^*=(x-p_1)/(p_2-p_1)$ used only at that state.", fontsize=9)
            fig.subplots_adjust(left=.085, right=.985, top=.85, bottom=.205,
                                hspace=.35, wspace=.28)
            for path in paths.values():
                fig.savefig(path, dpi=180, facecolor="white")
            if pdf_pages is not None:
                pdf_pages.savefig(fig, facecolor="white")
        finally:
            plt.close(fig)
    return paths
