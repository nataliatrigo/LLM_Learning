"""Figures and research interpretation for the policy comparison experiment.

All uncertainty bands come from independent simulation replications.  Continuous
policy extensions are displayed separately from validation of the finite grid.
This module never solves an equilibrium or changes source experiment outputs.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from model_names import MODEL_LABELS


LABELS = MODEL_LABELS
COLORS = {"oe": "#009E73", "p0": "#D55E00"}
MODES = {
    "grid": "Reputations restricted to saved values",
    "nearest": "Continuous reputations: closest saved rule",
    "linear": "Continuous reputations: blended saved rules",
}
MODE_TICKS = {
    "grid": "Saved reputation\nvalues",
    "nearest": "Continuous values\nClosest saved rule",
    "linear": "Continuous values\nBlended saved rules",
}


def _method_note(fig, modes):
    """Explain the reputation update labels without implying new optimization."""
    if len(modes) == 1:
        descriptions = {
            "grid": "Reputations move among the values used in the original experiments.",
            "nearest": "Reputations change continuously; use the product probability at the closest saved reputation.",
            "linear": "Reputations change continuously; average the product probabilities at neighboring saved reputations.",
        }
        text = descriptions[modes[0]] + " Both policies were computed earlier and stay fixed during simulation."
    else:
        text = (
            "Saved values: reputations move among the values used in the original experiments. Policies stay fixed during simulation.\n"
            "Continuous values: use the closest saved product probability, or blend neighboring saved probabilities.")
    fig.supxlabel(text, fontsize=8)


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=175,
                    bbox_inches="tight")
    plt.close(fig)


def _axes(rows=1, cols=1, width=5.5, height=3.5):
    fig, axs = plt.subplots(rows, cols, figsize=(width * cols, height * rows),
                            squeeze=False, layout="constrained")
    return fig, axs


def _style(ax):
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def _row(rows, mode, profile, metric):
    return next((row for row in rows
                 if row.get("mode") == mode and row.get("profile") == profile
                 and row.get("metric") == metric), None)


def _ci(row):
    mean = float(row["mean"]) if row.get("mean") is not None else np.nan
    low = row.get("ci_low", row.get("lower", mean))
    high = row.get("ci_high", row.get("upper", mean))
    low = float(low) if low is not None else mean
    high = float(high) if high is not None else mean
    return mean, low, high


def _comparison_points(ax, rows, modes, metric, ylabel=None, percent=False, *, by_seller=False):
    scale = 100 if percent else 1
    single_mode = len(modes) == 1
    sellers = (("A", -.055, "o"), ("B", .055, "s")) if by_seller else ((None, 0, "o"),)
    for index, (offset, profile) in enumerate([(-.13, "oe"), (.13, "p0")]):
        for k, mode in enumerate(modes):
            for seller, seller_offset, marker in sellers:
                seller_metric = f"{metric}_{seller}" if seller else metric
                row = _row(rows, mode, profile, seller_metric)
                if row is None:
                    continue
                mean, low, high = np.array(_ci(row)) * scale
                if not np.isfinite(mean):
                    continue
                position = (index if single_mode else k + offset) + seller_offset
                label = LABELS[profile] + (f" · Seller {seller}" if seller else "")
                ax.errorbar(position, mean,
                            yerr=[[max(0, mean-low)], [max(0, high-mean)]],
                            marker=marker, ms=6, capsize=4, color=COLORS[profile],
                            markerfacecolor="white" if seller == "B" else COLORS[profile],
                            label=label if k == 0 else None)
    if single_mode:
        ax.set_xticks([0, 1], ["OE distribution", "OE p0"])
        ax.set_xlim(-.5, 1.5)
        ax.set_xlabel("Policy used by both sellers")
    else:
        ax.set_xticks(range(len(modes)), [MODE_TICKS[mode] for mode in modes], fontsize=8)
        ax.set_xlabel("How reputations are simulated")
    if ylabel:
        ax.set_ylabel(ylabel)
    _style(ax)


def _line_label(profile, description):
    return f"{LABELS[profile]} — {description}"


def _profile_grid(profile):
    return np.asarray(profile["grid"], dtype=float)


def _predicted_mu(profile):
    return np.asarray(profile["mu"], dtype=float)


def _value_text(value, digits=5):
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "not available"


def _estimate_text(row, percent=False):
    if row is None:
        return "not available"
    scale = 100 if percent else 1
    mean, low, high = np.array(_ci(row)) * scale
    if not np.isfinite(mean):
        return "not estimable in these replications"
    unit = "%" if percent else ""
    return f"{mean:.5g}{unit} [95% CI {low:.5g}, {high:.5g}]"


def make_figures(pair, analysis, figures_dir):
    """Create the retained comparison and time-evolution figures from saved data."""
    profiles = pair["profiles"]
    modes = [mode for mode in MODES if mode in analysis["distributions"]]
    rows = [row for row in analysis["summary"]
            if row.get("window", "baseline") == "baseline"]
    main_mode = "nearest" if "nearest" in modes else modes[0]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "figure.titlesize": 13}):
        # 1. Stored probabilities, including all mixed nodes.
        fig, axs = _axes(width=9, height=4.2)
        ax = axs[0, 0]
        for key, profile in profiles.items():
            x, policy = _profile_grid(profile), np.asarray(profile["policy"])
            ax.plot(x, policy, ".-", ms=4, color=COLORS[key], label=LABELS[key])
            mixed = (policy > 1e-9) & (policy < 1 - 1e-9)
            ax.scatter(x[mixed], policy[mixed], s=40, marker="D",
                       facecolors="none", edgecolors=COLORS[key], zorder=4)
        ax.set(xlabel="Own reputation", ylabel="Probability of choosing product 2",
               xlim=(0, 1), ylim=(-.035, 1.035))
        ax.legend(loc="best")
        _style(ax)
        fig.suptitle("Product-choice policies saved from the original experiments\n"
                     "Diamonds mark randomized product choices")
        _save(fig, figures_dir, "01_product_policies")

        # 2. A few paths reveal stochastic variation, not stationary averages.
        samples = analysis["samples"][main_mode]
        reps = min(3, min(np.asarray(v["before"]).shape[0]
                          for v in samples.values()))
        fig, axs = _axes(rows=reps, cols=2, height=2.65)
        for col, key in enumerate(("oe", "p0")):
            sample = samples[key]
            state = np.asarray(sample["before"])
            for rep in range(reps):
                ax = axs[rep, col]
                time = np.arange(state.shape[1])
                ax.plot(time, state[rep, :, 0], color=COLORS[key], lw=.85,
                        label="Seller A")
                ax.plot(time, state[rep, :, 1], color="#6B64A1", lw=.85,
                        alpha=.85, label="Seller B")
                ax.set(title=f"{LABELS[key]} · simulation {rep + 1}",
                       ylabel="Reputation", ylim=(0, 1))
                if rep == reps - 1:
                    ax.set_xlabel("Period since the initial state")
                if rep == 0:
                    ax.legend(loc="best")
                _style(ax)
        fig.suptitle("Examples of reputation over time\n"
                     f"{MODES[main_mode]}; both sellers start at reputation 0.5")
        _save(fig, figures_dir, "02_sample_reputation_trajectories")

        # 3. CDFs are evaluated on the actual continuous states, not histogram bins.
        fig, axs = _axes(rows=len(modes), cols=2, height=3.1)
        for row_index, mode in enumerate(modes):
            for seller in range(2):
                ax = axs[row_index, seller]
                for key in ("oe", "p0"):
                    distribution = analysis["distributions"][mode][key]
                    x = np.asarray(distribution["grid"])
                    cdf = np.asarray(distribution["cdf"])[seller]
                    ax.plot(x, cdf, color=COLORS[key], lw=1.8,
                            label=_line_label(key, "simulated"))
                    ax.fill_between(x, np.asarray(distribution["cdf_low"])[seller],
                                    np.asarray(distribution["cdf_high"])[seller],
                                    color=COLORS[key], alpha=.13)
                    predicted = np.cumsum(np.asarray(distribution["predicted_mu"])[seller])
                    ax.step(x, predicted, where="post", ls="--", lw=1.25,
                            color=COLORS[key],
                            label=_line_label(key, "saved long-run prediction"))
                ax.set(title=f"{MODES[mode]} · seller {'AB'[seller]}",
                       xlabel="Reputation", ylabel="Share at or below this reputation",
                       xlim=(0, 1), ylim=(-.02, 1.02))
                if row_index == 0 and seller == 0:
                    ax.legend(loc="upper left", fontsize=7)
                _style(ax)
        fig.suptitle("Reputation distributions after the initial adjustment period\n"
                     "Solid: simulations; dashed: original predictions; shading: 95% confidence intervals")
        _save(fig, figures_dir, "03_stationary_marginal_distributions")

        # 4. Identical axes and color range throughout, including extensions.
        fig, axs = _axes(rows=len(modes), cols=2, height=3.8, width=4.7)
        vmax = max(float(np.max(analysis["distributions"][mode][key]["joint"]))
                   for mode in modes for key in ("oe", "p0"))
        for row_index, mode in enumerate(modes):
            for col, key in enumerate(("oe", "p0")):
                ax = axs[row_index, col]
                distribution = analysis["distributions"][mode][key]
                x = np.asarray(distribution["grid"])
                edges = np.r_[0., (x[:-1] + x[1:]) / 2, 1.]
                image = ax.pcolormesh(edges, edges,
                                     np.asarray(distribution["joint"]).T,
                                     cmap="viridis", vmin=0, vmax=vmax,
                                     rasterized=True)
                ax.set(title=f"{LABELS[key]}\n{MODES[mode]}",
                       xlabel="Seller A's reputation", ylabel="Seller B's reputation",
                       xlim=(0, 1), ylim=(0, 1), aspect="equal")
        fig.colorbar(image, ax=axs.ravel().tolist(), shrink=.8,
                     label="Share of periods in each reputation range")
        fig.suptitle("Frequency of the two sellers' reputation pairs")
        _save(fig, figures_dir, "04_joint_stationary_distributions")

        # 5. Compare A and B together, then show the market-wide frequency.
        fig, axs = _axes(cols=2, width=6.3, height=4.1)
        _comparison_points(axs[0, 0], rows, modes, "product2_conditional",
                           "Share of that seller's sales (%)", True, by_seller=True)
        axs[0, 0].set_title("Within each seller's sales")
        axs[0, 0].legend(loc="best", ncols=2, fontsize=8)
        _comparison_points(axs[0, 1], rows, modes, "product2_market",
                           "Share of all market sales (%)", True)
        axs[0, 1].set_title("Across all market sales")
        axs[0, 1].legend(loc="best")
        fig.suptitle("Product 2 purchases\n"
                     "After initial adjustment; dots: simulation averages; bars: 95% confidence intervals")
        _method_note(fig, modes)
        _save(fig, figures_dir, "05_product2_usage")

        # 6. Never mix steady-state flow profits with discounted initial values.
        fig, axs = _axes(rows=2, cols=2, width=6.3, height=3.5)
        _comparison_points(axs[0, 0], rows, modes, "profit",
                           "Average profit per period", by_seller=True)
        axs[0, 0].set_title("Average after initial adjustment\nSellers A and B")
        _comparison_points(axs[0, 1], rows, modes, "industry_profit",
                           "Average profit per period")
        axs[0, 1].set_title("Average after initial adjustment\nBoth sellers combined")
        _comparison_points(axs[1, 0], rows, modes, "discounted_profit",
                           "Discounted sum of profits", by_seller=True)
        axs[1, 0].set_title("Discounted total from the initial state\nSellers A and B")
        _comparison_points(axs[1, 1], rows, modes, "discounted_industry_profit",
                           "Discounted sum of profits")
        axs[1, 1].set_title("Discounted total from the initial state\nBoth sellers combined")
        axs[0, 0].legend(loc="best", ncols=2, fontsize=8)
        axs[0, 1].legend(loc="best")
        fig.suptitle("Seller profits under the saved policies\n95% confidence intervals across independent simulations; "
                     "discounted sums start from (0.5, 0.5)")
        _method_note(fig, modes)
        _save(fig, figures_dir, "06_seller_profits")

        # 7. Show seller A's selection frequency against the symmetric benchmark.
        plot_seller_a_selection(rows, modes, figures_dir)

        # 9. Distribution consistency and scalar consistency have different units.
        mode_titles = {
            "grid": "Saved reputation values",
            "nearest": "Continuous: closest saved rule",
            "linear": "Continuous: blended saved rules",
        }
        fig, axs = _axes(rows=len(modes), cols=2, width=5.8, height=3.2)
        for row_index, mode in enumerate(modes):
            ax = axs[row_index, 0]
            dist = analysis["distributions"][mode]["oe"]
            x = np.asarray(dist["grid"])
            for seller, style in ((0, "-"), (1, ":")):
                residual = np.asarray(dist["cdf"])[seller] - np.cumsum(
                    np.asarray(dist["predicted_mu"])[seller])
                ax.plot(x, residual, ls=style, color=COLORS["oe"],
                        label=f"Seller {'AB'[seller]}")
                ax.fill_between(x,
                    np.asarray(dist["cdf_low"])[seller] - np.cumsum(np.asarray(dist["predicted_mu"])[seller]),
                    np.asarray(dist["cdf_high"])[seller] - np.cumsum(np.asarray(dist["predicted_mu"])[seller]),
                    color=COLORS["oe"], alpha=.10)
            ax.axhline(0, color="black", ls="--", lw=.8)
            ax.set(title=f"OE distribution: distribution gap\n{mode_titles[mode]}",
                   xlabel="Reputation", ylabel="Difference in cumulative probability")
            if row_index == 0:
                ax.legend()
            _style(ax)
            ax = axs[row_index, 1]
            running = analysis["running"][mode]["p0"]
            time = np.asarray(running["time"])
            for variable, color, label in (
                    ("ts_B", COLORS["p0"], "Average selection draw for seller B"),
                    ("posterior_B", "#0072B2", "Average expected quality of seller B")):
                ax.plot(time, running[variable], color=color, label=label)
                ax.fill_between(time, running[f"{variable}_low"],
                                running[f"{variable}_high"], color=color, alpha=.12)
            ax.axhline(float(profiles["p0"]["p0"]), color="black", ls="--",
                       label="Saved p₀")
            ax.set(title=f"p₀: simulated averages\n{mode_titles[mode]}",
                   xlabel="Periods included after initial adjustment", ylabel="Average over the included periods")
            if row_index == 0:
                ax.legend(loc="best", fontsize=7)
            _style(ax)
        fig.suptitle("Simulation vs. saved predictions")
        fig.supxlabel("After initial adjustment; shading: 95% confidence intervals across simulations.",
                      fontsize=9)
        _save(fig, figures_dir, "09_consistency_diagnostics")

        _short_behavior(samples, main_mode, figures_dir)
        if analysis.get("product_usage"):
            plot_product_usage(analysis["product_usage"], figures_dir,
                stationary_frequencies={key: analysis["predicted"][key]["product2_market"]
                                        for key in LABELS})


def plot_seller_a_selection(rows, modes, directory):
    """One panel of selection percentages; uncertainty remains across paths."""
    single_mode = len(modes) == 1
    with plt.rc_context({"font.size": 10, "axes.titlesize": 10,
                         "axes.labelsize": 10, "figure.titlesize": 14}):
        fig, axs = _axes(width=9.8, height=3.5 if single_mode else 4.9)
        ax = axs[0, 0]
        for k, mode in enumerate(modes):
            for index, (profile, offset) in enumerate((("oe", -.18), ("p0", .18))):
                row = _row(rows, mode, profile, "market_share_A")
                if row is None:
                    continue
                mean, low, high = 100 * np.asarray(_ci(row))
                position = index if single_mode else k + offset
                ax.barh(position, mean, height=.30, color=COLORS[profile], alpha=.85,
                        xerr=[[max(0, mean-low)], [max(0, high-mean)]],
                        error_kw={"ecolor": "#333333", "elinewidth": 1.1, "capsize": 4},
                        label=LABELS[profile] if k == 0 and not single_mode else None)
                ax.text(high + 1.1, position, f"{mean:.2f}%",
                        va="center", ha="left", color=COLORS[profile], fontsize=10,
                        fontweight="bold")
        ax.axvline(50, color="#555555", ls="--", lw=1.2,
                   label="50% expected by symmetry", zorder=4)
        if single_mode:
            ax.set_yticks([0, 1], [LABELS["oe"], LABELS["p0"]])
            ax.set_ylim(-.55, 1.55)
        else:
            ax.set_yticks(range(len(modes)), [MODE_TICKS[mode] for mode in modes])
            ax.set_ylim(-.55, len(modes)-.45)
            ax.set_ylabel("How reputations are simulated")
        ax.invert_yaxis()
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set(xlim=(0, 100), xlabel="Share of periods in which seller A is selected",
               title="After initial adjustment · means and 95% confidence intervals across simulations")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_axisbelow(True)
        _style(ax)
        ax.grid(axis="y", visible=False)
        fig.suptitle("Seller A selection rate")
        _method_note(fig, modes)
        _save(fig, directory, "07_customer_outcomes")


def plot_product_usage(curves, directory, *, stationary_frequencies,
                       early_periods=500, initial=(.5, .5)):
    """Compare simulated time paths with frequencies from the saved joint laws.

    The saved laws are stationary for the original discrete reputation process.
    In continuous-reputation panels, their frequencies are reference benchmarks.
    """
    modes = [mode for mode in MODES if mode in curves]
    if not modes:
        return
    stationary_frequencies = {key: float(stationary_frequencies[key]) for key in LABELS}
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in stationary_frequencies.values()):
        raise ValueError("Stationary product-use frequencies must lie in [0, 1].")
    counts = sorted({curve["n_replications"] for mode in modes
                     for curve in curves[mode].values()})
    windows = sorted({curve["smoothing_window"] for mode in modes
                      for curve in curves[mode].values()})
    count_text = ", ".join(map(str, counts))
    window_text = ", ".join(map(str, windows))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 10,
                         "legend.fontsize": 9, "figure.titlesize": 14}):
        fig, axs = _axes(rows=2 * len(modes), cols=1, width=9.8, height=3.35)
        for row, mode in enumerate(modes):
            raw_ax, smoothed_ax = axs[2 * row, 0], axs[2 * row + 1, 0]
            for key in ("oe", "p0"):
                curve = curves[mode][key]
                time = np.asarray(curve["time"])
                early = time < early_periods
                raw_ax.plot(time[early], 100 * np.asarray(curve["q2"])[early],
                            color=COLORS[key], lw=.75, marker=".", ms=2.5,
                            alpha=.65, label=f"{LABELS[key]} · simulated")
                smoothed_ax.plot(time, 100 * np.asarray(curve["smoothed"]),
                                 color=COLORS[key], lw=1.5, label=f"{LABELS[key]} · simulated")
                smoothed_ax.fill_between(time, 100 * np.asarray(curve["ci_low"]),
                                         100 * np.asarray(curve["ci_high"]),
                                         color=COLORS[key], alpha=.15, linewidth=0)
                reference = 100 * stationary_frequencies[key]
                description = "stationary" if mode == "grid" else "original stationary reference"
                for ax in (raw_ax, smoothed_ax):
                    ax.axhline(reference, color=COLORS[key], ls="--", lw=1.5, zorder=4,
                               label=f"{LABELS[key]} · {description}: {reference:.2f}%")
            raw_ax.set(title=f"{MODES[mode]}\nFrequency in each period, without smoothing",
                       ylabel=r"$q_2(t)$ (% of simulations)", ylim=(-2, 103))
            smoothed_ax.set(title=f"{MODES[mode]}\nMoving average of product 2 use",
                            ylabel="Smoothed product 2 use (%)", ylim=(48, 103))
            for ax in (raw_ax, smoothed_ax):
                ax.set_xlabel("Period since the initial state (t = 0)")
                ax.set_xlim(left=0)
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                _style(ax)
            if row == 0:
                raw_ax.legend(loc="lower right", fontsize=8)
                smoothed_ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(
            f"Product 2 use from the common initial state ({initial[0]}, {initial[1]})\n"
            f"Saved policies from earlier experiments · M = {count_text} independent simulations per policy")
        reference_note = "Dashed lines: product 2 frequency calculated from each policy's saved joint stationary distribution."
        if any(mode != "grid" for mode in modes):
            reference_note += "\nFor continuous reputations, these are references from the original reputation values."
        fig.supxlabel(
            f"Smoothed panels: trailing average of up to {window_text} periods, with shorter windows at the start. Initial periods are included.\n"
            "Shading: pointwise 95% confidence intervals across simulations, keeping periods within each simulation together.\n"
            + reference_note, fontsize=8.5)
        _save(fig, directory, "14_product2_usage_over_time")


def _short_behavior(samples, mode, directory):
    fig, axs = _axes(rows=3, cols=2, width=5.5, height=3.0)
    seller_colors = ("#0072B2", "#D55E00")
    for col, key in enumerate(("oe", "p0")):
        sample = samples[key]
        product = np.asarray(sample["product"])
        profit = np.asarray(sample["profit"])
        success = np.asarray(sample["success"])
        reps, periods = product.shape
        time = np.arange(1, periods + 1)
        for rep in range(min(3, reps)):
            product2 = product[rep] == 2
            axs[0, col].scatter(time[product2], np.full(np.sum(product2), rep + 1),
                                 marker="|", s=18, lw=.6, color=COLORS[key])
            for seller, style in ((0, "-"), (1, "--")):
                axs[1, col].plot(time, np.cumsum(profit[rep, :, seller]) / time,
                                 color=seller_colors[seller], ls=style, alpha=.85, lw=1.15,
                                 label=f"Seller {'AB'[seller]}" if rep == 0 else None)
            axs[2, col].plot(time, np.cumsum(success[rep]) / time,
                             color=COLORS[key], alpha=.65, lw=1)
        axs[0, col].set(title=LABELS[key], ylabel="Simulation using product 2",
                        yticks=np.arange(1, min(3, reps)+1), ylim=(.5, min(3, reps)+.5))
        axs[1, col].set_ylabel("Average profit since the start")
        axs[1, col].legend(loc="best")
        axs[2, col].set(ylabel="Success rate since the start",
                        xlabel="Period since the initial state")
        for ax in axs[:, col]:
            _style(ax)
    fig.suptitle("Product choices, profits and success in sample simulations\n"
                 f"{MODES[mode]}; both sellers start at reputation 0.5")
    _save(fig, directory, "11_short_run_behavior")


def write_findings(pair, analysis, data_dir):
    """Write the numerical comparison table and ten requested research answers.

    Wording separates measured Monte Carlo precision from discretization and
    policy-extension effects; no automatic claim of continuous-grid convergence
    or equilibrium-approximation superiority is inferred from performance.
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [r for r in analysis["summary"]
            if r.get("window", "baseline") == "baseline"]
    differences = [r for r in analysis["differences"]
                   if r.get("window", "baseline") == "baseline"]
    modes = [mode for mode in MODES if any(r["mode"] == mode for r in rows)]
    main = "nearest" if "nearest" in modes else modes[0]
    params = pair["params"]

    def get(profile, metric, mode=main):
        return _row(rows, mode, profile, metric)

    def estimate(profile, metric, mode=main, percent=False):
        return _estimate_text(get(profile, metric, mode), percent)

    def difference(metric, mode=main):
        return next((r for r in differences if r["mode"] == mode
                     and r["metric"] == metric), None)

    def contrast(metric, mode=main, percent=False):
        row = difference(metric, mode)
        if row is None:
            return "not available"
        mean, low, high = _ci(row)
        scale = 100 if percent else 1
        unit = " percentage points" if percent else ""
        direction = ("the interval excludes zero" if low > 0 or high < 0
                     else "the interval includes zero")
        return (f"OE distribution minus OE p0 = {scale*mean:.5g}{unit} "
                f"[95% paired CI {scale*low:.5g}, {scale*high:.5g}]; {direction}")

    # A human-readable metric table plus machine-readable uncertainty columns.
    table_metrics = [
        "product2_conditional_A", "product2_conditional_B",
        "product2_calendar_A", "product2_calendar_B", "product2_market",
        "profit_A", "profit_B", "industry_profit", "success_rate", "failure_rate",
        "expected_success_probability", "market_share_A", "market_share_B",
        "mean_reputation_A", "mean_reputation_B", "variance_reputation_A",
        "variance_reputation_B", "correlation", "mean_ts_A", "mean_ts_B",
        "mean_posterior_A", "mean_posterior_B", "wasserstein_A", "wasserstein_B",
        "cdf_max_error_A", "cdf_max_error_B", "disagreement_occupancy_A",
        "disagreement_occupancy_B", "disagreement_selected_A",
        "disagreement_selected_B", "disagreement_market",
        "discounted_profit_A", "discounted_profit_B", "discounted_industry_profit",
    ]
    table_rows = []
    for mode in modes:
        for metric in table_metrics:
            row = {"mode": mode, "metric": metric}
            for key in ("oe", "p0"):
                result = get(key, metric, mode)
                for statistic in ("mean", "se", "ci_low", "ci_high"):
                    row[f"{key}_{statistic}"] = result.get(statistic) if result else None
            paired = difference(metric, mode)
            for statistic in ("mean", "se", "ci_low", "ci_high"):
                row[f"oe_minus_p0_{statistic}"] = paired.get(statistic) if paired else None
            table_rows.append(row)
    with (directory / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    config = analysis.get("configuration", {})
    primary_reps = next((r.get("n_valid") for r in rows if r["mode"] == main), None)
    audit = analysis.get("audit", pair.get("audit", {}))
    lines = [
        "RESEARCH SUMMARY: OE distribution AND OE p0", "",
        "Both profiles are simulated in the same actual two-seller game. Customer selection",
        "uses two independent Beta posterior draws in every calendar period. The saved",
        "optimization-demand approximations never determine simulated customer selections.", "",
        f"Primitives: R={params.R:g}, c1={params.c1:g}, c2={params.c2:g}, "
        f"p1={params.p1:g}, p2={params.p2:g}, gamma={params.gamma:g}, rho={params.rho:g}.",
        f"Primary grid: {len(pair['grid'])} points per seller; "
        f"independent replications: {_value_text(primary_reps)}.",
        f"Primary behavioral interpretation: {MODES[main]}. "
        "Nearest-node extension preserves each stored product probability.",
        "Linear extension is reported separately: between unequal pure nodes it creates",
        "additional mixing, so it is an additional policy approximation.",
        "All intervals below are based on independent replication means, not independent periods.",
        "Paired model differences use common random numbers within each replication.", "",
        "FINITE-HORIZON INTERPRETATION: removing burn-in does not guarantee stationarity.",
        "These are estimates over the recorded post-burn-in windows. Slow mixing can leave",
        "initial-condition bias that replication-based confidence intervals do not measure.",
        "In particular, precise aggregate flows need not imply convergence of the reputation law or p₀.",
        "If present, transient_validation.txt reports a separate deterministic finite-horizon check.", "",
        "INPUT VALIDATION",
    ]
    for key in ("oe", "p0"):
        a = audit.get("profiles", {}).get(key, {})
        source = pair["profiles"][key].get("source", {})
        lines.extend([
            f"{LABELS[key]}: verified saved candidate; source {source.get('run', '')}/"
            f"{source.get('case', '')}; candidate index {source.get('candidate_index', 'saved')}; "
            f"{source.get('verified_candidate_count', '?')} verified candidate(s) in this case.",
            f"  Consistency residual: {_value_text(a.get('consistency_residual'), 8)} "
            f"({a.get('consistency_condition', 'see validation audit')}).",
            f"  Actual joint-kernel stationarity residual (maximum absolute coordinate): "
            f"{_value_text(a.get('stationary_residual_max'), 8)}.",
        ])
    lines += ["Verification concerns the saved grid equilibrium and its consistency conditions.",
              "This comparison neither establishes uniqueness nor proves optimality in the full game.", "",
              "1. HOW DIFFERENT ARE THE POLICIES?",
              "The uniformly weighted mean absolute difference in the probability of product 2 is "
              f"{100*analysis['policy_comparison']['uniform_mean_absolute_difference']:.4g} "
              "percentage points over [0,1], using nearest-node policy extension."]
    for key in ("oe", "p0"):
        policy = np.asarray(pair["profiles"][key]["policy"])
        mixed = int(np.sum((policy > 1e-9) & (policy < 1 - 1e-9)))
        lines.append(f"{LABELS[key]} has {mixed} mixed grid node(s); mixed choices remain probabilistic.")
    lines += ["No threshold shape is imposed. Figure 1 shows the saved policy values.",
              "The numerical tables report policy differences, including partial differences in mixing.", "",
              "2. DO THEY DIFFER IN FREQUENTLY VISITED STATES?"]
    for key in ("oe", "p0"):
        lines += [f"Under {LABELS[key]}:",
                  f"  Post-burn-in-state-weighted difference, seller A: "
                  f"{estimate(key, 'disagreement_occupancy_A', percent=True)} percentage-probability units.",
                  f"  Difference conditional on A being selected: "
                  f"{estimate(key, 'disagreement_selected_A', percent=True)}.",
                  f"  Difference at all market purchases: {estimate(key, 'disagreement_market', percent=True)}."]
    lines += ["Each weighted measure uses the recorded post-burn-in behavior of that profile. It measures",
              "differences in product probabilities, rather than flagging every unequal mixture as a full disagreement.", "",
              "3. WHICH PROFILE GENERATES MORE PRODUCT 2?",
              f"OE distribution market frequency: {estimate('oe', 'product2_market', percent=True)}.",
              f"OE p0 market frequency: {estimate('p0', 'product2_market', percent=True)}.",
              contrast("product2_market", percent=True) + ".",
              "The comparison table separately reports seller-conditional and calendar-period frequencies.", "",
              "4. WHICH PROFILE GENERATES HIGHER SELLER PROFITS?"]
    for key in ("oe", "p0"):
        lines += [f"{LABELS[key]}: seller A {estimate(key, 'profit_A')}; "
                  f"seller B {estimate(key, 'profit_B')}; industry {estimate(key, 'industry_profit')}."]
    lines += [contrast("industry_profit") + ".",
              "These are post-burn-in average profits per calendar period, including zero profit when a seller is not selected.",
              "Higher seller profits do not establish that a policy is a more accurate equilibrium approximation.",
              "Discounted profits are separate initial-state outcomes in Figure 6 and the comparison table.",
              f"They use the same (0.5, 0.5) initial state and the first {config.get('short_periods', 1000)} periods.", "",
              "5. WHICH PROFILE GENERATES HIGHER CUSTOMER SUCCESS?",
              f"OE distribution success rate: {estimate('oe', 'success_rate', percent=True)}.",
              f"OE p0 success rate: {estimate('p0', 'success_rate', percent=True)}.",
              contrast("success_rate", percent=True) + ".",
              "Customer success, failure and the selected product's success probability are reported independently",
              "of seller profit. No monetary customer-welfare interpretation is imposed.", "",
              "6. DO THE REPUTATION DISTRIBUTIONS DIFFER?"]
    for key in ("oe", "p0"):
        lines += [f"{LABELS[key]}: mean reputation A {estimate(key, 'mean_reputation_A')}; "
                  f"B {estimate(key, 'mean_reputation_B')}.",
                  f"  Variance A {estimate(key, 'variance_reputation_A')}; "
                  f"B {estimate(key, 'variance_reputation_B')}; correlation {estimate(key, 'correlation')}."]
    lines += ["Figures 3 and 4 show the marginal CDFs and joint laws. Continuous CDF and Wasserstein",
              "calculations use the actual unrounded states. Only the joint heatmaps use nearest-node display bins.", "",
              "7. DOES OE distribution REPRODUCE ITS ASSUMED STATIONARY MARGINAL?"]
    for mode in modes:
        lines += [f"{MODES[mode]}:",
                  f"  Seller A mean error: {estimate('oe', 'mean_error_A', mode)}; "
                  f"variance error: {estimate('oe', 'variance_error_A', mode)}.",
                  f"  Wasserstein distance: {estimate('oe', 'wasserstein_A', mode)}; "
                  f"maximum CDF distance: {estimate('oe', 'cdf_max_error_A', mode)}."]
    lines += ["Grid simulation is the direct check against the equilibrium's numerical joint kernel.",
              "Continuous-simulation discrepancies also contain transition-grid and policy-extension effects;",
              "a nonzero continuous-versus-discrete CDF distance is not by itself a failure of OE distribution consistency.",
              "Distances are nonnegative: their intervals quantify the mean finite-simulation distance,",
              "not a hypothesis test that the underlying distributions differ. Seller B diagnostics are in the table.", "",
              "8. DOES OE p0 REPRODUCE THE LONG-RUN AVERAGE TS DRAW?",
              f"Saved equilibrium p₀ = {pair['profiles']['p0']['p0']:.10g}."]
    for mode in modes:
        lines += [f"{MODES[mode]}: mean TS draw of B {estimate('p0', 'mean_ts_B', mode)}; "
                  f"mean posterior expectation of B {estimate('p0', 'mean_posterior_B', mode)}.",
                  f"  TS-minus-p₀ error {estimate('p0', 'p0_ts_error', mode)}; "
                  f"posterior-mean-minus-p₀ error {estimate('p0', 'p0_mean_error', mode)}."]
    lines += ["TS draws contain additional posterior sampling noise. Posterior means remove that noise",
              "while retaining serially correlated reputation fluctuations. Figure 9 starts its running averages",
              "after burn-in and shows replication-based uncertainty; it does not reinterpret consistency as optimality.", "",
              "9. ARE THE DIFFERENCES LARGE RELATIVE TO MONTE CARLO UNCERTAINTY?",
              contrast("product2_market", percent=True) + ".",
              contrast("industry_profit") + ".",
              contrast("success_rate", percent=True) + ".",
              "The signs and intervals indicate Monte Carlo resolution. Economic importance should be judged",
              "from the reported effect sizes, in percentage points or profit units, rather than statistical significance alone.", "",
              "10. ARE THE CONCLUSIONS STABLE UNDER LONGER RUNS AND GRID CHANGES?",
    ]
    checks = [r for r in analysis["differences"] if r["mode"] == main]
    for metric in ("product2_market", "industry_profit", "success_rate"):
        matching = [r for r in checks if r["metric"] == metric]
        for row in matching:
            lines.append(f"{MODES[main]}, {row.get('window', 'baseline')}, {metric}: "
                         f"paired difference {_estimate_text(row)}.")
    lines += ["The longer and longer-burn windows share simulated paths. Their agreement is a stability",
              "diagnostic, not an independent replication of the experiment."]
    flow_metrics = ("product2_market", "industry_profit", "success_rate")
    same_sign = all(len({np.sign(r["mean"]) for r in checks if r["metric"] == metric
                        and r.get("mean") is not None}) == 1 for metric in flow_metrics)
    if same_sign:
        lines += ["The signs of the three flow-outcome differences agree across these recorded windows;",
                  "the estimates above show their magnitudes and Monte Carlo intervals."]
    lines += ["Stable flow-outcome differences do not establish that reputation distributions or p₀ have converged."]
    sensitivity = [r for r in analysis.get("sensitivity", [])
                   if r.get("window", "baseline") == "baseline"]
    grids = sorted(set(r["grid"] for r in sensitivity))
    if grids:
        lines.append("Available saved grids evaluated: " + ", ".join(map(str, grids)) + " points per seller.")
        for mode in [m for m in MODES if any(r["mode"] == m for r in sensitivity)]:
            for key in ("oe", "p0"):
                values = sorted((r for r in sensitivity if r["mode"] == mode
                                 and r["profile"] == key and r["metric"] == "product2_market"),
                                key=lambda r: r["grid"])
                if values:
                    lines.append(f"  {LABELS[key]}, {MODES[mode]}, product 2 market frequency: "
                        + "; ".join(f"n={r['grid']}: {100*r['mean']:.4g}%" for r in values) + ".")
                    if len(values) > 1:
                        spread = 100 * (max(r["mean"] for r in values) - min(r["mean"] for r in values))
                        lines.append(f"    Observed range across saved grids: {spread:.4g} percentage points.")
        lines += ["The sensitivity.csv table separates the saved grid sizes and simulation rules. Grid changes also change",
                  "the saved equilibrium policy and may select different verified equilibrium candidates;",
                  "those changes are not pure Monte Carlo noise or a controlled test on a fixed policy.",
                  "Agreement about which profile uses more product 2 does not imply stable absolute usage levels.",
                  f"No equilibrium beyond the available {max(grids)}-point grid was solved, so convergence at finer",
                  "resolutions is not established."]
    else:
        lines.append("No cross-grid sensitivity results are available in this run; grid refinement is untested.")
    lines += ["Nearest and linear extensions are kept distinct in every outcome figure. Disagreement between",
              "them indicates policy-extension sensitivity, especially near product-choice boundaries.", "",
              "INTERPRETATION",
              "Replacing the competitor's full marginal by an endogenous scalar changes the optimization problem.",
              "The comparison measures resulting differences in policies, post-burn-in behavior and outcomes.",
              "It does not identify an approximation error against a known exact equilibrium. Numerical kernel",
              "residuals, Monte Carlo uncertainty and discretization/policy-extension sensitivity are separate diagnostics.",
              "The optional asymmetric cross-policy experiment is not part of this symmetric-profile comparison.", "",
              "FILES",
              "comparison_table.csv: both models, standard errors, 95% intervals, and paired differences by mode.",
              "The runner also saves raw path arrays, per-replication statistics, configurations, seeds and input audits.",
              "Figures 1–10 address the requested comparisons; Figures 11–13 cover short-run behavior,",
              "profit dispersion across replications and available-grid sensitivity."]
    (directory / "research_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

