"""Isolated high-p0 monopoly comparison using the paper's calendar simulator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/high-p0-monte-carlo-mpl")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives
from forgetting.Code.simulation import simulate_calendar_paths


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def tag(value):
    return f"{value:g}".replace(".", "p")


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__)] + [
        root / "forgetting/Code" / name
        for name in ("model.py", "layers.py", "simulation.py")
    ]
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def path_statistics(paths, periods):
    sales = paths.a_chosen_count.to_numpy(dtype=float)
    high = paths.product2_count.to_numpy(dtype=float)
    assert np.all(high <= sales) and np.all(sales <= periods)
    rate = sales.mean() / periods
    calendar = high.mean() / periods
    mix = high.sum() / sales.sum() if sales.sum() else np.nan
    ratio_influence = (high - mix * sales) / sales.mean() if sales.mean() else np.full(len(paths), np.nan)
    output = {}
    for name, estimate, influence in [
        ("sale_rate", rate, sales / periods - rate),
        ("product2_calendar", calendar, high / periods - calendar),
        ("product2_conditional", mix, ratio_influence),
    ]:
        se = float(np.std(influence, ddof=1) / np.sqrt(len(paths)))
        output.update({
            name: float(estimate),
            name + "_se": se,
            name + "_ci_low": max(0., float(estimate - 1.96 * se)),
            name + "_ci_high": min(1., float(estimate + 1.96 * se)),
        })
    assert abs(calendar - rate * mix) < 1e-12
    return output


def save_figure(fig, path, *, pdf=False):
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render(output, config):
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .2, "legend.frameon": False,
    })
    figures = output / "figures"
    for rho in [config["primary_rho"], config["reference_rho"]]:
        frames = []
        for p0 in config["p0_values"]:
            directory = output / "data" / f"rho_{tag(rho)}_p0_{tag(p0)}"
            frame = pd.read_csv(directory / "by_period.csv")
            frame = frame[frame.period < config["paper_plot_periods"]]
            frames.append(frame)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
        for p0, color, frame in zip(config["p0_values"], COLORS, frames, strict=True):
            for ax, metric in zip(axes, ["a_market_share", "product2_unconditional_rate"], strict=True):
                ax.plot(frame.period + 1, frame[metric], color=color, lw=1.2, label=rf"$p_0={p0:g}$")
        for ax in axes:
            ax.set(xlabel="Calendar period", xlim=(1, config["paper_plot_periods"]), ylim=(0, .36))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set(title="(a) Seller demand", ylabel="Own sales / calendar period")
        axes[1].set(title="(b) Product 2 use", ylabel="Product 2 sales / calendar period")
        axes[1].legend(loc="upper right")
        fig.suptitle(
            rf"Monopoly Monte Carlo | $\rho={rho:g}$ | {config['calendar_paths']:,} paths"
            "\nInitial Beta(1,1); transient observation policies; no evidence update without a sale",
            fontsize=11,
        )
        save_figure(fig, figures / f"01_paper_style_rho_{tag(rho)}", pdf=True)

    rho = config["primary_rho"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    interval_columns = {
        "a_market_share": ("a_market_share_ci_lower", "a_market_share_ci_upper"),
        "product2_unconditional_rate": ("product2_unconditional_ci_lower", "product2_unconditional_ci_upper"),
        "product2_given_a_rate": ("product2_given_a_ci_lower", "product2_given_a_ci_upper"),
    }
    for p0, color in zip(config["p0_values"], COLORS, strict=True):
        frame = pd.read_csv(output / "data" / f"rho_{tag(rho)}_p0_{tag(p0)}" / "binned.csv")
        for ax, metric in zip(axes, ["a_market_share", "product2_unconditional_rate", "product2_given_a_rate"], strict=True):
            ax.plot(frame.bin_center + 1, frame[metric], color=color, lw=1.6, label=rf"$p_0={p0:g}$")
            lower, upper = interval_columns[metric]
            ax.fill_between(frame.bin_center + 1, frame[lower], frame[upper],
                            color=color, alpha=.12, linewidth=0)
    for ax, title in zip(axes, ["Own sales / period", "Product 2 / period", "Product 2 / own sales"], strict=True):
        ax.set(title=title, xlabel="Calendar period", ylim=(0, 1 if ax is axes[2] else .36))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper right")
    fig.suptitle(
        rf"Finite-horizon monopoly | $\rho={rho:g}$ | {config['calendar_paths']:,} paths"
        f"\n{config['bin_width']}-period blocks; shaded bands: nominal 95% path-cluster intervals; no stationarity assumption",
        fontsize=11,
    )
    save_figure(fig, figures / "02_extended_calendar_comparison")


def run(output, config):
    rows = []
    manifest = dict(configuration=config, source_sha256=source_hashes(), completed=[])
    for rho, periods in [
        (config["primary_rho"], config["primary_periods"]),
        (config["reference_rho"], config["reference_periods"]),
    ]:
        paired = {}
        for p0 in config["p0_values"]:
            directory = output / "data" / f"rho_{tag(rho)}_p0_{tag(p0)}"
            directory.mkdir(parents=True, exist_ok=False)
            start = time.monotonic()
            print(f"START rho={rho:g} p0={p0:g} T={periods}", flush=True)
            primitives = Primitives(p0=p0, **config["parameters"])
            period, binned, paths = simulate_calendar_paths(
                primitives, rho, n_paths=config["calendar_paths"], seed=config["seed"],
                max_periods=periods, bin_width=config["bin_width"],
                policy_nodes=config["policy_nodes"], terminal_gap=config["terminal_gap"],
                solver_tolerance=config["solver_tolerance"], action_tolerance=config["action_tolerance"],
            )
            for name, frame in [("by_period", period), ("binned", binned), ("paths", paths)]:
                frame.to_csv(directory / f"{name}.csv", index=False)
            stats = path_statistics(paths, periods)
            for stop in [config["paper_plot_periods"], periods]:
                window = period[period.period < stop]
                count = float(window.a_chosen_count.sum())
                count2 = float(window.product2_chosen_count.sum())
                rows.append(dict(
                    rho=rho, p0=p0, window_start=1, window_stop=stop,
                    n_paths=config["calendar_paths"], sale_rate=count / (len(window) * len(paths)),
                    product2_calendar=count2 / (len(window) * len(paths)),
                    product2_conditional=count2 / count if count else np.nan,
                ))
                if stop == periods:
                    rows[-1].update(stats)
            paired[p0] = paths
            elapsed = time.monotonic() - start
            print(f"DONE rho={rho:g} p0={p0:g} seconds={elapsed:.1f} P2/sale={stats['product2_conditional']:.6f} P2/period={stats['product2_calendar']:.6f}", flush=True)
            manifest["completed"].append(dict(rho=rho, p0=p0, periods=periods, seconds=elapsed, statistics=stats))
            write_json(output / "calendar_manifest.json", manifest)
            pd.DataFrame(rows).drop_duplicates(["rho", "p0", "window_stop"], keep="last").to_csv(output / "data/calendar_summary.csv", index=False)

        baseline = paired[config["p0_values"][0]]
        differences = []
        for p0, paths in paired.items():
            a0 = baseline.a_chosen_count.to_numpy(dtype=float)
            b0 = baseline.product2_count.to_numpy(dtype=float)
            a1 = paths.a_chosen_count.to_numpy(dtype=float)
            b1 = paths.product2_count.to_numpy(dtype=float)
            r0, r1 = b0.sum() / a0.sum(), b1.sum() / a1.sum()
            for metric, values, estimate in [
                ("sale_rate", (a1-a0)/periods, (a1-a0).mean()/periods),
                ("product2_calendar", (b1-b0)/periods, (b1-b0).mean()/periods),
                ("product2_conditional", (b1-r1*a1)/a1.mean()-(b0-r0*a0)/a0.mean(), r1-r0),
            ]:
                se = float(values.std(ddof=1)/np.sqrt(len(values)))
                differences.append(dict(rho=rho, p0=p0, baseline_p0=config["p0_values"][0],
                                        metric=metric, difference=float(estimate), se=se,
                                        ci_low=float(estimate-1.96*se), ci_high=float(estimate+1.96*se)))
        pd.DataFrame(differences).to_csv(output / f"data/paired_differences_rho_{tag(rho)}.csv", index=False)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "calendar_manifest.json", manifest)
    render(output, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.output / "configuration.json").read_text())
    if args.render_only:
        render(args.output, config)
    else:
        run(args.output, config)


if __name__ == "__main__":
    main()
