"""Verified monopoly/competition comparisons under the same outside benchmark.

This module only displays saved results.  It does not solve policies, infer
verification from residuals, or replace actual-environment discounted values
with the seller's perceived oblivious value.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("monopoly", "Monopoly", "#0072B2", "-", "o"),
    ("competition", "Competition (per seller)", "#D55E00", "--", "s"),
)


def _records(data):
    """Copy records while accepting either a DataFrame or a list of mappings."""
    if hasattr(data, "to_dict"):
        return [dict(row) for row in data.to_dict(orient="records")]
    return [dict(row) for row in data]


def _verified(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if value is not None and not np.isnan(value) else False


def _matches(row, p0, n):
    return int(row["n"]) == n and np.isclose(
        float(row["p0"]), p0, rtol=0, atol=1e-12
    )


def grid_sensitivity_selection(summary, *, p0, n,
                               probability_tolerance=.05, relative_tolerance=.05):
    """Screen both models using the two finest available verified grids.

    This is a presentation sensitivity criterion, not a convergence certificate.
    Probability metrics use absolute changes; earnings use symmetric relative
    changes so even tiny positive stationary profits are compared faithfully.
    """
    records = [row for row in _records(summary)
               if np.isclose(float(row['p0']), p0, rtol=0, atol=1e-12)
               and int(row['n']) <= n and _verified(row.get('verified', False))
               and row['model'] in {model for model, *_ in MODELS}]
    probability_metrics = ('product2_conditional', 'product2_calendar', 'sale_rate')
    relative_metrics = ('profit_calendar', 'value_initial')
    included, excluded, comparisons = [], {}, []
    rhos = sorted({float(row['rho']) for row in records if int(row['n']) == n})
    for rho in rhos:
        reasons = []
        for model, *_ in MODELS:
            by_grid = {}
            for row in records:
                if float(row['rho']) == rho and row['model'] == model:
                    size = int(row['n'])
                    if size in by_grid:
                        raise ValueError(f'Duplicate grid result: rho={rho}, model={model}, n={size}')
                    by_grid[size] = row
            coarse_grids = [size for size in by_grid if size < n]
            if n not in by_grid or not coarse_grids:
                reasons.append(dict(model=model, reason='Missing verified coarse/fine comparison'))
                continue
            coarse_n = max(coarse_grids)
            coarse, fine = by_grid[coarse_n], by_grid[n]
            changes = dict(rho=rho, model=model, coarse_n=coarse_n, fine_n=n,
                           absolute_probability_changes={}, relative_changes={})
            for metric in probability_metrics + relative_metrics:
                first, second = float(coarse[metric]), float(fine[metric])
                if not np.isfinite([first, second]).all():
                    reasons.append(dict(model=model, metric=metric, reason='Nonfinite metric'))
                    continue
                difference = abs(second-first)
                if metric in probability_metrics:
                    change, tolerance = difference, probability_tolerance
                    changes['absolute_probability_changes'][metric] = change
                else:
                    denominator = max(abs(first), abs(second))
                    change = difference/denominator if denominator else 0.
                    tolerance = relative_tolerance
                    changes['relative_changes'][metric] = change
                if change > tolerance + 1e-12:
                    reasons.append(dict(model=model, metric=metric, change=change, limit=tolerance))
            comparisons.append(changes)
        if reasons:
            excluded[str(rho)] = reasons
        else:
            included.append(rho)
    return dict(included_rhos=included, excluded_rhos=excluded,
                probability_change_limit=probability_tolerance,
                relative_earnings_change_limit=relative_tolerance,
                probability_metrics=list(probability_metrics), relative_metrics=list(relative_metrics),
                comparisons=comparisons,
                interpretation='Presentation filter on the two finest available verified grids; not a convergence proof')


def _style(ax):
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, directory, stem):
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _legend(fig, axes):
    handles = {}
    for ax in np.asarray(axes).ravel():
        artists, labels = ax.get_legend_handles_labels()
        for label, artist in zip(labels, artists):
            handles.setdefault(label, artist)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="outside lower center",
                   ncol=min(3, len(handles)), frameon=False)


def _rho_axis(ax, rhos):
    ax.set_xticks(np.arange(len(rhos)), [f"{rho:g}" for rho in rhos])
    ax.set_xlabel("Persistence ρ (equally spaced cases)")
    if len(rhos) > 1:
        ax.set_xlim(-.2, len(rhos) - .8)


def _calendar_scale(ax, values, *, force=False):
    """Keep rare calendar flows visible without discarding exact zeros."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    if (not len(positive) or np.any(values < 0)
            or (not force and np.log10(positive.max()) - np.log10(positive.min()) < 4)):
        return
    if np.any(values == 0):
        threshold = max(np.finfo(float).tiny, float(positive.min()) / 10)
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_ylabel(ax.get_ylabel() + "\n(symlog; linear near zero)")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(ax.get_ylabel() + "\n(log scale)")


def _calendar_groups(lookup, rhos, metric):
    """Show the ordinary-scale calendar panel; omit the rare-flow panel."""
    values = np.asarray([lookup.get((rho, model), {}).get(metric, np.nan)
                         for rho in rhos for model, *_ in MODELS], dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if (not len(positive)
            or np.log10(positive.max()) - np.log10(positive.min()) < 4):
        return [(rhos, "auto", "")]
    threshold = positive.max() * 1e-4
    rare = []
    for rho in rhos:
        pair = np.asarray([lookup.get((rho, model), {}).get(metric, np.nan)
                           for model, *_ in MODELS], dtype=float)
        if np.isfinite(pair).all() and pair.max() < threshold:
            rare.append(rho)
    common = [rho for rho in rhos if rho not in rare]
    if len(common) < 2 or not rare:
        return [(rhos, "auto", "")]
    return [(common, "linear", "")]


def _outcomes(rows, rhos, directory, *, p0, n, metrics, stem, title):
    lookup = {(float(row["rho"]), row["model"]): row for row in rows}
    panels = []
    for metric, scale, ylabel, subtitle in metrics:
        groups = (_calendar_groups(lookup, rhos, metric)
                  if metric in {"product2_calendar", "profit_calendar"}
                  else [(rhos, "linear", "")])
        for selected_rhos, axis_scale, scope in groups:
            if scope:
                short_title = ("Product 2 use" if metric == "product2_calendar"
                               else "Stationary profit")
                panel_title = f"{short_title} | {scope}"
            else:
                panel_title = subtitle
            panels.append((metric, scale, ylabel, panel_title, selected_rhos, axis_scale))
    width = 15 if len(panels) == 3 else 11
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.8),
                             squeeze=False, layout="constrained")
    for ax, (metric, scale, ylabel, subtitle, selected_rhos, axis_scale) in zip(axes.ravel(), panels):
        positions = np.arange(len(selected_rhos))
        displayed = []
        for model, label, color, linestyle, marker in MODELS:
            values = [lookup.get((rho, model), {}).get(metric, np.nan) for rho in selected_rhos]
            if not np.isfinite(np.asarray(values, dtype=float)).any():
                continue
            displayed.extend(scale * np.asarray(values, dtype=float))
            ax.plot(positions, scale * np.asarray(values, dtype=float),
                    color=color, ls=linestyle, marker=marker, ms=5, lw=1.8,
                    markerfacecolor="white" if model == "competition" else color,
                    label=label)
        ax.set(ylabel=ylabel, title=subtitle)
        if axis_scale in {"auto", "log"}:
            _calendar_scale(ax, displayed, force=axis_scale == "log")
        if axis_scale == "log":
            for position, rho in enumerate(selected_rhos):
                if any(lookup.get((rho, model), {}).get(metric, np.nan) == 0
                       for model, *_ in MODELS):
                    ax.annotate("0", (position, 0), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=8)
        _rho_axis(ax, selected_rhos)
        _style(ax)
    _legend(fig, axes)
    fig.suptitle(f"{title}\nFixed outside benchmark p₀ = {p0:g} | n = {n} | verified grid solutions",
                 fontsize=12)
    return _save(fig, directory, stem)


def make_figures(summary, policies, figures_dir, *, p0, n):
    """Write the three comparisons in PNG/PDF and return their six ``Path``s.

    Summary must include the comparison grids. Only rho values passing the
    sensitivity screen for both models appear, using verified records at p0,n.
    Their corresponding policy records are joined on ``rho,model``.
    Missing/unverified counterparts produce gaps; they are never interpolated
    across rho.  Lines in policy panels merely connect saved reputation nodes.
    Inputs are copied and are not changed.
    """
    p0, n = float(p0), int(n)
    selection = grid_sensitivity_selection(summary, p0=p0, n=n)
    rows = [row for row in _records(summary)
            if _matches(row, p0, n) and _verified(row.get("verified", False))
            and row["model"] in {"monopoly", "competition"}
            and float(row['rho']) in selection['included_rhos']]
    if not rows:
        raise ValueError('No rho passes the verified coarse/fine grid sensitivity screen')
    keys = [(float(row["rho"]), row["model"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Summary has duplicate verified rho/model records for this p0,n.")
    accepted = set(keys)
    policy_rows = [row for row in _records(policies)
                   if _matches(row, p0, n)
                   and (float(row["rho"]), row["model"]) in accepted]
    rhos = sorted({float(row["rho"]) for row in rows})
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'grid_sensitivity_selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    paths = []
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 9}):
        panel_rows = (len(rhos) + 1) // 2
        fig, axes = plt.subplots(panel_rows, 2, figsize=(11, 2.8 * panel_rows),
                                 sharex=True, sharey=True, squeeze=False,
                                 layout="constrained")
        for ax, rho in zip(axes.ravel(), rhos):
            for model, label, color, linestyle, _marker in MODELS:
                selected = sorted((row for row in policy_rows
                                   if float(row["rho"]) == rho and row["model"] == model),
                                  key=lambda row: float(row["reputation"]))
                if not selected:
                    continue
                reputation = np.asarray([row["reputation"] for row in selected], dtype=float)
                alpha = np.asarray([row["product2_probability"] for row in selected], dtype=float)
                ax.plot(reputation, alpha, color=color, ls=linestyle,
                        lw=1.8, label=label)
                mixed = (alpha > 1e-9) & (alpha < 1 - 1e-9)
                if mixed.any():
                    ax.scatter(reputation[mixed], alpha[mixed], marker="D", s=28,
                               facecolors="white", edgecolors=color, linewidths=1.2,
                               zorder=5, label="Mixed node")
            ax.set(title=f"ρ = {rho:g}", xlabel="Own reputation x",
                   ylabel="Pr(product 2 | own sale, x)",
                   xlim=(0, 1), ylim=(-.04, 1.04))
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_yticks([0, .5, 1])
            _style(ax)
        for ax in axes.ravel()[len(rhos):]:
            ax.set_visible(False)
        _legend(fig, axes)
        fig.suptitle(f"Product policies | fixed outside benchmark p₀ = {p0:g} | n = {n}\n"
                     "Verified grid solutions; diamonds mark mixtures; lines connect grid nodes.",
                     fontsize=12)
        paths += _save(fig, directory, "01_policy_comparison")
        paths += _outcomes(rows, rhos, directory, p0=p0, n=n,
            metrics=[
                ("product2_conditional", 100, "Product 2 (% of own sales)",
                 "Product mix when the seller sells"),
                ("product2_calendar", 100, "Product 2 sales per 100 calendar periods / seller",
                 "Product 2 use per seller, including waiting"),
            ], stem="02_product2_usage_comparison", title="Stationary product 2 use")
        paths += _outcomes(rows, rhos, directory, p0=p0, n=n,
            metrics=[
                ("profit_calendar", 1, "Profit / seller / calendar period",
                 "Stationary profit, including zero when not selected"),
                ("value_initial", 1, "Expected total discounted profit / seller",
                 "Actual value: initial reputations 0.5; γ = 0.98"),
            ], stem="03_profit_comparison", title="Profits in the actual environments")
    return paths
