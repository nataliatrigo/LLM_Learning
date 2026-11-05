"""Figures and tables for expected finite-window OE p0 consistency.

Plotting never solves policies or infers convergence from a finite window.
The baseline is the original stationary-closure OE p0 experiment.
"""

import csv
from functools import lru_cache
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


NEW = "#2466A8"
ORIGINAL = "#D87522"
NEW_LABEL = "OE p0: temporal closure"
ORIGINAL_LABEL = "OE p0: original stationary closure"


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=160,
                    bbox_inches="tight")
    plt.close(fig)


def _style(ax):
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def _panels(cases, height=2.7):
    rows = max(1, (len(cases) + 1) // 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, height * rows),
                             squeeze=False, layout="constrained")
    for ax in axes.ravel()[len(cases):]:
        ax.set_visible(False)
    return fig, axes.ravel()


def _grid(result, n):
    return np.asarray(result.get("grid", np.linspace(0, 1, n)), dtype=float)


def _law(result, key, n):
    value = result.get(key)
    if value is None:
        return None
    law = np.asarray(value, dtype=float).reshape(n, n)
    if not np.all(np.isfinite(law)) or law.min() < -1e-12:
        raise ValueError(f"Invalid {key} probability law.")
    return law


def _mixed(result):
    alpha = np.asarray(result["policy"], dtype=float)
    return (alpha > 1e-9) & (alpha < 1 - 1e-9)


def _is_verified(result):
    return bool(result.get("candidate_verified", result.get("verified", False)))


def _label(case):
    result = case["selected"]
    suffix = "" if _is_verified(result) else " | unverified diagnostic"
    return f"ρ = {case['rho']:g} | n = {case['n']}{suffix}"


def _window(config):
    nested = config.get("temporal", config.get("window", {}))
    if not isinstance(nested, dict):
        nested = {}
    burn = config.get("burn_in", config.get("burn", nested.get("burn_in", 10000)))
    periods = config.get("periods", config.get("horizon", nested.get("periods", 100000)))
    initial = config.get("initial_state", config.get("initial", nested.get("initial_state", (.5, .5))))
    return int(burn), int(periods), initial


def _window_note(cases):
    burn, periods, initial = _window(cases[0].get("configuration", {}))
    return (f"Expected calendar-time window: {periods:,} periods after {burn:,} burn-in periods; "
            f"initial reputations ({initial[0]:g}, {initial[1]:g}).")


def make_figures(cases, directory):
    """Export PNG/PDF figures; use the same model colors in every comparison."""
    cases = sorted((c for c in cases if c.get("selected") is not None),
                   key=lambda c: (c["rho"], c["n"]))
    if not cases:
        return
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8}):
        _plot_p0(cases, directory)
        _plot_policies(cases, directory)
        _plot_cdfs(cases, directory)
        _plot_maps(cases, directory)
        _plot_joint(cases, directory)
        _plot_advantages(cases, directory)
        _plot_outcomes(cases, directory)


def _plot_p0(cases, directory):
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    positions = np.arange(len(cases))
    old = [c.get("reference", {}).get("p0", np.nan) for c in cases]
    new = [c["selected"]["p0"] for c in cases]
    ax.plot(positions, old, "s--", color=ORIGINAL, lw=1.7, label=ORIGINAL_LABEL)
    ax.plot(positions, new, "o-", color=NEW, lw=1.7, label=NEW_LABEL)
    for x, case in zip(positions, cases):
        if not _is_verified(case["selected"]):
            ax.scatter(x, case["selected"]["p0"], marker="x", color="black", zorder=5)
    ax.set_xticks(positions, [f"{c['rho']:g}" for c in cases])
    ax.set(xlabel="Persistence ρ (equally spaced cases)", ylabel="Fixed benchmark p₀",
           title="Fixed benchmark under the two consistency definitions")
    ax.legend(loc="best")
    _style(ax)
    fig.supxlabel(_window_note(cases), fontsize=8)
    _save(fig, directory, "01_p0_comparison")


def _plot_policies(cases, directory):
    fig, axes = _panels(cases)
    for case, ax in zip(cases, axes):
        new, old = case["selected"], case.get("reference", {})
        if "policy" in old:
            ax.plot(_grid(old, case["n"]), old["policy"], "--", color=ORIGINAL,
                    lw=1.6, label=ORIGINAL_LABEL, zorder=3)
            mask = _mixed(old)
            ax.scatter(_grid(old, case["n"])[mask], np.asarray(old["policy"])[mask],
                       marker="D", s=23, facecolors="none", edgecolors=ORIGINAL, zorder=5)
        grid = _grid(new, case["n"])
        ax.plot(grid, new["policy"], color=NEW, lw=1.6, label=NEW_LABEL)
        mask = _mixed(new)
        ax.scatter(grid[mask], np.asarray(new["policy"])[mask],
                   marker="D", s=23, color=NEW, zorder=4)
        ax.set(title=_label(case), xlabel="Own reputation",
               ylabel="Probability of product 2", xlim=(0, 1), ylim=(-.04, 1.04))
        ax.set_yticks([0, .5, 1])
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Optimal product policies at the saved reputation nodes\n"
                 "Diamonds: mixed nodes. Lines connect grid values for display.", fontsize=12)
    _save(fig, directory, "02_policy_comparison")


def _plot_cdfs(cases, directory):
    fig, axes = _panels(cases, height=2.8)
    for case, ax in zip(cases, axes):
        n = case["n"]
        new, old = case["selected"], case.get("reference", {})
        old_window = case.get("reference_temporal", {})
        for result, key, color, style, label, width in [
                (old_window, "nu", ORIGINAL, "--", "Original policy: same finite window", 1.8),
                (new, "temporal_nu", NEW, "-", "New policy: same finite window", 1.8),
                (old, "nu", ORIGINAL, ":", "Original policy: stationary law", 1.1),
                (new, "nu", NEW, ":", "New policy: stationary diagnostic", 1.1)]:
            law = _law(result, key, n)
            if law is not None:
                ax.step(_grid(new, n), np.cumsum(law.sum(axis=0)), where="post",
                        color=color, ls=style, lw=width, label=label,
                        zorder=5 if style == "--" else (4 if style == "-" else 2))
        ax.set(title=_label(case), xlabel="Rival reputation", ylabel="Marginal CDF",
               xlim=(0, 1), ylim=(-.02, 1.02))
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Rival reputation: original and new policies\n" + _window_note(cases), fontsize=11)
    _save(fig, directory, "03_reputation_cdf_comparison")


def _map_points(ax, rows, color, label):
    valid = [r for r in rows if r.get("p0") is not None and r.get("F") is not None
             and np.isfinite(r["p0"]) and np.isfinite(r["F"])]
    if valid:
        ax.scatter([r["p0"] for r in valid], [r["F"] for r in valid],
                   edgecolors=color, facecolors="none" if color == ORIGINAL else color,
                   s=18 if color == ORIGINAL else 7, linewidths=.7,
                   alpha=.6, label=label, rasterized=True)


def _plot_maps(cases, directory):
    fig, axes = _panels(cases, height=2.9)
    for case, ax in zip(cases, axes):
        ax.plot([0, 1], [0, 1], "-", color=".55", lw=.8, label="Consistency: F = p₀")
        _map_points(ax, case.get("reference_map_rows", []), ORIGINAL,
                    "Original stationary map: evaluated policies")
        _map_points(ax, case.get("map_rows", []), NEW,
                    "New temporal map: evaluated policies")
        for candidates, color, marker, label in [
                (case.get("reference_candidates", [case.get("reference", {})]),
                 ORIGINAL, "s", "Original verified candidates"),
                (case.get("candidates", []), NEW, "o", "New verified candidates")]:
            valid = [c for c in candidates if "p0" in c and _is_verified(c)]
            if valid:
                ax.scatter([c["p0"] for c in valid], [c.get("F", c["p0"]) for c in valid],
                           marker=marker, s=34, facecolors="none", edgecolors=color,
                           linewidths=1.5, label=label, zorder=5)
        new, old = case["selected"], case.get("reference", {})
        ax.scatter(new["p0"], new["F"], marker="*", s=85, color=NEW, zorder=6)
        if "p0" in old:
            ax.scatter(old["p0"], old.get("F", old["p0"]), marker="*", s=145,
                       facecolors="none", edgecolors=ORIGINAL, linewidths=1.2, zorder=6)
        ax.set(title=_label(case), xlabel="Fixed benchmark p₀",
               ylabel="Generated rival TS mean F", xlim=(0, 1), ylim=(0, 1))
        _style(ax)
    handles = {}
    for ax in axes[:len(cases)]:
        hh, ll = ax.get_legend_handles_labels()
        handles.update(zip(ll, hh))
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2)
    fig.suptitle("Consistency maps under the two definitions | stars: selected candidates\n"
                 "Points show search evaluations; ties can admit multiple optimal mixtures.", fontsize=11)
    _save(fig, directory, "04_consistency_map_comparison")


def _plot_joint(cases, directory):
    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 2.8 * len(cases)),
                             squeeze=False, layout="constrained")
    laws = []
    for case in cases:
        n = case["n"]
        laws.append([
            _law(case.get("reference", {}), "nu", n),
            _law(case.get("reference_temporal", {}), "nu", n),
            _law(case["selected"], "temporal_nu", n),
            _law(case["selected"], "nu", n)])
    maximum = max((float(law.max()) for row in laws for law in row if law is not None),
                  default=1.)
    floor = min(1e-8, maximum / 10)
    norm = LogNorm(vmin=floor, vmax=maximum)
    artist = None
    titles = ["Original policy\nStationary law", "Original policy\nFinite-window law",
              "New policy\nFinite-window law", "New policy\nStationary diagnostic"]
    for row, (case, row_laws) in enumerate(zip(cases, laws)):
        for col, law in enumerate(row_laws):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[col], color=ORIGINAL if col < 2 else NEW, fontsize=11)
            if law is not None:
                artist = ax.imshow(np.ma.masked_less_equal(law, 0), origin="lower",
                                   extent=(0, 1, 0, 1), interpolation="nearest",
                                   norm=norm, cmap="viridis", aspect="equal")
            else:
                ax.text(.5, .5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("Rival reputation")
            ax.set_ylabel((f"ρ = {case['rho']:g}\n" if col == 0 else "") + "Own reputation")
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
    if artist is not None:
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.55, pad=.015,
                     label="Probability mass per grid pair (common log scale)", extend="min")
    fig.suptitle("Joint reputation laws under both policies\n" + _window_note(cases), fontsize=12)
    fig.supxlabel(f"Same color scale in every panel. Positive masses below {floor:g} share the lowest color; zero mass is blank.", fontsize=8)
    _save(fig, directory, "05_joint_law_comparison")


def _plot_advantages(cases, directory):
    fig, axes = _panels(cases)
    values = [np.asarray(result["advantage"], dtype=float)
              for case in cases for result in (case["selected"], case.get("reference", {}))
              if "advantage" in result]
    lower = min(0., min(float(v.min()) for v in values)) if values else -1.
    upper = max(0., max(float(v.max()) for v in values)) if values else 1.
    padding = max(1e-6, (upper - lower) * .06)
    for case, ax in zip(cases, axes):
        ax.axhline(0, color=".5", lw=.8)
        for result, color, style, label, zorder in [
                (case["selected"], NEW, "-", NEW_LABEL, 2),
                (case.get("reference", {}), ORIGINAL, "--", ORIGINAL_LABEL, 3)]:
            if "advantage" in result:
                grid = _grid(result, case["n"])
                ax.plot(grid, result["advantage"], color=color, ls=style,
                        lw=1.5, label=label, zorder=zorder)
        ax.set(title=_label(case), xlabel="Own reputation", xlim=(0, 1),
               ylabel="Product 2 value advantage\nwhen selected",
               ylim=(lower - padding, upper + padding))
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Action-value advantages under both benchmarks\n"
                 "Positive values favor product 2; zero permits indifference. Common vertical scale.", fontsize=11)
    _save(fig, directory, "06_advantage_comparison")


@lru_cache(maxsize=32)
def _cached_demand(grid, parameters, order):
    from oblivious.oe_p0.code.primitives import demand_matrix
    return demand_matrix(np.asarray(grid), parameters, order=order)[0]


def economic_outcomes(cases):
    """Expected market flows under the same finite window; no policy is solved."""
    from oblivious.oe_p0.code.primitives import Parameters
    rows = []
    for case in cases:
        if case.get("selected") is None:
            continue
        config = case.get("configuration", {})
        parameters = dict(config.get("parameters", {}), rho=case["rho"])
        params = Parameters(**parameters)
        n = case["n"]
        grid = _grid(case["selected"], n)
        demand = _cached_demand(tuple(grid), params, int(config.get("order", 128)))
        burn, periods, _ = _window(config)
        for profile, result, law_result, law_key in [
                ("original", case.get("reference", {}), case.get("reference_temporal", {}), "nu"),
                ("temporal", case["selected"], case["selected"], "temporal_nu")]:
            law = _law(law_result, law_key, n)
            if "policy" not in result or law is None:
                continue
            alpha = np.asarray(result["policy"], dtype=float)
            # D.T preserves rare selection tails represented in the game kernel.
            selection_a = law * demand
            selection_b = law * demand.T
            product2_a = float(np.sum(selection_a * alpha[:, None]))
            product2_b = float(np.sum(selection_b * alpha[None, :]))
            share_a, share_b = float(selection_a.sum()), float(selection_b.sum())
            cost = params.c1 + (params.c2 - params.c1) * alpha
            profit_a = float(np.sum(selection_a * (params.R - cost[:, None])))
            profit_b = float(np.sum(selection_b * (params.R - cost[None, :])))
            product2 = product2_a + product2_b
            success = params.p1 * (share_a + share_b) + (params.p2 - params.p1) * product2
            rows.append(dict(rho=case["rho"], n=n, profile=profile,
                p0=result["p0"], burn_in=burn, periods=periods,
                product2_market=product2, success_rate=success,
                profit_per_seller=(profit_a + profit_b) / 2,
                profit_A=profit_a, profit_B=profit_b,
                market_share_A=share_a, market_share_B=share_b,
                product2_calendar_A=product2_a, product2_calendar_B=product2_b,
                joint_probability_sum=float(law.sum()),
                definition="Expected calendar-period flows in the same finite window; no discounting"))
    return rows


def _plot_outcomes(cases, directory):
    rows = economic_outcomes(cases)
    lookup = {(row["rho"], row["n"], row["profile"]): row for row in rows}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    positions = np.arange(len(cases))
    for ax, metric, scale, ylabel in zip(axes,
            ("product2_market", "profit_per_seller", "success_rate"),
            (100, 1, 100),
            ("Product 2 (% of market purchases)", "Profit per seller / calendar period",
             "Customer success (% of purchases)")):
        for profile, color, style, marker, label in [
                ("temporal", NEW, "-", "o", NEW_LABEL),
                ("original", ORIGINAL, "--", "s", ORIGINAL_LABEL)]:
            values = [lookup.get((case["rho"], case["n"], profile), {}).get(metric, np.nan)
                      for case in cases]
            ax.plot(positions, scale * np.asarray(values), ls=style, marker=marker,
                    color=color, lw=1.6, ms=5, markerfacecolor="white" if profile == "original" else color,
                    label=label)
        ax.set_ylabel(ylabel)
        _style(ax)
    axes[-1].set_xticks(positions, [f"{case['rho']:g}" for case in cases])
    axes[-1].set_xlabel("Persistence ρ (equally spaced cases)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Expected outcomes: both policies in the same finite window\n" +
                 _window_note(cases), fontsize=11)
    _save(fig, directory, "07_economic_outcome_comparison")


def _number(value, digits=9):
    if value is None:
        return "no disponible"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _candidate_row(case, candidate, index=None):
    n = case["n"]
    grid = _grid(candidate, n)
    mask = _mixed(candidate)
    law = _law(candidate, "temporal_nu", n)
    stationary = _law(candidate, "nu", n)
    posterior = (1 - case["rho"] + grid) / (3 - 2 * case["rho"])
    dp = candidate.get("dp_diagnostics", {})
    diagnostic = candidate.get("temporal_diagnostics", {})
    row = dict(
        rho=case["rho"], n=n, candidate_index=index,
        candidate_verified=_is_verified(candidate), p0=candidate["p0"],
        F_temporal=candidate["F"], closure_residual=candidate.get("G", candidate["F"] - candidate["p0"]),
        mixed_nodes=int(mask.sum()),
        mixed_reputations=json.dumps(grid[mask].tolist()),
        mixed_probabilities=json.dumps(np.asarray(candidate["policy"])[mask].tolist()),
        F_stationary_new=(float(stationary.sum(axis=0) @ posterior) if stationary is not None else None),
        joint_tv_temporal_stationary=(float(.5 * np.sum(np.abs(law - stationary)))
                                      if law is not None and stationary is not None else None),
        bellman_residual=dp.get("bellman_residual"),
        mixed_indifference_residual=dp.get("mixed_indifference_residual"),
        conditional_regret=dp.get("conditional_regret"),
    )
    for key in ("occupation_truncation_l1_bound", "tolerance", "propagated_steps", "early_stopped",
                "last_step_l1", "row_sum_error", "occupation_mass_before_normalization",
                "occupation_normalization_l1_correction", "arithmetic_roundoff_certified",
                "arithmetic_roundoff_error_bound", "method"):
        row["temporal_" + key] = diagnostic.get(key)
    return row


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(cases, directory, config):
    """Write Spanish interpretation plus selected and full candidate tables."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = sorted(cases, key=lambda c: (c["rho"], c["n"]))
    outcomes = economic_outcomes(cases)
    outcome_lookup = {(row["rho"], row["n"], row["profile"]): row for row in outcomes}
    summary, candidates, sensitivity = [], [], []
    for case in cases:
        for measured in case.get("horizon_sensitivity", []):
            sensitivity.append(dict(rho=case["rho"], n=case["n"],
                burn_in=measured["burn_in"], periods=measured["periods"],
                F=measured["F"], residual_at_fixed_policy=measured["residual_at_fixed_policy"],
                interpretation=measured.get("interpretation", "Fixed selected policy; no equilibrium re-solved"),
                occupation_truncation_l1_bound=measured.get("diagnostics", {}).get("occupation_truncation_l1_bound")))
        for index, candidate in enumerate(case.get("candidates", [])):
            candidates.append(_candidate_row(case, candidate, index))
        selected = case.get("selected")
        if selected is None:
            summary.append(dict(rho=case["rho"], n=case["n"], candidate_verified=False,
                                candidate_count=0))
            continue
        row = _candidate_row(case, selected)
        old = case.get("reference", {})
        old_window = case.get("reference_temporal", {})
        independent = case.get("independent_validation", {})
        row.update(candidate_count=len(case.get("candidates", [])),
                   p0_original_stationary=old.get("p0"),
                   F_temporal_original_policy=old_window.get("F"),
                   delta_p0=(selected["p0"] - old["p0"] if "p0" in old else None),
                   original_window_closure_residual=(old_window["F"] - old["p0"]
                                                    if "F" in old_window and "p0" in old else None))
        for key in ("verified", "F", "consistency_residual", "occupation_l1_difference", "method"):
            row["independent_validation_" + key] = independent.get(key)
        if "policy" in old:
            difference = np.abs(np.asarray(selected["policy"]) - np.asarray(old["policy"]))
            row["policy_mean_abs_difference"] = float(difference.mean())
            row["policy_max_abs_difference"] = float(difference.max())
            law = _law(selected, "temporal_nu", case["n"])
            if law is not None:
                row["policy_abs_difference_weighted_new_window"] = float(law.sum(axis=1) @ difference)
        for metric in ("product2_market", "success_rate", "profit_per_seller"):
            original_value = outcome_lookup.get((case["rho"], case["n"], "original"), {}).get(metric)
            new_value = outcome_lookup.get((case["rho"], case["n"], "temporal"), {}).get(metric)
            row[metric + "_original"] = original_value
            row[metric + "_new"] = new_value
            row[metric + "_new_minus_original"] = (new_value - original_value
                if original_value is not None and new_value is not None else None)
        summary.append(row)
    _write_csv(directory / "summary.csv", summary)
    _write_csv(directory / "candidates.csv", candidates)
    _write_csv(directory / "horizon_sensitivity.csv", sensitivity)
    _write_csv(directory / "economic_outcomes.csv", outcomes)
    _write_report(cases, summary, candidates, directory, config)
    return summary


def _write_report(cases, summary, candidates, directory, config):
    burn, periods, initial = _window(config)
    parameters = {key: value for key, value in config.get("parameters", {}).items() if key != "rho"}
    lines = [
        "OE p0: CONSISTENCIA CON PROMEDIO TEMPORAL ESPERADO", "",
        "El benchmark p₀ es un escalar fijo dentro del problema de optimización.",
        "Se elige para que coincida con el promedio esperado del sorteo TS del rival",
        "en una ventana finita de períodos de calendario del juego real de dos vendedores.",
        "En cada período se comparan los dos sorteos TS; solo actualiza su reputación",
        "el vendedor seleccionado. El promedio incluye al rival en todos los períodos,",
        "también cuando no es seleccionado.", "",
        f"Descarte B={burn:,}; ventana T={periods:,}; reputaciones iniciales={initial}.",
        f"Estados preselección t={burn:,},...,{burn + periods - 1:,}, con X₀ como estado inicial.",
        "Para una política alpha y su matriz de transición P_alpha:",
        "  nu_bar = (1/T) sum_{t=B}^{B+T-1} nu_0 P_alpha^t;",
        "  F_T(p₀,alpha) = sum_{x_A,x_B} nu_bar(x_A,x_B) m_rho(x_B);",
        "  m_rho(x) = (1-rho+x)/(3-2*rho);   p₀ = F_T(p₀,alpha).", "",
        "La esperanza del sorteo TS condicional a la reputación es m_rho(x).",
        "Por eso propagar la ley y promediar m_rho calcula el promedio esperado",
        "del sorteo sin ruido Monte Carlo. No se está imponiendo la media de una",
        "única trayectoria, un promedio condicionado a ventas ni una media de calidad.",
        "Si se acelera una cola de propagación, la cota de truncamiento se reporta",
        "por separado: no es una certificación del error de redondeo aritmético.", "",
        "PARÁMETROS Y COMPARACIÓN",
        "Referencia: oblivious/02_oe_p0/data/rho_forgetting_mixed, candidato seleccionado",
        "en la misma grilla. Ese experimento usa la distribución estacionaria para",
        "cerrar p₀. Se conserva su política y se evalúa también en esta misma ventana.",
        "El comparador es OE p0 original; no es el modelo OE following monopoly.",
        "Parámetros: " + (json.dumps(parameters, ensure_ascii=False, sort_keys=True)
                          if parameters else "ver configuración guardada"),
        "Los valores de rho se indican en cada caso de la tabla siguiente.",
        "La demanda usada para elegir producto sigue siendo P(theta(x)>p₀),",
        "con cola Beta. Las mezclas interiores exigen indiferencia de valores.", "",
        "RESULTADOS POR RHO",
        "p₀ nuevo | p₀ original | diferencia | residual temporal nuevo | candidatos",
    ]
    for row in summary:
        lines.append(f"rho={row['rho']:g}, n={row['n']}: " + " | ".join([
            _number(row.get("p0")), _number(row.get("p0_original_stationary")),
            _number(row.get("delta_p0")), _number(row.get("closure_residual"), 4),
            str(row.get("candidate_count", 0))]))
        if row.get("p0") is None:
            lines.append("  No se guardó un candidato seleccionado.")
            continue
        lines += [
            f"  Candidato verificado: {row['candidate_verified']}; nodos mixtos: {row['mixed_nodes']}.",
            "  Política original evaluada en la misma ventana: F_T=" +
            _number(row.get("F_temporal_original_policy")) + "; F_T-p₀_original=" +
            _number(row.get("original_window_closure_residual"), 5) + ".",
            "  Nueva política: F estacionario=" + _number(row.get("F_stationary_new")) +
            "; distancia TV entre ley temporal y estacionaria=" +
            _number(row.get("joint_tv_temporal_stationary"), 5) + ".",
            "  Diferencia absoluta media entre políticas sobre los nodos=" +
            _number(row.get("policy_mean_abs_difference"), 5) + ".",
            "  Cota L1 de truncamiento temporal=" +
            _number(row.get("temporal_occupation_truncation_l1_bound"), 5) +
            "; pasos propagados=" + _number(row.get("temporal_propagated_steps")) + ".",
        ]
        if row.get("independent_validation_verified") is not None:
            lines += [
                "  Revalidación independiente del seleccionado: " +
                str(row["independent_validation_verified"]) + "; |F_T-p₀|=" +
                _number(row.get("independent_validation_consistency_residual"), 5) +
                "; diferencia L1 de leyes=" +
                _number(row.get("independent_validation_occupation_l1_difference"), 5) + ".",
                "  Método: " + str(row.get("independent_validation_method")) + ".",
            ]
    selected_verified = [r for r in summary if r.get("candidate_verified")]
    lines += ["", f"Casos con candidato seleccionado verificado: {len(selected_verified)}/{len(cases)}.",
              f"Candidatos guardados en total: {len(candidates)}."]
    if selected_verified:
        lines.append("Máximo residual absoluto temporal entre seleccionados verificados: " +
                     _number(max(abs(r["closure_residual"]) for r in selected_verified), 6) + ".")
    lines += ["", "RESULTADOS ECONÓMICOS EN LA MISMA VENTANA",
              "Ambas políticas se evalúan desde la misma condición inicial y con B,T comunes.",
              "Son flujos esperados por período de calendario, sin descuento ni ruido Monte Carlo.",
              "Producto 2 y éxito son fracciones de compras; el beneficio es por vendedor",
              "y período de calendario (incluye beneficio cero cuando no vende).",
              "Para cada rho se muestra original -> nuevo."]
    for row in summary:
        if row.get("product2_market_new") is not None:
            lines.append(f"rho={row['rho']:g}, n={row['n']}: " + "; ".join(
                label + " " + _number(row.get(metric + "_original"), 7) + " -> " +
                _number(row.get(metric + "_new"), 7)
                for metric, label in (("product2_market", "producto 2"),
                                      ("profit_per_seller", "beneficio"), ("success_rate", "éxito"))))
    if any(case.get("horizon_sensitivity") for case in cases):
        lines += ["", "SENSIBILIDAD DE LA VENTANA CON POLÍTICA FIJA",
                  "Se conserva la política nueva seleccionada; no se resuelve otro equilibrio."]
        for case in cases:
            for measured in case.get("horizon_sensitivity", []):
                lines.append(f"rho={case['rho']:g}, n={case['n']}, B={measured['burn_in']}, "
                             f"T={measured['periods']}: F_T={_number(measured['F'])}, "
                             f"F_T-p₀={_number(measured['residual_at_fixed_policy'], 6)}.")
    lines += ["", "ALCANCE DE LA INTERPRETACIÓN",
              "Una ventana finita depende del horizonte, el descarte y las condiciones iniciales.",
              "Descartar períodos no prueba estacionariedad. Un promedio escalar cercano",
              "al estacionario tampoco prueba que las distribuciones sean cercanas.",
              "Bajo condiciones ergódicas apropiadas, el promedio temporal de largo plazo",
              "puede coincidir con la media estacionaria. Eso no identifica automáticamente",
              "la ventana finita usada aquí con una distribución estacionaria.",
              "Los valores son deterministas para la cadena discretizada, sujetos a tolerancias",
              "numéricas. No son estimaciones de una simulación ni tienen intervalos Monte Carlo.",
              "La cota de cola solo controla esa aproximación; no incluye errores de cuadratura,",
              "redondeo, solución de la política ni discretización de reputaciones.",
              "La búsqueda guarda los candidatos encontrados y verificados; no demuestra",
              "completitud ni unicidad. Los candidatos originales se usan además como",
              "propuestas y se vuelven a verificar con el cierre temporal para aceptarlos.",
              "Una pequeña diferencia entre p₀ seleccionados no",
              "demuestra que se haya seguido la misma rama cuando hay varios candidatos.",
              "El certificado corresponde a la grilla y a la consistencia escalar especificada.",
              "No demuestra convergencia al modelo continuo ni equilibrio perfecto de Markov",
              "del juego en el que ambos vendedores observan las dos reputaciones.", "",
              "ARCHIVOS Y FIGURAS",
              "summary.csv: candidatos seleccionados, comparación original, cierres y diagnósticos.",
              "candidates.csv: todos los candidatos guardados, mezclas y residuos numéricos.",
              "horizon_sensitivity.csv: ventanas más largas o posteriores, con política fija.",
              "economic_outcomes.csv: uso de producto 2, beneficios y éxito esperado por política",
              "  en la misma ventana; incluye cuotas de mercado y beneficios de cada vendedor.",
              "01_p0_comparison: benchmarks de ambos experimentos para todos los rho.",
              "02_policy_comparison: políticas; rombos indican mezcla en nodos guardados.",
              "03_reputation_cdf_comparison: ambas políticas en la misma ventana, con leyes",
              "  estacionarias punteadas como diagnóstico adicional.",
              "04_consistency_map_comparison: mapa temporal nuevo y estacionario original;",
              "  puntos evaluados, candidatos verificados y seleccionados. Las mezclas en",
              "  estados indiferentes hacen que el cierre dependa de p₀ y de la política.",
              "05_joint_law_comparison: ambas políticas y ambas leyes, en una escala log común.",
              "06_advantage_comparison: ventaja del producto 2 bajo los dos benchmarks.",
              "07_economic_outcome_comparison: producto 2, beneficio por vendedor y éxito",
              "  del cliente bajo ambas políticas, en la misma ventana temporal.",
              "Todas las figuras se guardan en PNG y PDF; el color distingue el modelo."]
    (directory / "RESULTADOS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
