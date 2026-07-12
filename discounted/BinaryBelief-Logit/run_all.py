#!/usr/bin/env python3
"""Run all experiments (E0-E6) for the logit-demand reputation model with
stochastic per-period engagement (belief frozen when no transaction occurs).

Usage:
    python run_all.py                       # everything, default parameters
    python run_all.py --mu 0.6 --beta 16    # override any Config field
    python run_all.py --only E2 E3          # subset of experiments

Outputs: outputs/plots/*.png, outputs/tables/*.csv, outputs/SUMMARY.md.
Deterministic given Config.seed. matplotlib only, no notebooks.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from config import Config
from src.model import Band, SolveResult, demand, sigmoid, simulate, solve_dp

PLOTS = BASE / "outputs" / "plots"
TABLES = BASE / "outputs" / "tables"
OUT = BASE / "outputs"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
# Validated reference palette (light surface): categorical slots in fixed
# order, chrome ink recessive so the data carries the figure.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDC = "#e1e0d9"
AXISC = "#c3c2b7"
SURFACE = "#fcfcfb"
C_BLUE = "#2a78d6"    # slot 1: V, logit policy, first start in E5
C_AQUA = "#1baf7a"    # slot 2: comparison series (TS in E6)
C_VIOLET = "#4a3aa7"  # slot 5: reputational premium g(pi)
C_ORANGE = "#eb6834"  # slot 8: threshold dc/dp, second start in E5
BAND_FILL = C_BLUE
BAND_ALPHA = 0.13
# sequential blue ramp (one hue, light -> dark) for the ordered beta family
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 160,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXISC, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0,
})

# solves whose value-iteration residual did not reach tolerance (must stay
# empty; anything here is flagged loudly in SUMMARY.md)
NONCONVERGED: list = []


def _solve(cfg: Config, mode: str = "logit", tag: str = "") -> SolveResult:
    res = solve_dp(cfg, mode=mode)
    if not res.converged:
        NONCONVERGED.append(f"{tag or mode}: sup_err={res.sup_err:.2e} "
                            f"after {res.n_iter} iters")
        print(f"  *** VI DID NOT CONVERGE ({tag}): sup_err={res.sup_err:.2e}")
    return res


def _ax_clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _shade_band(ax, band: Band, label="investment band (x*=2)"):
    lab = label
    for a, b in band.intervals:
        ax.axvspan(a, b, color=BAND_FILL, alpha=BAND_ALPHA, lw=0, label=lab)
        lab = None  # legend entry once


def _param_str(cfg: Config, keys=("p1", "p2", "R", "c1", "c2", "gamma", "mu",
                                  "beta")):
    return ", ".join(f"{k}={getattr(cfg, k):g}" for k in keys)


def _save(fig, name):
    fig.savefig(PLOTS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote plots/{name}")


def _write_csv(name: str, header: list, rows: list):
    with open(TABLES / name, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                             for v in r) + "\n")
    print(f"  wrote tables/{name}")


def _band_note(band: Band) -> str:
    if band.empty:
        return "EMPTY (x*=1 everywhere)"
    s = "; ".join(f"[{a:.4f}, {b:.4f}]" for a, b in band.intervals)
    if not band.single_interval:
        s += "  *** NOT A SINGLE INTERVAL ***"
    return s


# =================================================================== E0
def e0_demand_shapes(cfg: Config, summary: list):
    """Sanity check of the microfoundation: D(pi) for several beta, with the
    greedy step at pibar(mu) as the beta -> infinity limit."""
    betas = [2, 4, 8, 16, 50]
    pi = np.linspace(0.0, 1.0, 801)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (b, col) in enumerate(zip(betas, SEQ_BLUE)):
        D = demand(pi, replace(cfg, beta=b))
        ax.plot(pi, D, color=col, label=f"beta = {b:g}")
        i = np.searchsorted(pi, min(0.97, cfg.pibar + 0.30 - 0.05 * k))
        ax.annotate(f"{b:g}", (pi[i], D[i]), textcoords="offset points",
                    xytext=(4, -2), color=col, fontsize=8.5)
    step = demand(pi, cfg, mode="greedy")
    ax.plot(pi, step, color=INK, ls="--", lw=1.4,
            label="greedy step (beta = inf)")
    ax.axvline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.annotate(f"pibar = {cfg.pibar:.2f}", (cfg.pibar, 0.03),
                textcoords="offset points", xytext=(5, 0), color=MUTED,
                fontsize=8.5)
    ax.set(xlabel="belief pi", ylabel="engagement probability D(pi)",
           title=f"E0  Logit demand D(pi) = sigmoid(beta (p1 + pi dp - mu))   "
                 f"[p1={cfg.p1:g}, p2={cfg.p2:g}, mu={cfg.mu:g}]")
    ax.legend(loc="upper left")
    _ax_clean(ax)
    _save(fig, "E0_demand_shapes.png")
    summary.append(("E0", "Demand microfoundation verified visually: logit "
                    "demand steepens toward the greedy step at pibar = "
                    f"{cfg.pibar:.3f} as beta grows; beta -> 0 flattens "
                    "toward 1/2."))


# =================================================================== E1
def e1_baseline(cfg: Config, summary: list) -> SolveResult:
    res = _solve(cfg, tag="E1 baseline")
    print(f"  VI converged in {res.n_iter} iters, sup_err = {res.sup_err:.2e}")
    band = res.band

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.3]})
    fig.suptitle(f"E1  Baseline solve   [{_param_str(cfg)}]",
                 fontsize=11, color=INK, y=0.995)

    # (a) value function
    ax = axes[0]
    ax.plot(res.pi, res.V, color=C_BLUE)
    _shade_band(ax, band)
    ax.axvline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.set(ylabel="V(pi)", title="(a) value function")
    ax.legend(loc="upper left")

    # (b) reputational premium vs the constant threshold
    ax = axes[1]
    ax.plot(res.pi, res.g, color=C_VIOLET,
            label="g(pi) = gamma [V(ell+dS) - V(ell+dF)]")
    ax.axhline(res.threshold, color=C_ORANGE, lw=1.8,
               label=f"dc/dp = {res.threshold:g}")
    _shade_band(ax, band, label=None)
    ax.axvline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.set(ylabel="value", title="(b) invest iff g >= dc/dp "
                                 "(shaded: investment band)")
    ax.legend(loc="upper right")

    # (c) policy
    ax = axes[2]
    ax.step(res.pi, res.policy, where="mid", color=C_BLUE)
    _shade_band(ax, band, label=None)
    ax.axvline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.annotate(f"pibar = {cfg.pibar:.2f}", (cfg.pibar, 1.45),
                textcoords="offset points", xytext=(5, 0), color=MUTED,
                fontsize=8.5)
    ax.set(xlabel="belief pi", ylabel="x*(pi)", yticks=[1, 2],
           title="(c) optimal product")
    for ax in axes:
        _ax_clean(ax)
    fig.tight_layout()
    _save(fig, "E1_baseline.png")

    _write_csv("E1_baseline_band.csv",
               ["pi_lo", "pi_hi", "width", "center", "pibar",
                "single_interval"],
               [[band.lo, band.hi, band.width, band.center, cfg.pibar,
                 band.single_interval]])
    # full g(pi) on the grid for later analysis
    _write_csv("E1_g_of_pi.csv",
               ["pi", "ell", "D", "V", "g", "policy"],
               [[float(res.pi[i]), float(res.ell[i]), float(res.D[i]),
                 float(res.V[i]), float(res.g[i]), int(res.policy[i])]
                for i in range(len(res.pi))])
    summary.append(("E1", f"Band = {_band_note(band)}; center "
                    f"{band.center:.4f} vs pibar {cfg.pibar:.4f} "
                    f"(offset {band.center - cfg.pibar:+.4f}); width "
                    f"{band.width:.4f}. Single interval: "
                    f"{band.single_interval}."))
    return res


# =================================================================== E2
def e2_sweep_mu(cfg: Config, summary: list) -> int:
    mus = np.linspace(0.32, 0.78, 24)
    rows, violations = [], 0
    for mu in mus:
        c = replace(cfg, mu=float(mu))
        r = _solve(c, tag=f"E2 mu={mu:.3f}")
        b = r.band
        violations += 0 if b.single_interval else 1
        rows.append([float(mu), c.pibar, b.lo, b.hi, b.width, b.center,
                     b.single_interval])
    _write_csv("E2_band_vs_mu.csv",
               ["mu", "pibar", "pi_lo", "pi_hi", "width", "center",
                "single_interval"], rows)
    arr = np.array([r[:6] for r in rows], dtype=float)
    mu_v, pibar_v, lo_v, hi_v, w_v, ctr_v = arr.T

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    ax.fill_between(mu_v, lo_v, hi_v, color=BAND_FILL, alpha=BAND_ALPHA, lw=0)
    ax.plot(mu_v, lo_v, color=C_BLUE, label="band endpoints pi-, pi+")
    ax.plot(mu_v, hi_v, color=C_BLUE)
    ax.plot(mu_v, ctr_v, color=C_VIOLET, label="band center")
    ax.plot(mu_v, pibar_v, color=INK, ls="--", lw=1.4,
            label="pibar(mu) = (mu - p1)/dp")
    ax.set(xlabel="outside-option center mu", ylabel="belief pi",
           title="(a) band location vs the greedy reference pibar(mu)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(mu_v, w_v, color=C_BLUE)
    ax.set(xlabel="outside-option center mu", ylabel="band width",
           title="(b) band width vs mu")
    for ax in axes:
        _ax_clean(ax)
    fig.suptitle(f"E2  Sweep in mu — localization   "
                 f"[{_param_str(cfg, ('p1','p2','R','c1','c2','gamma','beta'))}]",
                 fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "E2_sweep_mu.png")

    ok = ~np.isnan(ctr_v)
    if ok.sum() >= 2:
        corr = float(np.corrcoef(pibar_v[ok], ctr_v[ok])[0, 1])
        slope = float(np.polyfit(pibar_v[ok], ctr_v[ok], 1)[0])
        maxdev = float(np.nanmax(np.abs(ctr_v - pibar_v)))
        track = (f"corr(center, pibar) = {corr:.4f}, slope of center in "
                 f"pibar = {slope:.3f}, max |center - pibar| = {maxdev:.4f}")
    else:
        track = "band empty for (almost) all mu — nothing to track"
    summary.append(("E2", f"Localization: {track} over mu in [0.32, 0.78] "
                    f"({int((~np.isnan(ctr_v)).sum())}/24 nonempty bands). "
                    f"Single-interval violations: {violations}/24."))
    return violations


# =================================================================== E3
def _critical_scalar(cfg: Config, field: str, lo: float, hi: float,
                     tol: float = 1e-3, band_at_hi: bool = True) -> float:
    """Bisect (on [lo, hi], lo < hi) for the critical parameter value at which
    the band vanishes. Requires the band nonempty at exactly one end:
    at `hi` if band_at_hi, else at `lo` (monotone switch in between)."""
    def nonempty(v):
        return not solve_dp(replace(cfg, **{field: float(v)})).band.empty
    if not (nonempty(hi) == band_at_hi and nonempty(lo) != band_at_hi):
        return float("nan")   # bracket assumption failed; report and move on
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if nonempty(mid) == band_at_hi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def e3_sweep_beta(cfg: Config, summary: list):
    betas = [1, 2, 4, 8, 16, 32, 64, 128]
    rows, violations = [], 0
    for b in betas:
        r = _solve(replace(cfg, beta=float(b)), tag=f"E3 beta={b}")
        violations += 0 if r.band.single_interval else 1
        rows.append([float(b), r.band.lo, r.band.hi, r.band.width,
                     r.band.single_interval])
    greedy = _solve(cfg, mode="greedy", tag="E3 greedy limit")
    _write_csv("E3_band_vs_beta.csv",
               ["beta", "pi_lo", "pi_hi", "width", "single_interval"], rows)

    arr = np.array([r[:4] for r in rows], dtype=float)
    b_v, lo_v, hi_v, w_v = arr.T
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.fill_between(b_v, lo_v, hi_v, color=BAND_FILL, alpha=BAND_ALPHA, lw=0)
    ax.plot(b_v, lo_v, color=C_BLUE, marker="o", ms=4,
            label="band endpoints pi-, pi+")
    ax.plot(b_v, hi_v, color=C_BLUE, marker="o", ms=4)
    if not greedy.band.empty:
        ax.axhline(greedy.band.lo, color=INK, ls="--", lw=1.2,
                   label=f"greedy-limit band [{greedy.band.lo:.3f}, "
                         f"{greedy.band.hi:.3f}]")
        ax.axhline(greedy.band.hi, color=INK, ls="--", lw=1.2)
    ax.axhline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.annotate(f"pibar = {cfg.pibar:.2f}", (b_v[0], cfg.pibar),
                textcoords="offset points", xytext=(4, 4), color=MUTED,
                fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set(xlabel="demand precision beta (log scale)", ylabel="belief pi",
           title=f"E3  Band endpoints vs beta   "
                 f"[{_param_str(cfg, ('p1','p2','R','c1','c2','gamma','mu'))}]")
    ax.legend(loc="best")
    _ax_clean(ax)
    _save(fig, "E3_sweep_beta.png")

    # critical beta below which x* = 1 everywhere
    nonempty_b = [b for b, w in zip(b_v, w_v) if w > 0]
    empty_b = [b for b, w in zip(b_v, w_v) if w == 0]
    if nonempty_b:
        lo0 = max(empty_b) if empty_b else 0.02
        hi0 = min(nonempty_b)
        beta_crit = _critical_scalar(cfg, "beta", lo0, hi0, band_at_hi=True)
    else:
        beta_crit = float("nan")
    if greedy.band.empty:
        conv = "greedy-limit band is EMPTY"
    else:
        conv = (f"endpoints at beta=128: [{lo_v[-1]:.4f}, {hi_v[-1]:.4f}] vs "
                f"greedy limit [{greedy.band.lo:.4f}, {greedy.band.hi:.4f}]")
    summary.append(("E3", f"Critical beta (band vanishes below) ~= "
                    f"{beta_crit:.3f}. Convergence to greedy limit: {conv}. "
                    f"Single-interval violations: {violations}/{len(betas)}."))
    return beta_crit, violations


# =================================================================== E4
def e4_sweep_dc(cfg: Config, summary: list):
    # sweep the cost GAP dc so that dc/dp hits {0.2, 0.4, 0.6, 0.9, 1.2}
    # exactly (c2 = c1 + dc; with c1 = 0 this is the spec's "c2 in {...}")
    dcs = [0.1, 0.2, 0.3, 0.45, 0.6]
    rows, violations = [], 0
    for dc in dcs:
        c = replace(cfg, c2=cfg.c1 + dc)
        r = _solve(c, tag=f"E4 dc={dc}")
        violations += 0 if r.band.single_interval else 1
        rows.append([c.c2, dc, dc / c.dp, r.band.lo, r.band.hi,
                     r.band.width, r.band.single_interval])
    _write_csv("E4_band_vs_dc.csv",
               ["c2", "dc", "dc_over_dp", "pi_lo", "pi_hi", "width",
                "single_interval"], rows)

    arr = np.array([r[:6] for r in rows], dtype=float)
    c2_v, dc_v, ratio_v, lo_v, hi_v, w_v = arr.T
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ok = w_v > 0
    ax.fill_between(ratio_v[ok], lo_v[ok], hi_v[ok], color=BAND_FILL,
                    alpha=BAND_ALPHA, lw=0)
    ax.plot(ratio_v[ok], lo_v[ok], color=C_BLUE, marker="o", ms=4,
            label="band endpoints pi-, pi+")
    ax.plot(ratio_v[ok], hi_v[ok], color=C_BLUE, marker="o", ms=4)
    for rt in ratio_v[~ok]:
        ax.axvline(rt, color=MUTED, ls=":", lw=1.0)
        ax.annotate("band empty", (rt, 0.5), rotation=90, color=MUTED,
                    fontsize=8.5, ha="right", va="center",
                    textcoords="offset points", xytext=(-3, 0))
    ax.axhline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.annotate(f"pibar = {cfg.pibar:.2f}", (ratio_v[0], cfg.pibar),
                textcoords="offset points", xytext=(4, 4), color=MUTED,
                fontsize=8.5)
    ax.set(xlabel="cost-benefit ratio dc/dp", ylabel="belief pi",
           title=f"E4  Band vs investment cost   "
                 f"[{_param_str(cfg, ('p1','p2','R','c1','gamma','mu','beta'))}]")
    ax.legend(loc="best")
    _ax_clean(ax)
    _save(fig, "E4_sweep_dc.png")

    # critical dc above which the band is empty (bisect on c2; the band is
    # nonempty at the LOW end of the bracket). Expand the upper bracket in
    # case the band survives beyond the tested range.
    nonempty_c2 = [c for c, w in zip(c2_v, w_v) if w > 0]
    empty_c2 = [c for c, w in zip(c2_v, w_v) if w == 0]
    if not nonempty_c2:
        c2_crit = float("nan")
    else:
        lo0 = max(nonempty_c2)
        hi0 = min(empty_c2) if empty_c2 else 2.0 * max(max(c2_v), cfg.R)
        while (not solve_dp(replace(cfg, c2=hi0)).band.empty
               and hi0 < 8 * cfg.R):
            lo0, hi0 = hi0, 2.0 * hi0
        c2_crit = _critical_scalar(cfg, "c2", lo0, hi0, band_at_hi=False)
    dc_crit = c2_crit - cfg.c1
    monotone = bool(np.all(np.diff(w_v) <= 1e-9))
    summary.append(("E4", f"Band width monotone decreasing in dc: {monotone} "
                    f"(widths {np.array2string(w_v, precision=3)}). Critical "
                    f"dc ~= {dc_crit:.4f} (dc/dp ~= {dc_crit / cfg.dp:.3f}). "
                    f"Interior nontrivial bands across dc/dp in "
                    f"{np.array2string(ratio_v, precision=2)} (cf. Thompson "
                    f"demand, where dc/dp=0.6 gave nearly the whole space). "
                    f"Single-interval violations: {violations}/{len(dcs)}."))
    return dc_crit, violations


# =================================================================== E5
def e5_paths(cfg: Config, res: SolveResult, summary: list):
    rng = np.random.default_rng(cfg.seed)
    sims = [simulate(res, cfg, pi0, rng) for pi0 in cfg.sim_starts]
    start_cols = [C_BLUE, C_ORANGE]
    t = np.arange(cfg.sim_horizon + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    # (a) fan/spaghetti charts, one panel per start
    for ax, sim, col in zip(axes[0], sims, start_cols):
        for a, b in res.band.intervals:
            ax.axhspan(a, b, color=BAND_FILL, alpha=BAND_ALPHA, lw=0)
        for k in range(min(60, cfg.sim_paths)):   # spaghetti subset
            ax.plot(t, sim.pi_paths[k], color=col, alpha=0.10, lw=0.7)
        qs = np.percentile(sim.pi_paths, [10, 50, 90], axis=0)
        ax.fill_between(t, qs[0], qs[2], color=col, alpha=0.18, lw=0)
        ax.plot(t, qs[1], color=col, lw=2.0, label="median")
        ax.axhline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
        ax.set(xlabel="period t", ylabel="belief pi_t", ylim=(0, 1),
               title=f"(a) pi_0 = {sim.pi0:g}   "
                     f"(time in band: {100*sim.frac_in_band:.1f}%)")
        ax.legend(loc="lower right")

    # (b) fraction of periods with a transaction, over time
    ax = axes[1][0]
    for sim, col in zip(sims, start_cols):
        ax.plot(t[:-1], sim.engaged.mean(axis=0), color=col, lw=1.6,
                label=f"pi_0 = {sim.pi0:g}")
    ax.axhline(float(demand(cfg.pibar, cfg)), color=MUTED, ls=":", lw=1.0)
    ax.set(xlabel="period t", ylabel="fraction of paths transacting",
           ylim=(0, 1), title="(b) transaction frequency over time")
    ax.legend(loc="lower right")

    # (c) fraction of paths inside the band, over time
    ax = axes[1][1]
    for sim, col in zip(sims, start_cols):
        ax.plot(t, sim.frac_in_band_t, color=col, lw=1.6,
                label=f"pi_0 = {sim.pi0:g}")
    ax.set(xlabel="period t", ylabel="fraction of paths in band",
           ylim=(0, 1), title="(c) time in the investment band")
    ax.legend(loc="upper right")

    for ax in axes.ravel():
        _ax_clean(ax)
    fig.suptitle(f"E5  {cfg.sim_paths} simulated paths (T={cfg.sim_horizon}) "
                 f"under x*, true mechanism (frozen belief when no "
                 f"transaction)   [{_param_str(cfg)}]",
                 fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "E5_simulated_paths.png")

    # escape from the slow-learning trap: first passage to the band's lower
    # endpoint
    rows, esc_notes = [], []
    for sim in sims:
        esc = sim.escape_t
        n_esc = int(np.sum(~np.isnan(esc)))
        mean_esc = float(np.nanmean(esc)) if n_esc else float("nan")
        med_esc = float(np.nanmedian(esc)) if n_esc else float("nan")
        rows.append([sim.pi0, sim.frac_in_band,
                     float(np.mean(sim.engaged)),
                     float(sim.pi_paths[:, -1].mean()),
                     n_esc / cfg.sim_paths, mean_esc, med_esc])
        if not res.band.empty and sim.pi0 < res.band.lo:
            esc_notes.append(f"from pi0={sim.pi0:g}: {100*n_esc/cfg.sim_paths:.0f}% "
                             f"of paths reached the band (pi- = "
                             f"{res.band.lo:.3f}) within T={cfg.sim_horizon}, "
                             f"mean escape time {mean_esc:.0f} periods "
                             f"(median {med_esc:.0f})")
    _write_csv("E5_time_in_band.csv",
               ["pi0", "frac_in_band", "frac_transacting", "mean_pi_final",
                "frac_escaped", "mean_escape_t", "median_escape_t"], rows)
    fr = ", ".join(f"pi0={s.pi0:g}: {100*s.frac_in_band:.1f}%" for s in sims)
    summary.append(("E5", f"Fraction of time in band: {fr}. Mean terminal "
                    f"belief: "
                    + ", ".join(f"{float(s.pi_paths[:, -1].mean()):.3f}"
                                for s in sims)
                    + ". " + ("; ".join(esc_notes) + "." if esc_notes
                              else "Both starts lie at or above the band's "
                                   "lower endpoint — no escape problem.")))


# =================================================================== E6
def e6_ts_comparison(cfg: Config, res_logit: SolveResult, summary: list) -> int:
    res_ts = _solve(cfg, mode="ts", tag="E6 Thompson")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.step(res_logit.pi, res_logit.policy, where="mid", color=C_BLUE,
            label=f"logit demand (mu={cfg.mu:g}, beta={cfg.beta:g})")
    ax.step(res_ts.pi, res_ts.policy + 0.02, where="mid", color=C_AQUA,
            label="Thompson demand D(pi) = pi")
    _shade_band(ax, res_logit.band, label=None)
    for a, b in res_ts.band.intervals:
        ax.axvspan(a, b, color=C_AQUA, alpha=0.10, lw=0)
    ax.axvline(cfg.pibar, color=MUTED, ls=":", lw=1.0)
    ax.annotate(f"pibar = {cfg.pibar:.2f}", (cfg.pibar, 1.5),
                textcoords="offset points", xytext=(5, 0), color=MUTED,
                fontsize=8.5)
    ax.set(xlabel="belief pi", ylabel="x*(pi)", yticks=[1, 2],
           title=f"E6  Optimal policies under logit vs Thompson demand"
                 f"   [{_param_str(cfg)}]")
    ax.legend(loc="center left")
    _ax_clean(ax)
    _save(fig, "E6_ts_comparison.png")

    _write_csv("E6_bands.csv",
               ["demand", "pi_lo", "pi_hi", "width", "single_interval"],
               [["logit", res_logit.band.lo, res_logit.band.hi,
                 res_logit.band.width, res_logit.band.single_interval],
                ["ts", res_ts.band.lo, res_ts.band.hi, res_ts.band.width,
                 res_ts.band.single_interval]])
    summary.append(("E6", f"Logit band {_band_note(res_logit.band)} vs "
                    f"Thompson band {_band_note(res_ts.band)}: TS's region is "
                    f"mu-independent by construction; logit localizes near "
                    f"pibar = {cfg.pibar:.3f}."))
    return int(not res_logit.band.single_interval) + \
        int(not res_ts.band.single_interval)


# =================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for f in fields(Config):
        if f.name in ("sim_starts",):
            continue
        ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset, e.g. --only E1 E2")
    args = ap.parse_args()

    cfg = Config()
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)
                 if hasattr(args, f.name) and getattr(args, f.name) is not None}
    if overrides:
        cfg = replace(cfg, **overrides)
    run = (lambda e: args.only is None or e in args.only)

    t0 = time.time()
    summary: list = []
    violations = 0
    print(f"Config: {_param_str(cfg)}  (seed={cfg.seed}, N={cfg.n_grid}, "
          f"dc/dp={cfg.dc / cfg.dp:g}, pibar={cfg.pibar:g})")

    if run("E0"):
        print("E0 demand shapes...")
        e0_demand_shapes(cfg, summary)
    res = None
    if run("E1") or run("E5") or run("E6"):
        print("E1 baseline solve...")
        res = e1_baseline(cfg, summary)
        violations += int(not res.band.single_interval)
    if run("E2"):
        print("E2 sweep in mu (24 solves)...")
        violations += e2_sweep_mu(cfg, summary)
    if run("E3"):
        print("E3 sweep in beta (+ bisection for critical beta)...")
        violations += e3_sweep_beta(cfg, summary)[1]
    if run("E4"):
        print("E4 sweep in dc (+ bisection for critical dc)...")
        violations += e4_sweep_dc(cfg, summary)[1]
    if run("E5"):
        print("E5 simulated paths (true mechanism)...")
        e5_paths(cfg, res, summary)
    if run("E6"):
        print("E6 Thompson comparison...")
        violations += e6_ts_comparison(cfg, res, summary)

    # ---- SUMMARY.md
    lines = ["# Logit-demand reputation model — run summary", "",
             f"Parameters: `{_param_str(cfg)}`, seed={cfg.seed}, "
             f"grid N={cfg.n_grid} on pi in [{cfg.pi_eps:g}, {1-cfg.pi_eps:g}], "
             f"VI tol={cfg.vi_tol:g} (max {cfg.vi_max_iter} iters).", "",
             f"Derived: dp={cfg.dp:g}, dc={cfg.dc:g}, dc/dp={cfg.dc/cfg.dp:g}, "
             f"pibar(mu) = (mu - p1)/dp = {cfg.pibar:.4f}.", ""]
    for tag, text in summary:
        lines.append(f"- **{tag}** — {text}")
    lines.append("")
    if NONCONVERGED:
        lines.append("**WARNING: value iteration did NOT reach tolerance in "
                     f"{len(NONCONVERGED)} solve(s):** "
                     + "; ".join(NONCONVERGED))
    else:
        lines.append("Value iteration reached the 1e-10 residual tolerance "
                     "in every solve.")
    lines.append("")
    lines.append(f"**WARNING: single-interval (band) property VIOLATED in "
                 f"{violations} solve(s) — see flagged lines above.**"
                 if violations else
                 "Single-interval (band) property held in every solve.")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    if violations:
        print(f"\n*** WARNING: {violations} single-interval violation(s) ***")
    if NONCONVERGED:
        print(f"*** WARNING: {len(NONCONVERGED)} non-converged solve(s) ***")
    print(f"\nwrote outputs/SUMMARY.md   (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
