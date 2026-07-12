#!/usr/bin/env python3
"""Produce all figures and sanity checks for the 1-D binary-belief reputation model.

Usage:
    python run.py               # both configs; sweep figures (5-6) for the base config
    python run.py --sweeps-all  # also run the (slower) sweeps for gamma = 0.999

Outputs go to <outdir>/plots/ for each config in config.CONFIGS; band edges and
sanity checks are printed to stdout. Deterministic given the config seeds.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from config import CONFIGS, Config
from src.model import (Band, MCResult, SolveResult, demand, extract_band,
                       fluid_trajectories, sigmoid, simulate, solve_dp,
                       stationary_sample)

# ---------------------------------------------------------------- style
# Palette: validated categorical slots + chrome ink (light surface).
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDC = "#e1e0d9"
AXISC = "#c3c2b7"
SURFACE = "#fcfcfb"
RULE_COLOR = {"TS": "#2a78d6", "EG": "#1baf7a", "LOGIT": "#eda100"}
C_GAP = "#4a3aa7"                                   # continuation gap g(pi)
START_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7",
                "#e34948", "#e87ba4", "#eb6834"]     # per-start path identity
BAND_ALPHA = 0.14

RULE_LABEL = {"TS": "Thompson sampling", "EG": f"epsilon-greedy",
              "LOGIT": "smooth logit"}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXISC,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, outdir: Path, name: str):
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    wrote {path.relative_to(BASE)}")


def shade_band(ax, band: Band, color: str, axis: str = "x", label: str | None = None):
    if band.empty:
        return
    span = ax.axvspan if axis == "x" else ax.axhspan
    span(band.pi_lo, band.pi_hi, color=color, alpha=BAND_ALPHA, lw=0, label=label)


def mark_pibar(ax, cfg, axis: str = "x", label: str = r"$\bar\pi$"):
    line = ax.axvline if axis == "x" else ax.axhline
    line(cfg.pibar, color=INK2, ls=":", lw=1.3, label=label)


# ---------------------------------------------------------------- panel drawers
# Each drawer takes an Axes so single figures and the combined panel share code.

def draw_value(ax, cfg, rule, res: SolveResult, band: Band):
    ax.plot(res.pi, res.V, color=RULE_COLOR[rule])
    shade_band(ax, band, RULE_COLOR[rule], label="product-2 band")
    if rule == "EG":
        mark_pibar(ax, cfg)
    ax.set_xlabel(r"belief $\pi$")
    ax.set_ylabel(r"$V(\pi)$")
    ax.set_title(f"{rule}: seller value")
    ax.set_xlim(0, 1)


def draw_gap(ax, cfg, rule, res: SolveResult, band: Band):
    ax.plot(res.pi, res.g, color=C_GAP,
            label=r"$g(\pi)=\gamma[V(\ell+d_S)-V(\ell+d_F)]$")
    ax.axhline(cfg.threshold, color=INK2, ls="--", lw=1.2,
               label=rf"$\Delta c/\Delta p = {cfg.threshold:.3g}$")
    shade_band(ax, band, RULE_COLOR[rule])
    if rule == "EG":
        mark_pibar(ax, cfg)
    if not band.empty:
        ax.annotate(rf"$[\pi_-,\pi_+]=[{band.pi_lo:.3f},\,{band.pi_hi:.3f}]$",
                    xy=(0.02, 0.96), xycoords="axes fraction", va="top",
                    fontsize=9, color=INK2)
    ax.set_xlabel(r"belief $\pi$")
    ax.set_ylabel(r"$g(\pi)$")
    ax.set_title(f"{rule}: continuation gap vs threshold")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower center")


def draw_policy(ax, cfg, rule, res: SolveResult, band: Band):
    ax.step(res.pi, res.x, where="mid", color=RULE_COLOR[rule])
    shade_band(ax, band, RULE_COLOR[rule])
    if rule == "EG":
        mark_pibar(ax, cfg)
    ax.set_xlabel(r"belief $\pi$")
    ax.set_yticks([1, 2], ["1 (cheap)", "2 (good)"])
    ax.set_ylim(0.8, 2.2)
    ax.set_ylabel(r"$x^*(\pi)$")
    ax.set_title(f"{rule}: optimal product")
    ax.set_xlim(0, 1)


def draw_demand(ax, cfg, rule, res: SolveResult):
    ax.plot(res.pi, res.D, color=RULE_COLOR[rule], label=f"D, {RULE_LABEL[rule]}")
    D_logit = demand(cfg, "LOGIT", res.ell)
    ax.plot(res.pi, D_logit, color=RULE_COLOR["LOGIT"], ls="--", lw=1.4,
            label=rf"logit check ($\beta={cfg.beta:g}$)")
    mark_pibar(ax, cfg, label=r"$\bar\pi(p_0)$: E[quality] $= p_0$")
    if rule == "EG":
        ax.annotate(rf"$\varepsilon = {cfg.eps:g}$", xy=(0.02, cfg.eps),
                    xytext=(0.04, cfg.eps + 0.06), fontsize=9, color=INK2)
    ax.set_xlabel(r"belief $\pi$")
    ax.set_ylabel(r"$D(\pi)$")
    ax.set_title(f"{rule}: engagement probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left" if rule == "EG" else "lower right")


def draw_paths(ax, cfg, rule, mc: MCResult, band: Band, per_start: int = 1,
               t_max: int | None = 200):
    """A few paths per start, zoomed to the transient so the Bayes steps and the
    parking/ruin dynamics are actually legible (the full horizon is summarized
    by draw_fractions instead)."""
    T = mc.pi_paths.shape[0] if t_max is None else min(t_max, mc.pi_paths.shape[0])
    for s, pi0 in enumerate(mc.starts):
        cols = np.flatnonzero(mc.start_id == s)[:per_start]
        c = START_COLORS[s % len(START_COLORS)]
        for k, col in enumerate(cols):
            ax.plot(np.arange(T), mc.pi_paths[:T, col], color=c, lw=1.0,
                    alpha=0.85, label=rf"$\pi_0={pi0:g}$" if k == 0 else None)
    shade_band(ax, band, RULE_COLOR[rule], axis="y")
    if rule == "EG":
        mark_pibar(ax, cfg, axis="y")
    ax.set_xlabel("period $t$")
    ax.set_ylabel(r"$\pi_t$")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{rule}: sample belief paths (first {T - 1} periods, "
                 f"{per_start} per start)")
    ax.legend(loc="lower left", ncols=2, frameon=True, facecolor=SURFACE,
              framealpha=0.9, edgecolor=GRIDC)


def draw_paths_facets(fig, spec, cfg, rule, mc: MCResult, band: Band,
                      t_max: int = 200):
    """One mini-panel per start with a single path each — overlaid paths all
    orbit the same attractor and tangle, so facets are the legible option."""
    n = len(mc.starts)
    sub = spec.subgridspec(n, 1, hspace=0.35)
    T = min(t_max, mc.pi_paths.shape[0])
    axes = []
    for s, pi0 in enumerate(mc.starts):
        ax = fig.add_subplot(sub[s])
        col = np.flatnonzero(mc.start_id == s)[0]
        ax.plot(np.arange(T), mc.pi_paths[:T, col],
                color=START_COLORS[s % len(START_COLORS)], lw=1.1)
        shade_band(ax, band, RULE_COLOR[rule], axis="y")
        if rule == "EG":
            mark_pibar(ax, cfg, axis="y", label=None)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        ax.annotate(rf"$\pi_0={pi0:g}$", xy=(0.01, 0.95), xycoords="axes fraction",
                    va="top", fontsize=8, color=INK2,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1))
        if s < n - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[0].set_title(f"{rule}: one sample path per start "
                      f"(first {T - 1} periods)")
    axes[-1].set_xlabel("period $t$")
    axes[n // 2].set_ylabel(r"$\pi_t$")
    return axes


def draw_fractions(ax, cfg, rule, mc: MCResult, band: Band):
    """Ensemble summary over the full horizon: share of paths still alive,
    inside the product-2 band, and above the cliff, period by period."""
    P = mc.pi_paths
    ax.plot((P > 0.05).mean(axis=1), color=RULE_COLOR[rule],
            label=r"alive ($\pi_t > 0.05$)")
    if not band.empty:
        inband = ((P >= band.pi_lo) & (P <= band.pi_hi)).mean(axis=1)
        ax.plot(inband, color=C_GAP, label=r"in band $[\pi_-,\pi_+]$")
    ax.plot((P >= cfg.pibar).mean(axis=1), color=INK2, ls="--", lw=1.4,
            label=r"above cliff ($\pi_t \geq \bar\pi$)")
    ax.set_xlabel("period $t$")
    ax.set_ylabel("fraction of paths")
    ax.set_xlim(0, P.shape[0] - 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{rule}: ensemble over time ({P.shape[1]} paths)")
    ax.legend(loc="center right", frameon=True, facecolor=SURFACE,
              framealpha=0.9, edgecolor=GRIDC)


def draw_hist(ax, cfg, rule, mc: MCResult, band: Band):
    bins = np.linspace(0, 1, 81)
    sample = stationary_sample(mc)
    ax.hist(sample, bins=bins, density=True, color=RULE_COLOR[rule],
            alpha=0.85, label="stationary (last 25%)")
    occ = mc.pi_paths[50:].ravel()
    occ = occ[occ > 0.05]  # occupation before/without reputational ruin
    if occ.size:
        ax.hist(occ, bins=bins, density=True, histtype="step", color=C_GAP,
                lw=1.4, label=r"occupation | $\pi>0.05$ ($t>50$)")
    shade_band(ax, band, RULE_COLOR[rule])
    if rule == "EG":
        mark_pibar(ax, cfg)
    ax.legend(loc="upper right")
    ax.set_xlabel(r"belief $\pi$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_title(f"{rule}: distribution of $\\pi$ under $x^*$")


def draw_fluid(ax, cfg, rule, res: SolveResult, band: Band):
    starts = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
    n, traj = fluid_trajectories(cfg, res, starts)
    for s, pi0 in enumerate(starts):
        ax.plot(n, traj[:, s], color=START_COLORS[s % len(START_COLORS)],
                lw=1.4, label=rf"$\pi_0={pi0:g}$")
    shade_band(ax, band, RULE_COLOR[rule], axis="y")
    if rule == "EG":
        mark_pibar(ax, cfg, axis="y")
    ax.set_xlabel("observation time $n$")
    ax.set_ylabel(r"$\pi(n)$")
    ax.set_ylim(0, 1)
    ax.set_title(f"{rule}: fluid ODE under relaxed control $a^*(\\ell)$")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))


# ---------------------------------------------------------------- figures 1-4 (+fluid)

def per_rule_figures(cfg, rule, res, band, mc, outdir: Path):
    """Figures 1-4 + fluid + combined panel, saved under <outdir>/<rule>/."""
    rdir = outdir / rule
    rdir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.6))
    draw_value(ax1, cfg, rule, res, band)
    ax1.set_xlabel("")
    draw_gap(ax2, cfg, rule, res, band)
    fig.suptitle(f"{RULE_LABEL[rule]} — value and continuation gap "
                 f"($\\gamma={cfg.gamma}$)", fontsize=12, color=INK)
    save(fig, rdir, "value_and_gap.png")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    draw_policy(ax, cfg, rule, res, band)
    save(fig, rdir, "policy.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_demand(ax, cfg, rule, res)
    save(fig, rdir, "demand.png")

    fig = plt.figure(figsize=(17.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.22)
    draw_paths_facets(fig, gs[0], cfg, rule, mc, band)
    draw_fractions(fig.add_subplot(gs[1]), cfg, rule, mc, band)
    draw_hist(fig.add_subplot(gs[2]), cfg, rule, mc, band)
    fig.suptitle(f"{RULE_LABEL[rule]} — Monte Carlo under $x^*$ "
                 f"({mc.pi_paths.shape[1]} paths, {cfg.n_periods} periods)",
                 fontsize=12, color=INK)
    save(fig, rdir, "montecarlo.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    draw_fluid(ax, cfg, rule, res, band)
    save(fig, rdir, "fluid_check.png")

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.5))
    draw_value(axes[0, 0], cfg, rule, res, band)
    draw_gap(axes[0, 1], cfg, rule, res, band)
    draw_policy(axes[1, 0], cfg, rule, res, band)
    draw_demand(axes[1, 1], cfg, rule, res)
    draw_paths(axes[2, 0], cfg, rule, mc, band)
    draw_hist(axes[2, 1], cfg, rule, mc, band)
    fig.suptitle(f"{RULE_LABEL[rule]} — combined panel ($\\gamma={cfg.gamma}$)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, rdir, "combined_panel.png")


# ---------------------------------------------------------------- figure 5: band vs p0

def fig_band_vs_p0(cfg, outdir: Path):
    p0s = np.linspace(cfg.p1 + 0.02, cfg.p2 - 0.02, 25)
    # TS: p0 does not enter the TS HJB -> solve once, band is p0-free by construction
    ts_band = extract_band(cfg, solve_dp(cfg, "TS"))
    curves = {}
    for rule in ("EG", "LOGIT"):   # p0 enters both: cliff / smooth cliff
        lo, hi, V0 = [], [], None
        for p0 in p0s:
            c = replace(cfg, p0=float(p0))
            r = solve_dp(c, rule, V0=V0)
            V0 = r.V
            b = extract_band(c, r)
            lo.append(b.pi_lo)
            hi.append(b.pi_hi)
        curves[rule] = (np.array(lo), np.array(hi))
    pibars = (p0s - cfg.p1) / cfg.dp

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cts = RULE_COLOR["TS"]
    if ts_band.empty:
        ax.annotate("TS band empty at this $\\gamma$", xy=(0.02, 0.02),
                    xycoords="axes fraction", color=INK2, fontsize=9)
    else:
        ax.fill_between(p0s, ts_band.pi_lo, ts_band.pi_hi, color=cts,
                        alpha=BAND_ALPHA)
        ax.plot(p0s, np.full_like(p0s, ts_band.pi_lo), color=cts, lw=1.6)
        ax.plot(p0s, np.full_like(p0s, ts_band.pi_hi), color=cts, lw=1.6,
                label=r"TS band $[\pi_-,\pi_+]$ ($p_0$-free)")
    for rule, ls in (("EG", "-"), ("LOGIT", "-")):
        lo, hi = curves[rule]
        c = RULE_COLOR[rule]
        ax.fill_between(p0s, lo, hi, color=c, alpha=BAND_ALPHA)
        ax.plot(p0s, lo, color=c, lw=1.6, ls=ls)
        ax.plot(p0s, hi, color=c, lw=1.6, ls=ls,
                label=rf"{rule} band $[\pi_-,\pi_+]$")
    ax.plot(p0s, pibars, color=INK2, ls=(0, (4, 3)), lw=1.2,
            label=r"cliff $\bar\pi(p_0)=(p_0-p_1)/\Delta p$")
    ax.set_xlabel(r"outside option $p_0$")
    ax.set_ylabel(r"belief $\pi$")
    ax.set_ylim(0, 1)
    ax.set_title(rf"Product-2 band vs $p_0$  ($\gamma={cfg.gamma}$)")
    ax.legend(loc="lower right")
    save(fig, outdir, "band_vs_p0.png")


# ---------------------------------------------------------------- figure 6: band vs params

def _band_edges_sweep(cfgs, rule):
    lo, hi = [], []
    V0 = None
    for c in cfgs:
        r = solve_dp(c, rule, V0=V0)
        V0 = r.V
        b = extract_band(c, r)
        lo.append(b.pi_lo)
        hi.append(b.pi_hi)
    return np.array(lo), np.array(hi)


def _sweep_panel(ax, xvals, cfgs, rules, xlabel, base_x=None):
    for rule in rules:
        lo, hi = _band_edges_sweep(cfgs, rule)
        c = RULE_COLOR[rule]
        ax.fill_between(xvals, lo, hi, color=c, alpha=BAND_ALPHA)
        ax.plot(xvals, lo, color=c, lw=1.6)
        ax.plot(xvals, hi, color=c, lw=1.6, label=rule)
    if base_x is not None:
        ax.axvline(base_x, color=MUTED, ls=":", lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$[\pi_-,\ \pi_+]$")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")


def fig_band_vs_params(cfg, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    gammas = np.array([0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99])
    _sweep_panel(axes[0, 0], gammas, [replace(cfg, gamma=float(g)) for g in gammas],
                 ("TS", "EG", "LOGIT"), r"discount $\gamma$",
                 base_x=min(cfg.gamma, 0.99))
    axes[0, 0].set_title("vs discounting")

    epss = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    _sweep_panel(axes[0, 1], epss, [replace(cfg, eps=float(e)) for e in epss],
                 ("EG",), r"exploration $\varepsilon$ (EG only)", base_x=cfg.eps)
    axes[0, 1].axhline(cfg.pibar, color=INK2, ls=(0, (4, 3)), lw=1.1)
    axes[0, 1].annotate(r"$\bar\pi$", xy=(epss[-1], cfg.pibar),
                        xytext=(epss[-1] - 0.02, cfg.pibar + 0.03),
                        color=INK2, fontsize=9)
    axes[0, 1].set_title("vs exploration rate")

    ratios = np.array([0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40,
                       1.60, 1.80, 2.00])
    _sweep_panel(axes[1, 0], ratios,
                 [replace(cfg, c2=cfg.c1 + float(t) * cfg.dp) for t in ratios],
                 ("TS", "EG", "LOGIT"), r"cost/quality ratio $\Delta c/\Delta p$",
                 base_x=cfg.threshold)
    axes[1, 0].set_title("vs relative cost of quality")

    mid = 0.5 * (cfg.p1 + cfg.p2)
    widths = np.linspace(0.16, 0.50, 8)
    lam_cfgs, lams = [], []
    for w in widths:
        p1w, p2w = mid - w / 2.0, mid + w / 2.0
        c = replace(cfg, p1=float(p1w), p2=float(p2w),
                    c2=cfg.c1 + cfg.threshold * float(p2w - p1w))
        lam_cfgs.append(c)
        lams.append(c.Lambda)
    _sweep_panel(axes[1, 1], np.array(lams), lam_cfgs, ("TS", "EG", "LOGIT"),
                 r"distinguishability $\Lambda = d_S - d_F$", base_x=cfg.Lambda)
    axes[1, 1].set_title(r"vs distinguishability ($\Delta c/\Delta p$ held fixed)")

    fig.suptitle(rf"Product-2 band $[\pi_-,\pi_+]$ — comparative statics "
                 rf"(base: $\gamma={cfg.gamma}$, $\varepsilon={cfg.eps}$, "
                 rf"$\Delta c/\Delta p={cfg.threshold:.2f}$)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, outdir, "band_vs_params.png")


# ------------------------------------------------- extra experiment: LOGIT beta

def fig_logit_beta_sweep(cfg, bands, outdir: Path):
    """How the smooth-logit band converges to the epsilon-greedy cliff as the
    demand slope beta grows (beta -> inf is the eps = 0 hard cliff)."""
    betas = np.array([2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0])
    lo, hi, V0 = [], [], None
    for b in betas:
        c = replace(cfg, beta=float(b))
        r = solve_dp(c, "LOGIT", V0=V0)
        V0 = r.V
        bd = extract_band(c, r)
        lo.append(bd.pi_lo)
        hi.append(bd.pi_hi)
    lo, hi = np.array(lo), np.array(hi)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    c_lg = RULE_COLOR["LOGIT"]
    ax.fill_between(betas, lo, hi, color=c_lg, alpha=BAND_ALPHA)
    ax.plot(betas, lo, color=c_lg, lw=1.6)
    ax.plot(betas, hi, color=c_lg, lw=1.6, label=r"LOGIT band $[\pi_-,\pi_+]$")
    eg = bands["EG"]
    if not eg.empty:
        ax.axhline(eg.pi_lo, color=RULE_COLOR["EG"], ls="--", lw=1.3)
        ax.axhline(eg.pi_hi, color=RULE_COLOR["EG"], ls="--", lw=1.3,
                   label=rf"EG band edges ($\varepsilon={cfg.eps:g}$)")
    mark_pibar(ax, cfg, axis="y")
    ax.axvline(cfg.beta, color=MUTED, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"demand slope $\beta$ (log scale)")
    ax.set_ylabel(r"$[\pi_-,\ \pi_+]$")
    ax.set_ylim(0, 1)
    ax.set_title(rf"LOGIT band vs $\beta$ — sharpening toward the EG cliff "
                 rf"($\gamma={cfg.gamma}$)")
    ax.legend(loc="best")
    rdir = outdir / "LOGIT"
    rdir.mkdir(parents=True, exist_ok=True)
    save(fig, rdir, "beta_sweep.png")


# ---------------------------------------------------------------- sanity checks

def run_checks(cfg, results, bands, mcs):
    print("  sanity checks:")
    failures = []

    def check(name, ok, detail=""):
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def warn(name, ok, detail=""):
        print(f"    [{'PASS' if ok else 'WARN'}] {name}"
              + (f" — {detail}" if detail else ""))

    for rule, res in results.items():
        check(f"{rule}: value iteration converged", res.converged,
              f"{res.iterations} iters, final sup-diff {res.final_diff:.2e} < {cfg.tol:g}")
        check(f"{rule}: threshold policy == argmax policy (constant dc/dp = "
              f"{cfg.threshold:.4f})", res.policy_consistent)
        b = bands[rule]
        warn(f"{rule}: product-2 region is a single interval",
             b.is_interval,
             "empty band" if b.empty else f"{b.n_components} component(s)")
    b = bands["EG"]
    if b.empty:
        warn("EG: band brackets the cliff pibar", False,
             "band is empty at this gamma / dc/dp — never worth paying for quality")
    else:
        warn("EG: band brackets the cliff pibar",
             b.pi_lo <= cfg.pibar <= b.pi_hi,
             f"pi_- = {b.pi_lo:.4f} vs pibar = {cfg.pibar:.4f} vs pi_+ = {b.pi_hi:.4f}")

    # Phenomenology (economic expectations, reported but non-fatal): the belief
    # parks at the UPPER band edge pi_+ (above it the seller stops paying dc and
    # the belief drifts back down). "Parking near pibar" therefore holds exactly
    # when pi_+ is close to the cliff, i.e. when the insurance buffer
    # (log-odds distance from pibar to pi_+, in units of one failure |dF|)
    # is small — which happens for moderate gamma, not at gamma ~ 1.
    for rule, mc in mcs.items():
        s = stationary_sample(mc)
        alive = s > 0.05          # paths not absorbed in reputational ruin
        ruin = 1.0 - float(np.mean(alive))
        med = float(np.median(s))
        bb = bands[rule]
        if bb.empty:
            print(f"    [info] {rule}: band empty — seller never invests; "
                  f"ruin P(pi < 0.05) = {ruin:.2f}, median pi = {med:.3f}")
            continue
        near_top = (float(np.mean(np.abs(s[alive] - bb.pi_hi) <= 0.10))
                    if alive.any() else 0.0)
        warn(f"{rule}: surviving MC mass parks at the upper band edge pi_+",
             near_top >= 0.50,
             f"P(|pi - pi_+| <= 0.10 | alive) = {near_top:.2f}, "
             f"ruin P(pi < 0.05) = {ruin:.2f}, median pi = {med:.3f}")
        if rule == "EG":
            near_bar = (float(np.mean(np.abs(s[alive] - cfg.pibar) <= 0.15))
                        if alive.any() else 0.0)
            buffer_outcomes = (float(np.log(bb.pi_hi / (1 - bb.pi_hi)))
                               - float(np.log(cfg.pibar / (1 - cfg.pibar)))) / abs(cfg.dF)
            ok = buffer_outcomes <= 1.5 and (not alive.any() or near_bar >= 0.30)
            detail = (f"P(|pi - pibar| <= 0.15 | alive) = {near_bar:.2f}; buffer "
                      f"pi_+ over pibar = {buffer_outcomes:.1f} failures in log-odds")
            if buffer_outcomes > 1.5:
                detail += (" (deep reputation buffer at this gamma: the seller "
                           "parks the user well above the cliff; see the "
                           "gamma=0.70 config for cliff-hugging)")
            warn("EG: surviving MC mass parks near the cliff pibar", ok, detail)
    return failures


# ---------------------------------------------------------------- driver

def run_config(cfg: Config, do_sweeps: bool):
    outdir = BASE / cfg.outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== config: gamma={cfg.gamma}, p=({cfg.p1},{cfg.p0},{cfg.p2}), "
          f"dc/dp={cfg.threshold:.3f}, pibar={cfg.pibar:.3f}, "
          f"dS={cfg.dS:.4f}, dF={cfg.dF:.4f} -> {cfg.outdir}/ ===")

    results, bands, mcs = {}, {}, {}
    for rule in ("TS", "EG", "LOGIT"):
        t0 = time.perf_counter()
        res = solve_dp(cfg, rule)
        band = extract_band(cfg, res)
        results[rule], bands[rule] = res, band
        edge = ("(empty)" if band.empty
                else f"[pi_-, pi_+] = [{band.pi_lo:.4f}, {band.pi_hi:.4f}]")
        print(f"  {rule:5s} band {edge}   "
              f"({res.iterations} iters, {time.perf_counter() - t0:.1f}s)")

    for rule, offset in (("TS", 0), ("EG", 1), ("LOGIT", 2)):
        mcs[rule] = simulate(cfg, rule, results[rule], seed_offset=offset)
        print(f"  {rule:5s} MC engaged fraction = {mcs[rule].engaged_frac:.3f}")

    failures = run_checks(cfg, results, bands, mcs)

    for rule in ("TS", "EG", "LOGIT"):
        per_rule_figures(cfg, rule, results[rule], bands[rule], mcs[rule], outdir)

    if do_sweeps:
        print("  sweeps (figures 5-6 + LOGIT beta experiment):")
        fig_band_vs_p0(cfg, outdir)
        fig_band_vs_params(cfg, outdir)
        fig_logit_beta_sweep(cfg, bands, outdir)
    else:
        print("  sweeps skipped for this config (pass --sweeps-all to include)")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps-all", action="store_true",
                    help="run comparative-statics sweeps for every config "
                         "(default: base config only)")
    args = ap.parse_args()

    setup_style()
    all_failures = []
    for cfg in CONFIGS:
        do_sweeps = args.sweeps_all or cfg.gamma <= 0.99  # 0.999 sweeps are slow
        failures = run_config(cfg, do_sweeps=do_sweeps)
        all_failures += [f"[{cfg.outdir}] {f}" for f in failures]

    print()
    if all_failures:
        print("FAILED checks:")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
