#!/usr/bin/env python3
"""DECISION experiment on the 1-D logit reputation model (solver unchanged).

Question: does the investment band become INTERIOR, and does its center
TRACK the demand center pibar(mu), once the seller is impatient?

Sweep: gamma in {0.50 ... 0.95} x mu (12 points) x two quality-gap cases
(144 solves, existing solver from src/model.py, tolerance 1e-10). Results,
plots and SUMMARY.md (with an explicit PASS/FAIL verdict) go to
outputs/decision_sweep/.

Usage:
    python run_decision_sweep.py

Deterministic (no randomness involved); matplotlib only.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from config import Config
from src.model import solve_dp

OUT = BASE / "outputs" / "decision_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
# (same validated reference palette as run_all.py)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDC = "#e1e0d9"
AXISC = "#c3c2b7"
SURFACE = "#fcfcfb"
C_BLUE = "#2a78d6"    # slot 1: WIDE gap case
C_AQUA = "#1baf7a"    # slot 2: NARROW gap case
C_VIOLET = "#4a3aa7"  # slot 5: band center
BAND_ALPHA = 0.13
CMAP_WIDTH = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

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


def _ax_clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------- sweep spec
# Fixed economics: R = 1, c1 = 0.05, c2 = 0.65 => dc = 0.6; beta = 8.
# Quality-gap cases: in the NARROW case both the threshold dc/dp rises
# (0.6/0.2 = 3.0 vs 0.6/0.5 = 1.2) AND the belief steps dS, dF shrink
# (dS = log(0.65/0.45) ~ 0.37 vs log(0.8/0.3) ~ 0.98); both effects should
# shrink the band. NARROW uses a tighter mu grid so pibar = (mu-p1)/dp stays
# inside (0,1).
GAMMAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CASES = {
    "WIDE": dict(p1=0.3, p2=0.8, mus=np.linspace(0.32, 0.78, 12),
                 color=C_BLUE),
    "NARROW": dict(p1=0.45, p2=0.65, mus=np.linspace(0.47, 0.63, 12),
                   color=C_AQUA),
}
INTERIOR_LO, INTERIOR_HI = 0.02, 0.98

COLUMNS = ["gamma", "mu", "gap_case", "pibar", "band_exists", "pi_lo",
           "pi_hi", "center", "width", "interior", "single_interval"]


def run_sweep():
    """144 solves; returns list of row dicts and count of non-convergences."""
    rows, nonconv = [], []
    for case, spec in CASES.items():
        base = replace(Config(), p1=spec["p1"], p2=spec["p2"])
        assert (base.R, base.c1, base.c2, base.beta) == (1.0, 0.05, 0.65, 8.0)
        for gamma in GAMMAS:
            for mu in spec["mus"]:
                cfg = replace(base, gamma=gamma, mu=float(mu))
                res = solve_dp(cfg)
                if not res.converged:
                    nonconv.append(f"{case} gamma={gamma} mu={mu:.3f}: "
                                   f"sup_err={res.sup_err:.2e}")
                b = res.band
                interior = (not b.empty and b.lo > INTERIOR_LO
                            and b.hi < INTERIOR_HI)
                rows.append(dict(
                    gamma=gamma, mu=float(mu), gap_case=case,
                    pibar=cfg.pibar, band_exists=not b.empty,
                    pi_lo=b.lo, pi_hi=b.hi, center=b.center, width=b.width,
                    interior=interior, single_interval=b.single_interval))
    return rows, nonconv


def write_csv(rows):
    path = OUT / "decision_sweep.csv"
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                             else str(r[c]) for c in COLUMNS) + "\n")
    print(f"  wrote {path.relative_to(BASE)}")


def _case_rows(rows, case, gamma=None):
    out = [r for r in rows if r["gap_case"] == case]
    if gamma is not None:
        out = [r for r in out if r["gamma"] == gamma]
    return out


# ---------------------------------------------------------------- P1
def p1_band_maps(rows):
    """One figure per gap case, one panel per gamma: endpoints, center,
    and the reference pibar(mu). The eyeball version of the decision."""
    for case, spec in CASES.items():
        col = spec["color"]
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True,
                                 sharey=True)
        for ax, gamma in zip(axes.ravel(), GAMMAS):
            rs = _case_rows(rows, case, gamma)
            mu = np.array([r["mu"] for r in rs])
            lo = np.array([r["pi_lo"] for r in rs])
            hi = np.array([r["pi_hi"] for r in rs])
            ctr = np.array([r["center"] for r in rs])
            pibar = np.array([r["pibar"] for r in rs])
            n_bands = int(np.sum([r["band_exists"] for r in rs]))
            ax.fill_between(mu, lo, hi, color=col, alpha=BAND_ALPHA, lw=0)
            ax.plot(mu, lo, color=col, lw=1.6, label="band endpoints")
            ax.plot(mu, hi, color=col, lw=1.6)
            ax.plot(mu, ctr, color=C_VIOLET, lw=1.8, label="band center")
            ax.plot(mu, pibar, color=INK, ls="--", lw=1.3,
                    label="pibar(mu)")
            if n_bands == 0:
                ax.annotate("no band\n(x*=1 everywhere)",
                            (mu.mean(), 0.5), ha="center", va="center",
                            color=MUTED, fontsize=10)
            ax.set(title=f"gamma = {gamma:g}   ({n_bands}/12 bands)",
                   ylim=(0, 1))
            _ax_clean(ax)
        for ax in axes[1]:
            ax.set_xlabel("outside-option center mu")
        for ax in axes[:, 0]:
            ax.set_ylabel("belief pi")
        axes[0][0].legend(loc="upper left")
        p = spec
        fig.suptitle(f"P1  Band maps, {case} gap: p1={p['p1']:g}, "
                     f"p2={p['p2']:g} (dc/dp = {0.6/(p['p2']-p['p1']):g})   "
                     f"[R=1, c1=0.05, c2=0.65, beta=8]",
                     fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(OUT / f"P1_band_maps_{case}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote decision_sweep/P1_band_maps_{case}.png")


# ---------------------------------------------------------------- P2
def tracking_stats(rows):
    """kappa(gamma) per gap case: OLS slope of band center on pibar(mu)
    across the 12 mus (nonempty bands only), plus R^2 and cell flags."""
    cells = {}
    for case in CASES:
        for gamma in GAMMAS:
            rs = _case_rows(rows, case, gamma)
            ok = [r for r in rs if r["band_exists"]]
            x = np.array([r["pibar"] for r in ok])
            y = np.array([r["center"] for r in ok])
            if len(ok) >= 3 and np.ptp(x) > 1e-12:
                kappa, b0 = np.polyfit(x, y, 1)
                yhat = kappa * x + b0
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                kappa, r2 = np.nan, np.nan
            cells[(case, gamma)] = dict(
                kappa=float(kappa), r2=float(r2), n_bands=len(ok),
                all_interior=all(r["interior"] for r in rs) and len(ok) == 12,
                violations=sum(not r["single_interval"] for r in rs))
    return cells


def p2_tracking_plot(cells):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axhline(0.0, color=MUTED, ls=":", lw=1.0)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    ax.axhline(0.5, color=INK, ls="--", lw=1.2, label="PASS threshold 0.5")
    for case, spec in CASES.items():
        g = np.array(GAMMAS)
        k = np.array([cells[(case, gamma)]["kappa"] for gamma in GAMMAS])
        ax.plot(g, k, color=spec["color"], marker="o", ms=5,
                label=f"{case} gap (dc/dp = "
                      f"{0.6/(spec['p2']-spec['p1']):g})")
        for gi, ki in zip(g, k):     # mark undefined cells (no band)
            if np.isnan(ki):
                ax.plot(gi, 0, marker="x", color=spec["color"], ms=7,
                        mew=1.6)
    ax.annotate("x = no band (kappa undefined)", (GAMMAS[0], 0.03),
                color=MUTED, fontsize=8.5)
    ax.set(xlabel="discount factor gamma",
           ylabel="tracking coefficient kappa",
           title="P2  kappa(gamma): slope of band center on pibar(mu)   "
                 "[R=1, c1=0.05, c2=0.65, beta=8]")
    ax.legend(loc="best")
    _ax_clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "P2_tracking_kappa.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote decision_sweep/P2_tracking_kappa.png")


# ---------------------------------------------------------------- P3
def p3_interiority_maps(rows):
    """Heatmap of band width over (gamma, mu) per gap case; 'x' marks cells
    with NO band, open triangles mark bands that touch the grid edges."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, (case, spec) in zip(axes, CASES.items()):
        mus = spec["mus"]
        W = np.full((len(GAMMAS), len(mus)), np.nan)
        for i, gamma in enumerate(GAMMAS):
            for j, mu in enumerate(mus):
                r = next(r for r in _case_rows(rows, case, gamma)
                         if abs(r["mu"] - mu) < 1e-12)
                W[i, j] = r["width"]
                if not r["band_exists"]:
                    ax.plot(mu, gamma, marker="x", color=MUTED, ms=7,
                            mew=1.4)
                elif not r["interior"]:
                    ax.plot(mu, gamma, marker="^", mfc="none", mec=INK,
                            ms=7, mew=1.2)
        dmu = mus[1] - mus[0]
        pc = ax.pcolormesh(
            np.append(mus - dmu / 2, mus[-1] + dmu / 2),
            [0.45, 0.55, 0.65, 0.75, 0.85, 0.925, 0.975],
            W, cmap=CMAP_WIDTH, vmin=0.0, vmax=1.0, shading="flat")
        fig.colorbar(pc, ax=ax, label="band width")
        ax.set(xlabel="outside-option center mu",
               ylabel="discount factor gamma",
               yticks=GAMMAS,
               title=f"{case} gap (dc/dp = "
                     f"{0.6/(spec['p2']-spec['p1']):g})")
        ax.grid(False)
        _ax_clean(ax)
    fig.suptitle("P3  Interiority map: band width over (gamma, mu); "
                 "x = no band, open triangle = band touches grid edge   "
                 "[R=1, c1=0.05, c2=0.65, beta=8]",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "P3_interiority_map.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote decision_sweep/P3_interiority_map.png")


# ---------------------------------------------------------------- verdict
def decide(cells):
    """Apply the decision rule; return verdict bool, passing cells, best
    cell, and the kappa-vs-gamma trend per case."""
    passing = [(case, gamma) for (case, gamma), c in cells.items()
               if c["all_interior"] and not np.isnan(c["kappa"])
               and c["kappa"] >= 0.5 and c["r2"] >= 0.8]

    def rank(key):
        c = cells[key]
        return (c["r2"] >= 0.8, c["kappa"])   # prefer meaningful fits

    defined = [k for k, c in cells.items() if not np.isnan(c["kappa"])]
    best = max(defined, key=rank) if defined else None

    trend = {}
    for case in CASES:
        pts = [(g, cells[(case, g)]["kappa"]) for g in GAMMAS
               if not np.isnan(cells[(case, g)]["kappa"])]
        if len(pts) >= 2:
            slope = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
            trend[case] = float(slope)   # negative => kappa rises as gamma falls
        else:
            trend[case] = float("nan")
    return bool(passing), passing, best, trend


def write_summary(rows, cells, passing, verdict, best, trend, nonconv):
    total_viol = sum(not r["single_interval"] for r in rows)
    L = ["# Decision sweep: interior band + tracking of pibar(mu)?", ""]
    L.append(f"## VERDICT: **{'PASS' if verdict else 'FAIL'}**")
    L.append("")
    L.append("Rule: PASS iff some gamma has, in at least one gap case, "
             "(i) an interior band (pi- > 0.02, pi+ < 0.98) for all 12 mus "
             "AND (ii) kappa >= 0.5 with R^2 >= 0.8.")
    if passing:
        L.append("Passing cells: " + ", ".join(
            f"{case} gamma={g:g} (kappa={cells[(case, g)]['kappa']:.3f}, "
            f"R^2={cells[(case, g)]['r2']:.3f})" for case, g in passing))
    if best:
        bc = cells[best]
        L.append(f"Best cell: {best[0]} gamma={best[1]:g} — kappa = "
                 f"{bc['kappa']:.3f}, R^2 = {bc['r2']:.3f}, "
                 f"{bc['n_bands']}/12 bands, "
                 f"{'all interior' if bc['all_interior'] else 'NOT all interior'}.")
    else:
        L.append("No cell has a defined kappa (no bands anywhere).")
    for case in CASES:
        t = trend[case]
        n_pts = sum(1 for g in GAMMAS
                    if not np.isnan(cells[(case, g)]["kappa"]))
        if np.isnan(t):
            L.append(f"Trend ({case}): kappa defined for <2 gammas — "
                     "no trend to report.")
        else:
            L.append(f"Trend ({case}): d kappa / d gamma = {t:+.2f} over "
                     f"{n_pts} defined point(s) — kappa "
                     + ("INCREASES as gamma falls" if t < 0 else
                        "does NOT increase as gamma falls")
                     + (" (thin evidence: only 2 points)." if n_pts == 2
                        else "."))
    L += ["", "Fixed: R=1, c1=0.05, c2=0.65 (dc=0.6), beta=8; grid N=3000, "
          "tol 1e-10. WIDE: (p1,p2)=(0.3,0.8), dc/dp=1.2, mu in "
          "[0.32,0.78]. NARROW: (p1,p2)=(0.45,0.65), dc/dp=3.0, mu in "
          "[0.47,0.63]. 144 solves.", ""]

    L.append("## kappa table (R^2) — gamma x gap case")
    L.append("")
    L.append("| gamma | " + " | ".join(CASES) + " |")
    L.append("|---|" + "---|" * len(CASES))
    for g in GAMMAS:
        row = []
        for case in CASES:
            c = cells[(case, g)]
            if np.isnan(c["kappa"]):
                row.append(f"no band ({c['n_bands']}/12)")
            else:
                mark = " **interior**" if c["all_interior"] else ""
                row.append(f"{c['kappa']:.3f} ({c['r2']:.3f}){mark}")
        L.append(f"| {g:g} | " + " | ".join(row) + " |")
    L.append("")
    L.append("Interior cells (band interior for all 12 mus): " + (", ".join(
        f"{case} gamma={g:g}" for case in CASES for g in GAMMAS
        if cells[(case, g)]["all_interior"]) or "none"))
    L.append("")
    L.append(f"Single-interval violations: {total_viol}/{len(rows)}"
             + (" — **FLAGGED, see decision_sweep.csv**" if total_viol
                else "."))
    if nonconv:
        L.append(f"**WARNING: {len(nonconv)} non-converged solve(s):** "
                 + "; ".join(nonconv))
    else:
        L.append("Value iteration reached tolerance 1e-10 in all 144 solves.")

    # three one-line takeaways, filled from the data
    L += ["", "## Takeaways"]
    n_exist = {case: sum(r["band_exists"] for r in _case_rows(rows, case))
               for case in CASES}
    L.append(f"1. Band existence needs patience: WIDE has bands in "
             f"{n_exist['WIDE']}/72 solves, NARROW in {n_exist['NARROW']}/72 "
             f"— impatient sellers (low gamma) never invest, so 'impatience "
             f"buys localization' only works while the band survives.")
    if best:
        bc = cells[best]
        L.append(f"2. Tracking is strongest at {best[0]} gamma={best[1]:g} "
                 f"(kappa={bc['kappa']:.2f}, R^2={bc['r2']:.2f}); the PASS "
                 f"bar (kappa>=0.5, R^2>=0.8, interior) is "
                 f"{'met' if verdict else 'NOT met'} on this grid.")
    else:
        L.append("2. No tracking statistic is defined anywhere — the band "
                 "never exists on this grid.")
    wide_t = trend.get("WIDE", float("nan"))
    L.append(f"3. In the WIDE case kappa "
             + ("rises as gamma falls toward the existence edge — the "
                "interior/tracking regime is a narrow patience window "
                "just above the no-investment region."
                if (not np.isnan(wide_t) and wide_t < 0) else
                "does not rise as gamma falls — patience does not trade "
                "off against localization here.")
             + " The NARROW gap (higher threshold, smaller belief steps) "
             + ("kills the band entirely." if n_exist["NARROW"] == 0 else
                "shrinks the band as predicted."))

    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"  wrote decision_sweep/SUMMARY.md")


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("Decision sweep: 6 gammas x 12 mus x 2 gap cases = 144 solves...")
    rows, nonconv = run_sweep()
    write_csv(rows)
    p1_band_maps(rows)
    cells = tracking_stats(rows)
    p2_tracking_plot(cells)
    p3_interiority_maps(rows)
    verdict, passing, best, trend = decide(cells)
    write_summary(rows, cells, passing, verdict, best, trend, nonconv)
    viol = sum(not r["single_interval"] for r in rows)
    if viol:
        print(f"*** WARNING: {viol} single-interval violation(s) ***")
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}   "
          f"(total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
