"""Show actual action advantages, not fitted bands, on a saved fixed-n slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from figures import STYLE, COLORS, _save
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=float, default=20.)
    parser.add_argument("--baseline-dir", type=Path, default=root / "outputs/p0_comparison")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/interval_search/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(args.baseline_dir.glob("p0_*/data/solution.npz"))
    if not 1 <= len(cases) <= 5:
        raise ValueError("Expected one to five saved p0 scenarios")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, path in zip(axes.flat, cases):
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"]))
                level = round(args.n/meta["h"])
                if not np.isclose(level*meta["h"], args.n) or args.n > meta["report_n"]:
                    raise ValueError("Requested n must be a saved grid level")
                start, end = saved["row_offsets"][[level, level+1]]
                gap = saved["advantages"][start:end]
            x = (np.arange(level+1)*meta["h"]+1)/(args.n+2)
            ax.axhline(0, color="#64717C", linewidth=.8)
            ax.fill_between(x, 0, gap, where=gap > 0, color=COLORS[2], alpha=.6, interpolate=True)
            ax.fill_between(x, 0, gap, where=gap < 0, color=COLORS[1], alpha=.25, interpolate=True)
            ax.plot(x, gap, color="#263642", linewidth=1.3)
            ax.set(title=f"p₀={meta['parameters']['p0']:g}", xlim=(0, 1),
                   xlabel="Media posterior x", ylabel="Ventaja Aₕ del producto 2")
            ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_axis_off()
        axes.flat[5].set_axis_off()
        axes.flat[5].text(0, .9,
            "Criterio de contraejemplo\n\n"
            "En la misma diagonal n buscamos\n"
            "x₁ < x₂ < x₃ con signos +, −, +.\n\n"
            "Naranja: producto 2 preferido.\n"
            "Verde: producto 1 preferido.\n\n"
            "Estas curvas son casos de referencia;\n"
            "no representan todas las calibraciones\n"
            "del barrido ni prueban un teorema.",
            transform=axes.flat[5].transAxes, va="top", fontsize=11, linespacing=1.45)
        fig.suptitle(f"Qué se comprueba en una sección de n=S+F={args.n:g}", fontsize=16, fontweight="semibold")
        fig.supxlabel("Calibración base: p₁=.35, p₂=.8, c₁=.05, c₂=.65, R=1, γ=.98; h=.125. Datos completos de la solución guardada.",
                      fontsize=9, color="#52616B")
        print(_save(fig, args.output_dir, "advantage_sections"))


if __name__ == "__main__":
    main()
