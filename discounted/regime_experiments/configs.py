from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    parameter_id: str
    experiment: str
    p0: float
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.98
    representative: bool = False

    @property
    def regime(self) -> str:
        if self.p0 < self.p1:
            return "I: p0<p1<p2"
        if self.p0 < self.p2:
            return "II: p1<p0<p2"
        return "III: p1<p2<p0"


def dense_p0_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(f"A_{i:03d}", "A_dense_p0", round(0.05 + 0.025 * i, 3),
                         representative=round(0.05 + 0.025 * i, 3) in {0.20, 0.50, 0.90})
        for i in range(37)
    ]


def representative_quality_configs() -> list[ExperimentConfig]:
    pairs = [(0.15, 0.45), (0.15, 0.80), (0.35, 0.60),
             (0.35, 0.80), (0.55, 0.80), (0.70, 0.90)]
    out: list[ExperimentConfig] = []
    for j, (p1, p2) in enumerate(pairs):
        gap = p2 - p1
        values = [max(0.03, p1 - 0.15), max(0.02, p1 - 0.03),
                  p1 + 0.15 * gap, (p1 + p2) / 2, p2 - 0.15 * gap,
                  min(0.98, p2 + 0.03), min(0.97, p2 + 0.12)]
        for k, p0 in enumerate(values):
            out.append(ExperimentConfig(
                f"B_{j:02d}_{k:02d}", "B_quality_pairs", round(p0, 4), p1, p2,
                representative=(p1, p2) == (0.35, 0.80) and k in {0, 3, 6},
            ))
    return out


def robustness_configs() -> list[ExperimentConfig]:
    triples = [(0.20, 0.35, 0.80), (0.50, 0.35, 0.80), (0.90, 0.35, 0.80)]
    gammas = [0.80, 0.90, 0.95, 0.98, 0.99]
    costs = [0.20, 0.60, 0.90]
    out: list[ExperimentConfig] = []
    for j, (p0, p1, p2) in enumerate(triples):
        # A stratified, non-Cartesian design: all patience levels at the
        # intermediate cost and all costs at gamma=.90 and .98.
        design = [(g, 0.60) for g in gammas]
        design += [(g, dc) for g in (0.90, 0.98) for dc in costs if dc != 0.60]
        for k, (gamma, dc) in enumerate(design):
            out.append(ExperimentConfig(
                f"C_{j:02d}_{k:02d}", "C_cost_patience", p0, p1, p2,
                0.05, 0.05 + dc, 1.0, gamma,
            ))
    return out


def all_configs() -> list[ExperimentConfig]:
    configs = dense_p0_configs() + representative_quality_configs() + robustness_configs()
    seen: set[tuple] = set()
    unique: list[ExperimentConfig] = []
    for cfg in configs:
        key = (cfg.experiment, cfg.p0, cfg.p1, cfg.p2, cfg.c1, cfg.c2, cfg.gamma)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique
