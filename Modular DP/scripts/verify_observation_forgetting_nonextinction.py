#!/usr/bin/env python3
"""Computer-assisted non-extinction proof for observation-time forgetting.

The normalized boundary ``x+y=1`` is invariant under an observation.  On that
boundary the two-dimensional Bellman equation reduces exactly to one state
coordinate ``x``.  This script encloses the one-dimensional fixed point on
uniform cells, certifies a strict product-2 interval, and constructs a finite
outcome word that maps every prior state into that interval.

The final probabilistic step is analytic: if product 2 were used only finitely
often, outcomes would eventually be i.i.d. Bernoulli(p1).  The certified word
would then occur infinitely often and force another product-2 use, a
contradiction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/modular_dp_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaincc


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.outputs import read_config  # noqa: E402
from modular_dp.primitives import SellerPrimitives  # noqa: E402


@dataclass(frozen=True)
class BoundaryCertificate:
    edges: np.ndarray
    lower_value: np.ndarray
    upper_value: np.ndarray
    lower_advantage: np.ndarray
    upper_advantage: np.ndarray
    iterations: int
    final_enclosure_width: float
    certified_interval: tuple[float, float]
    minimum_certified_advantage: float
    demand_floor: float
    contraction_upper_bound: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "full.json",
    )
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--cells", type=int, default=50_000)
    parser.add_argument("--interval", default="0.33,0.42")
    parser.add_argument("--failures", type=int, default=19)
    parser.add_argument("--successes", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument(
        "--demand-padding",
        type=float,
        default=5e-13,
        help="outward padding applied to SciPy beta-tail evaluations",
    )
    return parser.parse_args()


def _interval(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or not 0.0 < parts[0] < parts[1] < 1.0:
        raise ValueError("--interval must be two ordered values strictly inside (0,1)")
    return parts[0], parts[1]


def boundary_demand(x: np.ndarray, rho: float, p0: float) -> np.ndarray:
    """Thompson demand on the invariant boundary ``x+y=1``."""

    innovation = 1.0 - rho
    alpha = 1.0 + x / innovation
    beta = 1.0 + (1.0 - x) / innovation
    return np.asarray(betaincc(alpha, beta, p0), dtype=float)


def _mapped_cell_extrema(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    take_minimum: bool,
) -> np.ndarray:
    """Enclose a cellwise function over affine images narrower than one cell."""

    cells = len(values)
    scaled_lower = np.nextafter(lower, -np.inf) * cells
    scaled_upper = np.nextafter(upper, np.inf) * cells
    lower_index = np.clip(np.floor(scaled_lower).astype(np.int64), 0, cells - 1)
    upper_index = np.clip(np.floor(scaled_upper).astype(np.int64), 0, cells - 1)
    if np.any(upper_index - lower_index > 1):
        raise RuntimeError("an affine image unexpectedly spans more than two cells")
    first = values[lower_index]
    second = values[upper_index]
    if take_minimum:
        return np.minimum(first, second)
    return np.maximum(first, second)


def solve_boundary_enclosure(
    primitives: SellerPrimitives,
    *,
    rho: float,
    cells: int,
    certified_interval: tuple[float, float],
    tolerance: float,
    max_iterations: int,
    demand_padding: float,
) -> BoundaryCertificate:
    """Enclose the exact boundary Bellman fixed point on uniform cells."""

    if not 0.5 <= rho < 1.0:
        raise ValueError("the synchronizing-word certificate requires 0.5 <= rho < 1")
    if cells < 100:
        raise ValueError("--cells must be at least 100")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    if demand_padding < 0.0:
        raise ValueError("demand padding must be nonnegative")
    if min(primitives.net_rewards) < 0.0:
        raise ValueError(
            "the current product-form enclosure requires nonnegative net rewards"
        )

    edges = np.linspace(0.0, 1.0, cells + 1)
    demand_at_edges = boundary_demand(edges, rho, primitives.p0)
    if np.any(np.diff(demand_at_edges) < -1e-12):
        raise RuntimeError("boundary demand is not numerically monotone")
    demand_lower = np.maximum(0.0, demand_at_edges[:-1] - demand_padding)
    demand_upper = np.minimum(1.0, demand_at_edges[1:] + demand_padding)
    denominator_lower = 1.0 - primitives.gamma * (1.0 - demand_lower)
    denominator_upper = 1.0 - primitives.gamma * (1.0 - demand_upper)
    waiting_lower = demand_lower / denominator_lower
    waiting_upper = demand_upper / denominator_upper

    lower_edge = edges[:-1]
    upper_edge = edges[1:]
    innovation = 1.0 - rho
    failure_lower = rho * lower_edge
    failure_upper = rho * upper_edge
    success_lower = failure_lower + innovation
    success_upper = failure_upper + innovation

    lower_value = np.zeros(cells, dtype=float)
    upper_value = np.full(
        cells,
        max(primitives.net_rewards) / (1.0 - primitives.gamma),
        dtype=float,
    )
    final_change = math.inf
    for iteration in range(1, max_iterations + 1):
        lower_success = _mapped_cell_extrema(
            lower_value, success_lower, success_upper, take_minimum=True
        )
        lower_failure = _mapped_cell_extrema(
            lower_value, failure_lower, failure_upper, take_minimum=True
        )
        upper_success = _mapped_cell_extrema(
            upper_value, success_lower, success_upper, take_minimum=False
        )
        upper_failure = _mapped_cell_extrema(
            upper_value, failure_lower, failure_upper, take_minimum=False
        )
        lower_q1 = primitives.net_rewards[0] + primitives.gamma * (
            primitives.p1 * lower_success
            + (1.0 - primitives.p1) * lower_failure
        )
        lower_q2 = primitives.net_rewards[1] + primitives.gamma * (
            primitives.p2 * lower_success
            + (1.0 - primitives.p2) * lower_failure
        )
        upper_q1 = primitives.net_rewards[0] + primitives.gamma * (
            primitives.p1 * upper_success
            + (1.0 - primitives.p1) * upper_failure
        )
        upper_q2 = primitives.net_rewards[1] + primitives.gamma * (
            primitives.p2 * upper_success
            + (1.0 - primitives.p2) * upper_failure
        )
        candidate_lower = np.nextafter(
            waiting_lower * np.maximum(lower_q1, lower_q2), -np.inf
        )
        candidate_upper = np.nextafter(
            waiting_upper * np.maximum(upper_q1, upper_q2), np.inf
        )
        # Both the previous and candidate arrays are valid enclosures. Their
        # intersection preserves validity and makes monotone tightening
        # explicit, rather than relying on roundoff-level monotonicity.
        updated_lower = np.maximum(lower_value, candidate_lower)
        updated_upper = np.minimum(upper_value, candidate_upper)
        if np.any(updated_lower > updated_upper):
            raise RuntimeError("invalid boundary value enclosure")
        final_change = max(
            float(np.max(np.abs(updated_lower - lower_value))),
            float(np.max(np.abs(updated_upper - upper_value))),
        )
        lower_value = updated_lower
        upper_value = updated_upper
        if final_change <= tolerance:
            break
    else:
        raise RuntimeError(
            f"boundary enclosure did not settle in {max_iterations} iterations; "
            f"last change={final_change:.3e}"
        )

    lower_success = _mapped_cell_extrema(
        lower_value, success_lower, success_upper, take_minimum=True
    )
    upper_success = _mapped_cell_extrema(
        upper_value, success_lower, success_upper, take_minimum=False
    )
    lower_failure = _mapped_cell_extrema(
        lower_value, failure_lower, failure_upper, take_minimum=True
    )
    upper_failure = _mapped_cell_extrema(
        upper_value, failure_lower, failure_upper, take_minimum=False
    )
    delta_probability = primitives.p2 - primitives.p1
    delta_cost = primitives.c2 - primitives.c1
    lower_advantage = np.nextafter(
        -delta_cost
        + primitives.gamma
        * delta_probability
        * (lower_success - upper_failure),
        -np.inf,
    )
    upper_advantage = np.nextafter(
        -delta_cost
        + primitives.gamma
        * delta_probability
        * (upper_success - lower_failure),
        np.inf,
    )
    interval_lower, interval_upper = certified_interval
    selected = (upper_edge > interval_lower) & (lower_edge < interval_upper)
    if not np.any(selected):
        raise RuntimeError("no cells intersect the requested interval")
    minimum_certified_advantage = float(np.min(lower_advantage[selected]))
    # Beta tails increase with alpha and decrease with beta.  The simplex-wide
    # minimum is therefore Beta(1, 1+1/(1-rho)), whose tail is closed form.
    demand_floor = float(
        (1.0 - primitives.p0) ** (1.0 + 1.0 / (1.0 - rho))
    )
    if minimum_certified_advantage <= 0.0:
        raise RuntimeError(
            "requested interval is not certified: minimum lower advantage "
            f"is {minimum_certified_advantage:.6g}; increase --cells or narrow "
            "--interval"
        )
    if demand_floor <= 0.0:
        raise RuntimeError("failed to certify a positive demand floor")
    return BoundaryCertificate(
        edges=edges,
        lower_value=lower_value,
        upper_value=upper_value,
        lower_advantage=lower_advantage,
        upper_advantage=upper_advantage,
        iterations=iteration,
        final_enclosure_width=float(np.max(upper_value - lower_value)),
        certified_interval=certified_interval,
        minimum_certified_advantage=minimum_certified_advantage,
        demand_floor=demand_floor,
        contraction_upper_bound=primitives.gamma,
    )


def synchronizing_word(
    *,
    rho: float,
    failures: int,
    successes: int,
    repetitions: int = 1,
) -> dict[str, float | int | str]:
    if (
        failures < 0
        or successes < 0
        or failures + successes == 0
        or repetitions <= 0
    ):
        raise ValueError("the synchronizing word must be nonempty")
    lower, upper = 0.0, 1.0
    innovation = 1.0 - rho
    base_word = [0] * failures + [1] * successes
    for outcome in base_word * repetitions:
        lower = rho * lower + innovation * outcome
        upper = rho * upper + innovation * outcome
    base_notation = f"0^{failures} 1^{successes}"
    notation = base_notation if repetitions == 1 else f"({base_notation})^{repetitions}"
    return {
        "notation": notation,
        "base_block_failures_then_successes": True,
        "base_word": base_notation,
        "repetitions": repetitions,
        "length": repetitions * (failures + successes),
        "failures": repetitions * failures,
        "successes": repetitions * successes,
        "image_lower": lower,
        "image_upper": upper,
        "image_width": upper - lower,
    }


def word_probabilities(
    primitives: SellerPrimitives,
    word: dict[str, float | int | str],
) -> tuple[float, float]:
    """Return the product-1 probability and adaptive-policy lower bound."""

    total_failures = int(word["failures"])
    total_successes = int(word["successes"])
    under_product1 = (1.0 - primitives.p1) ** total_failures * (
        primitives.p1**total_successes
    )
    uniform_lower_bound = min(
        1.0 - primitives.p1, 1.0 - primitives.p2
    ) ** total_failures * min(primitives.p1, primitives.p2) ** total_successes
    if uniform_lower_bound <= 0.0:
        raise ValueError(
            "the recurrence proof requires both outcomes to have positive "
            "probability under both products"
        )
    return under_product1, uniform_lower_bound


def _plot_certificate(
    certificate: BoundaryCertificate,
    word: dict[str, float | int | str],
    path: Path,
) -> None:
    centers = 0.5 * (certificate.edges[:-1] + certificate.edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    axes[0].fill_between(
        centers,
        certificate.lower_advantage,
        certificate.upper_advantage,
        color="#99f6e4",
        alpha=0.7,
        linewidth=0.0,
        label="cellwise enclosure",
    )
    axes[0].plot(centers, certificate.lower_advantage, color="#0f766e", lw=1.0)
    axes[0].axhline(0.0, color="#475569", ls="--", lw=0.9)
    axes[0].axvspan(
        *certificate.certified_interval,
        color="#22c55e",
        alpha=0.12,
        label="certified product-2 interval",
    )
    axes[0].set(
        xlabel="Boundary success coordinate x",
        ylabel="Product-2 advantage",
        title="Strict optimality on the invariant boundary",
    )
    axes[0].legend(fontsize=8)

    axes[1].axhspan(
        *certificate.certified_interval,
        color="#22c55e",
        alpha=0.14,
        label="certified interval",
    )
    axes[1].plot([0, 1], [0, 1], color="#94a3b8", lw=4, alpha=0.35)
    axes[1].plot(
        [0, 1],
        [word["image_lower"], word["image_upper"]],
        color="#0f766e",
        lw=4,
        solid_capstyle="round",
        label=f"image of {word['notation']}",
    )
    axes[1].set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Any initial x in [0,1]",
        ylabel="x after the outcome word",
        title="A finite word synchronizes every initial belief",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, color="#e2e8f0", linewidth=0.7)
    fig.suptitle(
        "Observation forgetting: computer-assisted non-extinction certificate",
        x=0.01,
        ha="left",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: dict, path: Path) -> None:
    primitives = summary["primitives"]
    word = summary["synchronizing_word"]
    interval = summary["certified_interval"]
    report = f"""# Observation-time forgetting: non-extinction certificate

## Claim

For the reported primitives, `rho={summary['rho']}`, and the stationary optimal
policy, product 2 is used infinitely often almost surely from the uniform prior.
Indeed, the argument gives a strictly positive (though extremely conservative)
lower bound on its asymptotic realized frequency.

## Exact reduction

At Seller-A observation times the normalized state follows

```text
x' = rho*x + (1-rho)*Y,
y' = rho*y + (1-rho)*(1-Y).
```

The total coordinate is deterministic:
`x_k+y_k=1-rho^k`. Hence it approaches the invariant boundary `x+y=1`.
On that boundary the Bellman equation is exactly one-dimensional in `x`; both
success and failure transitions remain on the boundary. The boundary Bellman
operator is a contraction with modulus at most `gamma={primitives['gamma']}`.

## Strict product-2 region

A cellwise lower/upper Bellman enclosure with `{summary['cells']:,}` cells
certifies that product 2 is strictly optimal throughout
`x in [{interval[0]}, {interval[1]}]` on the invariant boundary. The minimum
certified advantage is `{summary['minimum_certified_advantage']:.6g}` and the
largest final value-enclosure width is
`{summary['maximum_value_enclosure_width']:.6g}`. SciPy beta-tail endpoint
evaluations were padded outward by `{summary['demand_evaluation_padding']:.1e}`.
Strict positivity and continuity imply the same action on a sufficiently thin
interior neighborhood of this boundary interval.

## Synchronizing word

The outcome word `{word['notation']}` (the failure/success block repeated
`{word['repetitions']}` times) maps
every initial boundary coordinate `x in [0,1]` into
`[{word['image_lower']:.9f}, {word['image_upper']:.9f}]`, which lies strictly
inside the certified product-2 interval. If product 1 is used, this particular
word has probability `{summary['word_probability_under_product1']:.6g}` in
each disjoint block of `{word['length']}` Seller-A observations. Uniformly over
all adaptive product choices, its conditional block probability is at least
`{summary['word_probability_lower_bound_any_policy']:.6g}`.

## Almost-sure recurrence argument

1. Thompson demand is continuous and strictly positive on the compact
   forgetting simplex. Its exact global floor is
   `{summary['demand_floor']:.6g}`, so Seller A is observed infinitely often
   almost surely.
2. Divide Seller-A interactions into blocks of `{word['length'] + 1}`. In the
   first `{word['length']}` outcomes of every block, conditional on the entire
   preceding history, the synchronizing word has probability at least the
   positive policy-uniform number above.
3. The block indicators have conditional expectations bounded below by that
   number. The strong law for bounded martingale differences therefore gives a
   lower asymptotic block frequency at least as large as the bound; independence
   of the adaptive blocks is not required.
4. Once `x+y` is sufficiently close to one, the word enters the strict
   product-2 neighborhood. Observation-time idling leaves that state unchanged,
   so the final Seller-A interaction in each successful block uses product 2.
5. Consequently the asymptotic product-2 frequency among Seller-A interactions
   has lower bound `{summary['asymptotic_product2_frequency_lower_bound_among_A']:.6g}`.
   Since the asymptotic Seller-A frequency is at least the demand floor, the
   corresponding calendar-time lower bound is
   `{summary['asymptotic_product2_frequency_lower_bound_calendar']:.6g}`.
   These tiny bounds certify persistence; they are not estimates of the actual
   long-run frequency.

## Scope

This certificate is parameter-specific. It proves pathwise non-extinction for
observation-time forgetting at the values above; it does not automatically
extend to calendar-time forgetting or arbitrary primitives. The only
computer-assisted step is the strict Bellman action sign. The recurrence step
is analytic. The Bellman enclosure is rigorous in exact arithmetic; this
implementation evaluates beta tails and arithmetic in double precision and
uses the stated outward padding. A publication-grade machine-verified proof
would replace that numerical layer with directed-rounding interval special
functions. The positive advantage margin is reported so this remaining
numerical assumption is explicit rather than hidden.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = read_config(args.config.resolve())
    primitives = SellerPrimitives(**config.get("seller", {}))
    interval = _interval(args.interval)
    certificate = solve_boundary_enclosure(
        primitives,
        rho=args.rho,
        cells=args.cells,
        certified_interval=interval,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        demand_padding=args.demand_padding,
    )
    word = synchronizing_word(
        rho=args.rho,
        failures=args.failures,
        successes=args.successes,
        repetitions=args.repetitions,
    )
    if not (
        interval[0] < float(word["image_lower"])
        and float(word["image_upper"]) < interval[1]
    ):
        raise RuntimeError("the requested word does not map strictly inside the interval")
    word_probability, word_probability_lower_bound = word_probabilities(
        primitives, word
    )
    summary = {
        "profile": str(config["profile"]),
        "primitives": primitives.as_dict(),
        "rho": args.rho,
        "cells": args.cells,
        "iterations": certificate.iterations,
        "contraction_upper_bound": certificate.contraction_upper_bound,
        "demand_evaluation_padding": args.demand_padding,
        "demand_floor": certificate.demand_floor,
        "certified_interval": list(interval),
        "minimum_certified_advantage": certificate.minimum_certified_advantage,
        "maximum_value_enclosure_width": certificate.final_enclosure_width,
        "synchronizing_word": word,
        "word_probability_under_product1": word_probability,
        "word_probability_lower_bound_any_policy": word_probability_lower_bound,
        "asymptotic_product2_frequency_lower_bound_among_A": (
            word_probability_lower_bound / (int(word["length"]) + 1)
        ),
        "asymptotic_product2_frequency_lower_bound_calendar": (
            certificate.demand_floor
            * word_probability_lower_bound
            / (int(word["length"]) + 1)
        ),
        "word_strictly_inside_certified_interval": True,
        "seller_A_observed_infinitely_often_almost_surely": True,
        "product2_used_infinitely_often_almost_surely": True,
        "certificate_type": (
            "exact boundary reduction + padded cellwise Bellman enclosure + "
            "analytic recurrence argument"
        ),
        "floating_point_scope": (
            "exact-arithmetic enclosure implemented in float64; SciPy beta-tail "
            "endpoint values padded outward, without a directed-rounding interval "
            "special-function library"
        ),
    }
    results = EXPERIMENT_ROOT / "results" / str(config["profile"])
    figures = EXPERIMENT_ROOT / "figures" / str(config["profile"])
    results.mkdir(parents=True, exist_ok=True)
    json_path = results / "observation_forgetting_nonextinction_certificate.json"
    report_path = results / "OBSERVATION_FORGETTING_NONEXTINCTION.md"
    figure_path = figures / "diagnostics" / "25_observation_forgetting_nonextinction"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(summary, report_path)
    _plot_certificate(certificate, word, figure_path)
    print(json_path)
    print(report_path)
    print(figure_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
