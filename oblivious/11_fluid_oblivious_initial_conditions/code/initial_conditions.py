"""Exact phase-line initial_conditions for saved, numerically computed fluid policies.

On the symmetric diagonal, lambda*D(x,x;p0) is strictly positive for the
experiment's rho,p0 in (0,1). It changes the speed, but not the direction or
limit, of xdot = lambda*D(x,x;p0)*(p1+(p2-p1)*sigma(x)-x).

The algorithm partitions [0,1] at every policy root and pure stationary
reputation. Each open cell has one phase-line destination; the boundary
points are evaluated separately. Thus the initial_conditions is an interval union with
explicit endpoint membership, rather than a classification of sampled x0s.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Mapping


TIE_CONVENTION = (
    "Explicit verified holding points hold. Other attractive roots use the "
    "holding mixture. Repelling roots select product 1. Crossing roots select "
    "the downstream action. At a root equal to p1 or p2, the corresponding "
    "available stationary pure action is selected. Endpoint membership is "
    "conditional on this selector; alternative optimal tie selectors may "
    "change isolated boundary points, but not the initial_conditions's measure. At a "
    "repelling discontinuity the selected action also specifies the immediate "
    "forward-event continuation; a Caratheodory (ODE almost everywhere) "
    "interpretation alone need not give a unique path from that exact point."
)


@dataclass(frozen=True)
class _PhasePolicy:
    p1: float
    p2: float
    roots: tuple[float, ...]
    products: tuple[int, ...]
    holding: Mapping[float, float]

    def probability(self, product):
        return self.p1 if product == 1 else self.p2

    def kind(self, index):
        root = self.roots[index]
        left = self.probability(self.products[index])-root
        right = self.probability(self.products[index+1])-root
        if left > 0 and right < 0:
            return "attractive"
        if left < 0 and right > 0:
            return "repelling"
        return "crossing"

    def choice(self, x):
        """Return (region, alpha, holds), without thickening point sets."""
        region = bisect_right(self.roots, x)
        if x not in self.roots:
            product = self.products[region]
            return region, float(product == 2), x == self.probability(product)
        index = self.roots.index(x)
        if x in self.holding:
            return index, self.holding[x], True
        kind = self.kind(index)
        if kind == "attractive":
            return index, (x-self.p1)/(self.p2-self.p1), True
        # A degenerate switch at a pure stationary state has a valid stationary
        # tie selector. Resolve it explicitly before selecting a departure.
        for product in (1, 2):
            if x == self.probability(product) and product in self.products[index:index+2]:
                chosen = index if self.products[index] == product else index+1
                return chosen, float(product == 2), True
        if kind == "repelling":
            chosen = index if self.products[index] == 1 else index+1
        elif self.probability(self.products[index+1]) > x:
            chosen = index+1
        else:
            chosen = index
        return chosen, float(self.products[chosen] == 2), False


def _prepare(params, policy):
    if isinstance(policy, _PhasePolicy):
        return policy
    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a saved policy mapping")
    if isinstance(policy.get("policy"), Mapping):
        policy = policy["policy"]
    p1, p2 = float(params.p1), float(params.p2)
    if not 0 < p1 < p2 < 1:
        raise ValueError("Require 0 < p1 < p2 < 1")
    roots = tuple(float(x) for x in policy.get("roots", []))
    if (any(not math.isfinite(x) or not 0 < x < 1 for x in roots)
            or any(a >= b for a, b in zip(roots, roots[1:]))):
        raise ValueError("Policy roots must increase strictly in (0,1)")
    if "products" in policy:
        products = tuple(policy["products"])
    else:
        products = tuple(row["product"] for row in policy["intervals"])
    if len(products) != len(roots)+1 or any(x not in (1, 2) for x in products):
        raise ValueError("One pure product (1 or 2) is required for every interval")
    holding = {}
    for point in policy.get("holding_points", []):
        x, alpha = float(point["x"]), float(point["alpha"])
        # Input files sometimes differ by the last decimal from a snapped root.
        # Normalize only the supplied holding record; never x0 neighborhoods.
        closest = min(roots, key=lambda root: abs(root-x), default=None)
        if (closest is None or abs(closest-x) > 1e-12
                or not math.isfinite(alpha) or not 0 <= alpha <= 1
                or abs(p1+(p2-p1)*alpha-closest) > 1e-10):
            raise ValueError("An explicit holding point must be a zero-drift policy root")
        if closest in holding and holding[closest] != alpha:
            raise ValueError("Conflicting holding choices at the same root")
        holding[closest] = alpha
    return _PhasePolicy(p1, p2, roots, products, holding)


def phase_limit(params, policy_mapping, x0):
    """Asymptotic destination for a common mature initial reputation x0.

    This follows the scalar phase line, including exact switch-point choices.
    No finite time horizon or x0 mesh is used.
    """
    policy = _prepare(params, policy_mapping)
    x = float(x0)
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("x0 must lie in [0,1]")
    region, _alpha, holding = policy.choice(x)
    if holding:
        return x
    for _ in range(len(policy.roots)+2):
        target = policy.probability(policy.products[region])
        if target == x:
            return x
        direction = 1 if target > x else -1
        index = region if direction > 0 else region-1
        if not 0 <= index < len(policy.roots):
            return target
        root = policy.roots[index]
        # A pure target before/on the next switch is approached asymptotically;
        # it is never crossed, regardless of the selector exactly at that point.
        if direction*(target-root) <= 0:
            return target
        if direction*(root-x) < 0:
            raise AssertionError("Phase-line region is inconsistent with direction")
        chosen, _alpha, holding = policy.choice(root)
        if holding:
            return root
        if chosen != region+direction:
            raise ValueError("A switch does not admit the required forward continuation")
        x, region = root, chosen
    raise AssertionError("A scalar phase line cannot cycle through policy roots")


def contains(intervals, x0):
    x = float(x0)
    return any((x > row["lo"] or row["left_closed"] and x == row["lo"])
               and (x < row["hi"] or row["right_closed"] and x == row["hi"])
               for row in intervals)


def _merge(atoms):
    merged = []
    for atom in atoms:
        row = {name: atom[name] for name in ("lo", "hi", "left_closed", "right_closed")}
        if (merged and merged[-1]["hi"] == row["lo"]
                and (merged[-1]["right_closed"] or row["left_closed"])):
            merged[-1]["hi"] = row["hi"]
            merged[-1]["right_closed"] = row["right_closed"]
        else:
            merged.append(row)
    return merged


def interval_notation(intervals, *, digits=9):
    def number(x):
        return f"{x:.{digits}g}"
    pieces = []
    for row in intervals:
        if row["lo"] == row["hi"]:
            pieces.append("{"+number(row["lo"])+"}")
        else:
            pieces.append(("[" if row["left_closed"] else "(")
                          + number(row["lo"])+", "+number(row["hi"])
                          + ("]" if row["right_closed"] else ")"))
    return " ∪ ".join(pieces) if pieces else "∅"


def initial_conditions_for_equilibrium(params, policy_mapping, x_bar, *, tolerance=1e-9):
    """Compute both consistency conditions for one optimal-policy candidate.

    Optimality is inherited from the caller's HJB verification. This function
    verifies stationarity and computes the full initial_conditions of the supplied selector.
    Its interval endpoints retain the numerical policy-root precision.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    policy = _prepare(params, policy_mapping)
    x_bar = float(x_bar)
    if not math.isfinite(x_bar) or not 0 <= x_bar <= 1:
        raise ValueError("x_bar must lie in [0,1]")
    _region, alpha, _holding = policy.choice(x_bar)
    defect = abs(policy.p1+(policy.p2-policy.p1)*alpha-x_bar)
    stationary = defect <= tolerance
    breaks = sorted({0., 1., policy.p1, policy.p2, *policy.roots})
    partition = []
    for index, point in enumerate(breaks):
        limit = phase_limit(params, policy, point)
        partition.append(dict(kind="point", lo=point, hi=point,
                              left_closed=True, right_closed=True,
                              long_run_limit=limit,
                              in_initial_conditions=bool(stationary and abs(limit-x_bar) <= tolerance)))
        if index < len(breaks)-1:
            right = breaks[index+1]
            limit = phase_limit(params, policy, (point+right)/2)
            partition.append(dict(kind="open_interval", lo=point, hi=right,
                                  left_closed=False, right_closed=False,
                                  long_run_limit=limit,
                                  in_initial_conditions=bool(stationary and abs(limit-x_bar) <= tolerance)))
    intervals = _merge([row for row in partition if row["in_initial_conditions"]])
    root_actions = []
    for index, root in enumerate(policy.roots):
        _region, action, holds = policy.choice(root)
        root_actions.append(dict(x=root, alpha=action, kind=policy.kind(index),
                                 explicit_holding=root in policy.holding, holding=holds,
                                 long_run_limit=phase_limit(params, policy, root),
                                 in_initial_conditions=contains(intervals, root)))
    return dict(intervals=intervals, text=interval_notation(intervals),
                measure=float(sum(row["hi"]-row["lo"] for row in intervals)),
                stationarity_defect=defect, stationary=bool(stationary),
                target_alpha=alpha, partition=partition, root_actions=root_actions,
                boundary_points=[dict(x=x, in_initial_conditions=contains(intervals, x),
                                      long_run_limit=phase_limit(params, policy, x))
                                 for x in breaks],
                tie_convention=TIE_CONVENTION,
                interpretation="Exact phase-line topology for a numerical policy; "
                               "endpoint values are numerical approximations, not certified bounds.")
