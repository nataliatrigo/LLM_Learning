"""Exact rational certificates for random-replacement policy intervals.

Uses only the standard library, without loading saved policies or floating-point
solvers. Rewards are normalized by R-c1, so theta=(c2-c1)/(R-c1).
For a fixed policy its value is Z-theta*W, and its product-2 advantage is
A-theta*B. Every sign assertion below is an exact Fraction comparison.

Run from the repository root:
    python -B forgetting_random_replacement/code/verify_interval_theorem.py
"""

from dataclasses import dataclass
from fractions import Fraction as F
from math import comb


@dataclass(frozen=True)
class Model:
    n: int
    p0: F
    p1: F
    p2: F
    gamma: F

    def demand(self, m):
        """Beta(s+1,m-s+1) survival, as a binomial CDF."""
        probabilities = [
            F(comb(m + 1, j)) * self.p0**j * (1 - self.p0)**(m + 1 - j)
            for j in range(m + 1)
        ]
        cumulative = F(0)
        result = []
        for probability in probabilities:
            cumulative += probability
            result.append(cumulative)
        return result


def solve_tridiagonal(lower, diagonal, upper, rhs):
    """Thomas elimination over the rationals; no numerical tolerance."""
    b, y = diagonal.copy(), rhs.copy()
    for s in range(1, len(b)):
        assert b[s - 1] > 0
        multiplier = lower[s] / b[s - 1]
        b[s] -= multiplier * upper[s - 1]
        y[s] -= multiplier * y[s - 1]
    assert b[-1] > 0
    value = [F(0)] * len(b)
    value[-1] = y[-1] / b[-1]
    for s in range(len(b) - 2, -1, -1):
        value[s] = (y[s] - upper[s] * value[s + 1]) / b[s]
    return value


def mature_components(model, actions):
    """Solve (I-gamma*K)Z=d and (I-gamma*K)W=d*action."""
    d = model.demand(model.n)
    lower, diagonal, upper = [], [], []
    for s, action in enumerate(actions):
        p = model.p2 if action else model.p1
        down = model.gamma * d[s] * (1 - p) * F(s, model.n)
        up = model.gamma * d[s] * p * F(model.n - s, model.n)
        lower.append(-down)
        diagonal.append(1 - model.gamma + down + up)
        upper.append(-up)
    z = solve_tridiagonal(lower, diagonal, upper, d)
    w = solve_tridiagonal(
        lower, diagonal, upper, [prob * action for prob, action in zip(d, actions)]
    )
    return z, w


def replacement_difference(value, s):
    n = len(value) - 1
    return (
        F(n - s, n) * (value[min(s + 1, n)] - value[s])
        + F(s, n) * (value[s] - value[max(s - 1, 0)])
    )


def advantage_components(model, z, w, mature):
    multiplier = model.gamma * (model.p2 - model.p1)
    if mature:
        dz = [replacement_difference(z, s) for s in range(model.n + 1)]
        dw = [replacement_difference(w, s) for s in range(model.n + 1)]
    else:
        dz = [right - left for left, right in zip(z, z[1:])]
        dw = [right - left for left, right in zip(w, w[1:])]
    return [multiplier * x for x in dz], [1 + multiplier * x for x in dw]


def startup_components(model, m, actions, next_z, next_w):
    z, w = [], []
    for s, (d, action) in enumerate(zip(model.demand(m), actions)):
        waiting = d / (1 - model.gamma * (1 - d))
        p = model.p2 if action else model.p1
        z.append(waiting * (1 + model.gamma * (p * next_z[s + 1] + (1 - p) * next_z[s])))
        w.append(waiting * (action + model.gamma * (p * next_w[s + 1] + (1 - p) * next_w[s])))
    return z, w


def policy_and_advantages(model, mature_actions, theta):
    """Evaluate a mature candidate and optimize startup by exact induction.

    Returns fixed-policy affine advantages at every state. Their signs at the
    requested theta certify the mature candidate and every startup decision.
    The same affine expressions can subsequently certify a whole cost range.
    """
    z, w = mature_components(model, mature_actions)
    aa, bb = advantage_components(model, z, w, mature=True)
    policies = {model.n: mature_actions}
    coefficients = {model.n: (aa, bb)}
    for m in range(model.n - 1, -1, -1):
        aa, bb = advantage_components(model, z, w, mature=False)
        actions = [int(a - theta * b > 0) for a, b in zip(aa, bb)]
        policies[m] = actions
        coefficients[m] = aa, bb
        z, w = startup_components(model, m, actions, z, w)
    return policies, coefficients


def intervals(actions):
    pieces = []
    for s, action in enumerate(actions):
        if action:
            if pieces and pieces[-1][1] == s - 1:
                pieces[-1] = pieces[-1][0], s
            else:
                pieces.append((s, s))
    return pieces


def signed_margins(policies, coefficients, theta):
    return [
        (2 * action - 1) * (a - theta * b)
        for m, actions in policies.items()
        for action, a, b in zip(actions, *coefficients[m])
    ]


def verify_baseline_and_cost_family():
    model = Model(20, F(1, 2), F(7, 20), F(4, 5), F(49, 50))
    mature_actions = [int(5 <= s <= 12) for s in range(21)]
    theta = F(12, 19)  # (0.65-0.05)/(1-0.05).
    policies, coefficients = policy_and_advantages(model, mature_actions, theta)
    expected = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 5),
        (0, 6), (0, 6), (1, 7), (1, 7), (2, 8), (2, 8), (2, 9),
        (3, 9), (3, 10), (3, 10), (4, 11), (4, 11), (4, 11), (5, 12),
    ]
    assert all(intervals(policies[m]) == [expected[m]] for m in range(21))
    baseline_margin = min(signed_margins(policies, coefficients, theta))
    assert baseline_margin > 0
    endpoint_margins = []
    for endpoint in (F(61, 100), F(16, 25)):
        margin = min(signed_margins(policies, coefficients, endpoint))
        assert margin > F(1, 2500)
        endpoint_margins.append(margin)
    # Every signed gap is affine in theta, so its minimum on the interval is
    # attained at an endpoint. Bellman verification proves unique optimality.
    print("Baseline: exact Bellman inequalities certify every layer; mature interval S=5..12.")
    print("Startup intervals (m=0..20):", expected)
    print("Cost family: theta in [61/100, 16/25]; every signed normalized gap > 1/2500.")
    print("Endpoint minimum margins (decimal display only):", *(float(x) for x in endpoint_margins))


def verify_transient_counterexample():
    model = Model(30, F(9, 20), F(1, 50), F(3, 20), F(19997, 20000))
    theta = F(53, 100)  # R=1, c1=0, c2=0.53.
    policies, coefficients = policy_and_advantages(model, [0] * 31, theta)
    assert min(signed_margins(policies, coefficients, theta)) > 0
    assert intervals(policies[30]) == []
    assert policies[3] == [1, 0, 0, 1]
    a, b = coefficients[3]
    gaps = [x - theta * y for x, y in zip(a, b)]
    assert min((2 * action - 1) * gap for action, gap in zip(policies[3], gaps)) > F(9, 1000)
    print("Counterexample: mature policy is always product 1, but at m=3 product 2 is optimal only at S=0,3.")
    print("m=3 advantages (decimal display only):", *(float(x) for x in gaps))
    print("All assertions used exact rational arithmetic.")


def main():
    verify_baseline_and_cost_family()
    verify_transient_counterexample()


if __name__ == "__main__":
    main()
