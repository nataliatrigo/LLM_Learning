"""Monopoly optimization with a true-quality, rather than mean-TS, closure.

The requested default averages true success probabilities over rival sales.
Exploratory search can use the exact own-sales embedded chain. Every retained
candidate is freshly certified in the full actual two-seller TS game.
"""

from dataclasses import asdict

import numpy as np

from oblivious.oe_p0.code.core import ScalarExperiment, find_fixed_points
from oblivious.oe_p0.code.primitives import Parameters, probability_array
from forgetting.Code.model import Primitives, _bellman_update
from .event_chain import purchase_chain_statistics


LABEL = "OE quality"


class QualityExperiment(ScalarExperiment):
    def __init__(self, n=21, params=None, *, weighting="purchase", **kwargs):
        if weighting not in ("purchase", "calendar"):
            raise ValueError("weighting must be purchase or calendar")
        super().__init__(n=n, params=params, **kwargs)
        self.weighting = weighting
        self.p0_bounds = (self.params.p1, self.params.p2)
        self._event_cache = {}

    def quality_statistics(self, policy, nu):
        alpha = probability_array(policy)
        nu = np.asarray(nu, dtype=float)
        if (alpha.shape != (self.n,) or nu.shape != (self.n, self.n)
                or not np.isfinite(nu).all() or np.any(nu < 0)
                or abs(nu.sum() - 1) > 1e-10):
            raise ValueError("Require a grid policy and normalized nonnegative joint law")
        quality = self.params.p1 + (self.params.p2 - self.params.p1) * alpha
        mu_b = nu.sum(axis=0)
        sales_b = (nu * self.game.D.T).sum(axis=0)
        share_b = float(sales_b.sum())
        if share_b <= 0:
            raise ValueError("Rival must have positive market share")
        calendar = float(mu_b @ quality)
        purchase = float(sales_b @ quality / share_b)
        return dict(calendar_quality_B=calendar, purchase_quality_B=purchase,
                    market_share_B=share_b,
                    generated_quality=purchase if self.weighting == "purchase" else calendar,
                    posterior_mean_B=float(mu_b @ ((1-self.params.rho+self.grid)
                                                   / (3-2*self.params.rho))))

    def _generated_scalar(self, policy, invariant):
        return self.quality_statistics(policy, invariant["nu"])["generated_quality"]

    def _event_statistics(self, policy):
        key = np.asarray(policy, dtype=float).tobytes()
        if key not in self._event_cache:
            self._event_cache[key] = purchase_chain_statistics(self.grid, policy, self.params)
        return self._event_cache[key]

    def _exploratory_scalar(self, policy):
        if self.weighting != "purchase":
            return super()._exploratory_scalar(policy)
        event = self._event_statistics(policy)
        return event["generated_quality"], dict(
            verified=event["stationary_verified"], residual=event["stationary_residual"],
            scope="Exact embedded own-sales chain; full two-seller certification follows")

    def _pure_probe(self, p0):
        """Cheap search evidence; never a full-game equilibrium certificate."""
        solution = self._optimal(p0)
        policy = solution["policy"]
        event = self._event_statistics(policy)
        generated = event["generated_quality"]
        return dict(p0=float(p0), F=generated, G=generated-float(p0),
                    policy=policy.copy(), value=solution["value"].copy(),
                    advantage=solution["advantage"].copy(), grid=self.grid.copy(),
                    verified=bool(event["stationary_verified"]),
                    exploratory=True, full_game_verified=False,
                    stationarity_scope="Own-sales chain only; candidates need joint certification")

    def search_pure(self, p0_points=41, consistency_tol=1e-7):
        evaluator = self._pure_probe if self.weighting == "purchase" else self.evaluate
        return find_fixed_points(evaluator, self.p0_bounds, p0_points, consistency_tol)

    def _annotate(self, result):
        result["quality_summary"] = self.quality_statistics(result["policy"], result["nu"])
        result["closure"] = self.weighting
        result["F_definition"] = "True success probability among rival sales" if self.weighting == "purchase" else "Calendar-time mean true quality"
        primitives = Primitives(p0=result["p0"], p1=self.params.p1, p2=self.params.p2,
                                c1=self.params.c1, c2=self.params.c2,
                                revenue=self.params.R, gamma=self.params.gamma)
        demand = result["d_fixed"]
        updated, advantage = _bellman_update(result["value"], self.grid, demand,
                                             primitives, self.params.rho, optimize=True)
        certificate = dict(
            bellman_residual=float(np.max(np.abs(updated-result["value"]))),
            advantage_agreement=float(np.max(np.abs(advantage-result["advantage"]))),
            source="forgetting.Code.model._bellman_update on the identical grid")
        certificate["verified"] = bool(certificate["bellman_residual"] <= self.dp_tol
                                       and certificate["advantage_agreement"] <= self.dp_tol)
        result["monopoly_certificate"] = certificate
        result["verified"] = bool(result["verified"] and certificate["verified"])
        if "candidate_verified" in result:
            result["candidate_verified"] = bool(result["candidate_verified"] and certificate["verified"])
        result["summary"].update(quality=result["quality_summary"], closure=self.weighting,
                                 monopoly_certificate=certificate)
        return result

    def evaluate(self, p0):
        return self._annotate(super().evaluate(p0))

    def evaluate_mixed(self, p0, alpha, consistency_tol=1e-7):
        return self._annotate(super().evaluate_mixed(p0, alpha, consistency_tol))

    def configuration(self):
        return dict(super().configuration(), model=LABEL, quality_weighting=self.weighting,
                    p0_bounds=list(self.p0_bounds),
                    policy_problem="Original forgetting monopoly Bellman equation",
                    competition="Actual independent two-seller Beta TS draws",
                    parameters=asdict(self.params))

    def iteration_history(self, start=.5, *, damping=1., max_iterations=100, tolerance=1e-7):
        if not 0 < damping <= 1 or max_iterations < 1:
            raise ValueError("Require positive iterations and damping in (0,1]")
        point = float(np.clip(start, *self.p0_bounds))
        rows = []
        for iteration in range(max_iterations):
            result = self._pure_probe(point) if self.weighting == "purchase" else self.evaluate(point)
            rows.append(dict(iteration=iteration, p0=point, F=result["F"], G=result["G"],
                             damping=damping, policy_signature=result["policy"].astype(int).tolist()))
            if abs(result["G"]) <= tolerance:
                return dict(status="converged", start=start, damping=damping, rows=rows)
            new_point = float((1-damping)*point+damping*result["F"])
            if any(abs(new_point-old["p0"]) < 1e-12 for old in rows[-10:]):
                return dict(status="cycle detected", start=start, damping=damping, rows=rows)
            point = new_point
        return dict(status="iteration limit", start=start, damping=damping, rows=rows)
