"""Fixed benchmark calibrated to an expected finite-window TS time average.

Only the consistency operator changes. The underlying infinite-horizon seller
problem and mixed-action optimality conditions are inherited without changes.
"""

import numpy as np

from oblivious.oe_p0.code.core import ScalarExperiment, posterior_mean
from .occupation import finite_window_law


LABEL = "OE temporal p0"


class TemporalExperiment(ScalarExperiment):
    def __init__(self, n=51, params=None, *, periods=100000, burn_in=10000,
                 initial=.5, temporal_tolerance=1e-10, **kwargs):
        super().__init__(n=n, params=params, **kwargs)
        if (isinstance(periods, bool) or int(periods) != periods or periods < 1
                or isinstance(burn_in, bool) or int(burn_in) != burn_in or burn_in < 0
                or not np.isfinite(initial) or not 0 <= initial <= 1
                or not np.isfinite(temporal_tolerance) or temporal_tolerance <= 0):
            raise ValueError("Require positive integer periods, nonnegative integer burn-in, valid initial state and tolerance")
        self.periods, self.burn_in = int(periods), int(burn_in)
        self.initial, self.temporal_tolerance = float(initial), float(temporal_tolerance)
        self._temporal_cache = {}
        self._probe_cache = {}
        self._search_progress = None

    def temporal_statistics(self, policy):
        key = np.asarray(policy, dtype=float).tobytes()
        if key not in self._temporal_cache:
            result = finite_window_law(
                self.game, policy, periods=self.periods, burn_in=self.burn_in,
                initial=self.initial, tolerance=self.temporal_tolerance)
            result["F"] = float(result["mu_b"] @ posterior_mean(self.grid, self.params))
            self._temporal_cache[key] = result
        return self._temporal_cache[key]

    def _generated_scalar(self, policy, invariant):
        # The stationary law is a diagnostic, never the closure in this model.
        return self.temporal_statistics(policy)["F"]

    def _exploratory_scalar(self, policy):
        result = self.temporal_statistics(policy)
        return result["F"], result["summary"]

    def _annotate(self, result):
        temporal = self.temporal_statistics(result["policy"])
        result["verified"] = bool(result["verified"] and temporal["summary"]["verified"])
        if "candidate_verified" in result:
            result["candidate_verified"] = bool(result["candidate_verified"] and result["verified"])
        result["temporal_nu"] = temporal["nu"]
        result["temporal_mu_a"] = temporal["mu_a"]
        result["temporal_mu_b"] = temporal["mu_b"]
        result["temporal_diagnostics"] = temporal["summary"]
        result["stationary_F"] = float(result["mu_b"] @ posterior_mean(self.grid, self.params))
        result["F_definition"] = "Expected average of B's pre-selection TS draws over the retained finite calendar window"
        result["closure"] = "expected finite-window TS time average"
        result["summary"].update(temporal=temporal["summary"], stationary_F=result["stationary_F"],
                                 closure=result["closure"])
        return result

    def evaluate(self, p0):
        return self._annotate(super().evaluate(p0))

    def evaluate_mixed(self, p0, alpha, consistency_tol=1e-7):
        return self._annotate(super().evaluate_mixed(p0, alpha, consistency_tol))

    def _pure_probe(self, p0):
        p0 = float(p0)
        if p0 not in self._probe_cache:
            solution = self._optimal(p0)
            policy = solution["policy"].copy()
            generated, temporal_summary = self._exploratory_scalar(policy)
            regret = np.maximum(solution["advantage"], 0) - policy * solution["advantage"]
            verified = bool(temporal_summary["verified"]
                            and solution["residual"] <= self.dp_tol
                            and np.max(regret) <= self.dp_tol)
            self._probe_cache[p0] = dict(
                p0=p0, F=generated, G=generated-p0, policy=policy,
                value=solution["value"].copy(), advantage=solution["advantage"].copy(),
                grid=self.grid.copy(), verified=verified, exploratory=True,
                full_game_verified=False, temporal_diagnostics=temporal_summary,
                strategy_mode="pure temporal map probe")
            if self._search_progress and len(self._probe_cache) % 10 == 0:
                self._search_progress(f"Temporal map: {len(self._probe_cache)} scalar probes; "
                                      f"{len(self._temporal_cache)} distinct policies")
        return self._probe_cache[p0]

    def search_pure(self, p0_points=41, consistency_tol=1e-7):
        """Bounded scan plus generated values and sign-change localization.

        Policy discontinuities are only search brackets, never accepted roots.
        Roots without a detected sign change and unsampled narrow branches may
        be missed. This restriction is explicitly retained in the saved audit.
        """
        if p0_points < 3 or int(p0_points) != p0_points or consistency_tol <= 0:
            raise ValueError("Require at least three scan points and positive consistency tolerance")
        failures, cache, history = [], {}, []

        def evaluate(point):
            point = float(np.clip(point, *self.p0_bounds))
            if point not in cache:
                try:
                    record = dict(self._pure_probe(point))
                    if not record["verified"] or not np.isfinite(record["F"]):
                        raise ArithmeticError("Temporal pure-policy probe did not verify")
                    record["candidate_verified"] = abs(record["G"]) <= consistency_tol
                    cache[point] = record
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    failures.append(dict(p0=point, error=str(exc)))
                    cache[point] = None
            return cache[point]

        for point in np.linspace(*self.p0_bounds, int(p0_points)):
            evaluate(point)
        # F is constant on a pure-policy plateau; its generated value is the
        # only possible fixed point on that plateau. Bound discovery to two
        # passes so long finite-window propagations do not proliferate.
        signatures = set()
        for _ in range(2):
            for record in list(cache.values()):
                if record is None:
                    continue
                signature = record["policy"].tobytes()
                if signature not in signatures:
                    signatures.add(signature)
                    evaluate(record["F"])
        rows = sorted((record for record in cache.values() if record is not None), key=lambda r:r["p0"])
        intervals = [(left, right) for left, right in zip(rows[:-1], rows[1:])
                     if left["G"] * right["G"] < 0
                     and not left["candidate_verified"] and not right["candidate_verified"]]
        brackets = []
        for left, right in intervals:
            initial = [left["p0"], right["p0"]]
            found = False
            for _ in range(60):
                middle = evaluate((left["p0"] + right["p0"])/2)
                if middle is None:
                    break
                if middle["candidate_verified"]:
                    found = True
                    break
                if left["G"] * middle["G"] < 0:
                    right = middle
                else:
                    left = middle
                if right["p0"] - left["p0"] <= max(1e-12, consistency_tol*.001):
                    break
            brackets.append(dict(
                initial_bracket=initial, final_bracket=[left["p0"], right["p0"]],
                left_G=left["G"], right_G=right["G"], left_F=left["F"], right_F=right["F"],
                candidate_detected=found,
                policy_jump=not np.array_equal(left["policy"], right["policy"]),
                status="pure candidate detected" if found else "sign change requires mixed-face verification"))
        # Check exact generated values of policies revealed by refinement.
        for record in list(cache.values()):
            if record is not None and record["policy"].tobytes() not in signatures:
                signatures.add(record["policy"].tobytes())
                evaluate(record["F"])
        rows = sorted((record for record in cache.values() if record is not None), key=lambda r:r["p0"])
        candidates = []
        for record in sorted((r for r in rows if r["candidate_verified"]), key=lambda r:abs(r["G"])):
            if not any(abs(record["p0"]-old["p0"]) <= 5*consistency_tol
                       and np.array_equal(record["policy"], old["policy"]) for old in candidates):
                candidates.append(record)
        candidates.sort(key=lambda r:r["p0"])
        best = min(rows, key=lambda r:abs(r["G"])) if rows else None
        history.append(dict(method="uniform temporal map scan, two generated-policy passes, sign-change bisection",
                            generated_policy_passes=2, mixed_face_scope="Only localized sign-changing policy jumps"))
        return dict(evaluations=rows, candidates=candidates, best_diagnostic=best,
                    failures=failures, search_log=history, brackets=brackets,
                    summary=dict(verified_candidates=0, exploratory_candidates=len(candidates),
                                 evaluations=len(rows), detected_policies=len(signatures),
                                 failed_evaluations=len(failures), scan_points=int(p0_points),
                                 consistency_tolerance=consistency_tol,
                                 minimum_consistency_residual=None if best is None else abs(best["G"]),
                                 p0_bounds=list(self.p0_bounds),
                                 completeness="Finite restricted search; no completeness or uniqueness claim"))

    def search(self, p0_points=41, consistency_tol=1e-7, pure_seed=None, progress=None):
        if pure_seed is not None:
            raise ValueError("Stationary search seeds cannot replace the temporal consistency map")
        self._search_progress = progress
        pure = self.search_pure(p0_points, consistency_tol)
        configuration = super().configuration()
        def seed(rows, candidates):
            return dict(configuration=configuration, search_summary=pure["summary"],
                        brackets=[], search_log=[], failures=[], map_rows=rows,
                        candidates=candidates, selected=pure["best_diagnostic"])
        # Freshly certify pure candidates in the full game first.
        merged = super().search(p0_points, consistency_tol,
                                pure_seed=seed([], pure["candidates"]), progress=progress)
        for bracket in pure["brackets"]:
            if bracket["candidate_detected"] or not bracket["policy_jump"]:
                continue
            endpoints = [self._pure_probe(p) for p in bracket["final_bracket"]]
            branch = super().search(p0_points, consistency_tol,
                                    pure_seed=seed(endpoints, []), progress=progress)
            for key in ("evaluations", "failures", "search_log", "mixing_branches"):
                merged[key].extend(branch[key])
            for candidate in branch["candidates"]:
                if not any(abs(candidate["p0"]-old["p0"]) <= 5*consistency_tol
                           and np.max(np.abs(candidate["policy"]-old["policy"])) <= 1e-6
                           for old in merged["candidates"]):
                    merged["candidates"].append(candidate)
        merged["candidates"].sort(key=lambda r:(r["p0"], tuple(r["policy"])))
        merged["evaluations"] = sorted(pure["evaluations"] + merged["evaluations"], key=lambda r:r["p0"])
        merged["failures"] = pure["failures"] + merged["failures"]
        merged["search_log"] = pure["search_log"] + merged["search_log"]
        merged["brackets"] = pure["brackets"]
        if merged["candidates"]:
            merged["best_diagnostic"] = min(merged["candidates"], key=lambda r:abs(r["G"]))
        else:
            merged["best_diagnostic"] = pure["best_diagnostic"]
        merged["summary"].update(
            verified_candidates=len(merged["candidates"]),
            verified_mixed_candidates=sum(c["mixed_nodes"] > 0 for c in merged["candidates"]),
            evaluations=len(merged["evaluations"]), failed_evaluations=len(merged["failures"]),
            mixing_branches=len(merged["mixing_branches"]),
            localized_ties=len(set(b["p0"] for b in merged["mixing_branches"])),
            minimum_consistency_residual=None if merged["best_diagnostic"] is None else abs(merged["best_diagnostic"]["G"]),
            reused_pure_seed=False, distinct_temporal_policies=len(self._temporal_cache),
            closure="expected finite-window TS time average",
            exploratory_stationarity="No stationary law used in temporal map probes",
            completeness="Uniform scan and generated pure-policy values; mixed faces only at detected sign-changing jumps. Narrow, tangent, non-sign-changing or multiple-mixing branches may be missed; not exhaustive.")
        self._search_progress = None
        return merged

    def configuration(self):
        return dict(super().configuration(), model=LABEL, periods=self.periods,
                    burn_in=self.burn_in, initial_state=[self.initial, self.initial],
                    temporal_tolerance=self.temporal_tolerance,
                    closure="Expected mean of B's pre-selection TS draws in the retained calendar window",
                    fixed_benchmark=True)
