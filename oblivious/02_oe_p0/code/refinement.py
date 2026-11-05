"""Restartable scalar-model searches with the existing certification rules.

The OE p0 and OE quality searches enumerate optimal-policy switches. An
interpolated mixed policy is not generally indifferent on a finer grid, so it
cannot replace that search. Same-grid maps are usable search proposals; maps
from a different grid are deliberately not passed as ``pure_seed``.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid

import numpy as np

from oblivious.oe_p0.code.core import ScalarExperiment
from oblivious.oe_p0.code.run_experiment import (
    evaluation_row, load_bundle, save_bundle, select_candidate, write_json,
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprint(experiment_class):
    """Do not silently reuse partial evaluations after numerical code changes."""
    folder = Path(__file__).resolve().parent
    paths = {folder / name for name in
             ("core.py", "game.py", "policy.py", "primitives.py", "refinement.py",
              "positive_stationary.py", "positive_stationary.cpp")}
    for cls in experiment_class.__mro__:
        if cls is not object:
            paths.add(Path(inspect.getfile(cls)).resolve())
    # QualityExperiment also uses an embedded chain and an independent
    # monopoly certificate. These are inputs to its cached evaluations.
    if experiment_class.__module__.startswith("oblivious.oe_following_monopoly."):
        root = folder.parents[2]
        paths.update((root / "oblivious/03_oe_following_monopoly/code/event_chain.py",
                      root / "forgetting/Code/model.py"))
    return {str(path): _digest(path) for path in sorted(paths)}


class EvaluationCheckpoint:
    """Atomic index of independently written, hash-checked numerical entries.

    An interruption before the index update leaves an unreferenced entry;
    reloading never mistakes that entry for a completed evaluation.
    """

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "evaluations.json"
        self.identity = identity
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())
            if self.index.get("identity") != identity:
                raise ValueError("Checkpoint configuration or numerical sources differ; use a new checkpoint directory.")
        else:
            self.index = dict(version=1, identity=identity, entries={})
            write_json(self.index_path, self.index)
        self.hits = 0
        self.writes = 0

    def get(self, kind, key):
        token = kind + ":" + key
        entry = self.index["entries"].get(token)
        if entry is None:
            return None
        base = self.directory / entry["stem"]
        for suffix in (".json", ".npz"):
            path = base.with_suffix(suffix)
            if not path.is_file() or _digest(path) != entry["sha256"][suffix]:
                raise ValueError(f"Checkpoint entry is missing or corrupted: {path.name}")
        self.hits += 1
        return load_bundle(base)

    def put(self, kind, key, result):
        token = kind + ":" + key
        # Immutable generations also protect a previously indexed evaluation
        # if an explicit replacement of the same token is interrupted.
        stem = kind + "_" + hashlib.sha256(key.encode()).hexdigest() + "_" + uuid.uuid4().hex
        base = self.directory / stem
        save_bundle(base, result)
        self.index["entries"][token] = dict(
            stem=stem, sha256={suffix: _digest(base.with_suffix(suffix))
                              for suffix in (".json", ".npz")})
        write_json(self.index_path, self.index)
        self.writes += 1

    def summary(self):
        return dict(directory=str(self.directory), entries=len(self.index["entries"]),
                    reused_evaluations=self.hits, written_evaluations=self.writes,
                    scope="Completed evaluations only; an interrupted numerical solve restarts")


class CheckpointedScalarMixin:
    """Memoize exact inputs after the original solver certifies/evaluates them."""

    def configure_checkpoint(self, directory=None, progress=None, *, consistency_tol=1e-7,
                             experiment_class=ScalarExperiment):
        self._refinement_progress = progress
        self._refinement_memory = {}
        identity = dict(configuration=self.configuration(), consistency_tolerance=consistency_tol,
                        experiment_class=experiment_class.__module__ + "." + experiment_class.__qualname__,
                        numerical_sources=_source_fingerprint(experiment_class))
        self.evaluation_checkpoint = (None if directory is None else
                                      EvaluationCheckpoint(directory, identity))

    def _checkpoint_call(self, kind, key, calculation):
        token = (kind, key)
        if token in self._refinement_memory:
            return self._refinement_memory[token]
        store = self.evaluation_checkpoint
        result = None if store is None else store.get(kind, key)
        if result is None:
            result = calculation()
            if store is not None:
                store.put(kind, key, result)
            if self._refinement_progress is not None:
                self._refinement_progress(f"Checkpoint: {kind} evaluation completed")
        self._refinement_memory[token] = result
        return result

    def evaluate(self, p0):
        p0 = float(p0)
        result = self._checkpoint_call("pure", p0.hex(), lambda: super(CheckpointedScalarMixin, self).evaluate(p0))
        # Preserve core's reuse of an identical pure-policy stationary law at
        # other benchmark values, including after a process restart.
        policy = np.asarray(result["policy"])
        if np.all((policy == 0) | (policy == 1)):
            self._policy_cache[policy.astype(np.uint8).tobytes()] = dict(
                nu=result["nu"], mu_a=result["mu_a"], mu_b=result["mu_b"],
                summary=result["stationary_diagnostics"])
        self._evaluations[p0] = result
        return result

    def evaluate_mixed(self, p0, alpha, consistency_tol=1e-7):
        alpha = np.asarray(alpha, dtype=float)
        key = ":".join((float(p0).hex(), float(consistency_tol).hex(),
                        hashlib.sha256(alpha.tobytes()).hexdigest(), str(alpha.shape)))
        return self._checkpoint_call(
            "mixed", key,
            lambda: super(CheckpointedScalarMixin, self).evaluate_mixed(p0, alpha, consistency_tol))

    def _exploratory_scalar(self, policy):
        policy = np.asarray(policy, dtype=float)
        key = hashlib.sha256(policy.tobytes()).hexdigest() + ":" + str(policy.shape)

        def calculate():
            generated, diagnostics = super(CheckpointedScalarMixin, self)._exploratory_scalar(policy)
            return dict(generated=generated, diagnostics=diagnostics)

        result = self._checkpoint_call("exploratory", key, calculate)
        return result["generated"], result["diagnostics"]


def _certified(candidate, n, consistency_tol, dp_tol, stationary_tol, stability_tol):
    if not isinstance(candidate, dict):
        return False
    if not (candidate.get("verified") and candidate.get("candidate_verified")):
        return False
    policy = np.asarray(candidate.get("policy", []), dtype=float)
    grid = np.asarray(candidate.get("grid", []), dtype=float)
    nu = np.asarray(candidate.get("nu", []), dtype=float)
    if (policy.shape != (n,) or grid.shape != (n,) or nu.shape != (n, n)
            or not np.all(np.isfinite(policy)) or np.any((policy < 0) | (policy > 1))
            or not np.array_equal(grid, np.linspace(0, 1, n))
            or not np.all(np.isfinite(nu)) or np.any(nu < 0)
            or abs(float(nu.sum()) - 1) > stationary_tol):
        return False
    if (not np.isfinite(candidate.get("G", np.inf))
            or abs(candidate["G"]) > consistency_tol
            or abs(candidate.get("F", np.inf) - candidate.get("p0", -np.inf) - candidate["G"]) > consistency_tol):
        return False
    dp = candidate.get("dp_diagnostics", {})
    stationary = candidate.get("stationary_diagnostics", {})
    if not (dp.get("verified") and stationary.get("verified")):
        return False
    for name in ("bellman_residual", "conditional_regret", "policy_evaluation_residual",
                 "complementarity_residual", "mixed_indifference_residual"):
        if name in dp and (not np.isfinite(dp[name]) or dp[name] > dp_tol):
            return False
    if dp.get("best_response_gain_upper_bound", 0) > consistency_tol:
        return False
    if (stationary.get("residual", np.inf) > stationary_tol
            or stationary.get("stability_l1", np.inf) > stability_tol):
        return False
    if "monopoly_certificate" in candidate and not candidate["monopoly_certificate"].get("verified"):
        return False
    return True


def is_verified_case(case, n, params, *, order=128, consistency_tol=1e-7,
                     dp_tol=1e-10, stationary_tol=1e-11, stability_tol=1e-7,
                     near_tie_tol=1e-7, expected_configuration=None):
    """Inspect stored strong certificates; this does not recompute a law.

    The caller remains responsible for checking the saved run's provenance.
    ``expected_configuration`` can additionally require model-specific closure
    settings, for example the purchase weighting in OE quality.
    """
    if not isinstance(case, dict):
        return False
    expected = (dict(parameters=asdict(params), n=int(n), order=order,
                     dp_tolerance=dp_tol, stationary_tolerance=stationary_tol,
                     stationary_stability_tolerance=stability_tol,
                     near_tie_tolerance=near_tie_tol)
                if expected_configuration is None else expected_configuration)
    configuration = case.get("configuration", {})
    if (case.get("n") != n or case.get("rho") != params.rho
            or any(configuration.get(key) != value for key, value in expected.items())
            or (expected_configuration is not None and configuration != expected_configuration)):
        return False
    candidates = case.get("candidates", [])
    return bool(candidates and all(_certified(c, n, consistency_tol, dp_tol,
                                             stationary_tol, stability_tol)
                                   for c in candidates)
                and _certified(case.get("selected"), n, consistency_tol, dp_tol,
                               stationary_tol, stability_tol))


def solve_refinement(n, params, previous=None, progress=None, *, checkpoint_dir=None,
                     p0_points=41, consistency_tol=1e-7, order=128,
                     dp_tol=1e-10, stationary_tol=1e-11, stability_tol=1e-7,
                     near_tie_tol=1e-7, experiment_class=ScalarExperiment,
                     experiment_kwargs=None, reuse_verified=True):
    """Return a runner-compatible case, preserving the original finite search.

    OE quality is supported with ``experiment_class=QualityExperiment`` and
    ``experiment_kwargs={"weighting": "purchase"}``. Different-grid previous
    cases are recorded as references, never relabelled as a valid pure map.
    The search's existing non-exhaustive scope and tolerances are unchanged.
    """
    started = time.monotonic()
    cls = type("Checkpointed" + experiment_class.__name__,
               (CheckpointedScalarMixin, experiment_class), {})
    experiment = cls(n=n, params=params, order=order, dp_tol=dp_tol,
                     stationary_tol=stationary_tol, stability_tol=stability_tol,
                     near_tie_tol=near_tie_tol, **(experiment_kwargs or {}))
    if reuse_verified and is_verified_case(
            previous, n, params, order=order, consistency_tol=consistency_tol,
            dp_tol=dp_tol, stationary_tol=stationary_tol, stability_tol=stability_tol,
            near_tie_tol=near_tie_tol, expected_configuration=experiment.configuration()):
        if progress is not None:
            progress("Reusing same-grid case with stored strong certificates")
        result = dict(previous)
        result["refinement"] = dict(reused_verified_case=True,
                                    interpretation="Stored verified result; no new numerical certification")
        return result
    experiment.configure_checkpoint(checkpoint_dir, progress,
                                    consistency_tol=consistency_tol,
                                    experiment_class=experiment_class)
    pure_seed = None
    if previous is not None and previous.get("configuration") == experiment.configuration():
        if all(key in previous for key in ("search_summary", "brackets", "map_rows")):
            pure_seed = previous
    elif previous is not None and progress is not None:
        progress("Previous grid differs: retaining the full scalar scan and optimal tie-face search")
    search = experiment.search(p0_points=p0_points, consistency_tol=consistency_tol,
                               pure_seed=pure_seed, progress=progress)
    selected = (select_candidate(search["candidates"]) if search["candidates"]
                else search["best_diagnostic"])
    return dict(rho=params.rho, n=int(n), selected=selected, candidates=search["candidates"],
                map_rows=[evaluation_row(e) for e in search["evaluations"]],
                search_log=search["search_log"], failures=search["failures"],
                brackets=search["brackets"], search_summary=search["summary"],
                mixing_branches=search.get("mixing_branches", []),
                configuration=experiment.configuration(), demand_audit=experiment.demand_audit(),
                elapsed_seconds=time.monotonic() - started,
                refinement=dict(reused_verified_case=False, reused_same_grid_map=pure_seed is not None,
                                previous_grid=None if previous is None else previous.get("n"),
                                previous_p0=None if previous is None or previous.get("selected") is None
                                else previous["selected"].get("p0"),
                                interpolated_policy_used=False,
                                method="Original scalar scan and optimal tie-face mixed search",
                                checkpoint=None if experiment.evaluation_checkpoint is None
                                else experiment.evaluation_checkpoint.summary()))
