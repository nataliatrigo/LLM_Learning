"""Small-grid checks for optional local continuation and its certificate gate.

Run from the project root with one BLAS thread:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest \
        oblivious.oe_distribution.code.test_local_continuation -v
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import issparse

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution.code.model import JointMarginalExperiment, _JointResponse
from oblivious.oe_conditional_distribution.code.model import ConditionalExperiment, _ConditionalResponse


class LocalContinuation(unittest.TestCase):
    def test_interpolated_policy_skips_smoothing_and_satisfies_action_maximum(self):
        for experiment_type in (JointMarginalExperiment, ConditionalExperiment):
            with self.subTest(model=experiment_type.__name__):
                coarse = experiment_type(n=7, params=Parameters(rho=.6)).solve(max_evaluations=500)
                self.assertTrue(coarse["verified"])
                experiment = experiment_type(n=11, params=Parameters(rho=.6))
                warm = np.interp(experiment.grid, coarse["grid"], coarse["policy"])
                original = experiment.solve(initial_policy=warm, max_evaluations=500)
                local = experiment.solve(initial_policy=warm, max_evaluations=500, prefer_local=True)
                self.assertTrue(original["verified"])
                self.assertTrue(local["verified"])
                metadata = local["search_metadata"]
                self.assertTrue(metadata["local_verified"])
                self.assertFalse(metadata["fallback_used"])
                self.assertLess(metadata["evaluations"], original["search_metadata"]["evaluations"])
                self.assertFalse(any(row.get("temperature", 0) > 0 for row in local["history"]))
                assert_allclose(local["policy"], original["policy"], atol=1e-10, rtol=0)
                assert_allclose(local["nu"], original["nu"], atol=1e-10, rtol=0)
                # Reconstruct both selected-product values independently of
                # the complementarity search and its analytic sensitivities.
                p, value, grid = experiment.params, local["value"], experiment.grid
                success = np.interp(p.rho*grid+1-p.rho, grid, value)
                failure = np.interp(p.rho*grid, grid, value)
                q1 = p.R-p.c1+p.gamma*(p.p1*success+(1-p.p1)*failure)
                q2 = p.R-p.c2+p.gamma*(p.p2*success+(1-p.p2)*failure)
                chosen = (1-local["policy"])*q1+local["policy"]*q2
                assert_allclose(chosen, np.maximum(q1, q2), atol=1e-10, rtol=0)
                assert_allclose(value, p.gamma*(1-local["demand"])*value
                                +local["demand"]*np.maximum(q1, q2), atol=1e-10, rtol=0)

    def test_exact_initial_policy_gets_one_fresh_full_certificate(self):
        for experiment_type in (JointMarginalExperiment, ConditionalExperiment):
            with self.subTest(model=experiment_type.__name__):
                experiment = experiment_type(n=7, params=Parameters(rho=.6, gamma=0.))
                with patch.object(experiment, "evaluate", wraps=experiment.evaluate) as certify:
                    result = experiment.solve(initial_policy=np.zeros(7), max_evaluations=80,
                                              prefer_local=True)
                self.assertTrue(result["verified"])
                self.assertEqual(certify.call_count, 1)
                self.assertEqual(result["search_metadata"]["local_certifications"], 1)
                self.assertEqual(result["search_metadata"]["exploratory_evaluations"], 1)

    def test_failed_final_certificate_forces_original_search_and_stays_unverified(self):
        for experiment_type in (JointMarginalExperiment, ConditionalExperiment):
            with self.subTest(model=experiment_type.__name__):
                experiment = experiment_type(n=3, params=Parameters(rho=.6, gamma=0.))
                failed = experiment.evaluate(np.full(3, .4))
                self.assertFalse(failed["verified"])
                with patch.object(experiment, "evaluate",
                                  side_effect=lambda *_a, **_kw: deepcopy(failed)):
                    result = experiment.solve(initial_policy=np.zeros(3), max_evaluations=80,
                                              prefer_local=True)
                self.assertFalse(result["verified"])
                self.assertFalse(result["search_metadata"]["local_verified"])
                self.assertTrue(result["search_metadata"]["fallback_used"])
                self.assertEqual(result["search_metadata"]["max_evaluations"], 80)
                self.assertTrue(any(row.get("temperature", 0) > 0 for row in result["history"]))
                if experiment_type is ConditionalExperiment:
                    self.assertFalse(result["converged"])
                    self.assertEqual(result["status"], "unresolved")

    def test_local_numerical_failure_still_allows_original_search(self):
        for experiment_type, response_type in ((JointMarginalExperiment, _JointResponse),
                                                (ConditionalExperiment, _ConditionalResponse)):
            with self.subTest(model=experiment_type.__name__):
                experiment = experiment_type(n=3, params=Parameters(rho=.6, gamma=0.))
                calculate = response_type.calculate
                first = [True]

                def failing_once(engine, alpha):
                    if first[0]:
                        first[0] = False
                        raise ArithmeticError("Injected local numerical failure")
                    return calculate(engine, alpha)

                with patch.object(response_type, "calculate", new=failing_once):
                    result = experiment.solve(initial_policy=np.full(3, .5), max_evaluations=80,
                                              prefer_local=True)
                self.assertTrue(result["verified"])
                self.assertTrue(result["search_metadata"]["fallback_used"])
                self.assertTrue(any(row.get("stage") == "local numerical failure"
                                    for row in result["history"]))

    def test_sparse_projections_match_full_joint_marginals_and_selected_mass(self):
        experiment = ConditionalExperiment(n=7, params=Parameters(rho=.6))
        for response_type in (_JointResponse, _ConditionalResponse):
            with self.subTest(response=response_type.__name__):
                engine = response_type(experiment)
                self.assertTrue(issparse(engine.marginal))
                mass = np.arange(1, len(engine.i)+1, dtype=float)
                mass /= mass.sum()
                joint = np.zeros((7, 7))
                shares = mass/np.where(engine.i == engine.j, 1, 2)
                joint[engine.i, engine.j] = shares
                joint[engine.j, engine.i] = shares
                assert_allclose(engine.marginal@mass, joint.sum(axis=1), atol=1e-16, rtol=0)
                self.assertLessEqual(engine.marginal.nnz, 2*len(mass))
                if response_type is _ConditionalResponse:
                    self.assertTrue(issparse(engine.sale_marginal))
                    assert_allclose(engine.sale_marginal@mass,
                                    (joint*experiment.game.D).sum(axis=1), atol=1e-16, rtol=0)


if __name__ == "__main__":
    unittest.main()
