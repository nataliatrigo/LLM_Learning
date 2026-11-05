"""Independent numerical checks for the endogenous scalar benchmark.

Run from the project root with:
    .venv/bin/python -m unittest oblivious.oe_p0.code.test_experiment -v

These checks use analytical special cases and observed simulated events. They
do not claim continuous-state equilibrium or prove long-run ergodicity.
"""

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betaln
from scipy.sparse import csr_matrix, eye, kron
from scipy.stats import beta

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_p0.code.core import (
    ScalarExperiment, checked_joint_stationary, find_fixed_points,
    posterior_mean,
)
from oblivious.oe_p0.code.diagnostics import (
    compare_existing, demand_diagnostics, load_existing_references, simulate_ts,
)


class ScalarModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.myopic = ScalarExperiment(n=9, params=Parameters(gamma=0, rho=.6))
        cls.myopic_result = cls.myopic.evaluate(.43)
        cls.dynamic = ScalarExperiment(n=11, params=Parameters(rho=.95))
        cls.dynamic_result = cls.dynamic.evaluate(.5)

    def test_posterior_mean_is_not_reputation(self):
        x = np.array([0., .17, .5, .83, 1.])
        for rho in (.6, .95, .999):
            p = Parameters(rho=rho)
            a, b = posterior(x, p)
            actual = posterior_mean(x, p)
            assert_allclose(actual, a / (a + b), atol=2e-16)
            assert_allclose(actual, (1-rho+x)/(3-2*rho), atol=2e-16)
            self.assertGreater(actual[0], 0.)
            self.assertLess(actual[-1], 1.)

    def test_myopic_dp_has_analytical_solution_and_beta_tail(self):
        exp, result = self.myopic, self.myopic_result
        p, grid = exp.params, exp.grid
        a, b = posterior(grid, p)
        exact_demand = beta.sf(result["p0"], a, b)
        assert_allclose(result["d_fixed"], exact_demand, atol=1e-14)
        assert_allclose(result["value"], (p.R-p.c1)*exact_demand, atol=1e-10)
        assert_allclose(result["advantage"], -(p.c2-p.c1), atol=1e-14)
        assert_array_equal(result["policy"], np.zeros(len(grid)))
        # A mean-threshold indicator would fail at every interior tail value.
        interior = exact_demand[(grid > .2) & (grid < .8)]
        self.assertTrue(np.all((interior > 0) & (interior < 1)))
        self.assertGreater(np.max(np.abs(
            exact_demand-(posterior_mean(grid, p)>result["p0"]))), .1)

    def test_dynamic_bellman_and_conditional_advantage_independently(self):
        exp, result = self.dynamic, self.dynamic_result
        p, grid, value = exp.params, exp.grid, result["value"]
        vs = np.interp(p.rho*grid+1-p.rho, grid, value)
        vf = np.interp(p.rho*grid, grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*vs+(1-p.p1)*vf)
        q2 = p.R-p.c2+p.gamma*(p.p2*vs+(1-p.p2)*vf)
        demand = result["d_fixed"]
        updated = p.gamma*(1-demand)*value+demand*np.maximum(q1,q2)
        assert_allclose(updated, value, atol=2e-9, rtol=0)
        assert_allclose(result["advantage"], q2-q1, atol=2e-13, rtol=0)
        assert_array_equal(result["policy"], result["advantage"] >= 0)

    def test_generated_p0_uses_actual_joint_marginal_and_posterior_mean(self):
        exp, result = self.dynamic, self.dynamic_result
        nu = result["nu"]
        assert_allclose(nu.sum(axis=1), result["mu_a"], atol=1e-12)
        assert_allclose(nu.sum(axis=0), result["mu_b"], atol=1e-12)
        assert_allclose(result["mu_a"], result["mu_b"], atol=1e-9)
        expected = result["mu_b"] @ posterior_mean(exp.grid, exp.params)
        self.assertAlmostEqual(result["F"], expected, places=12)
        self.assertAlmostEqual(result["G"], expected-result["p0"], places=12)
        self.assertTrue(result["verified"])


class ActualGameChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = Game(np.linspace(0, 1, 7), Parameters(rho=.95), order=128)

    def test_ts_matrix_symmetry_and_independent_beta_moment(self):
        g, p = self.game, self.game.params
        assert_allclose(g.D.diagonal(), .5, atol=1e-13)
        assert_allclose(g.D+g.D.T, 1, atol=1e-13)
        # Against Beta(1,b0), the other draw's CDF is 1-(1-t)**b0.
        a, b = posterior(g.grid, p)
        _, b0 = posterior(0., p)
        exact = -np.expm1(betaln(a, b+b0)-betaln(a,b))
        assert_allclose(g.D[:, 0], exact, atol=2e-12, rtol=0)

    def test_joint_kernel_updates_exactly_one_seller_and_is_symmetric(self):
        g = self.game
        own_policy = np.array([0, 1, 1, 0, 0, 1, 0], dtype=float)
        a = np.broadcast_to(own_policy[:, None], (g.n, g.n))
        b = a.T
        matrix = g.transition(a,b).tocoo()
        self.assertGreaterEqual(matrix.data.min(), 0)
        assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1, atol=2e-15)
        source_a, source_b = np.divmod(matrix.row, g.n)
        dest_a, dest_b = np.divmod(matrix.col, g.n)
        positive = matrix.data > 0
        self.assertTrue(np.all(((source_a == dest_a) | (source_b == dest_b))[positive]))
        permutation = np.arange(g.n*g.n).reshape(g.n,g.n).T.ravel()
        dense = matrix.toarray()
        assert_allclose(dense, dense[np.ix_(permutation,permutation)], atol=2e-15)

    def test_stationary_known_law_including_slow_chain(self):
        g = Game(np.linspace(0,1,3), Parameters(rho=.6), order=32)
        marginal = np.array([.2,.3,.5])
        refresh = csr_matrix(np.tile(marginal,(3,1)))
        identity = eye(3,format="csr")
        updates = .5*(kron(refresh,identity)+kron(identity,refresh))
        exact = np.outer(marginal,marginal)
        for epsilon in (1.,1e-8):
            with self.subTest(epsilon=epsilon):
                matrix = (1-epsilon)*eye(9,format="csr")+epsilon*updates
                answer = checked_joint_stationary(g,matrix)
                nu = answer["nu"]
                assert_allclose(nu, exact, atol=1e-7, rtol=0)
                self.assertGreaterEqual(nu.min(),0)
                self.assertAlmostEqual(nu.sum(),1.,places=12)
                self.assertLess(np.abs(nu.ravel()@matrix-nu.ravel()).sum(),1e-10)

    def test_stationary_rejects_invalid_transition_probabilities(self):
        g = Game(np.linspace(0,1,3), Parameters(rho=.6), order=32)
        matrices = []
        negative = np.eye(9)
        negative[0,:2] = [1.1,-.1]
        matrices.append(negative)
        matrices.append(np.eye(9)*.9)
        nonfinite = np.eye(9)
        nonfinite[0,0] = np.nan
        matrices.append(nonfinite)
        for matrix in matrices:
            with self.subTest(matrix=matrix[0].tolist()):
                with self.assertRaises(ValueError):
                    checked_joint_stationary(g,csr_matrix(matrix))

    def test_stationary_does_not_certify_unresolved_reducible_chain(self):
        g = Game(np.linspace(0,1,3), Parameters(rho=.6), order=32)
        # Every distribution is invariant for the identity matrix. A tiny
        # residual here cannot select or certify the desired long-run law.
        with self.assertRaises(ArithmeticError):
            checked_joint_stationary(g,eye(9,format="csr"))


class MixedPolicyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = ScalarExperiment(n=9,params=Parameters(rho=.6))
        cls.myopic = ScalarExperiment(n=9,params=Parameters(rho=.6,gamma=0))

    def test_fractional_policy_uses_expected_cost_and_success_probability(self):
        exp = self.experiment
        alpha = np.linspace(.1,.9,exp.n)
        result = exp.evaluate_mixed(.43,alpha)
        p,grid,value = exp.params,exp.grid,result["value"]
        vs=np.interp(p.rho*grid+1-p.rho,grid,value)
        vf=np.interp(p.rho*grid,grid,value)
        success=p.p1+(p.p2-p.p1)*alpha
        cost=p.c1+(p.c2-p.c1)*alpha
        updated=p.gamma*(1-result["d_fixed"])*value+result["d_fixed"]*(
            p.R-cost+p.gamma*(success*vs+(1-success)*vf))
        assert_allclose(updated,value,atol=2e-10,rtol=0)
        q1=p.R-p.c1+p.gamma*(p.p1*vs+(1-p.p1)*vf)
        q2=p.R-p.c2+p.gamma*(p.p2*vs+(1-p.p2)*vf)
        conditional_regret=np.max(np.maximum(q1,q2)-((1-alpha)*q1+alpha*q2))
        self.assertGreater(conditional_regret,.01)
        self.assertFalse(result["verified"])
        assert_allclose(result["policy"],alpha,atol=0,rtol=0)
        assert_allclose(result["advantage"],q2-q1,atol=2e-13,rtol=0)

    def test_scalar_consistency_cannot_certify_a_profitable_deviation(self):
        exp=self.myopic
        alpha=np.full(exp.n,.5)
        first=exp.evaluate_mixed(.43,alpha)
        result=exp.evaluate_mixed(first["F"],alpha)
        # For a fixed mixed policy, its actual-game invariant law is independent
        # of the assumed p0. This construction makes G zero despite a strictly
        # suboptimal randomization when future rewards have zero weight.
        self.assertLess(abs(result["G"]),1e-12)
        assert_allclose(result["value"],result["d_fixed"]*(exp.params.R-
                        .5*(exp.params.c1+exp.params.c2)),atol=1e-12,rtol=0)
        assert_allclose(result["advantage"],-(exp.params.c2-exp.params.c1),atol=1e-14)
        self.assertFalse(result["verified"])
        self.assertFalse(result["candidate_verified"])

    def test_mixed_joint_kernel_averages_actions_only_for_the_selected_seller(self):
        g=self.experiment.game
        alpha=.37
        mixed=g.transition(alpha,alpha)
        exact=(1-alpha)*g.transition(0.,0.)+alpha*g.transition(1.,1.)
        assert_allclose(mixed.toarray(),exact.toarray(),atol=2e-15,rtol=0)
        assert_allclose(g.rewards(alpha,alpha),(1-alpha)*g.rewards(0.,0.)+
                        alpha*g.rewards(1.,1.),atol=2e-15,rtol=0)

    def test_mixed_evaluation_rejects_invalid_probabilities(self):
        exp=self.experiment
        for alpha in (np.full(exp.n,-.01),np.full(exp.n,1.01),
                      np.full(exp.n,np.nan),np.ones(exp.n+1)):
            with self.subTest(alpha=alpha.tolist()):
                with self.assertRaises(ValueError):
                    exp.evaluate_mixed(.43,alpha)

    def test_high_rho_mixture_gets_strong_stationary_check_without_false_consistency(self):
        exp=ScalarExperiment(n=7,params=Parameters(rho=.99))
        # This benchmark makes product choices exactly indifferent at x=1/6.
        # Half mixing there is optimal, but it does NOT make F equal p0.
        p0=.20904026228447609
        alpha=exp.evaluate(p0)["policy"].copy()
        alpha[1]=.5
        result=exp.evaluate_mixed(p0,alpha)
        self.assertTrue(result["verified"])
        self.assertFalse(result["candidate_verified"])
        self.assertGreater(abs(result["G"]),.001)
        self.assertLess(abs(result["advantage"][1]),1e-10)
        stationary=result["stationary_diagnostics"]
        self.assertFalse(stationary["exploratory"])
        self.assertTrue(any("GTH" in method for method in stationary["methods"]))
        sigma=np.broadcast_to(alpha[:,None],(exp.n,exp.n))
        transition=exp.game.transition(sigma,sigma.T)
        self.assertLess(np.abs(transition.T@result["nu"].ravel()-result["nu"].ravel()).sum(),1e-11)


class MixedSearchChecks(unittest.TestCase):
    def test_genuine_interior_candidates_fill_the_known_policy_jumps(self):
        for n,expected_p0 in ((31,.662667766),(51,.680474794)):
            with self.subTest(n=n):
                exp=ScalarExperiment(n=n,params=Parameters(rho=.95))
                search=exp.search(p0_points=9,consistency_tol=1e-7)
                interior=[r for r in search["candidates"] if r["mixed_nodes"]>0
                          and abs(r["p0"]-expected_p0)<1e-6]
                self.assertTrue(interior,"A genuine interior candidate must resolve this known jump")
                for r in interior:
                    self.assertTrue(r["verified"])
                    self.assertTrue(r["candidate_verified"])
                    p,grid,value=exp.params,exp.grid,r["value"]
                    alpha=r["policy"]
                    fractional=(alpha>0)&(alpha<1)
                    vs=np.interp(p.rho*grid+1-p.rho,grid,value)
                    vf=np.interp(p.rho*grid,grid,value)
                    q1=p.R-p.c1+p.gamma*(p.p1*vs+(1-p.p1)*vf)
                    q2=p.R-p.c2+p.gamma*(p.p2*vs+(1-p.p2)*vf)
                    assert_allclose((q2-q1)[fractional],0,atol=1e-10,rtol=0)
                    assert_allclose(p.gamma*(1-r["d_fixed"])*value+
                                    r["d_fixed"]*np.maximum(q1,q2),value,atol=1e-10,rtol=0)
                    sigma_a=np.broadcast_to(alpha[:,None],(n,n))
                    transition=exp.game.transition(sigma_a,sigma_a.T)
                    self.assertLess(np.abs(transition.T@r["nu"].ravel()-r["nu"].ravel()).sum(),1e-11)
                    generated=r["nu"].sum(axis=0)@posterior_mean(grid,p)
                    self.assertLess(abs(generated-r["p0"]),1e-7)
                    self.assertLess(r["dp_diagnostics"]["conditional_regret"],1e-10)
                    self.assertLess(r["dp_diagnostics"]["mixed_indifference_residual"],1e-10)


class WorkflowChecks(unittest.TestCase):
    def test_cli_places_one_final_run_in_separate_data_and_figures_folders(self):
        root=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="endogenous_p0_cli_") as temp:
            temp=Path(temp)
            environment=dict(os.environ,MPLBACKEND="Agg",MPLCONFIGDIR=str(temp/"mpl"),
                             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
            command=[sys.executable,"-B","-m","oblivious.oe_p0.code.run_experiment",
                     "--output",str(temp/"data"),"--figures-output",str(temp/"figures"),
                     "--name","check","--rhos",".6","--grids","3","--p0-points","3",
                     "--gamma","0","--periods","20","--burn-in","0","--trajectories","2"]
            completed=subprocess.run(command,cwd=root,env=environment,capture_output=True,
                                     text=True,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            data,figures=temp/"data"/"check",temp/"figures"/"check"
            manifest=json.loads((data/"manifest.json").read_text())
            self.assertEqual(manifest["status"],"complete")
            self.assertIn("Mixed",manifest["strategy_space"])
            self.assertEqual(len(manifest["completed_cases"]),1)
            self.assertEqual(len(manifest["completed_diagnostics"]),1)
            self.assertFalse((data/"figures").exists())
            expected_figures = {"01_fixed_point_map", "02_policy", "03_advantage",
                                "04_joint_stationary", "07_demand_comparison",
                                "08_reference_policies"}
            self.assertEqual({p.stem for p in figures.glob("*.png")}, expected_figures)
            self.assertEqual({p.stem for p in figures.glob("*.pdf")}, expected_figures)
            with (data/"fixed_points.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),1)
            with np.load(data/"arrays"/"rho_0p6_n003.npz") as stored:
                assert_array_equal(stored["root/selected/policy"],np.zeros(3))


class FixedPointSearchChecks(unittest.TestCase):
    @staticmethod
    def evaluate_step(p0, *, increasing):
        low = p0 < .5
        generated = (.3 if low else .7) if increasing else (.75 if low else .25)
        return dict(p0=float(p0), F=generated, G=generated-p0,
                    policy=np.array([not low],dtype=float), verified=True)

    def test_policy_jump_sign_change_is_not_certified_as_root(self):
        result = find_fixed_points(
            lambda p: self.evaluate_step(p,increasing=False),
            (.1,.9),p0_points=17,consistency_tol=1e-8)
        self.assertEqual(len(result["candidates"]),0)
        self.assertGreaterEqual(abs(result["best_diagnostic"]["G"]),.249999)

    def test_multiple_policy_consistent_roots_are_retained(self):
        result = find_fixed_points(
            lambda p: self.evaluate_step(p,increasing=True),
            (.1,.9),p0_points=17,consistency_tol=1e-8)
        roots = sorted(r["p0"] for r in result["candidates"])
        assert_allclose(roots,[.3,.7],atol=1e-8,rtol=0)
        for candidate in result["candidates"]:
            self.assertLessEqual(abs(candidate["G"]),1e-8)

    def test_unverified_map_values_cannot_become_fixed_points(self):
        def unverified(p0):
            return dict(p0=p0,F=p0,G=0.,policy=np.zeros(1),verified=False)
        result = find_fixed_points(unverified,(.1,.9),p0_points=5)
        self.assertEqual(len(result["candidates"]),0)


class DiagnosticsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = ScalarExperiment(n=9,params=Parameters(gamma=0,rho=.6))
        cls.evaluation = cls.experiment.evaluate(.43)
        cls.simulation = simulate_ts(cls.experiment.game,cls.evaluation,
                                     periods=2500,burn_in=100,trajectories=3,seed=14731)

    def test_simulation_records_two_ts_draws_and_selects_actual_winner(self):
        a = self.simulation["arrays"]
        self.assertEqual(a["theta"].shape,(3,2500,2))
        self.assertTrue(np.all((a["theta"]>0)&(a["theta"]<1)))
        assert_array_equal(a["selected"],a["theta"][:,:,1]>a["theta"][:,:,0])
        expected_mean = posterior_mean(a["states"],self.experiment.params)
        assert_allclose(a["posterior_mean"],expected_mean,atol=1e-14)
        # Unselected sellers also have fresh recorded draws, not missing data.
        self.assertTrue(np.isfinite(a["theta"]).all())

    def test_simulation_updates_only_the_selected_seller(self):
        a, p = self.simulation["arrays"], self.experiment.params
        expected = a["states"].copy()
        path, period = np.indices(a["selected"].shape)
        seller = a["selected"].astype(int)
        expected[path,period,seller] = (
            p.rho*expected[path,period,seller]+(1-p.rho)*a["success"])
        observed = np.concatenate((a["states"][:,1:],a["final_state"][:,None,:]),axis=1)
        assert_allclose(observed,expected,atol=1e-14,rtol=0)
        own = a["states"][path,period,seller]
        probability = np.interp(own,self.experiment.grid,self.evaluation["policy"])
        assert_allclose(a["chosen_probability"],probability,atol=1e-14,rtol=0)
        assert_array_equal(a["product"],1+(a["product_uniform"]<probability))

    def test_recorded_running_averages_and_ts_mean_difference(self):
        a, p = self.simulation["arrays"], self.experiment.params
        times = a["running_periods"].astype(int)
        assert_allclose(a["running_ts_b"],np.cumsum(a["theta"][:,:,1],axis=1)[:,times-1]/times)
        assert_allclose(a["running_posterior_b"],np.cumsum(a["posterior_mean"][:,:,1],axis=1)[:,times-1]/times)
        shape_a, shape_b = posterior(a["states"],p)
        conditional_variance = shape_a*shape_b/((shape_a+shape_b)**2*(shape_a+shape_b+1))
        residual = a["theta"]-a["posterior_mean"]
        for seller in (0,1):
            # Conditional differences form a bounded martingale difference;
            # this uncertainty concerns draws minus means, not mixing of X.
            standardized = residual[:,:,seller].sum()/np.sqrt(conditional_variance[:,:,seller].sum())
            self.assertLess(abs(standardized),6.)

    def test_conditional_demand_uses_joint_dependence(self):
        g = Game(np.linspace(0,1,3),Parameters(rho=.6),order=64)
        marginal = np.array([.2,.5,.3])
        fixed = beta.sf(.43,*posterior(g.grid,g.params))
        evaluation = dict(grid=g.grid,p0=.43,F=.4,G=-.03,nu=np.diag(marginal),
                          mu_a=marginal,mu_b=marginal,d_fixed=fixed,
                          policy=np.zeros(3),advantage=-np.ones(3))
        result = demand_diagnostics(g,evaluation)["arrays"]
        assert_allclose(result["d_conditional"],.5,atol=1e-13)
        assert_allclose(result["d_marginal"],g.D@marginal,atol=1e-13)
        assert_allclose(result["d_fixed"],fixed,atol=1e-13)
        self.assertGreater(np.max(np.abs(result["d_conditional"]-result["d_marginal"])),.1)

    def test_simulation_preserves_multiple_product_two_intervals(self):
        g = Game(np.linspace(0,1,9),Parameters(rho=.6),order=32)
        alpha = np.array([0.,1.,1.,0.,0.,1.,1.,0.,0.])
        evaluation = dict(p0=.5,F=.5,policy=alpha,nu=np.ones((9,9))/81)
        a = simulate_ts(g,evaluation,periods=1500,burn_in=0,trajectories=2,
                        seed=771)["arrays"]
        path,period = np.indices(a["selected"].shape)
        own = a["states"][path,period,a["selected"].astype(int)]
        probability = np.interp(own,g.grid,alpha)
        assert_allclose(a["chosen_probability"],probability,atol=1e-14,rtol=0)
        chosen_two = a["product_uniform"]<probability
        assert_array_equal(a["product"],1+chosen_two)
        self.assertTrue(np.any(chosen_two & (own<.5)))
        self.assertTrue(np.any(chosen_two & (own>.5)))

    def test_mixed_simulation_randomizes_product_independently_and_uses_realized_product(self):
        g=Game(np.linspace(0,1,9),Parameters(rho=.6),order=32)
        alpha=np.linspace(.15,.85,g.n)
        evaluation=dict(p0=.5,F=.5,policy=alpha,nu=np.ones((g.n,g.n))/g.n**2)
        a=simulate_ts(g,evaluation,periods=6000,burn_in=100,trajectories=3,seed=9831)["arrays"]
        path,period=np.indices(a["selected"].shape)
        own=a["states"][path,period,a["selected"].astype(int)]
        probability=np.interp(own,g.grid,alpha)
        assert_allclose(a["chosen_probability"],probability,atol=1e-14,rtol=0)
        chosen_two=a["product_uniform"]<probability
        assert_array_equal(a["product"],1+chosen_two)
        success_probability=g.params.p1+(g.params.p2-g.params.p1)*chosen_two
        assert_array_equal(a["success"],a["success_uniform"]<success_probability)
        self.assertTrue(np.any(chosen_two))
        self.assertTrue(np.any(~chosen_two))
        # The random product coin is separate from the customer selection and
        # success coins. A shared uniform stream would fail these correlations.
        for other in (a["success_uniform"],a["theta"][:,:,0],a["theta"][:,:,1]):
            self.assertLess(abs(np.corrcoef(a["product_uniform"].ravel(),other.ravel())[0,1]),.05)
        for product,p_success in ((1,g.params.p1),(2,g.params.p2)):
            mask=a["product"]==product
            z=(a["success"][mask].sum()-p_success*mask.sum())/np.sqrt(mask.sum()*p_success*(1-p_success))
            self.assertLess(abs(z),6.)

    @staticmethod
    def write_real_marginal_reference(root, name, params, *, n=3, order=256,
                                      verified=True, policy=None):
        # Synthetic certificates exercise file loading and scope, not optimality.
        folder = root / "oblivious/01_oe_distribution" / "data" / name
        arrays = folder / "arrays"
        arrays.mkdir(parents=True)
        grid = np.linspace(0, 1, n)
        alpha = np.linspace(0, .8, n) if policy is None else np.asarray(policy)
        mu = np.arange(1, n + 1, dtype=float)
        mu /= mu.sum()
        nu = np.outer(mu, mu)
        stored = {key: value for key, value in dict(
            grid=grid, policy=alpha, mu=mu, demand=np.linspace(.1, .9, n),
            nu=nu, mu_a=mu, mu_b=mu).items()}
        selected = {key: {"__array__": key} for key in stored}
        selected.update(verified=verified, candidate_verified=verified,
                        dp_diagnostics=dict(verified=verified),
                        stationary_diagnostics=dict(verified=verified),
                        summary=dict(verified=verified))
        tag = "rho_" + str(params.rho).replace(".", "p") + f"_n{n:03d}"
        metadata = dict(rho=params.rho, n=n, selected=selected,
                        configuration=dict(parameters=asdict(params), n=n,
                                           order=order, tolerance=1e-7))
        (folder / "manifest.json").write_text(json.dumps(dict(status="complete",
            configuration=dict(parameters=asdict(params)), completed_cases=[tag])))
        (arrays / (tag + ".json")).write_text(json.dumps(metadata))
        np.savez(arrays / (tag + ".npz"), **stored)
        return arrays / tag

    def test_reference_loader_excludes_mismatch_unverified_and_invalid_probabilities(self):
        from dataclasses import replace

        p = self.experiment.params
        with tempfile.TemporaryDirectory(prefix="endogenous_p0_references_") as temp:
            root = Path(temp)
            self.write_real_marginal_reference(root, "wrong_cost", replace(p, c2=.7))
            self.write_real_marginal_reference(root, "wrong_rho", replace(p, rho=.7))
            self.write_real_marginal_reference(root, "unverified", p, verified=False)
            self.write_real_marginal_reference(root, "invalid", p, policy=[0., 2., 1.])
            result = load_existing_references(p, root)
            self.assertEqual(result["references"], [])
            reasons = " ".join(row["reason"] for row in result["excluded"])
            self.assertIn("parameters", reasons)
            self.assertIn("rho", reasons)
            self.assertIn("not verified", reasons)
            self.assertIn("probabilities", reasons)

    def test_reference_loader_prefers_same_grid_then_finest_verified_grid(self):
        p = self.experiment.params
        with tempfile.TemporaryDirectory(prefix="endogenous_p0_selection_") as temp:
            root = Path(temp)
            self.write_real_marginal_reference(root, "coarse", p, n=3)
            matched_path = self.write_real_marginal_reference(root, "matching", p, n=5, order=64)
            with np.load(matched_path.with_suffix(".npz"), allow_pickle=False) as stored:
                native = {key: stored[key].copy() for key in stored.files}
            native["demand"][-1] = np.nextafter(1., 2.)
            np.savez(matched_path.with_suffix(".npz"), **native)
            self.write_real_marginal_reference(root, "finest", p, n=7)
            self.write_real_marginal_reference(root, "unverified", p, n=9, verified=False)
            matching = load_existing_references(p, root, n=5, order=64)
            ref = matching["references"][0]
            self.assertEqual(ref["kind"], "OE_REAL_MARGINAL")
            self.assertEqual(ref["label"], "OE distribution")
            self.assertEqual(len(ref["grid"]), 5)
            self.assertEqual(ref["source_order"], 64)
            self.assertEqual(ref["demand"][-1], np.nextafter(1., 2.))
            self.assertEqual(sum(item["included"] for item in matching["catalog"]), 1)
            self.assertTrue(matching["provenance"])
            self.assertTrue(all(path.startswith("oblivious/01_oe_distribution/data/")
                                for path in matching["provenance"]))
            self.assertEqual(len(load_existing_references(p, root, n=11)["references"][0]["grid"]), 7)
            self.assertEqual(len(load_existing_references(p, root)["references"][0]["grid"]), 7)

    def test_reference_transfer_preserves_own_reputation_axes_and_certificate_scope(self):
        p = self.experiment.params
        with tempfile.TemporaryDirectory(prefix="endogenous_p0_transfer_") as temp:
            root = Path(temp)
            alpha = np.array([0., .2, .8])
            self.write_real_marginal_reference(root, "fixture", p, policy=alpha)
            result = compare_existing(self.experiment.game, self.evaluation, root)
            row = result["summary"]["references"][0]
            self.assertEqual(row["model"], "OE_REAL_MARGINAL")
            self.assertTrue(row["native_certificate_verified"])
            self.assertIn("certificate not transferred", row["evaluation_profile"])
            self.assertEqual(row["status"], "evaluated")
            self.assertEqual(row["symmetry_error"], 0.)
            arrays, grid = result["arrays"], self.experiment.grid
            native_grid = np.linspace(0, 1, 3)
            transferred = np.interp(grid, native_grid, alpha)
            assert_allclose(arrays["reference_000_sigma_a"],
                            np.broadcast_to(transferred[:, None], (9, 9)))
            assert_allclose(arrays["reference_000_sigma_b"],
                            np.broadcast_to(transferred[None, :], (9, 9)))
            assert_allclose(arrays["reference_000_native_policy"], alpha)
            assert_allclose(arrays["reference_000_native_nu"].sum(axis=0),
                            arrays["reference_000_native_mu"])
            self.assertNotIn("equilibrium_verified", row)


if __name__ == "__main__":
    unittest.main()
