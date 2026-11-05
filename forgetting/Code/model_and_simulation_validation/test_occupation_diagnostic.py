"""Strict region, tolerance discrepancies, and recorded interaction histories."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from dataclasses import replace
import unittest
import numpy as np

from forgetting.Code.experiments.occupation_diagnostic import (RECORD_DTYPE, distance_to_region,
    positive_components, occupation_statistics, stream_history)
from forgetting.Code.layers import solve_observation_layers
from forgetting.Code.model import Primitives, solve_boundary


class OccupationDiagnosticTests(unittest.TestCase):
    def test_disconnected_components_and_distances(self):
        solution=SimpleNamespace(rho=.95,coordinate=np.array([0.,.2,.4,.6,.8,1.]),
                                 advantage=np.array([-1.,1.,-1.,-1.,1.,-1.]))
        components=positive_components(solution)
        np.testing.assert_allclose([[c['lower'],c['upper']] for c in components],[[.1,.3],[.7,.9]])
        np.testing.assert_allclose(distance_to_region([0,.2,.5,.8,1],components),[.1,0,.2,0,.1])
        self.assertTrue(np.isinf(distance_to_region(.5,[])))
        self.assertTrue(all(c['reachable_intersection']=='yes: K=[0,1]' for c in components))

    def test_strict_region_and_tie_tolerance_are_reported_separately(self):
        records=np.zeros((4,2),dtype=RECORD_DTYPE)
        records['x_before']=[[.1,.5000005],[.4,.6],[.9,.2],[.8,.3]]
        records['mature_advantage_fine']=records['x_before']-.5
        records['product2']=records['mature_advantage_fine']>1e-6
        components=[{'lower':.5,'upper':1.}]
        policy={'coordinate':np.array([0.,1.]),'advantage_table':np.array([[-.5,.5]])}
        paths,stats,blocks,hist=occupation_statistics(records,components,policy,burn_in=10,
            neighborhoods=[.001,.01,.05],action_tolerance=1e-6,block_size=2)
        self.assertEqual(stats['strict_fine_occupation'],.5)
        self.assertEqual(stats['product2_frequency'],.375)
        self.assertEqual(stats['fine_strict_action_mismatch_count'],1)
        self.assertEqual(stats['policy_rule_mismatch_count'],0)
        self.assertAlmostEqual(stats['fine_occupation_minus_product2'],.125)
        self.assertEqual(hist['count'].sum(),8)
        self.assertAlmostEqual(float((hist.density*(hist.upper-hist.lower)).sum()),1)
        self.assertEqual(blocks.start_interaction.min(),11)
        self.assertEqual(blocks.end_interaction.max(),14)

    def test_recorded_states_are_pre_choice_and_outcomes_use_existing_policy(self):
        p=replace(Primitives(),gamma=0)
        rho=.8
        _,policy=solve_observation_layers(p,rho,nodes=101,terminal_gap=.1,return_policy=True)
        fine=solve_boundary(p,rho,central_nodes=201)
        with TemporaryDirectory() as temporary:
            records,steps,pre,last=stream_history(p,rho,policy,fine,Path(temporary),paths=2,
                interactions=20,burn_in=10,seed=123,action_tolerance=1e-6)
            rng=np.random.default_rng(123)
            x=np.zeros(2)
            expected_steps=np.zeros(2)
            for n in range(1,21):
                success=rng.random(2)<p.p1
                next_x=rho*x+(1-rho)*success
                if n>10:
                    row=records[n-11]
                    np.testing.assert_allclose(row['x_before'],x)
                    np.testing.assert_array_equal(row['success'],success)
                    self.assertFalse(row['product2'].any())
                    expected_steps+=abs(next_x-x)
                x=next_x
            np.testing.assert_allclose(steps,expected_steps/10)
            self.assertFalse(pre.any())
            self.assertFalse(last.any())
            saved=np.load(Path(temporary)/'observations.npy',mmap_mode='r')
            self.assertEqual(saved.shape,(10,2))
            self.assertEqual(saved.dtype,RECORD_DTYPE)
