"""Independent slot-replacement and seeded Monte Carlo verification."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import betaincc

from forgetting_random_replacement.code.model import Primitives,solve_counts
from forgetting_random_replacement.code.simulation import (
    CountState,_mean_interval,_ratio_interval,simulate_calendar,simulate_interactions,
)


def scalar_slot_update(m,successes,incoming,capacity,uniform):
    """Use an actual vector of integer outcomes and select one slot uniformly.

    Reordering slots before a uniform draw has no effect on its distribution;
    arranging successes first also permits exact common-uniform replay.
    """
    slots = [1]*successes+[0]*(m-successes)
    if m < capacity:
        slots.append(int(incoming))
        removed = -1
    else:
        index = int(uniform*capacity)
        removed = slots[index]
        slots[index] = int(incoming)
    return len(slots),sum(slots),removed


class SimulationTests(unittest.TestCase):
    def test_user_example_failure_removes_success_with_probability_point9(self):
        paths = 10000
        state = CountState(SimpleNamespace(n=20),paths,initial_successes=18)
        uniforms = (np.arange(paths)+.5)/paths
        removed = state.advance(True,False,uniforms)
        self.assertEqual(np.count_nonzero(removed == 1),9000)
        self.assertEqual(np.count_nonzero(state.counts == 17),9000)
        self.assertEqual(np.count_nonzero(state.counts == 18),1000)
        np.testing.assert_array_equal(state.lengths,20)

    def test_removal_distribution_is_independent_of_new_outcome(self):
        uniforms = np.array([.01,.25,.49,.5,.75,.99])
        failures = CountState(SimpleNamespace(n=20),6,initial_successes=10)
        successes = CountState(SimpleNamespace(n=20),6,initial_successes=10)
        old0 = failures.advance(True,False,uniforms)
        old1 = successes.advance(True,True,uniforms)
        np.testing.assert_array_equal(old0,old1)
        np.testing.assert_array_equal(successes.counts,failures.counts+1)

    def test_scalar_slots_growth_boundaries_and_frozen_outside_paths(self):
        rng = np.random.default_rng(7721)
        for n,initial in ((1,None),(3,None),(20,0),(20,20),(1000,None),(1000,500)):
            with self.subTest(n=n,initial=initial):
                state = CountState(SimpleNamespace(n=n),7,initial_successes=initial)
                reference = [(n,initial) if initial is not None else (0,0)]*7
                for _ in range(2*n+19):
                    buyers,successes,uniforms = rng.random(7)<.57,rng.random(7)<.61,rng.random(7)
                    old_m,old_s = state.lengths.copy(),state.counts.copy()
                    removed = state.advance(buyers,successes,uniforms)
                    for path in range(7):
                        expected_old = -1
                        if buyers[path]:
                            m,s,expected_old = scalar_slot_update(*reference[path],successes[path],n,uniforms[path])
                            reference[path] = m,s
                        self.assertEqual((state.lengths[path],state.counts[path]),reference[path])
                        self.assertEqual(removed[path],expected_old)
                    np.testing.assert_array_equal(state.lengths[~buyers],old_m[~buyers])
                    np.testing.assert_array_equal(state.counts[~buyers],old_s[~buyers])

    def test_interaction_records_precede_outcome_and_keep_removal_type(self):
        paths,times,seed = 4,75,724
        for n,initial in ((1,None),(3,None),(20,18),(1000,500)):
            with self.subTest(n=n,initial=initial):
                policy = solve_counts(Primitives(),n)
                records,summary = simulate_interactions(policy,paths=paths,interactions=times,
                    seed=seed,burn_in=7,initial_successes=initial)
                rng = np.random.default_rng(seed)
                states = [(n,initial) if initial is not None else (0,0)]*paths
                for time in range(times):
                    outcome_u,removal_u = rng.random(paths),rng.random(paths)
                    for path,(m,s) in enumerate(states):
                        row = records[time,path]
                        self.assertEqual((row['m_before'],row['S_before']),(m,s))
                        p2 = policy.advantages[m][s] > policy.action_tolerance
                        self.assertEqual(row['product2'],p2)
                        demand = betaincc(1+s,1+m-s,policy.primitives.p0)
                        self.assertAlmostEqual(float(row['demand']),demand,places=7)
                        y = outcome_u[path] < (policy.primitives.p2 if p2 else policy.primitives.p1)
                        self.assertEqual(row['success'],y)
                        nm,ns,removed = scalar_slot_update(m,s,y,n,removal_u[path])
                        self.assertEqual(row['removed_success'],removed)
                        states[path] = nm,ns
                np.testing.assert_array_equal(summary.final_remembered_successes,[s for _,s in states])
                np.testing.assert_array_equal(summary.final_remembered_successes,
                    summary.initial_successes+summary.final_successes-summary.removed_success_count)
                np.testing.assert_array_equal(summary.final_window_length,
                    summary.initial_window_length+times-summary.replacement_count)

    def test_calendar_matches_independent_scalar_slot_replay_and_profit(self):
        paths,periods,seed,width = 7,61,98217,8
        for n,initial in ((1,None),(3,None),(20,18),(1000,500)):
            with self.subTest(n=n,initial=initial):
                p = Primitives()
                policy = solve_counts(p,n)
                actual,bins,summary = simulate_calendar(policy,n_paths=paths,max_periods=periods,
                    seed=seed,bin_width=width,initial_successes=initial)
                states = [(n,initial) if initial is not None else (0,0)]*paths
                observed,total_y,removed_y,replacements = [np.zeros(paths,int) for _ in range(4)]
                profit = np.zeros(paths)
                all_a,all_p2 = np.zeros((periods,paths),bool),np.zeros((periods,paths),bool)
                rng = np.random.default_rng(seed)
                for time in range(periods):
                    demand_u,outcome_u,removal_u = rng.random(paths),rng.random(paths),rng.random(paths)
                    self.assertEqual(actual.loc[time,'mean_observation_n'],observed.mean())
                    self.assertEqual(actual.loc[time,'mean_remembered_successes'],np.mean([s for _,s in states]))
                    for path,(m,s) in enumerate(states):
                        p2 = policy.advantages[m][s] > policy.action_tolerance
                        d = betaincc(1+s,1+m-s,p.p0)
                        buys = demand_u[path] < d
                        y = outcome_u[path] < (p.p2 if p2 else p.p1)
                        all_a[time,path],all_p2[time,path] = buys,buys and p2
                        if buys:
                            m,s,removed = scalar_slot_update(m,s,y,n,removal_u[path])
                            states[path] = m,s
                            observed[path] += 1
                            total_y[path] += y
                            removed_y[path] += removed == 1
                            replacements[path] += removed >= 0
                            profit[path] += p.gamma**time*(p.net_rewards[1] if p2 else p.net_rewards[0])
                    self.assertEqual(actual.loc[time,'a_chosen_count'],all_a[time].sum())
                    self.assertEqual(actual.loc[time,'product2_chosen_count'],all_p2[time].sum())
                for column,expected in [('final_observation_n',observed),('final_successes',total_y),
                    ('removed_success_count',removed_y),('replacement_count',replacements),
                    ('final_remembered_successes',[s for _,s in states]),('product2_count',all_p2.sum(axis=0))]:
                    np.testing.assert_array_equal(summary[column],expected)
                np.testing.assert_allclose(summary.discounted_profit,profit,rtol=1e-14,atol=1e-14)
                np.testing.assert_array_equal(summary.final_remembered_successes,
                    summary.initial_successes+summary.final_successes-summary.removed_success_count)
                np.testing.assert_array_equal(summary.final_window_length,
                    summary.initial_window_length+summary.final_observation_n-summary.replacement_count)
                for b,row in bins.iterrows():
                    start,stop = b*width,min((b+1)*width,periods)
                    a,p2 = all_a[start:stop],all_p2[start:stop]
                    self.assertEqual(row.a_chosen_count,a.sum())
                    self.assertEqual(row.product2_chosen_count,p2.sum())
                    self.assertAlmostEqual(row.a_market_share,a.mean())
                    self.assertAlmostEqual(row.product2_unconditional_rate,p2.mean())

    def test_cluster_intervals_and_input_validation(self):
        numerator,denominator = np.array([1.,5.,2.,0.]),np.array([3.,8.,5.,1.])
        estimate,se,lower,upper = _ratio_interval(numerator,denominator)
        expected = numerator.sum()/denominator.sum()
        expected_se = np.std(numerator-expected*denominator,ddof=1)/(np.sqrt(4)*denominator.mean())
        self.assertAlmostEqual(estimate,expected)
        self.assertAlmostEqual(se,expected_se)
        self.assertLess(lower,estimate)
        self.assertGreater(upper,estimate)
        self.assertTrue(np.isnan(_ratio_interval([0,0],[0,0])[0]))
        self.assertTrue(np.isnan(_mean_interval([1])[1]))
        for initial in (-1,4,1.5,True):
            with self.assertRaises(ValueError):
                CountState(SimpleNamespace(n=3),2,initial_successes=initial)
        state = CountState(SimpleNamespace(n=3),2)
        for uniform in (-.1,1.,np.nan):
            with self.assertRaises(ValueError):
                state.advance(True,True,uniform)


if __name__ == '__main__':
    unittest.main()
