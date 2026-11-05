"""Bitwise equivalence and memory-lease checks for dense-storage GTH."""

from contextlib import contextmanager
import hashlib
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, diags, eye

from oblivious.oe_p0.code import positive_stationary as original
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_gth as dense
from oblivious.oe_distribution_fixed_p0.code.test_sweep201_parallel_gth import (
    raw_kernel, symmetric_jump,
)


@contextmanager
def uncontented_lease(n):
    yield


class DenseGTHChecks(TestCase):
    def test_small_real_kernels_both_orderings_and_calendar_bitwise(self):
        with patch.object(dense, '_memory_lease', uncontented_lease):
            for n in (5, 11, 31, 51):
                jump, rates = symmetric_jump(n)
                for permutation in (None, np.arange(len(rates))[::-1]):
                    with self.subTest(n=n, reverse=permutation is not None):
                        status, reference = raw_kernel(original._kernel, jump, permutation)
                        new_status, actual = raw_kernel(dense._kernel, jump, permutation)
                        self.assertEqual((status, new_status), (0, 0))
                        self.assertTrue(np.array_equal(actual.view(np.uint64), reference.view(np.uint64)))
                        expected = original.positive_invariant(jump, rates, permutation)
                        with patch.object(original, '_kernel', dense._kernel):
                            actual = original.positive_invariant(jump, rates, permutation)
                        self.assertTrue(np.array_equal(actual.view(np.uint64), expected.view(np.uint64)))
                        self.assertTrue(np.all(actual > 0))

    def test_n101_recorded_original_raw_hashes_and_calendar_residual(self):
        # Original sparse CPP outputs independently recorded before this backend
        # existed; each ordering uses its own original floating-point result.
        expected = {
            False:'0e2e0524444ce3f041253b7f5adeac99ceb2f80d0e3d8f71e74a81d63b466080',
            True:'93fb2f9443a335e9a95744d39c655943a68365b34b1c7e473c8f0304db71d00d',
        }
        jump, rates = symmetric_jump(101)
        grid = np.linspace(0, 1, 101)
        game = OutsideOptionGame(grid, Parameters(rho=.9), p0=.9, order=256)
        policy = np.where((grid >= .2) & (grid <= .7), 1., 0.)
        sigma = np.broadcast_to(policy[:, None], (101, 101))
        full = game.transition(sigma, sigma.T)
        i, j = np.triu_indices(101)
        calendar = []
        with patch.object(dense, '_memory_lease', uncontented_lease):
            for reverse in (False, True):
                perm = np.arange(len(rates))[::-1] if reverse else np.arange(len(rates))
                status, raw = raw_kernel(dense._kernel, jump, perm)
                self.assertEqual(status, 0)
                self.assertEqual(hashlib.sha256(raw.tobytes()).hexdigest(), expected[reverse])
                law = original._calendar(raw.copy(), rates, perm)
                self.assertTrue(np.all(law > 0))
                self.assertAlmostEqual(float(law.sum()), 1., places=14)
                transition_residual = np.sum(np.abs((diags(rates)@jump-diags(rates)).T@law))
                self.assertLess(transition_residual, 1e-11)
                nu = np.zeros((101, 101))
                shares = law/np.where(i == j, 1., 2.)
                nu[i, j], nu[j, i] = shares, shares
                self.assertLess(float(np.sum(np.abs(full.T@nu.ravel()-nu.ravel()))), 1e-11)
                calendar.append(law)
        self.assertLess(float(np.sum(np.abs(calendar[0]-calendar[1]))), 1e-11)

    def test_rare_positive_masses_remain_bitwise_equal(self):
        n = 21
        target = np.logspace(-240, 0, n)
        target /= target.sum()
        up = np.full(n-1, .25)
        down = .25*target[:-1]/target[1:]
        off = diags((down, up), (-1, 1), shape=(n, n), format='csr')
        rates = np.asarray(off.sum(axis=1)).ravel()
        jump = (diags(1/rates)@off).tocsr()
        for perm in (None, np.arange(n)[::-1]):
            reference = original.positive_invariant(jump, rates, perm)
            with patch.object(dense, '_memory_lease', uncontented_lease):
                with patch.object(original, '_kernel', dense._kernel):
                    actual = original.positive_invariant(jump, rates, perm)
            self.assertTrue(np.array_equal(actual.view(np.uint64), reference.view(np.uint64)))
            self.assertTrue(np.all(actual > 0))
            self.assertLess(actual.min(), 1e-239)

    def test_memory_lease_gets_actual_state_count_and_releases_on_status(self):
        events = []

        @contextmanager
        def lease(n):
            events.append(('enter', n))
            try:
                yield
            finally:
                events.append(('leave', n))

        with patch.object(dense, '_memory_lease', lease):
            status, _ = raw_kernel(dense._kernel, eye(3, format='csr'))
        self.assertEqual(status, 1)
        self.assertEqual(events, [('enter', 3), ('leave', 3)])

    def test_original_compile_failure_fallback_and_thread_contract(self):
        jump = csr_matrix([[0., 1.], [1., 0.]])
        with patch.object(original, '_kernel', side_effect=RuntimeError('compiler unavailable')):
            with self.assertWarns(RuntimeWarning):
                law = original.positive_invariant(jump, np.array([.2, .8]))
        np.testing.assert_allclose(law, [.8, .2], atol=1e-15)
        for invalid in (1, 4, None):
            with self.assertRaises(ValueError):
                dense.install(invalid)


if __name__ == '__main__':
    main()
