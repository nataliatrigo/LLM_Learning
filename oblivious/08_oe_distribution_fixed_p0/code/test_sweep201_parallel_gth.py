"""Exact-operation checks for the optional row-parallel GTH backend."""

from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, diags

from oblivious.oe_p0.code import positive_stationary as original
from oblivious.oe_p0.code.core import _jump_system
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_distribution_fixed_p0.code import sweep201_parallel_gth as parallel


def symmetric_jump(n, p0=.9, rho=.9):
    grid = np.linspace(0, 1, n)
    game = OutsideOptionGame(grid, Parameters(rho=rho), p0=p0, order=256)
    policy = np.where((grid >= .2) & (grid <= .7), 1., 0.)
    sigma = np.broadcast_to(policy[:, None], (n, n))
    matrix = game.transition(sigma, sigma.T)
    i, j = np.triu_indices(n)
    groups = np.empty((n, n), dtype=int)
    groups[i, j] = np.arange(len(i))
    groups[j, i] = np.arange(len(i))
    membership = csr_matrix((np.ones(n*n), (np.arange(n*n), groups.ravel())),
                            shape=(n*n, len(i)))
    return _jump_system((matrix[i*n+j]@membership).tocsr())


def raw_kernel(kernel, jump, permutation=None):
    n = jump.shape[0]
    permutation = np.arange(n) if permutation is None else permutation
    reduced = jump[permutation, :][:, permutation].tocsr()
    law = np.empty(n)
    _, function = kernel()
    status = function(n, np.ascontiguousarray(reduced.indptr, dtype=np.int64),
                      np.ascontiguousarray(reduced.indices, dtype=np.int64),
                      np.ascontiguousarray(reduced.data, dtype=np.float64), law)
    return status, law


class ParallelGTHChecks(TestCase):
    def test_real_symmetric_jump_raw_and_calendar_bitwise(self):
        for n in (5, 11, 31, 51):
            jump, rates = symmetric_jump(n)
            for permutation in (None, np.arange(len(rates))[::-1]):
                expected_status, expected = raw_kernel(original._kernel, jump, permutation)
                self.assertEqual(expected_status, 0)
                calendar = original.positive_invariant(jump, rates, permutation)
                for threads in (1, 2, 4):
                    with self.subTest(n=n, threads=threads, reverse=permutation is not None):
                        parallel.set_threads(threads)
                        status, actual = raw_kernel(parallel._kernel, jump, permutation)
                        self.assertEqual(status, 0)
                        self.assertTrue(np.array_equal(actual.view(np.uint64), expected.view(np.uint64)))
                        with patch.object(original, '_kernel', parallel._kernel):
                            actual_calendar = original.positive_invariant(jump, rates, permutation)
                        self.assertTrue(np.array_equal(actual_calendar.view(np.uint64), calendar.view(np.uint64)))
                        self.assertTrue(np.all(actual_calendar > 0))

    def test_rare_positive_masses_and_zero_structure_retained(self):
        n = 21
        target = np.logspace(-240, 0, n)
        target /= target.sum()
        up = np.full(n-1, .25)
        down = .25*target[:-1]/target[1:]
        off = diags((down, up), (-1, 1), shape=(n, n), format='csr')
        rates = np.asarray(off.sum(axis=1)).ravel()
        jump = (diags(1/rates)@off).tocsr()
        expected = original.positive_invariant(jump, rates)
        parallel.set_threads(4)
        with patch.object(original, '_kernel', parallel._kernel):
            actual = original.positive_invariant(jump, rates)
        self.assertTrue(np.array_equal(actual.view(np.uint64), expected.view(np.uint64)))
        self.assertTrue(np.all(actual > 0))
        self.assertLess(actual.min(), 1e-239)

    def test_reducibility_status_and_explicit_opt_in(self):
        kernel_before = original._kernel
        self.assertIsNot(kernel_before, parallel._kernel)
        for invalid in (0, 65, 1.5):
            with self.assertRaises(ValueError):
                parallel.set_threads(invalid)
        status, _ = raw_kernel(parallel._kernel, csr_matrix(np.eye(3)))
        self.assertEqual(status, 1)
        self.assertIs(original._kernel, kernel_before)


if __name__ == '__main__':
    main()
