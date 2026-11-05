"""Budget routing, source coverage, and restoration for the high-p0 worker."""

from hashlib import sha256
from math import comb
from unittest.mock import Mock, patch
import pickle
import unittest

from . import high_p0_worker as entry
from . import expanded_run, expanded_stationary, model, stationary


class HighP0WorkerTests(unittest.TestCase):
    def test_worker_forces_budget_and_restores_installer_after_success(self):
        restore = Mock()
        installer = Mock(return_value=restore)
        task = {'task': 'sentinel', 'rho': .6}

        def frozen_worker(actual):
            self.assertIs(actual, task)
            self.assertIsNot(expanded_stationary.install, installer)
            callback = expanded_stationary.install()
            callback()
            return {'status': 'complete'}

        with patch.object(expanded_stationary, 'install', installer), \
                patch.object(entry, '_FROZEN_WORK_CASE', side_effect=frozen_worker):
            self.assertEqual(entry.work_case(task), {'status': 'complete'})
            self.assertIs(expanded_stationary.install, installer)
        installer.assert_called_once_with(reference_max_states=6000, threads=2)
        restore.assert_called_once_with()

    def test_worker_restores_installer_when_frozen_worker_raises(self):
        installer = Mock()
        with patch.object(expanded_stationary, 'install', installer), \
                patch.object(entry, '_FROZEN_WORK_CASE', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                entry.work_case({'rho': .6})
            self.assertIs(expanded_stationary.install, installer)

    def test_entrypoint_redirect_does_not_recurse_and_worker_is_picklable(self):
        original = entry._FROZEN_WORK_CASE
        with patch.object(expanded_run, 'work_case', entry.work_case):
            self.assertIs(entry._FROZEN_WORK_CASE, original)
            self.assertIs(pickle.loads(pickle.dumps(entry.work_case)), entry.work_case)

    def test_original_installer_retains_tolerances_and_positive_work_budgets(self):
        checker = Mock(return_value={'law': 'sentinel'})

        def frozen_worker(task):
            restore = expanded_stationary.install()
            try:
                return model.checked_symmetric_stationary('jump', 'rates', (41,)*3,
                    tolerance=1e-11, stability_tolerance=1e-7)
            finally:
                restore()

        with patch.object(model, 'checked_symmetric_stationary', checker), \
                patch.object(entry, '_FROZEN_WORK_CASE', side_effect=frozen_worker):
            self.assertEqual(entry.work_case({'rho': .6}), {'law': 'sentinel'})
        kwargs = checker.call_args.kwargs
        self.assertEqual(kwargs, dict(tolerance=1e-11, stability_tolerance=1e-7,
            reference_max_states=6000, positive_max_iterations=1_000_000,
            positive_time_limit=240))

    def test_three_seller_quotient_routes_41_and_51_to_original_large_branch(self):
        self.assertLessEqual(comb(31+2, 3), entry.reference_budget(.6))
        for n in (41, 51):
            self.assertGreater(comb(n+2, 3), entry.reference_budget(.7))
            self.assertLessEqual(comb(n+2, 3), entry.reference_budget(.8))
        self.assertEqual(entry.CERTIFICATION['failure_policy'],
                         'Preserve explicit failure; no dense retry or tolerance relaxation')

    def test_reference_selection_is_fixed_by_rho_before_solving(self):
        for rho, expected in ((.6, 6000), (.7, 6000), (.8, 25000), (.9, 25000)):
            installer = Mock(return_value=Mock())
            def frozen_worker(task):
                expanded_stationary.install()()
                return task
            with self.subTest(rho=rho), patch.object(expanded_stationary, 'install', installer), \
                    patch.object(entry, '_FROZEN_WORK_CASE', side_effect=frozen_worker):
                entry.work_case({'rho': rho})
            installer.assert_called_once_with(reference_max_states=expected, threads=2)
        for rho in (True, -.1, 1., float('nan'), float('inf'), '.8'):
            with self.subTest(invalid_rho=rho), self.assertRaises(ValueError):
                entry.reference_budget(rho)

    def test_sparse_ordering_preserves_all_other_arguments_and_restores_solver(self):
        sparse_solver = Mock(return_value='solution')
        def frozen_worker(task):
            self.assertEqual(stationary.spsolve('matrix', 'rhs', permc_spec='COLAMD',
                                                use_umfpack=False), 'solution')
            self.assertEqual(stationary.spsolve('matrix2', 'rhs2', 'COLAMD', False), 'solution')
            return 'complete'
        with patch.object(stationary, 'spsolve', sparse_solver), \
                patch.object(entry, '_FROZEN_WORK_CASE', side_effect=frozen_worker):
            self.assertEqual(entry.work_case({'rho': .9}), 'complete')
            self.assertIs(stationary.spsolve, sparse_solver)
        first, second = sparse_solver.call_args_list
        self.assertEqual(first.args, ('matrix', 'rhs'))
        self.assertEqual(first.kwargs, dict(permc_spec='MMD_AT_PLUS_A', use_umfpack=False))
        self.assertEqual(second.args, ('matrix2', 'rhs2', 'MMD_AT_PLUS_A', False))
        self.assertEqual(second.kwargs, {})

    def test_sparse_solver_is_restored_when_worker_fails(self):
        sparse_solver = Mock()
        with patch.object(stationary, 'spsolve', sparse_solver), \
                patch.object(entry, '_FROZEN_WORK_CASE', side_effect=RuntimeError('certificate failed')):
            with self.assertRaisesRegex(RuntimeError, 'certificate failed'):
                entry.work_case({'rho': .8})
            self.assertIs(stationary.spsolve, sparse_solver)

    def test_adapter_source_is_fingerprinted(self):
        reference = entry.source_reference()
        self.assertEqual(len(reference), 1)
        for path, digest in reference.items():
            self.assertTrue(path.endswith('/high_p0_worker.py'))
            self.assertEqual(sha256((expanded_run.ROOT/path).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
