"""Publication gates for the complete n201/n251 independent review."""

from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import sweep201_audit as audit


class CompleteCampaignGate(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.manifest = dict(status='complete', current_grid=None, current_model=None,
            completed=sorted(audit.required_tags(.9)), requested_grids=[201, 251],
            numerical_sources={'unchanged-kernel.py':'fingerprint'},
            configuration=dict(p0=.8, rho=.9,
                parameters={k:v for k,v in asdict(Parameters()).items() if k != 'rho'}))

    def tearDown(self):
        self.temporary.cleanup()

    def read(self):
        (self.directory/'manifest.json').write_text(json.dumps(self.manifest))
        return audit.closed_manifest(self.directory, .8, .9,
                                     {'unchanged-kernel.py':'fingerprint'})

    def test_requires_conditional_validation_even_when_201_is_complete(self):
        self.manifest['completed'].remove('rho_0p9_n251_conditional')
        with self.assertRaisesRegex(ArithmeticError, 'All n201/n251'):
            self.read()

    def test_cannot_review_an_active_worker_with_complete_old_files(self):
        self.manifest.update(status='running', current_grid=251, current_model='conditional')
        with self.assertRaisesRegex(ArithmeticError, 'closed campaign'):
            self.read()

    def test_rejects_changed_kernel_and_economic_calibration(self):
        self.manifest['numerical_sources']['unchanged-kernel.py'] = 'changed'
        with self.assertRaisesRegex(ArithmeticError, 'numerical source'):
            self.read()
        self.manifest['numerical_sources']['unchanged-kernel.py'] = 'fingerprint'
        self.manifest['configuration']['p0'] = .9
        with self.assertRaisesRegex(ArithmeticError, 'economic calibration'):
            self.read()

    def test_historical_grids_cannot_masquerade_as_the_requested_refinement(self):
        self.manifest['requested_grids'] = [51, 101, 201, 251]
        with self.assertRaisesRegex(ArithmeticError, 'exactly grids 201 and 251'):
            self.read()

    def test_accepts_closed_complete_matching_campaign(self):
        self.assertEqual(self.read(), self.manifest)


class ExecutionPhaseProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.home = self.root/'experiment'
        (self.home/'code').mkdir(parents=True)
        self.files = {}
        for name in ('sweep201_batch.py', 'sweep201_run.py', 'optional_phase.py', 'parallel.cpp'):
            path = self.home/'code'/name
            path.write_text('original '+name)
            self.files[name] = sha256(path.read_bytes()).hexdigest()
        phase = dict(source_input_sha256={
            str((self.home/'code'/name).relative_to(self.root)):self.files[name]
            for name in ('optional_phase.py', 'parallel.cpp')},
            immutable_scheduler_sha256=self.files['sweep201_batch.py'],
            immutable_worker_sha256=self.files['sweep201_run.py'])
        self.master = dict(scheduler_sha256=self.files['sweep201_batch.py'],
            worker_sha256=self.files['sweep201_run.py'],
            execution_entrypoint_history=[phase], execution_entrypoint=deepcopy(phase))
        self.checked = {}

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        def add_input(path, expected):
            actual = sha256(path.read_bytes()).hexdigest()
            audit.raw.require(actual == expected, 'Input/source hash differs: '+path.name)
            self.checked[str(path.relative_to(self.root))] = actual
        with patch.object(audit, 'ROOT', self.root), patch.object(audit, 'HOME', self.home):
            return audit.verify_execution_history(self.master, add_input)

    def test_verifies_python_cpp_and_frozen_controller_worker_without_checkpoints(self):
        self.assertEqual(self.verify(), 1)
        self.assertEqual(len(self.checked), 4)
        (self.home/'code/parallel.cpp').write_text('changed even though this phase saved no checkpoint')
        with self.assertRaisesRegex(ArithmeticError, 'hash differs: parallel.cpp'):
            self.verify()

    def test_checks_older_phase_sources_even_when_current_phase_is_unchanged(self):
        old = deepcopy(self.master['execution_entrypoint'])
        old['source_input_sha256'] = {'experiment/code/optional_phase.py':'0'*64}
        self.master['execution_entrypoint_history'].insert(0, old)
        with self.assertRaisesRegex(ArithmeticError, 'hash differs: optional_phase.py'):
            self.verify()

    def test_rejects_current_history_disagreement_and_altered_immutable_reference(self):
        self.master['execution_entrypoint']['source_input_sha256'] = {}
        with self.assertRaisesRegex(ArithmeticError, 'last history record'):
            self.verify()
        self.master['execution_entrypoint'] = deepcopy(self.master['execution_entrypoint_history'][-1])
        self.master['execution_entrypoint_history'][-1]['immutable_worker_sha256'] = '0'*64
        self.master['execution_entrypoint'] = deepcopy(self.master['execution_entrypoint_history'][-1])
        with self.assertRaisesRegex(ArithmeticError, 'immutable worker differs'):
            self.verify()

    def test_accepts_base_execution_and_verifies_a_current_only_record(self):
        self.master.pop('execution_entrypoint_history')
        self.assertEqual(self.verify(), 1)
        self.master.pop('execution_entrypoint')
        self.assertEqual(self.verify(), 0)


class ParallelAuditGates(unittest.TestCase):
    def test_missing_cases_cannot_reach_the_pool_or_publish_a_review(self):
        with tempfile.TemporaryDirectory(dir=audit.DATA.parent) as temporary:
            directory = Path(temporary)
            master = dict(status='complete_certified',
                configuration=dict(p0s=list(audit.P0S), rhos=list(audit.RHOS),
                    grids=list(audit.GRIDS), order=256, tol=1e-7, dp_tol=1e-10),
                completed_p0_rho_pairs=24, certified_checkpoint_bundles=96,
                certified_model_grid_solutions=144,
                numerical_sources=audit.source_fingerprints(),
                scheduler_sha256=audit.raw.digest(audit.HOME/'code/sweep201_batch.py'),
                worker_sha256=audit.raw.digest(audit.HOME/'code/sweep201_run.py'))
            (directory/'sweep_manifest.json').write_text(json.dumps(master))
            output = directory/'independent_review.json'
            with patch.object(audit, 'DATA', directory), patch.object(audit, 'audit_jobs') as pool:
                with self.assertRaisesRegex(ArithmeticError, 'campaign is missing'):
                    audit.run_audit(output, workers=6)
                pool.assert_not_called()
            self.assertFalse(output.exists())

    def test_worker_count_is_bounded_before_any_process_is_started(self):
        with patch.object(audit, 'ProcessPoolExecutor') as pool:
            for workers in (0, 7, True, 1.5):
                with self.assertRaisesRegex(ArithmeticError, 'integer from 1 to 6'):
                    audit.audit_jobs([], workers=workers)
            pool.assert_not_called()

    def test_jobs_keep_input_order_and_child_blas_environment_is_restored(self):
        observed = []
        def fake_worker(job):
            observed.append(tuple(audit.os.environ.get(key) for key in
                                  ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')))
            return dict(record=job)
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.map.side_effect = lambda function, jobs: map(function, jobs)
        with patch.dict(audit.os.environ, OPENBLAS_NUM_THREADS='3', OMP_NUM_THREADS='4'):
            before = dict(audit.os.environ)
            with patch.object(audit, 'ProcessPoolExecutor', return_value=executor) as pool, \
                 patch.object(audit, 'audit_saved_checkpoints', fake_worker):
                self.assertEqual(audit.audit_jobs([3, 1, 2], workers=2),
                                 [dict(record=n) for n in (3, 1, 2)])
                self.assertEqual(pool.call_args.kwargs['max_workers'], 2)
            self.assertEqual(dict(audit.os.environ), before)
        self.assertEqual(observed, [('1', '1', '1')]*3)


if __name__ == '__main__':
    unittest.main()
