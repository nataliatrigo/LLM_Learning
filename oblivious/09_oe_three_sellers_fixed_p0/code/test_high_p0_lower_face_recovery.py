"""The optional second continuation preserves prior data and recorded phases."""

import fcntl
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from . import audit
from . import high_p0_lower_face_recovery as entry
from . import test_high_p0_face_recovery as fixtures


class LowerFaceRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.FaceRecoveryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_prepare()
        self.fixture,self.root,self.data = fixture,fixture.root,fixture.data
        self.before_bytes = (self.data/'manifest.json').read_bytes()
        self.before = json.loads(self.before_bytes)
        self.worker_before = (self.data/'p0_0p8/rho_0p8/worker_state.json').read_bytes()
        helper = self.root/'high_p0_lower_face_recovery.py'
        helper.write_text('lower-face recovery fixture\n')
        for owner,name,value in ((entry,'ROOT',self.root),(entry,'HOME',fixture.fixture.home),
                                 (entry,'__file__',str(helper))):
            patcher = patch.object(owner,name,value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self,*extra):
        return entry.main(['--prepare-only',*extra])

    def test_registers_new_phase_and_exact_backups_preserving_all_179_checkpoints(self):
        self.prepare()
        after = json.loads((self.data/'manifest.json').read_bytes())
        self.assertEqual(after['configuration'],self.before['configuration'])
        self.assertEqual(after['completed'],self.before['completed'])
        self.assertEqual(after['import_provenance'],self.before['import_provenance'])
        self.assertEqual(after['execution_entrypoint_history'][:3],self.before['execution_entrypoint_history'])
        self.assertEqual({path:path.read_bytes() for path in self.fixture.raw},self.fixture.raw)
        self.assertEqual(len(after['import_provenance']['imported_stems']),144)
        phase = after['execution_entrypoint_history'][-1]
        self.assertEqual(phase['continuation_protocol'],'lower_face_v2')
        self.assertEqual(phase['initial_active_columns'],[13])
        self.assertEqual(phase['initial_seed_overrides'],{'35':1.})
        self.assertEqual((self.root/phase['before_manifest']).read_bytes(),self.before_bytes)
        self.assertEqual((self.root/phase['before_worker_state']).read_bytes(),self.worker_before)
        self.assertEqual(phase['numerical_sources'],after['numerical_sources'])
        self.assertEqual(len(after['numerical_sources'])-len(self.before['numerical_sources']),1)
        self.assertTrue(audit.verify_import_provenance(self.data,after)['verified'])
        entry.io.check_hashes(after['input_sha256'])
        entry.io.check_hashes(after['numerical_sources'])

    def test_prepare_resume_is_idempotent(self):
        self.prepare()
        first = (self.data/'manifest.json').read_bytes()
        backups = {path:path.read_bytes() for path in (self.data/'provenance').glob('*lower_face*')}
        self.prepare('--resume')
        self.assertEqual((self.data/'manifest.json').read_bytes(),first)
        after = json.loads(first)
        self.assertEqual(len(after['execution_entrypoint_history']),4)
        self.assertEqual(len(after['recorded_continuations']),2)
        self.assertEqual({path:path.read_bytes() for path in (self.data/'provenance').glob('*lower_face*')},backups)

    def test_original_partial_failed_status_is_accepted_and_preserved_in_history(self):
        manifest = dict(self.before,status='partial_failed')
        entry.io.write_json(self.data/'manifest.json',manifest)
        self.prepare()
        after = json.loads((self.data/'manifest.json').read_bytes())
        self.assertEqual(after['recorded_continuations'][-1]['before_status'],'partial_failed')

    def test_missing_first_continuation_is_rejected(self):
        manifest = dict(self.before,execution_entrypoint_history=self.before['execution_entrypoint_history'][:2])
        entry.io.write_json(self.data/'manifest.json',manifest)
        before = (self.data/'manifest.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError,'first active-face continuation'):
            self.prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),before)
        self.assertEqual(list((self.data/'provenance').glob('*lower_face*')),[])

    def test_running_controller_and_orphan_worker_locks_are_rejected(self):
        with (self.data/'.controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.prepare()
        with (self.data/'p0_0p8/rho_0p8/.worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),self.before_bytes)
        self.assertEqual(list((self.data/'provenance').glob('*lower_face*')),[])

    def test_seed_override_copies_input_and_requests_only_lower_initial_face(self):
        instance = entry.RecoveryExperiment.__new__(entry.RecoveryExperiment)
        instance.n,instance.sellers,instance.closure,instance.p0 = 41,3,'marginal',.8
        instance.params = SimpleNamespace(rho=.8)
        policy = np.linspace(0,1,41)
        original = policy.copy()
        instance.solve_active_face = Mock(return_value={'verified':True})
        self.assertTrue(instance.solve(policy,max_evaluations=234)['verified'])
        np.testing.assert_array_equal(policy,original)
        seed,columns = instance.solve_active_face.call_args.args
        expected = original.copy()
        expected[35] = 1.
        np.testing.assert_array_equal(seed,expected)
        self.assertEqual(columns,[13])
        self.assertEqual(instance.solve_active_face.call_args.kwargs,dict(max_evaluations=234,progress=None))

    def test_worker_restores_original_experiment_after_certificate_failure(self):
        original = entry.campaign.FastExperiment
        def failed(task):
            self.assertIs(entry.campaign.FastExperiment,entry.RecoveryExperiment)
            raise RuntimeError('full certificate rejected')
        with patch.object(entry,'_HYBRID_WORK_CASE',side_effect=failed):
            with self.assertRaisesRegex(RuntimeError,'full certificate rejected'):
                entry.work_case(dict(p0=.8,rho=.8))
        self.assertIs(entry.campaign.FastExperiment,original)


if __name__ == '__main__':
    unittest.main()
