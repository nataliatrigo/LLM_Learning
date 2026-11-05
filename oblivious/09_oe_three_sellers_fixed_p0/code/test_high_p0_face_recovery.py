"""Optional continuation preserves provenance and rejects live controllers."""

import fcntl
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import audit
from . import high_p0_face_recovery as entry
from . import test_high_p0_refine as fixtures


class FaceRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.HighP0RefinementTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root,self.data = fixture.root,fixture.data
        fixture.source_campaign()
        manifest = fixture.prepare()
        worker_inputs,worker_completed = {},[]
        for p0 in (.7,.8,.9):
            for rho in entry.campaign.RHOS:
                for sellers,closure in entry.campaign.MODELS:
                    if (p0,rho,sellers) == (.8,.8,3):
                        continue
                    stem,inputs = fixture.checkpoint(self.data,p0,rho,41,sellers,closure)
                    manifest['completed'].append(stem)
                    manifest['input_sha256'].update(inputs)
                    if (p0,rho) == (.8,.8):
                        worker_inputs.update(inputs)
                        worker_completed.append(stem)
        manifest.update(status='interrupted',active_workers=0,current_case=None)
        entry.io.export(self.data,manifest)
        entry.io.write_json(self.data/'manifest.json',manifest)
        entry.io.write_json(self.data/'p0_0p8/rho_0p8/worker_state.json',dict(
            status='failed',p0=.8,rho=.8,completed=worker_completed,input_sha256=worker_inputs,
            current_case=entry.io.key(**entry.RECOVERY_CASE),error='KeyboardInterrupt()'))
        self.before_manifest = (self.data/'manifest.json').read_bytes()
        self.before_worker = (self.data/'p0_0p8/rho_0p8/worker_state.json').read_bytes()
        self.before = json.loads(self.before_manifest)
        self.raw = {self.data/(stem+suffix):(self.data/(stem+suffix)).read_bytes()
                    for stem in self.before['completed'] for suffix in ('.json','.npz')}
        source_files = {}
        for name in ('high_p0_face_recovery.py','active_face_search.py'):
            path = self.root/name
            path.write_text(name+' fixture\n')
            source_files[name] = entry.io.digest(path)
        for owner,name,value in ((entry,'ROOT',self.root),(entry,'HOME',fixture.home),
                                 (entry,'__file__',str(self.root/'high_p0_face_recovery.py'))):
            patcher = patch.object(owner,name,value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(entry,'source_reference',return_value=source_files)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_prepare(self,*extra):
        return entry.main(['--prepare-only',*extra])

    def test_registration_preserves_179_certificates_and_exact_before_states(self):
        self.run_prepare()
        after = json.loads((self.data/'manifest.json').read_bytes())
        self.assertEqual(set(after['completed']),set(self.before['completed']))
        self.assertEqual(len(after['completed']),179)
        self.assertEqual(after['configuration'],self.before['configuration'])
        self.assertEqual(after['import_provenance'],self.before['import_provenance'])
        self.assertEqual(after['execution_entrypoint_history'][:2],self.before['execution_entrypoint_history'])
        self.assertEqual({path:path.read_bytes() for path in self.raw},self.raw)
        phase = after['execution_entrypoint_history'][-1]
        self.assertEqual((self.root/phase['before_manifest']).read_bytes(),self.before_manifest)
        self.assertEqual((self.root/phase['before_worker_state']).read_bytes(),self.before_worker)
        self.assertEqual(phase['initial_active_columns'],[13,35])
        self.assertEqual(phase['numerical_sources'],after['numerical_sources'])
        self.assertEqual(phase['stationary_certification'],entry.hybrid.CERTIFICATION)
        self.assertEqual(len(after['import_provenance']['imported_stems']),144)
        self.assertEqual(len(after['execution_entrypoint_history'][0]['completed_stems']),144)
        self.assertTrue(audit.verify_import_provenance(self.data,after)['verified'])
        entry.io.check_hashes(after['input_sha256'])
        entry.io.check_hashes(after['numerical_sources'])

    def test_prepare_then_resume_is_idempotent_without_duplicate_phase_or_backup(self):
        self.run_prepare()
        first = (self.data/'manifest.json').read_bytes()
        backups = {path:path.read_bytes() for path in (self.data/'provenance').glob('*active_face*')}
        self.run_prepare('--resume')
        after = json.loads((self.data/'manifest.json').read_bytes())
        self.assertEqual((self.data/'manifest.json').read_bytes(),first)
        self.assertEqual(len(after['execution_entrypoint_history']),3)
        self.assertEqual(len(after['recorded_continuations']),1)
        self.assertEqual({path:path.read_bytes() for path in (self.data/'provenance').glob('*active_face*')},backups)

    def test_live_controller_lock_rejected_before_metadata_or_source_changes(self):
        with (self.data/'.controller.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),self.before_manifest)
        self.assertEqual(list((self.data/'provenance').glob('*active_face*')),[])

    def test_running_status_and_live_orphan_worker_are_both_rejected(self):
        manifest = dict(self.before,status='running')
        entry.io.write_json(self.data/'manifest.json',manifest)
        with self.assertRaisesRegex(RuntimeError,'interrupted or failed'):
            self.run_prepare()
        (self.data/'manifest.json').write_bytes(self.before_manifest)
        with (self.data/'p0_0p8/rho_0p8/.worker.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),self.before_manifest)
        self.assertEqual(list((self.data/'provenance').glob('*active_face*')),[])

    def test_changed_frozen_source_is_rejected_without_writing_a_backup(self):
        (self.root/'frozen.py').write_text('changed numerical source\n')
        with self.assertRaisesRegex(RuntimeError,'input changed'):
            self.run_prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),self.before_manifest)
        self.assertEqual(list((self.data/'provenance').glob('*active_face*')),[])

    def test_different_remaining_case_is_rejected(self):
        manifest = json.loads(self.before_manifest)
        removed = entry.io.key(.7,.6,41,3,'marginal')
        manifest['completed'].remove(removed)
        stem,inputs = self.fixture.checkpoint(self.data,**entry.RECOVERY_CASE)
        manifest['completed'].append(stem)
        manifest['input_sha256'].update(inputs)
        entry.io.export(self.data,manifest)
        entry.io.write_json(self.data/'manifest.json',manifest)
        before = (self.data/'manifest.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError,'single declared missing case'):
            self.run_prepare()
        self.assertEqual((self.data/'manifest.json').read_bytes(),before)
        self.assertEqual(list((self.data/'provenance').glob('*active_face*')),[])

    def test_worker_restores_frozen_experiment_on_failure(self):
        original = entry.campaign.FastExperiment
        def failed(task):
            self.assertIs(entry.campaign.FastExperiment,entry.RecoveryExperiment)
            raise RuntimeError('certificate rejected')
        with patch.object(entry,'_HYBRID_WORK_CASE',side_effect=failed):
            with self.assertRaisesRegex(RuntimeError,'certificate rejected'):
                entry.work_case(dict(p0=.8,rho=.8))
        self.assertIs(entry.campaign.FastExperiment,original)

    def test_recovery_calls_adaptive_search_and_requires_the_recorded_case(self):
        instance = entry.RecoveryExperiment.__new__(entry.RecoveryExperiment)
        instance.n,instance.sellers,instance.closure,instance.p0 = 41,3,'marginal',.8
        instance.params = SimpleNamespace(rho=.8)
        policy = np.full(41,.5)
        result = {'verified':True}
        instance.solve_active_face = Mock(return_value=result)
        self.assertIs(instance.solve(policy,max_evaluations=123),result)
        instance.solve_active_face.assert_called_once_with(policy,[13,35],max_evaluations=123,progress=None)
        instance.params.rho = .7
        with self.assertRaisesRegex(RuntimeError,'single declared missing case'):
            instance.solve(policy)


if __name__ == '__main__':
    unittest.main()
