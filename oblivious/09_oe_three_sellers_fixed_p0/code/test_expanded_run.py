"""Import, ownership, fingerprint, and interrupted-worker checks."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import expanded_run as entry


def configuration(p0s=None, rhos=None, grids=None):
    p0s = entry.P0S if p0s is None else p0s
    rhos = entry.RHOS if rhos is None else rhos
    grids = [21, 31, 41] if grids is None else grids
    return dict(p0s=p0s, rhos=rhos, grids=grids,
                case_grids={entry.pair_key(p, r):list(grids) for p in p0s for r in rhos},
                models=[list(x) for x in entry.MODELS], order=256, tolerance=1e-7)


class ExpandedRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root/'experiment'
        self.data = self.home/'data/expanded'
        self.data.mkdir(parents=True)
        self.source = self.root/'frozen.py'
        self.source.write_text('frozen numerical source\n')
        self.sources = {'frozen.py':entry.io.digest(self.source)}
        for owner, name, value in ((entry, 'HOME', self.home), (entry, 'ROOT', self.root),
                                    (entry, '__file__', str(self.root/'expanded_run.py')),
                                    (entry.io, 'ROOT', self.root)):
            patcher = patch.object(owner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def saved(self, data, p0, rho, n, sellers, closure, policy=.2):
        stem = entry.io.key(p0, rho, n, sellers, closure)
        path = data/stem
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path)+'.npz', grid=np.linspace(0, 1, n), policy=np.full(n, policy))
        summary = dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure, verified=True)
        entry.io.write_json(str(path)+'.json', dict(summary=summary,
            checkpoint_sha256=entry.io.digest(str(path)+'.npz')))
        return stem, {str(Path(str(path)+suffix).relative_to(self.root)):
                      entry.io.digest(str(path)+suffix) for suffix in ('.npz', '.json')}

    def pilot(self):
        origin = self.home/'data/marginal_pilot'
        completed, inputs = [], {}
        for p0 in (.1, .5, .8):
            for rho in (.6, .8):
                for n in (21, 31):
                    for sellers, closure in entry.MODELS:
                        stem, fingerprints = self.saved(origin, p0, rho, n, sellers, closure)
                        completed.append(stem)
                        inputs.update(fingerprints)
        manifest = dict(status='complete', completed=completed, input_sha256=inputs,
                        numerical_sources=self.sources, configuration={})
        entry.io.write_json(origin/'manifest.json', manifest)
        return origin, manifest

    def test_default_scope_has_216_model_grid_cases_and_case_extension_adds_three(self):
        config = configuration()
        self.assertEqual(len(entry.expected(config)), 216)
        config['case_grids'][entry.pair_key(.8, .9)].append(51)
        self.assertEqual(len(entry.expected(config)), 219)

    def test_imports_36_checkpoints_byte_identically_and_keeps_source_snapshot(self):
        origin, original = self.pilot()
        original_hashes = {p:entry.io.digest(self.root/p) for p in original['input_sha256']}
        with patch.object(entry, 'new_sources', side_effect=lambda previous:dict(previous)):
            imported = entry.prepare(self.data, configuration())
        self.assertEqual(len(imported['completed']), 36)
        for stem in original['completed']:
            for suffix in ('.npz', '.json'):
                self.assertEqual((origin/(stem+suffix)).read_bytes(), (self.data/(stem+suffix)).read_bytes())
        self.assertEqual((origin/'manifest.json').read_bytes(),
                         (self.data/'provenance/source_manifest.json').read_bytes())
        self.assertEqual(original_hashes, {p:entry.io.digest(self.root/p) for p in original_hashes})
        entry.io.check_hashes(imported['input_sha256'])

    def test_seed_uses_highest_previous_same_equilibrium_grid_before_same_grid_n2(self):
        stems = []
        for n, sellers, closure, policy in ((21, 3, 'marginal', .21),
                                             (31, 2, 'marginal', .82),
                                             (31, 1, 'monopoly', .91)):
            stem, _ = self.saved(self.data, .5, .6, n, sellers, closure, policy)
            stems.append(stem)
        seed, source = entry.io.load_seed(self.data, {'completed':stems}, .5, .6, 31, 3, 'marginal')
        self.assertEqual(source, stems[0])
        np.testing.assert_allclose(seed, .21)
        stem, _ = self.saved(self.data, .5, .6, 31, 3, 'marginal', .31)
        stems.append(stem)
        seed, source = entry.io.load_seed(self.data, {'completed':stems}, .5, .6, 41, 3, 'marginal')
        self.assertEqual(source, stem)
        np.testing.assert_allclose(seed, .31)

    def result(self, n, sellers, closure, p0, rho):
        grid = np.linspace(0, 1, n)
        return dict(grid=grid, policy=np.zeros(n), joint=np.full((n,)*sellers, n**(-sellers)),
                    value=np.zeros(n), actual_value=np.zeros((n,)*sellers), demand=np.ones(n)/sellers,
                    advantage=-np.ones(n), verified=True, diagnostics={},
                    summary=dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure, verified=True))

    def fake_experiment(self, failures=None, calls=None):
        owner = self
        failures = set() if failures is None else failures
        calls = [] if calls is None else calls
        class FakeExperiment:
            def __init__(self, n, sellers, closure, p0, params, order):
                self.n, self.sellers, self.closure, self.p0, self.rho = n, sellers, closure, p0, params.rho
                self.game = SimpleNamespace(demand_diagnostics={})
            def configuration(self):
                return dict(n=self.n, sellers=self.sellers, closure=self.closure, p0=self.p0, rho=self.rho)
            def solve(self, initial, **kwargs):
                calls.append((self.n, self.sellers))
                if self.sellers in failures:
                    raise RuntimeError('deliberate interrupted solve')
                return owner.result(self.n, self.sellers, self.closure, self.p0, self.rho)
        return FakeExperiment

    def task(self):
        return dict(data=str(self.data), p0=.5, rho=.6, grids=[5], completed=[],
                    sources=self.sources, order=256, max_evaluations=30)

    def test_checkpoint_refuses_overwrite_and_preserves_bytes(self):
        stem, _ = self.saved(self.data, .5, .6, 5, 3, 'marginal')
        before = {suffix:(self.data/(stem+suffix)).read_bytes() for suffix in ('.npz', '.json')}
        experiment = SimpleNamespace(configuration=lambda:{}, game=SimpleNamespace(demand_diagnostics={}))
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            entry.checkpoint(self.data, stem, self.result(5, 3, 'marginal', .5, .6), experiment, None, 0.)
        for suffix, contents in before.items():
            self.assertEqual((self.data/(stem+suffix)).read_bytes(), contents)

    def test_worker_resumes_state_without_rewriting_certified_results_or_parent_manifest(self):
        parent_path = self.data/'manifest.json'
        parent_path.write_text('parent owns this exact file\n')
        restore = Mock()
        first_calls = []
        with patch.object(entry, 'FastExperiment', self.fake_experiment({2}, first_calls)), \
                patch.object(entry.expanded_stationary, 'install', return_value=restore):
            failed = entry.work_case(self.task())
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(first_calls, [(5, 1), (5, 2)])
        self.assertEqual(len(failed['completed']), 1)
        preserved = dict(failed['input_sha256'])
        next_calls = []
        with patch.object(entry, 'FastExperiment', self.fake_experiment(calls=next_calls)), \
                patch.object(entry.expanded_stationary, 'install', return_value=restore):
            complete = entry.work_case(self.task())
        self.assertEqual(complete['status'], 'complete')
        self.assertEqual(next_calls, [(5, 2), (5, 3)])
        self.assertEqual(len(complete['completed']), 3)
        self.assertEqual(restore.call_count, 2)
        entry.io.check_hashes(preserved)
        self.assertEqual(parent_path.read_text(), 'parent owns this exact file\n')

    def test_same_case_cannot_be_owned_by_two_workers(self):
        directory = self.data/entry.pair_key(.5, .6)
        directory.mkdir(parents=True)
        with (directory/'.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                entry.work_case(self.task())

    def test_orphan_complete_bundle_is_recovered_without_changing_its_bytes(self):
        stem = entry.io.key(.5, .6, 5, 3, 'marginal')
        result = self.result(5, 3, 'marginal', .5, .6)
        experiment = SimpleNamespace(configuration=lambda:dict(parameters=dict(rho=.6)),
                                     game=SimpleNamespace(demand_diagnostics={}))
        fingerprints = entry.checkpoint(self.data, stem, result, experiment, None, 0.)
        state = dict(completed=[], input_sha256={})
        self.assertTrue(entry.recover_checkpoint(self.data, stem, .5, .6, 5, 3, 'marginal', state))
        self.assertEqual(state['completed'], [stem])
        self.assertEqual(state['input_sha256'], fingerprints)
        self.assertEqual(state['recovered_atomic_bundles'], [stem])
        entry.io.check_hashes(fingerprints)

    def test_orphan_partial_bundle_is_quarantined_byte_identically(self):
        stem = entry.io.key(.5, .6, 5, 3, 'marginal')
        partial = self.data/(stem+'.npz')
        partial.parent.mkdir(parents=True)
        contents = b'interrupted compressed array bytes'
        partial.write_bytes(contents)
        state = dict(completed=[], input_sha256={})
        self.assertFalse(entry.recover_checkpoint(self.data, stem, .5, .6, 5, 3, 'marginal', state))
        self.assertFalse(partial.exists())
        self.assertEqual(state['completed'], [])
        archive = state['uncommitted_partial_files'][0]['files'][0]
        self.assertEqual((self.root/archive['path']).read_bytes(), contents)
        self.assertEqual(entry.io.digest(self.root/archive['path']), archive['sha256'])

    def test_orphan_wrong_configuration_is_rejected_without_modifying_bundle(self):
        stem = entry.io.key(.5, .6, 5, 3, 'marginal')
        experiment = SimpleNamespace(configuration=lambda:dict(parameters=dict(rho=.7)),
                                     game=SimpleNamespace(demand_diagnostics={}))
        fingerprints = entry.checkpoint(self.data, stem, self.result(5, 3, 'marginal', .5, .6), experiment, None, 0.)
        state = dict(completed=[], input_sha256={})
        with self.assertRaisesRegex(RuntimeError, 'orphan'):
            entry.recover_checkpoint(self.data, stem, .5, .6, 5, 3, 'marginal', state)
        self.assertEqual(state['completed'], [])
        entry.io.check_hashes(fingerprints)

    def test_worker_restores_process_even_when_existing_state_hash_is_invalid(self):
        directory = self.data/entry.pair_key(.5, .6)
        entry.io.write_json(directory/'worker_state.json', dict(completed=[], input_sha256={'frozen.py':'bad'}))
        restore = Mock()
        stdout, stderr = sys.stdout, sys.stderr
        try:
            with patch.object(entry.expanded_stationary, 'install', return_value=restore):
                try:
                    entry.work_case(self.task())
                except RuntimeError:
                    pass
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        restore.assert_called_once()

    def test_merge_rejects_replacing_an_existing_parent_fingerprint(self):
        stem, inputs = self.saved(self.data, .5, .6, 5, 3, 'marginal')
        manifest = dict(configuration=configuration([.5], [.6], [5]), completed=[stem], input_sha256=dict(inputs))
        record_path = self.data/(stem+'.json')
        saved = json.loads(record_path.read_text())
        saved['new_metadata'] = 'different bytes'
        entry.io.write_json(record_path, saved)
        changed = dict(inputs)
        changed[str(record_path.relative_to(self.root))] = entry.io.digest(record_path)
        entry.io.write_json(record_path.parent/'worker_state.json', dict(p0=.5, rho=.6, completed=[stem], input_sha256=changed))
        with self.assertRaises(RuntimeError):
            entry.merge_states(self.data, manifest)


if __name__ == '__main__':
    unittest.main()
