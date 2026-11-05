"""Validation-budget changes preserve tolerances, sources and checkpoints."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import validation_entry as entry


class ValidationEntryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=entry.HOME/'data', prefix='_validation_test_')
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        frozen = self.data/'base.py'
        frozen.write_text('original mathematical source\n')
        self.raw = []
        for name in ('checkpoint.json','checkpoint.npz'):
            path = self.data/name
            path.write_bytes(b'previous certified bytes '+name.encode())
            self.raw.append(path)
        self.manifest = dict(configuration=dict(models=[list(model) for model in entry.marginal_entry.MODELS]),
                             status='failed',current_case='last-case',completed=['earlier-case'],
                             failures={'last-case':'positive iteration budget exhausted'},
                             numerical_sources={str(frozen.relative_to(entry.ROOT)):entry.runner.digest(frozen)},
                             input_sha256={str(path.relative_to(entry.ROOT)):entry.runner.digest(path) for path in self.raw},
                             execution_entrypoint_history=[])
        self.save()

    def save(self):
        (self.data/'manifest.json').write_text(json.dumps(self.manifest,indent=2)+'\n')

    def test_wrapper_changes_only_budgets(self):
        underlying = Mock(return_value={'law':'unchanged'})
        with patch.object(entry.model,'checked_symmetric_stationary',underlying):
            entry.install()
            result = entry.model.checked_symmetric_stationary('jump','rates',(31,)*3,
                tolerance=3e-12,stability_tolerance=7e-8,reference_max_states=2000)
        self.assertEqual(result,{'law':'unchanged'})
        kwargs = underlying.call_args.kwargs
        self.assertEqual(kwargs['positive_max_iterations'],1_000_000)
        self.assertEqual(kwargs['positive_time_limit'],240)
        self.assertEqual(kwargs['tolerance'],3e-12)
        self.assertEqual(kwargs['stability_tolerance'],7e-8)
        self.assertEqual(kwargs['reference_max_states'],2000)

    def test_larger_preexisting_budget_is_not_lowered(self):
        underlying = Mock()
        with patch.object(entry.model,'checked_symmetric_stationary',underlying):
            entry.install()
            entry.model.checked_symmetric_stationary('jump','rates',(31,)*3,
                positive_max_iterations=2_000_000,positive_time_limit=None)
        self.assertEqual(underlying.call_args.kwargs['positive_max_iterations'],2_000_000)
        self.assertIsNone(underlying.call_args.kwargs['positive_time_limit'])

    def test_prepare_preserves_snapshot_checkpoints_sources_and_failure(self):
        before = (self.data/'manifest.json').read_bytes()
        raw_before = {path:path.read_bytes() for path in self.raw}
        base_sources = deepcopy(self.manifest['numerical_sources'])
        saved = entry.prepare(self.data)
        self.assertEqual((self.data/entry.SNAPSHOT).read_bytes(),before)
        self.assertEqual({path:path.read_bytes() for path in self.raw},raw_before)
        for path,sha in base_sources.items():
            self.assertEqual(saved['numerical_sources'][path],sha)
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(saved['failures'],self.manifest['failures'])
        self.assertEqual(saved['failure_history'][-1]['failures'],self.manifest['failures'])
        self.assertEqual(saved['execution_entrypoint_history'][-1]['positive_max_iterations'],1_000_000)
        self.assertEqual(saved['execution_entrypoint_history'][-1]['numerical_sources'],saved['numerical_sources'])
        self.assertEqual(entry.prepare(self.data),saved)

    def test_prepare_rejects_running_campaign(self):
        self.manifest['status'] = 'running'; self.save()
        with self.assertRaisesRegex(RuntimeError,'Close the active campaign'):
            entry.prepare(self.data)
        self.assertFalse((self.data/entry.SNAPSHOT).exists())

    def test_prepare_rejects_changed_checkpoint_fingerprint(self):
        self.raw[0].write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'Frozen experiment input changed'):
            entry.prepare(self.data)
        self.assertFalse((self.data/entry.SNAPSHOT).exists())


if __name__ == '__main__':
    unittest.main()
