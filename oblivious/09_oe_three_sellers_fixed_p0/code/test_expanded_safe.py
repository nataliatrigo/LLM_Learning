"""Regression checks for immutable import lineage and the metadata-only repair."""

import json
import unittest
from unittest.mock import patch

from . import expanded_safe as safe
from . import expanded_run as campaign
from . import test_expanded_run as fixtures


class ImportScopeRepairTests(unittest.TestCase):
    saved = fixtures.ExpandedRunTests.saved
    pilot = fixtures.ExpandedRunTests.pilot

    def setUp(self):
        fixtures.ExpandedRunTests.setUp(self)
        self.helper = self.root/'expanded_safe.py'
        self.helper.write_text('new metadata repair helper\n')
        for name, value in (('ROOT', self.root), ('HOME', self.home), ('__file__', str(self.helper))):
            patcher = patch.object(safe, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def broken_campaign(self):
        self.pilot()
        runner = self.root/'expanded_run.py'
        runner.write_text('frozen runner containing its historical list alias\n')
        config = fixtures.configuration(grids=[21, 31])
        with patch.object(campaign, 'new_sources', side_effect=lambda previous:
                          dict(previous, **{'expanded_run.py':safe.io.digest(runner)})):
            manifest = campaign.prepare(self.data, config)
        for stem in campaign.expected(config):
            if stem in manifest['completed']:
                continue
            parts = stem.split('/')
            p0 = float(parts[0].removeprefix('p0_').replace('p', '.'))
            rho = float(parts[1].removeprefix('rho_').replace('p', '.'))
            grid, count, closure = parts[2].split('_')
            _, hashes = self.saved(self.data, p0, rho, int(grid[1:]), int(count[1:]), closure)
            manifest['completed'].append(stem)
            manifest['input_sha256'].update(hashes)
        manifest['status'] = 'complete'
        safe.io.export(self.data, manifest)
        safe.io.write_json(self.data/'manifest.json', manifest)
        self.assertEqual(len(manifest['completed']), 144)
        self.assertEqual(len(manifest['import_provenance']['imported_stems']), 144)
        self.assertEqual(len(manifest['execution_entrypoint_history'][0]['completed_stems']), 144)
        return manifest

    def test_repair_144_results_keeps_only_36_imports_and_preserves_every_numerical_byte(self):
        old = self.broken_campaign()
        before = (self.data/'manifest.json').read_bytes()
        numerical = dict(old['numerical_sources'])
        current_phase = json.loads(json.dumps(old['execution_entrypoint_history'][1]))
        original_inputs = dict(old['input_sha256'])
        result = safe.repair_imports(self.data)
        repaired = json.loads((self.data/'manifest.json').read_text())
        self.assertEqual(result['status'], 'repaired')
        self.assertEqual(len(repaired['completed']), 144)
        self.assertEqual(len(repaired['import_provenance']['imported_stems']), 36)
        self.assertEqual(len(repaired['execution_entrypoint_history'][0]['completed_stems']), 36)
        self.assertEqual(repaired['numerical_sources'], numerical)
        self.assertEqual(repaired['execution_entrypoint_history'][1], current_phase)
        self.assertEqual((self.root/result['before_manifest']).read_bytes(), before)
        self.assertEqual(len(repaired['metadata_repairs']), 1)
        safe.io.check_hashes(original_inputs)
        safe.io.check_hashes(repaired['input_sha256'])

    def test_repair_is_idempotent_without_restamping_master_or_snapshot(self):
        self.broken_campaign()
        first = safe.repair_imports(self.data)
        before = (self.data/'manifest.json').read_bytes()
        snapshot_before = (self.root/first['before_manifest']).read_bytes()
        second = safe.repair_imports(self.data)
        self.assertEqual(second['status'], 'already_correct')
        self.assertFalse(second['manifest_changed'])
        self.assertEqual((self.data/'manifest.json').read_bytes(), before)
        self.assertEqual((self.root/first['before_manifest']).read_bytes(), snapshot_before)

    def test_tampered_import_is_rejected_before_any_metadata_mutation(self):
        manifest = self.broken_campaign()
        before = (self.data/'manifest.json').read_bytes()
        stem = next(stem for stem in manifest['completed'] if stem.endswith('N3_marginal'))
        (self.data/(stem+'.npz')).write_bytes(b'changed certified arrays')
        with self.assertRaises(RuntimeError):
            safe.repair_imports(self.data)
        self.assertEqual((self.data/'manifest.json').read_bytes(), before)
        self.assertFalse((self.data/'provenance/manifest_before_import_scope_repair.json').exists())

    def test_future_wrapper_json_roundtrip_breaks_all_shared_import_lists(self):
        shared = ['pilot'+str(i) for i in range(36)]
        manifest = dict(completed=shared, import_provenance=dict(imported_stems=shared),
                        execution_entrypoint_history=[dict(completed_stems=shared)])
        with patch.object(campaign, 'prepare', return_value=manifest), \
                patch.object(campaign, 'new_sources', return_value=dict(self.sources)):
            restore = safe.install()
            try:
                isolated = campaign.prepare(self.data, {})
                references = campaign.new_sources({})
            finally:
                restore()
        isolated['completed'].append('newly solved case')
        self.assertEqual(len(isolated['completed']), 37)
        self.assertEqual(len(isolated['import_provenance']['imported_stems']), 36)
        self.assertEqual(len(isolated['execution_entrypoint_history'][0]['completed_stems']), 36)
        self.assertEqual(references['expanded_safe.py'], safe.io.digest(self.helper))


if __name__ == '__main__':
    unittest.main()
