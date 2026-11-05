"""Import lineage, immutable checkpoints and selective refinement checks."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import audit
from . import high_p0_refine as entry


class HighP0RefinementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root/'experiment'
        self.source = self.home/'data/rho_p0_sweep_31'
        self.data = self.home/'data/high_p0_refinement_41'
        self.data.mkdir(parents=True)
        helper = self.root/'high_p0_refine.py'
        helper.write_text('refinement adapter fixture\n')
        frozen = self.root/'frozen.py'
        frozen.write_text('frozen equations fixture\n')
        self.sources = {'frozen.py':entry.io.digest(frozen)}
        source_patcher = patch.object(entry.worker,'source_reference',return_value=dict(self.sources))
        source_patcher.start()
        self.addCleanup(source_patcher.stop)
        for owner, name, value in ((entry, 'HOME', self.home), (entry, 'ROOT', self.root),
                                  (entry, '__file__', str(helper)),
                                  (entry.campaign, 'HOME', self.home), (entry.campaign, 'ROOT', self.root),
                                  (entry.io, 'ROOT', self.root), (audit, 'ROOT', self.root)):
            patcher = patch.object(owner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = dict(p0s=list(entry.campaign.P0S), rhos=list(entry.campaign.RHOS), grids=[21,31],
            case_grids={entry.campaign.pair_key(p,r):[21,31]
                        for p in entry.campaign.P0S for r in entry.campaign.RHOS},
            models=[list(model) for model in entry.campaign.MODELS], order=256, tolerance=1e-7,
            economics=dict(R=1.,c1=.05,c2=.65,p1=.35,p2=.8,gamma=.98),
            purpose='Compare 1, 2 and 3 strategic sellers with fixed p0 across persistence and outside benchmarks')
        self.refined = deepcopy(self.config)
        self.refined['grids'] = [21,31,41]
        for p0 in entry.HIGH_P0S:
            for rho in entry.campaign.RHOS:
                self.refined['case_grids'][entry.campaign.pair_key(p0,rho)].append(41)

    def checkpoint(self, data, p0, rho, n, sellers, closure):
        stem = entry.io.key(p0,rho,n,sellers,closure)
        path = data/stem
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path)+'.npz', grid=np.linspace(0,1,n), policy=np.full(n,.5))
        entry.io.write_json(str(path)+'.json', dict(
            summary=dict(p0=p0,rho=rho,n=n,sellers=sellers,closure=closure,verified=True),
            checkpoint_sha256=entry.io.digest(str(path)+'.npz')))
        return stem, {str(Path(str(path)+suffix).relative_to(self.root)):entry.io.digest(str(path)+suffix)
                      for suffix in ('.json','.npz')}

    def source_campaign(self):
        inputs, completed = {}, []
        for p0 in self.config['p0s']:
            for rho in self.config['rhos']:
                for n in (21,31):
                    for sellers,closure in entry.campaign.MODELS:
                        stem, fingerprints = self.checkpoint(self.source,p0,rho,n,sellers,closure)
                        completed.append(stem)
                        inputs.update(fingerprints)
        metadata = self.source/'provenance/earlier_source.json'
        entry.io.write_json(metadata, {'historical':True})
        inputs[str(metadata.relative_to(self.root))] = entry.io.digest(metadata)
        manifest = dict(status='complete',completed=completed,input_sha256=inputs,
                        numerical_sources=self.sources,configuration=self.config,failures={})
        entry.io.export(self.source,manifest)
        entry.io.write_json(self.source/'manifest.json',manifest)
        return manifest

    def prepare(self):
        return entry.prepare_refinement(self.data,self.refined,self.source,lambda sources:dict(sources))

    def test_imports_all_144_byte_identically_with_auditable_generic_source_lineage(self):
        original = self.source_campaign()
        before = {path:path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        result = self.prepare()
        self.assertEqual(len(result['completed']),144)
        self.assertEqual(len(entry.campaign.expected(result['configuration'])),180)
        for stem in original['completed']:
            for suffix in ('.json','.npz'):
                self.assertEqual((self.source/(stem+suffix)).read_bytes(),(self.data/(stem+suffix)).read_bytes())
        self.assertEqual((self.source/'manifest.json').read_bytes(),(self.data/'provenance/source_manifest.json').read_bytes())
        self.assertEqual({path:path.read_bytes() for path in before},before)
        entry.io.check_hashes(result['input_sha256'])
        lineage = audit.verify_import_provenance(self.data,result)
        self.assertTrue(lineage['verified'])
        self.assertEqual(lineage['imported_model_grid_cases'],144)
        self.assertEqual(lineage['identical_checkpoint_files'],288)
        self.assertEqual(result['stationary_certification'],entry.worker.CERTIFICATION)
        self.assertEqual(result['execution_entrypoint_history'][-1]['stationary_certification'],entry.worker.CERTIFICATION)
        for name,fingerprint in original['input_sha256'].items():
            self.assertEqual(result['input_sha256'][name],fingerprint)

    def test_future_merge_does_not_expand_historical_import_lists(self):
        self.source_campaign()
        manifest = self.prepare()
        stem, fingerprints = self.checkpoint(self.data,.8,.6,41,1,'monopoly')
        entry.io.write_json(self.data/'p0_0p8/rho_0p6/worker_state.json',dict(
            completed=[stem],input_sha256=fingerprints,p0=.8,rho=.6))
        entry.campaign.merge_states(self.data,manifest)
        self.assertEqual(len(manifest['completed']),145)
        self.assertEqual(len(manifest['import_provenance']['imported_stems']),144)
        self.assertEqual(len(manifest['execution_entrypoint_history'][0]['completed_stems']),144)
        self.assertIsNot(manifest['completed'],manifest['import_provenance']['imported_stems'])
        self.assertIsNot(manifest['completed'],manifest['execution_entrypoint_history'][0]['completed_stems'])
        self.assertTrue(audit.verify_import_provenance(self.data,manifest)['verified'])

    def test_changed_source_checkpoint_is_rejected_before_any_destination_write(self):
        source = self.source_campaign()
        stem = source['completed'][0]
        (self.source/(stem+'.npz')).write_bytes(b'corrupted compressed array')
        with self.assertRaisesRegex(RuntimeError,'input changed'):
            self.prepare()
        self.assertEqual(list(self.data.iterdir()),[])

    def test_uncertified_record_is_rejected_even_when_all_fingerprints_match(self):
        source = self.source_campaign()
        stem = source['completed'][0]
        path = self.source/(stem+'.json')
        record = json.loads(path.read_bytes())
        record['summary']['verified'] = False
        entry.io.write_json(path,record)
        source['input_sha256'][str(path.relative_to(self.root))] = entry.io.digest(path)
        entry.io.write_json(self.source/'manifest.json',source)
        with self.assertRaisesRegex(RuntimeError,'uncertified'):
            self.prepare()
        self.assertEqual(list(self.data.iterdir()),[])

    def test_partial_source_and_economic_changes_are_rejected(self):
        source = self.source_campaign()
        source['status'] = 'running'
        entry.io.write_json(self.source/'manifest.json',source)
        with self.assertRaisesRegex(RuntimeError,'closed and complete'):
            self.prepare()
        source['status'] = 'complete'
        entry.io.write_json(self.source/'manifest.json',source)
        self.refined['economics']['c2'] = .66
        with self.assertRaisesRegex(RuntimeError,'economics'):
            self.prepare()
        self.assertEqual(list(self.data.iterdir()),[])

    def test_conflicting_existing_import_is_preserved_and_never_overwritten(self):
        source = self.source_campaign()
        stem = source['completed'][0]
        path = self.data/(stem+'.json')
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preserve these conflicting bytes')
        with self.assertRaisesRegex(RuntimeError,'Existing import bytes differ'):
            self.prepare()
        self.assertEqual(path.read_bytes(),b'preserve these conflicting bytes')
        self.assertFalse((self.data/'manifest.json').exists())

    def test_prepare_cli_adds_only_36_high_p0_cases_and_can_resume_without_solves(self):
        self.source_campaign()
        # The test exercises the real controller lock, config and CSV export;
        # only source registration is replaced with the fixture's source map.
        original_worker = entry.campaign.work_case
        with patch.object(entry.campaign,'new_sources',side_effect=lambda previous:dict(previous)):
            entry.main(['--prepare-only'])
            self.assertIs(entry.campaign.work_case,original_worker)
            manifest = json.loads((self.data/'manifest.json').read_bytes())
            self.assertEqual(len(manifest['completed']),144)
            self.assertEqual(len(entry.campaign.expected(manifest['configuration'])),180)
            entry.main(['--resume','--prepare-only'])
        saved = json.loads((self.data/'manifest.json').read_bytes())
        self.assertEqual(saved['completed'],manifest['completed'])
        self.assertEqual(len(saved['import_provenance']['imported_stems']),144)
        self.assertEqual(len(saved['execution_entrypoint_history'][0]['completed_stems']),144)

    def test_resume_rejects_a_changed_source_manifest_and_restores_worker(self):
        self.source_campaign()
        with patch.object(entry.campaign,'new_sources',side_effect=lambda previous:dict(previous)):
            entry.main(['--prepare-only'])
            source_manifest = self.source/'manifest.json'
            source_manifest.write_bytes(source_manifest.read_bytes()+b'\n')
            original_worker = entry.campaign.work_case
            with self.assertRaisesRegex(RuntimeError,'source manifest fingerprint'):
                entry.main(['--resume','--prepare-only'])
            self.assertIs(entry.campaign.work_case,original_worker)

    def test_51_is_explicit_and_retains_intermediate_41(self):
        args = SimpleNamespace(name=None,target_grid=51,workers=1,max_evaluations=800,resume=False,prepare_only=True)
        arguments = entry.controller_arguments(args)
        extensions = [arguments[i+1] for i,value in enumerate(arguments) if value == '--add-case-grid']
        self.assertEqual(len(extensions),24)
        self.assertEqual({item.split(':')[0] for item in extensions},{'0.7','0.8','0.9'})
        self.assertEqual({item.split(':')[2] for item in extensions},{'41','51'})
        self.assertEqual(arguments[arguments.index('--name')+1],'high_p0_refinement_51')


if __name__ == '__main__':
    unittest.main()
