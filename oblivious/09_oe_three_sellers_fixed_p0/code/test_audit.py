"""The independent audit rejects consistent-hash numerical corruption."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from .audit import (AuditFailure, HOME, ROOT, audit_checkpoint, audit_pass,
                    case_stem, digest, verify_import_provenance)
from .model import Experiment


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        experiment = Experiment(n=5, sellers=1, closure='monopoly', p0=.5,
                                params=Parameters(rho=.6), order=64)
        cls.result = experiment.solve()
        cls.configuration = experiment.configuration()
        cls.stem = case_stem(.5,.6,5,1,'monopoly')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HOME/'data', prefix='_audit_test_')
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.arrays = {name:self.result[name].copy() for name in
                       ['grid','policy','joint','value','actual_value','demand','advantage']}
        self.summary = dict(self.result['summary'])

    def save(self):
        npz = self.data/(self.stem+'.npz')
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **self.arrays)
        json_path = self.data/(self.stem+'.json')
        json_path.write_text(json.dumps(dict(configuration=self.configuration,
                                             summary=self.summary,
                                             diagnostics=self.result['diagnostics'],
                                             checkpoint_sha256=digest(npz)),
                                        default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item()))
        return {str(path.relative_to(ROOT)):digest(path) for path in (json_path,npz)}

    def test_fresh_valid_bundle(self):
        result = audit_checkpoint(self.data,self.stem,self.save())
        self.assertTrue(result['verified'])
        self.assertLess(result['actual_discounted_error_bound'],1e-8)

    def test_fingerprint_tampering_is_rejected(self):
        fingerprints = self.save()
        fingerprints[next(iter(fingerprints))] = '0'*64
        with self.assertRaisesRegex(AuditFailure,'Fingerprint mismatch'):
            audit_checkpoint(self.data,self.stem,fingerprints)

    def test_actual_values_shifted_with_updated_hashes_are_rejected(self):
        self.arrays['actual_value'] += .01
        self.summary['value_initial'] += .01
        with self.assertRaisesRegex(AuditFailure,'Actual discounted-value equation failed'):
            audit_checkpoint(self.data,self.stem,self.save())

    def test_product_share_corruption_is_rejected(self):
        self.summary['product2_conditional'] += .01
        with self.assertRaisesRegex(AuditFailure,'Summary product2_conditional'):
            audit_checkpoint(self.data,self.stem,self.save())

    def test_noninvariant_normalized_joint_is_rejected(self):
        self.arrays['joint'][0] += .01
        self.arrays['joint'] /= self.arrays['joint'].sum()
        with self.assertRaisesRegex(AuditFailure,'Full invariant law failed'):
            audit_checkpoint(self.data,self.stem,self.save())

    def marginal_manifest(self):
        return dict(configuration=dict(p0s=[.1,.5,.8],rhos=[.6,.8],grids=[21,31],
                                        models=[[1,'monopoly'],[2,'marginal'],[3,'marginal']]),
                    status='running',completed=[],numerical_sources={},input_sha256={})

    def save_manifest(self, manifest):
        (self.data/'manifest.json').write_text(json.dumps(manifest))

    def test_marginal_scope_has_36_cases(self):
        self.save_manifest(self.marginal_manifest())
        result = audit_pass(self.data)
        self.assertEqual(result['expected_model_grid_cases'],36)
        self.assertEqual(result['audited_model_grid_cases'],0)
        self.assertFalse(result['verified'])

    def test_conditional_checkpoint_is_outside_marginal_scope(self):
        manifest = self.marginal_manifest()
        manifest['completed'] = [case_stem(.1,.6,21,3,'conditional')]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(AuditFailure,'outside the configured experiment scope'):
            audit_pass(self.data)

    def test_closed_marginal_scope_cannot_omit_cases(self):
        manifest = self.marginal_manifest()
        manifest['status'] = 'complete'
        self.save_manifest(manifest)
        with self.assertRaisesRegex(AuditFailure,'incomplete checkpoint count'):
            audit_pass(self.data)

    def test_duplicate_completed_cases_do_not_inflate_count(self):
        manifest = self.marginal_manifest()
        manifest['completed'] = [case_stem(.1,.6,21,3,'marginal')]*36
        self.save_manifest(manifest)
        with self.assertRaisesRegex(AuditFailure,'Duplicate completed checkpoint stems'):
            audit_pass(self.data)

    def imported_manifest(self):
        source, target = self.data/'source', self.data/'target'
        source.mkdir(); target.mkdir()
        original_inputs, target_inputs = {}, {}
        for suffix in ('.json','.npz'):
            original, clone = source/(self.stem+suffix), target/(self.stem+suffix)
            original.parent.mkdir(parents=True, exist_ok=True)
            clone.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b'immutable checkpoint '+suffix.encode())
            shutil.copyfile(original,clone)
            original_inputs[str(original.relative_to(ROOT))] = digest(original)
            target_inputs[str(clone.relative_to(ROOT))] = digest(clone)
        source_manifest = source/'manifest.json'
        source_manifest.write_text(json.dumps(dict(status='failed',completed=[self.stem],
                                                   input_sha256=original_inputs,numerical_sources={},
                                                   failures={'other_conditional_case':'row instability'})))
        snapshot = target/'source_manifest.json'
        shutil.copyfile(source_manifest,snapshot)
        inputs = dict(original_inputs,**target_inputs)
        inputs[str(snapshot.relative_to(ROOT))] = digest(snapshot)
        manifest = dict(completed=[self.stem],numerical_sources={},input_sha256=inputs,
                        execution_entrypoint_history=[dict(numerical_sources={},completed_stems=[self.stem]),
                                                      dict(numerical_sources={})],
                        import_provenance=dict(source_manifest=str(source_manifest.relative_to(ROOT)),
                            source_manifest_sha256=digest(source_manifest),
                            source_manifest_snapshot=str(snapshot.relative_to(ROOT)),
                            imported_stems=[self.stem],source_input_sha256=original_inputs))
        return target,manifest

    def test_imports_preserve_source_failure_and_checkpoint_bytes(self):
        target,manifest = self.imported_manifest()
        result = verify_import_provenance(target,manifest)
        self.assertTrue(result['verified'])
        self.assertEqual(result['identical_checkpoint_files'],2)
        self.assertEqual(result['source_campaign_status'],'failed')
        self.assertIn('other_conditional_case',result['source_failures'])

    def test_imported_bytes_cannot_be_changed_by_updating_target_hash(self):
        target,manifest = self.imported_manifest()
        clone = target/(self.stem+'.npz')
        clone.write_bytes(b'changed')
        manifest['input_sha256'][str(clone.relative_to(ROOT))] = digest(clone)
        with self.assertRaisesRegex(AuditFailure,'bytes differ from the source'):
            verify_import_provenance(target,manifest)

    def test_additional_frozen_validation_phase_is_supported(self):
        target,manifest = self.imported_manifest()
        extra = self.data/'validation_entry.py'
        extra.write_text('longer work budget; same acceptance criteria\n')
        sources = {str(extra.relative_to(ROOT)):digest(extra)}
        manifest['numerical_sources'] = sources
        manifest['execution_entrypoint_history'].append(dict(numerical_sources=sources,
            completed_stems_at_start=[self.stem],positive_max_iterations=1_000_000))
        result = verify_import_provenance(target,manifest)
        self.assertTrue(result['verified'])
        self.assertEqual(len(result['execution_entrypoint_history']),3)

    def test_current_validation_phase_must_match_current_sources(self):
        target,manifest = self.imported_manifest()
        extra = self.data/'validation_entry.py'
        extra.write_text('longer work budget\n')
        manifest['numerical_sources'] = {str(extra.relative_to(ROOT)):digest(extra)}
        manifest['execution_entrypoint_history'].append(dict(numerical_sources={}))
        with self.assertRaisesRegex(AuditFailure,'Current execution phase source fingerprints differ'):
            verify_import_provenance(target,manifest)


if __name__ == '__main__':
    unittest.main()
