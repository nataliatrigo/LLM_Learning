"""Scope and provenance checks for per-case grid refinements."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import expanded_audit as audit


def configuration():
    return dict(p0s=[.1,.3,.5,.7,.8,.9],rhos=[.6,.7,.8,.9],grids=[21,31,41],
                models=[[1,'monopoly'],[2,'marginal'],[3,'marginal']])


class ExpandedAuditScopeChecks(unittest.TestCase):
    def test_global_schedule_has_216_model_grid_cases(self):
        scope = audit.expected_scope(configuration())
        self.assertEqual(scope['parameter_pairs'],24)
        self.assertEqual(len(scope['stems']),216)
        self.assertEqual(len(scope['case_grids']),24)

    def test_adaptive_grid_map_adds_only_selected_case(self):
        config = configuration()
        config['case_grids'] = audit.expected_scope(config)['case_grids']
        config['case_grids'][audit.pair_key(.8,.9)].append(51)
        scope = audit.expected_scope(config)
        self.assertEqual(len(scope['stems']),219)
        self.assertIn('p0_0p8/rho_0p9/n51_N3_marginal',scope['stems'])
        self.assertNotIn('p0_0p1/rho_0p6/n51_N3_marginal',scope['stems'])

    def test_case_map_rejects_unknown_or_missing_pairs(self):
        for change in ('missing','unknown'):
            config = configuration()
            config['case_grids'] = audit.expected_scope(config)['case_grids']
            if change == 'missing':
                del config['case_grids']['p0_0p1/rho_0p6']
            else:
                config['case_grids']['p0_0p2/rho_0p6'] = [21,31,41]
            with self.subTest(change=change):
                with self.assertRaisesRegex(audit.original.AuditFailure,'exactly all configured'):
                    audit.expected_scope(config)

    def test_duplicate_parameters_or_grids_are_rejected(self):
        for field in ('p0s','rhos','grids'):
            config = configuration();config[field].append(config[field][0])
            with self.subTest(field=field):
                with self.assertRaises(audit.original.AuditFailure):
                    audit.expected_scope(config)

    def test_invalid_adaptive_grid_is_rejected(self):
        for bad in ([21,31,31],[21,32],[True],[]):
            config = configuration();config['case_grids'] = audit.expected_scope(config)['case_grids']
            config['case_grids']['p0_0p1/rho_0p6'] = bad
            with self.subTest(grids=bad):
                with self.assertRaises(audit.original.AuditFailure):
                    audit.expected_scope(config)

    def test_audit_source_map_includes_frozen_auditor_and_new_adapter(self):
        references = audit.source_map()
        self.assertTrue(any(name.endswith('/code/audit.py') for name in references))
        self.assertTrue(any(name.endswith('/code/expanded_audit.py') for name in references))
        audit.original.verify_hashes(references)


class ExpandedAuditPassChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=audit.HOME/'data',prefix='_expanded_audit_test_')
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.config = dict(p0s=[.1],rhos=[.6],grids=[21],models=[[1,'monopoly']])
        self.stem = 'p0_0p1/rho_0p6/n21_N1_monopoly'
        self.manifest = dict(configuration=self.config,status='running',completed=[],
                             numerical_sources={},input_sha256={})

    def save(self):
        (self.data/'manifest.json').write_text(json.dumps(self.manifest))

    def test_closed_missing_checkpoint_cannot_pass(self):
        self.manifest['status'] = 'complete';self.save()
        with self.assertRaisesRegex(audit.original.AuditFailure,'incomplete or incorrect'):
            audit.audit_pass(self.data)

    def test_unchanged_original_checkpoint_checker_is_invoked(self):
        self.manifest['status'] = 'complete';self.manifest['completed'] = [self.stem];self.save()
        proof = dict(verified=True,input_sha256={})
        with patch.object(audit.original,'audit_checkpoint',return_value=proof) as checker:
            result = audit.audit_pass(self.data)
        checker.assert_called_once_with(self.data,self.stem,{})
        self.assertTrue(result['verified'])
        self.assertEqual(result['audited_model_grid_cases'],1)
        self.assertEqual(result['cases'][self.stem],proof)

    def test_stale_cached_checkpoint_cannot_disappear_silently(self):
        self.save()
        previous = dict(audit_sources=audit.source_map(),cases={self.stem:dict(input_sha256={})})
        with self.assertRaisesRegex(audit.original.AuditFailure,'disappeared'):
            audit.audit_pass(self.data,previous)

    def test_refinement_can_extend_scope_without_discarding_prior_checks(self):
        self.manifest['completed'] = [self.stem]
        self.manifest['configuration']['case_grids'] = {'p0_0p1/rho_0p6':[21,31]}
        self.save()
        proof = dict(verified=True,input_sha256={})
        previous = dict(audit_sources=audit.source_map(),cases={self.stem:proof})
        with patch.object(audit.original,'audit_checkpoint') as checker:
            result = audit.audit_pass(self.data,previous)
        checker.assert_not_called()
        self.assertEqual(result['expected_model_grid_cases'],2)
        self.assertEqual(result['audited_model_grid_cases'],1)
        self.assertFalse(result['verified'])

    def test_watch_finalizes_when_master_closes_during_partial_pass(self):
        partial = dict(status='running_partial_verified',verified=False,audited_model_grid_cases=0,
                       expected_model_grid_cases=1,current_manifest_status='complete')
        final = dict(status='complete_verified',verified=True,audited_model_grid_cases=1,
                     expected_model_grid_cases=1,current_manifest_status='complete')
        with (patch.object(audit,'HOME',self.data),
              patch.object(audit,'audit_pass',side_effect=[partial,final]) as check,
              patch.object(audit.original,'write_json'),
              patch.object(audit.time,'sleep') as sleep):
            code = audit.main(['--data','new','--watch'])
        self.assertEqual(code,0)
        self.assertEqual(check.call_count,2)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
