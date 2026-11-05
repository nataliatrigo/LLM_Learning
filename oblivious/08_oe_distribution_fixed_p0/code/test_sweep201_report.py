"""Publication safeguards for original n201 policies and forward n251 checks."""

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.sweep201_report import (
    GRIDS, MODELS, P0S, RHOS, case_tag, common_rhos, digest, read_campaign,
    read_required_audit, read_sweep_execution, select_stable, validate_display_records,
)


MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_report'
READER = 'oblivious.oe_distribution_fixed_p0.code.multi_p0_report'


class Sweep201ReportTests(unittest.TestCase):
    def rows(self, p0=.3, rho=.9):
        return [dict(p0=p0, rho=rho, n=n, model=model, verified=True,
            product2_conditional=.4, product2_calendar=.2 if model == 'monopoly' else .12,
            sale_rate=.5 if model == 'monopoly' else .3,
            outside_rate=.5 if model == 'monopoly' else .4,
            profit_calendar=.355 if model == 'monopoly' else .213, value_initial=10.)
            for n in GRIDS for model, *_ in MODELS]

    def policies(self, p0=.3, rho=.9):
        return [dict(p0=p0, rho=rho, n=n, model=model, reputation=float(x),
                     product2_probability=.4 if n == 201 else .8)
                for n in GRIDS for model, *_ in MODELS for x in np.linspace(0, 1, n)]

    def test_fine_grid_validates_without_replacing_the_original_201_nodes(self):
        rows, policies = self.rows(), self.policies()
        before = copy.deepcopy((rows, policies))
        selected = select_stable(rows, [.9])
        self.assertEqual(selected['included_rhos'], [.9])
        self.assertEqual(selected['displayed_grid'], 201)
        self.assertEqual({(row['coarse_n'], row['fine_n'], row['displayed_n'])
                          for row in selected['comparisons']}, {(201, 251, 201)})
        displayed, displayed_policies = validate_display_records(rows, policies, .3, [.9], grid=201)
        self.assertEqual({row['n'] for row in displayed+displayed_policies}, {201})
        self.assertEqual({row['product2_probability'] for row in displayed_policies}, {.4})
        self.assertEqual((rows, policies), before)
        # An earlier, stable 101-point result cannot fill a missing target grid.
        missing = [dict(row, n=101) if row['n'] == 201 else row for row in rows]
        self.assertEqual(select_stable(missing, [.9])['included_rhos'], [])

    def test_missing_251_is_not_certified_and_original_nodes_cannot_be_resampled(self):
        rows = [row for row in self.rows() if not (row['n'] == 251 and row['model'] == 'competition')]
        self.assertEqual(select_stable(rows, [.9])['included_rhos'], [])
        policies = self.policies()
        own = next(row for row in policies if row['n'] == 201 and row['model'] == 'monopoly'
                   and row['reputation'] > 0)
        own['reputation'] += 1e-6
        with self.assertRaisesRegex(ValueError, 'original common grid'):
            validate_display_records(self.rows(), policies, .3, [.9], grid=201)

    def test_same_forward_threshold_and_global_rho_set_apply_to_every_p0_model(self):
        rows = self.rows()
        fine = next(row for row in rows if row['n'] == 251 and row['model'] == 'competition')
        fine['profit_calendar'] = .213/.95
        self.assertEqual(select_stable(rows, [.9])['included_rhos'], [.9])
        fine['profit_calendar'] = .213/.949
        failed = select_stable(rows, [.9])
        self.assertEqual(failed['included_rhos'], [])
        record = next(row for row in failed['comparisons'] if row['model'] == 'competition')
        self.assertAlmostEqual(record['profit_calendar_relative_change'], .051)
        self.assertEqual(common_rhos({'.3': failed, '.9': {'included_rhos': [.9]}}, [.9]), [])
        fine['profit_calendar'] = .213
        fine['outside_rate'] += .050001
        self.assertEqual(select_stable(rows, [.9])['included_rhos'], [])

    def environment(self, root):
        stack = ExitStack()
        stack.enter_context(patch(MODULE+'.ROOT', root))
        stack.enter_context(patch(MODULE+'.HOME', root))
        stack.enter_context(patch(READER+'.ROOT', root))
        return stack

    def saved_case(self, n, closure):
        grid, policy = np.linspace(0, 1, n), np.full(n, .4)
        configuration = dict(n=n, p0=.3, order=256, tolerance=1e-7, dp_tolerance=1e-10,
            stationary_tolerance=1e-11, stationary_stability_tolerance=1e-7,
            strategy='Own reputation', market='Two sellers and fixed outside',
            outside_event='Both freeze', forgetting_clock='Own observations',
            p0_role='Fixed known benchmark', closure=closure+' distribution',
            parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98, rho=.9))
        models = {'monopoly', 'competition'} if closure == 'marginal' else {'conditional_competition'}
        rows = [row for row in self.rows() if row['n'] == n and row['model'] in models]
        policies = [dict(row, product2_probability=.4) for row in self.policies()
                    if row['n'] == n and row['model'] in models]
        return dict(p0=.3, rho=.9, n=n, configuration=configuration,
            demand_audit={'verified': True}, outcomes=rows, policy_rows=policies,
            monopoly=dict(verified=True, grid=grid, policy=policy, value=np.full(n, 10.)),
            competition=dict(verified=True, grid=grid, policy=policy,
                             actual_values=np.full((2, n, n), 10.)))

    def campaign(self, root):
        directory = root/'data/p0_0p3/rho_0p9'
        (directory/'arrays').mkdir(parents=True)
        source = root/'engine.py'
        source.write_text('unchanged numerical engine')
        sources = {'engine.py': digest(source)}
        configuration = self.saved_case(201, 'marginal')['configuration']
        manifest = dict(status='complete', current_grid=None, current_model=None,
            requested_grids=GRIDS,
            configuration=dict(p0=.3, rho=.9, order=256, tol=1e-7, dp_tol=1e-10,
                parameters={key: value for key, value in configuration['parameters'].items() if key != 'rho'}),
            numerical_sources=sources, reference_input_sha256={}, completed=[], checkpoint_sha256={})
        for n in GRIDS:
            for closure in ('marginal', 'conditional'):
                tag = case_tag(.9, n)+'_'+closure
                manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, self.saved_case(n, closure)))
                manifest['completed'].append(tag)
        (directory/'manifest.json').write_text(json.dumps(manifest))
        return directory, manifest, sources

    def test_reader_accepts_actual_201_251_bundles_and_rejects_stale_metadata_or_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest, sources = self.campaign(root)
            with self.environment(root), patch(MODULE+'.source_fingerprints', return_value=sources):
                rows, policies, inputs, statuses, _ = read_campaign(root/'data', .3, [.9])
                self.assertEqual({row['n'] for row in rows+policies}, {201, 251})
                self.assertEqual(len(rows), 6)
                self.assertEqual(statuses['0.9']['status'], 'complete')
                self.assertIn('data/p0_0p3/rho_0p9/manifest.json', inputs)
                tag = case_tag(.9, 201)+'_marginal'
                wrong = self.saved_case(201, 'marginal')
                wrong['configuration']['n'] = 101
                manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, wrong))
                (directory/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'p0/rho/grid mismatch'):
                    read_campaign(root/'data', .3, [.9])
                manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, self.saved_case(201, 'marginal')))
                (directory/'manifest.json').write_text(json.dumps(manifest))
                (directory/'arrays'/(tag+'.npz')).write_bytes(b'changed checkpoint')
                with self.assertRaisesRegex(ArithmeticError, 'Checkpoint changed'):
                    read_campaign(root/'data', .3, [.9])

    def test_every_execution_phase_source_is_verified_even_without_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'code').mkdir()
            (root/'data').mkdir()
            for name in ('sweep201_batch.py', 'sweep201_run.py', 'old_phase.py', 'parallel.cpp'):
                (root/'code'/name).write_text(name)
            scheduler, worker = (digest(root/'code'/name)
                                 for name in ('sweep201_batch.py', 'sweep201_run.py'))
            old_phase = dict(entry_module='old_phase', source_input_sha256={
                'code/old_phase.py': digest(root/'code/old_phase.py')},
                immutable_scheduler_sha256=scheduler, immutable_worker_sha256=worker)
            phase = dict(old_phase, entry_module='parallel_phase', source_input_sha256={
                'code/parallel.cpp': digest(root/'code/parallel.cpp')})
            master = dict(status='running', numerical_sources={},
                configuration=dict(p0s=P0S, rhos=RHOS, grids=GRIDS, order=256, tol=1e-7, dp_tol=1e-10),
                scheduler_sha256=scheduler, worker_sha256=worker,
                execution_entrypoint=phase, execution_entrypoint_history=[old_phase, phase])
            path = root/'data/sweep_manifest.json'
            path.write_text(json.dumps(master))
            with self.environment(root), patch(MODULE+'.source_fingerprints', return_value={}):
                summary, inputs = read_sweep_execution(root/'data', require_complete=False)
                self.assertEqual(summary['verified_execution_phase_count'], 2)
                self.assertEqual(inputs['code/parallel.cpp'], digest(root/'code/parallel.cpp'))
                (root/'code/old_phase.py').write_text('changed prior phase with no checkpoint')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: code/old_phase.py'):
                    read_sweep_execution(root/'data', require_complete=False)
                (root/'code/old_phase.py').write_text('old_phase.py')
                (root/'code/parallel.cpp').write_text('changed without a committed checkpoint')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: code/parallel.cpp'):
                    read_sweep_execution(root/'data', require_complete=False)
                (root/'code/parallel.cpp').write_text('parallel.cpp')
                master['execution_entrypoint'] = dict(phase, immutable_worker_sha256='different')
                path.write_text(json.dumps(master))
                with self.assertRaisesRegex(ValueError, 'final recorded phase'):
                    read_sweep_execution(root/'data', require_complete=False)
                master['execution_entrypoint_history'] = [master['execution_entrypoint']]
                path.write_text(json.dumps(master))
                with self.assertRaisesRegex(ValueError, 'immutable worker/scheduler'):
                    read_sweep_execution(root/'data', require_complete=False)

    def review(self, root):
        for name in ('audit.py', 'wrapper.py', 'case.json'):
            (root/name).write_text(name)
        review = dict(verified=True, p0s=P0S, rhos=RHOS, grids=GRIDS,
            publication_grid=201, validation_grid=251,
            publication_requirement_base_checkpoints_complete=True,
            audit_source='audit.py', audit_source_sha256=digest(root/'audit.py'),
            audit_entrypoint='wrapper.py', audit_entrypoint_sha256=digest(root/'wrapper.py'),
            audited_input_sha256={'case.json': digest(root/'case.json')},
            audited_bundle_count=96, audited_file_count=192, audited_input_count=1,
            cases=[dict(p0=p0, rho=rho, n=n, closure=closure, verified=True)
                   for p0 in P0S for rho in RHOS for n in GRIDS
                   for closure in ('marginal', 'conditional')], residual_maxima={'value': 1e-12})
        (root/'independent_review.json').write_text(json.dumps(review))
        return review

    def test_publication_requires_all_24_cases_both_grids_and_current_audited_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.environment(root):
                with self.assertRaisesRegex(ValueError, 'required before publication'):
                    read_required_audit(root, {})
                review = self.review(root)
                required = review['audited_input_sha256']
                summary, _ = read_required_audit(root, required)
                self.assertEqual(summary['audited_bundle_count'], 96)
                review['cases'].pop()
                (root/'independent_review.json').write_text(json.dumps(review))
                with self.assertRaisesRegex(ValueError, 'all 24'):
                    read_required_audit(root, required)
                self.review(root)
                with self.assertRaisesRegex(ValueError, 'every current checkpoint'):
                    read_required_audit(root, {**required, 'new_case.json': 'not audited'})
                (root/'case.json').write_text('changed after audit')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: case.json'):
                    read_required_audit(root, required)


if __name__ == '__main__':
    unittest.main()
