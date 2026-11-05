"""Safeguards specific to comparisons across the known fixed outside option."""

import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code.multi_p0_report import (
    MODELS, P0S, arguments, common_rhos, digest, read_independent_audit, rendering_rows,
    select_stable, sweep_execution_provenance, validate_display_records, validate_saved_case,
    validate_market_outside_rows,
)


class MultiP0ReportTests(unittest.TestCase):
    def test_user_selected_p0_point_eight_replaces_point_nine(self):
        self.assertEqual(P0S, [.1, .3, .5, .7, .8])
        with patch('sys.argv', ['multi_p0_report', '--p0s', '0.8']):
            self.assertEqual(arguments().p0s, [.8])
        with patch('sys.argv', ['multi_p0_report', '--p0s', '0.9']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                arguments()
            self.assertEqual(stopped.exception.code, 2)

    def rows(self, p0=.3):
        return [dict(p0=p0, rho=.9, n=n, model=model, verified=True,
                     product2_conditional=.4, product2_calendar=.2, sale_rate=.5,
                     outside_rate=.5 if model == 'monopoly' else 0.,
                     profit_calendar=.3, value_initial=10.)
                for n in (51, 101) for model, *_ in MODELS]

    def policies(self, p0=.3):
        return [dict(p0=p0, rho=.9, n=101, model=model, reputation=float(x), product2_probability=.4)
                for model, *_ in MODELS for x in np.linspace(0, 1, 101)]

    def test_mixed_p0_is_rejected_before_plotting(self):
        rows, policies = self.rows(), self.policies()
        rows[-1]['p0'] = .7
        with self.assertRaisesRegex(ValueError, 'mix p0/grid'):
            validate_display_records(rows, policies, .3, [.9])
        rows = self.rows()
        policies[-1]['p0'] = .7
        with self.assertRaisesRegex(ValueError, 'mix p0/grid'):
            validate_display_records(rows, policies, .3, [.9])

    def test_outside_is_whole_market_not_focal_nonselection(self):
        rows = self.rows()
        for row in rows:
            if row['model'] != 'monopoly':
                row.update(sale_rate=.3, outside_rate=.4)
        validate_display_records(rows, self.policies(), .3, [.9])
        rows[-1]['outside_rate'] = .7
        with self.assertRaisesRegex(ValueError, 'whole-market'):
            validate_display_records(rows, self.policies(), .3, [.9])

    def test_endpoint_roundoff_is_accepted_but_meaningful_excess_is_rejected(self):
        row = dict(model='competition', rho=.97, outside_rate=1+2*np.finfo(float).eps,
                   sale_rate=4.15e-24)
        validate_market_outside_rows([row])
        row['outside_rate'] = 1+1e-8
        with self.assertRaisesRegex(ValueError, 'whole-market'):
            validate_market_outside_rows([row])

    def test_rendering_clips_copies_and_keeps_raw_saved_probabilities(self):
        row = dict(model='competition', rho=.97, outside_rate=1+2*np.finfo(float).eps,
                   sale_rate=4.15e-24, product2_conditional=.4, profit_calendar=1e-24)
        before = copy.deepcopy(row)
        displayed = rendering_rows([row])[0]
        self.assertEqual(row, before)
        self.assertIsNot(displayed, row)
        self.assertEqual(displayed['outside_rate'], 1.)
        self.assertEqual({key: value for key, value in displayed.items() if key != 'outside_rate'},
                         {key: value for key, value in row.items() if key != 'outside_rate'})

    def test_finer_validation_never_substitutes_displayed_n101(self):
        rows = self.rows()
        conditional = next(row for row in rows if row['n'] == 101 and row['model'] == 'conditional_competition')
        conditional['product2_conditional'] = .47
        rows.append(dict(conditional, n=151, product2_conditional=.48))
        policies = self.policies()+[dict(p0=.3, rho=.9, n=151, model='conditional_competition',
                                         reputation=float(x), product2_probability=.9)
                                     for x in np.linspace(0, 1, 151)]
        before = copy.deepcopy(rows)
        selection = select_stable(rows, [.9])
        self.assertEqual(selection['included_rhos'], [.9])
        checks = [row for row in selection['comparisons'] if row['model'] == 'conditional_competition']
        self.assertFalse(checks[0]['passed'])
        self.assertFalse(checks[0]['used_for_selection'])
        self.assertTrue(checks[1]['passed'])
        self.assertEqual((checks[1]['coarse_n'], checks[1]['fine_n']), (101, 151))
        fine, fine_policies = validate_display_records(rows, policies, .3, [.9])
        self.assertEqual({row['n'] for row in fine+fine_policies}, {101})
        self.assertEqual(next(row['product2_conditional'] for row in fine
                              if row['model'] == 'conditional_competition'), .47)
        self.assertEqual(rows, before)

    def test_failed_finer_check_is_excluded_from_common_rhos(self):
        rows = self.rows()
        conditional = next(row for row in rows if row['n'] == 101 and row['model'] == 'conditional_competition')
        conditional['value_initial'] = 12.
        rows.append(dict(conditional, n=151, value_initial=15.))
        selection = select_stable(rows, [.9])
        self.assertEqual(selection['included_rhos'], [])
        self.assertEqual(common_rhos({'0.3': selection, '0.5': {'included_rhos': [.9]}}, [.9]), [])

    def test_missing_n101_is_not_replaced_by_n151(self):
        rows = [dict(row, n=151) if row['n'] == 101 and row['model'] == 'conditional_competition'
                else row for row in self.rows()]
        self.assertEqual(select_stable(rows, [.9])['included_rhos'], [])
        with self.assertRaisesRegex(ValueError, 'omit/duplicate'):
            validate_display_records(rows, self.policies(), .3, [.9])

    def test_truncated_policy_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original common grid'):
            validate_display_records(self.rows(), self.policies()[:-1], .3, [.9])

    def saved_case(self):
        grid = np.linspace(0, 1, 3)
        policy = np.full(3, .4)
        config = dict(n=3, p0=.3, order=256, tolerance=1e-7, dp_tolerance=1e-10,
            stationary_tolerance=1e-11, stationary_stability_tolerance=1e-7,
            strategy='Own reputation', market='Two sellers and fixed outside',
            outside_event='Both freeze', forgetting_clock='Own observations',
            p0_role='Fixed known benchmark', closure='Full marginal of joint invariant law',
            parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98, rho=.9))
        rows = [dict(row, n=3, profit_calendar=.355) for row in self.rows()
                if row['n'] == 101 and row['model'] != 'conditional_competition']
        policies = [dict(p0=.3, rho=.9, n=3, model=model, reputation=float(x), product2_probability=.4)
                    for model in ('monopoly', 'competition') for x in grid]
        return dict(p0=.3, rho=.9, n=3, configuration=config, demand_audit=dict(verified=True),
                    monopoly=dict(verified=True, grid=grid, policy=policy, value=np.full(3, 10.)),
                    competition=dict(verified=True, grid=grid, policy=policy,
                                     actual_values=np.full((2, 3, 3), 10.)),
                    outcomes=rows, policy_rows=policies)

    def test_saved_checkpoint_metadata_and_accounting_are_validated(self):
        case = self.saved_case()
        validate_saved_case(case, .3, .9, 3, 'marginal')
        case['configuration']['n'] = 101
        with self.assertRaisesRegex(ValueError, 'p0/rho/grid mismatch'):
            validate_saved_case(case, .3, .9, 3, 'marginal')
        case = self.saved_case()
        case['outcomes'][1]['profit_calendar'] = .4
        with self.assertRaisesRegex(ValueError, 'accounting or actual-value'):
            validate_saved_case(case, .3, .9, 3, 'marginal')

    def test_perceived_value_cannot_replace_actual_game_value(self):
        case = self.saved_case()
        case['outcomes'][1].update(value_initial=12., perceived_value_initial=12.)
        with self.assertRaisesRegex(ValueError, 'actual-value'):
            validate_saved_case(case, .3, .9, 3, 'marginal')

    def test_available_independent_audit_must_match_current_inputs_and_selection(self):
        module = 'oblivious.oe_distribution_fixed_p0.code.multi_p0_report'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = root/'audit.py', root/'case.json'
            source.write_text('audit source')
            checkpoint.write_text('saved checkpoint')
            review = dict(verified=True, p0s=[.1, .3, .7, .8], rhos=[.9],
                audit_source='audit.py', audit_source_sha256=digest(source),
                audited_input_sha256={'case.json': digest(checkpoint)},
                audited_bundle_count=1, audited_file_count=2, audited_input_count=1,
                residual_maxima={'actual_value_bellman_max': 1e-14})
            path = root/'independent_review.json'
            path.write_text(json.dumps(review))
            with patch(module+'.ROOT', root):
                summary, inputs = read_independent_audit(root, [.3, .5, .8], [.9])
                self.assertEqual(summary['audited_bundle_count'], 1)
                self.assertEqual(inputs['audit.py'], digest(source))
                self.assertEqual(inputs['independent_review.json'], digest(path))
                review['p0s'] = [.1, .3, .7]
                path.write_text(json.dumps(review))
                with self.assertRaisesRegex(ValueError, 'current p0/rho selection'):
                    read_independent_audit(root, [.3, .8], [.9])
                review['p0s'].append(.8)
                path.write_text(json.dumps(review))
                checkpoint.write_text('changed checkpoint')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs'):
                    read_independent_audit(root, [.3, .8], [.9])

    def test_archived_execution_and_partial_supplement_hashes_are_verified(self):
        module = 'oblivious.oe_distribution_fixed_p0.code.multi_p0_report'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executed, supplement = root/'executed.py', root/'monopoly.json'
            executed.write_text('archived executed scheduler')
            supplement.write_text('standalone monopoly certificate')
            sweep = dict(status='complete_comparison_scope_with_globally_excluded_supplements_stopped',
                prior_run_input_sha256={'executed.py': digest(executed)},
                supplemental_monopoly_validation_sha256={'monopoly.json': digest(supplement)},
                all_supplemental_151_models_completed=False,
                global_exclusion_witnesses={'0.9': 'Monopoly validation fails'},
                globally_excluded_rhos=[.9])
            with patch(module+'.ROOT', root):
                summary, inputs = sweep_execution_provenance(sweep)
                self.assertEqual(set(inputs), {'executed.py', 'monopoly.json'})
                self.assertFalse(summary['all_supplemental_151_models_completed'])
                self.assertIn('were stopped', summary['supplemental_stop_reason'])
                self.assertEqual(summary['global_exclusion_witnesses'], sweep['global_exclusion_witnesses'])
                supplement.write_text('changed certificate')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs'):
                    sweep_execution_provenance(sweep)


if __name__ == '__main__':
    unittest.main()
