"""Scientific reporting safeguards for the three-model fixed-p0 comparison."""

import copy
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import (
    MODELS, coincidence_diagnostics, displayed_records, render_figures, sensitivity_selection,
    validate_market_outside_rows,
)


class ConditionalComparisonReportTests(unittest.TestCase):
    def rows(self):
        return [dict(rho=.9, p0=.5, n=n, model=model, verified=True,
                     product2_conditional=.4, product2_calendar=.2, sale_rate=.5,
                     outside_rate=.5 if model == 'monopoly' else 0.,
                     profit_calendar=.3, value_initial=10.)
                for n in (51, 101) for model, *_ in MODELS]

    def test_outside_is_no_market_purchase_not_no_focal_seller_purchase(self):
        rows = [dict(model='monopoly', rho=.9, sale_rate=.7, outside_rate=.3),
                dict(model='competition', rho=.9, sale_rate=.3, outside_rate=.4),
                dict(model='conditional_competition', rho=.9, sale_rate=.3, outside_rate=.4)]
        validate_market_outside_rows(rows)
        # With two sellers each selling 30%, another seller's 30% is part of
        # focal non-selection, but not part of the 40% outside share.
        rows[1]['outside_rate'] = .7
        with self.assertRaisesRegex(ValueError, 'whole-market'):
            validate_market_outside_rows(rows)

    def test_conditional_sensitivity_is_checked_even_when_other_models_pass(self):
        rows = self.rows()
        conditional = next(row for row in rows
                           if row['n'] == 101 and row['model'] == 'conditional_competition')
        conditional['product2_conditional'] += .051
        selected = sensitivity_selection(rows, [.9], 51, 101)
        self.assertEqual(selected['included_rhos'], [])
        self.assertEqual(selected['excluded_rhos']['0.9'][0]['model'], 'conditional_competition')

    def test_actual_value_is_screened_even_if_perceived_value_is_unchanged(self):
        rows = self.rows()
        for row in rows:
            row['perceived_value_initial'] = 10.
            if row['n'] == 101 and row['model'] == 'conditional_competition':
                row['value_initial'] = 11.
        selected = sensitivity_selection(rows, [.9], 51, 101)
        self.assertEqual(selected['included_rhos'], [])
        self.assertEqual(selected['excluded_rhos']['0.9'][0]['metric'], 'value_initial')

    def test_missing_model_cannot_pass_as_a_two_model_comparison(self):
        rows = [row for row in self.rows() if not (
            row['n'] == 51 and row['model'] == 'conditional_competition')]
        selected = sensitivity_selection(rows, [.9], 51, 101)
        self.assertEqual(selected['included_rhos'], [])
        self.assertEqual(selected['excluded_rhos']['0.9'][0]['reason'],
                         'Missing verified coarse/fine comparison')

    def test_coincident_policies_do_not_claim_coincident_outcomes(self):
        rows = self.rows()
        for row in rows:
            if row['model'] == 'conditional_competition':
                row['value_initial'] = 9.
        policies = [dict(rho=.9, n=101, model=model, reputation=x, product2_probability=alpha)
                    for model, *_ in MODELS for x, alpha in [(0., 0.), (.5, 1.), (1., 0.)]]
        before = copy.deepcopy(policies)
        diagnostics = coincidence_diagnostics(rows, policies, [.9], 101)
        pair = next(item for item in diagnostics['pairs'] if item['first_model'] == 'competition'
                    and item['second_model'] == 'conditional_competition')
        self.assertTrue(pair['policy_exactly_equal'])
        self.assertFalse(pair['outcomes_equal_at_display_precision'])
        self.assertEqual(policies, before)

    def test_explicit_finer_check_preserves_failed_history_and_accepts_n101(self):
        rows = self.rows()
        conditional101 = next(row for row in rows
                              if row['n'] == 101 and row['model'] == 'conditional_competition')
        conditional101['product2_conditional'] = .468
        rows.append(dict(conditional101, n=151, product2_conditional=.475))
        self.assertEqual(sensitivity_selection(rows, [.9], 51, 101)['included_rhos'], [])
        selection = sensitivity_selection(rows, [.9], 51, 101,
                                          validation_pairs={(.9, 'conditional_competition'): (101, 151)})
        self.assertEqual(selection['included_rhos'], [.9])
        checked = [row for row in selection['comparisons'] if row['model'] == 'conditional_competition']
        self.assertEqual(len(checked), 2)
        historical, finer = checked
        self.assertFalse(historical['passed'])
        self.assertFalse(historical['used_for_selection'])
        self.assertEqual((historical['coarse_n'], historical['fine_n']), (51, 101))
        self.assertTrue(finer['passed'])
        self.assertTrue(finer['used_for_selection'])
        self.assertEqual((finer['coarse_n'], finer['fine_n'], finer['displayed_n']), (101, 151, 101))
        rows[-1]['product2_conditional'] = .6
        rejected = sensitivity_selection(rows, [.9], 51, 101,
                                         validation_pairs={(.9, 'conditional_competition'): (101, 151)})
        self.assertEqual(rejected['included_rhos'], [])

    def test_n151_cannot_replace_n101_in_rendered_lines_or_exported_records(self):
        rows = self.rows()
        conditional101 = next(row for row in rows
                              if row['n'] == 101 and row['model'] == 'conditional_competition')
        rows.append(dict(conditional101, n=151, product2_conditional=.99,
                         profit_calendar=99., value_initial=999.))
        policies = [dict(rho=.9, n=101, model=model, reputation=float(x),
                         product2_probability=float(.2 < x < .7))
                    for model, *_ in MODELS for x in np.linspace(0, 1, 101)]
        policies += [dict(rho=.9, n=151, model='conditional_competition',
                          reputation=float(x), product2_probability=.25)
                     for x in np.linspace(0, 1, 151)]
        exported_rows, exported_policies = displayed_records(rows, policies, 101)
        self.assertEqual(len(exported_rows), 3)
        self.assertEqual(len(exported_policies), 303)
        self.assertEqual({row['n'] for row in exported_rows + exported_policies}, {101})
        captures = {}

        def inspect_figure(fig, _directory, stem):
            captures[stem] = [[(line.get_label(), line.get_xdata().copy(), line.get_ydata().copy())
                               for line in ax.lines] for ax in fig.axes]
            plt.close(fig)
            return []

        module = 'oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report'
        with tempfile.TemporaryDirectory() as directory, patch(module+'.save', side_effect=inspect_figure):
            render_figures(rows, policies, directory, [.9], n=101,
                           validation_note='Conditional n101 checked against n151')
        conditional_policy = next(line for line in captures['01_policy_comparison'][0]
                                  if line[0] == 'Conditional OE')
        self.assertEqual(len(conditional_policy[1]), 101)
        self.assertEqual(set(conditional_policy[2]), {0., 1.})
        usage = next(line for line in captures['02_product2_usage_comparison'][0]
                     if line[0] == 'Conditional OE')
        self.assertEqual(usage[2].tolist(), [40.])
        profits = next(line for line in captures['03_profit_comparison'][0]
                       if line[0] == 'Conditional OE')
        self.assertEqual(profits[2].tolist(), [.3])
        outside = next(line for line in captures['04_outside_option_comparison'][0]
                       if line[0] == 'Conditional OE')
        self.assertEqual(outside[2].tolist(), [0.])


if __name__ == '__main__':
    unittest.main()
