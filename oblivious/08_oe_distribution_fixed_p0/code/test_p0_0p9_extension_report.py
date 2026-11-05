"""Publication gates for the separately requested p0=.9 comparison."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oblivious.oe_distribution_fixed_p0.code.p0_0p9_extension_report import (
    digest, exclusion_lines, read_prior_comparison, read_required_audit,
)


MODULE = 'oblivious.oe_distribution_fixed_p0.code.p0_0p9_extension_report'
READER = 'oblivious.oe_distribution_fixed_p0.code.multi_p0_report'


class P0PointNineExtensionReportTests(unittest.TestCase):
    def environment(self, root):
        stack = ExitStack()
        stack.enter_context(patch(MODULE+'.ROOT', root))
        stack.enter_context(patch(MODULE+'.HOME', root))
        stack.enter_context(patch(READER+'.ROOT', root))
        return stack

    def review(self, root):
        for name in ('engine.py', 'wrapper.py', 'case.json'):
            (root/name).write_text(name)
        review = dict(verified=True, p0s=[.9], rhos=[.6, .7, .8],
            publication_requirement_base_checkpoints_complete=True,
            audit_source='engine.py', audit_source_sha256=digest(root/'engine.py'),
            audit_entrypoint='wrapper.py', audit_entrypoint_sha256=digest(root/'wrapper.py'),
            audited_input_sha256={'case.json': digest(root/'case.json')},
            audited_bundle_count=12, audited_file_count=24, audited_input_count=1,
            residual_maxima={'value': 1e-12})
        (root/'independent_review.json').write_text(json.dumps(review))
        return review

    def test_dedicated_review_must_cover_all_current_inputs_and_base_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.environment(root):
                with self.assertRaisesRegex(ValueError, 'required before publication'):
                    read_required_audit(root, [.6], {})
                review = self.review(root)
                required = review['audited_input_sha256']
                summary, inputs = read_required_audit(root, [.6, .7, .8], required)
                self.assertTrue(summary['verified'])
                self.assertEqual(inputs['wrapper.py'], digest(root/'wrapper.py'))
                with self.assertRaisesRegex(ValueError, 'every current checkpoint'):
                    read_required_audit(root, [.6], {**required, 'missing.json': 'unknown'})
                review['publication_requirement_base_checkpoints_complete'] = False
                (root/'independent_review.json').write_text(json.dumps(review))
                with self.assertRaisesRegex(ValueError, 'base grids/rhos'):
                    read_required_audit(root, [.6], required)

    def test_audit_entrypoint_and_current_checkpoint_cannot_change_after_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = self.review(root)
            with self.environment(root):
                (root/'wrapper.py').write_text('changed wrapper')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: wrapper.py'):
                    read_required_audit(root, [.6], review['audited_input_sha256'])
                review = self.review(root)
                (root/'case.json').write_text('changed checkpoint')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: case.json'):
                    read_required_audit(root, [.6], review['audited_input_sha256'])

    def test_adding_rho_point_nine_requires_its_own_current_independent_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = self.review(root)
            with self.environment(root):
                requested = [.6, .7, .8, .9]
                with self.assertRaisesRegex(ValueError, 'base grids/rhos'):
                    read_required_audit(root, requested, review['audited_input_sha256'])
                review['rhos'] = requested
                (root/'independent_review.json').write_text(json.dumps(review))
                summary, _ = read_required_audit(root, requested, review['audited_input_sha256'])
                self.assertEqual(summary['rhos'], requested)

    def test_prior_comparison_is_checked_and_its_unhashed_sidecars_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, figures = root/'data/p0_sweep', root/'figures/p0_sweep/p0_0p8'
            data.mkdir(parents=True)
            figures.mkdir(parents=True)
            for name in ('input.json', 'source.py', 'summary.csv'):
                (root/name).write_text(name)
            figure = figures/'01_policy_comparison.png'
            figure.write_bytes(b'prior figure')
            sidecar = data/'comparison_coincidence.json'
            sidecar.write_text('{}')
            (data/'COMPARACION.txt').write_text('prior text')
            prior = dict(requested_p0s=[.1, .3, .5, .7, .8],
                common_retained_rhos=[.6, .7, .8], displayed_n=101,
                source_input_sha256={'input.json': digest(root/'input.json')},
                reporting_source_sha256={'source.py': digest(root/'source.py')},
                figure_files={str(figure.relative_to(root)): digest(figure)},
                generated_data_sha256={'summary.csv': digest(root/'summary.csv')})
            path = data/'comparison_report_manifest.json'
            path.write_text(json.dumps(prior))
            with self.environment(root):
                summary, inputs, protected = read_prior_comparison()
                self.assertEqual(summary['p0s'], [.1, .3, .5, .7, .8])
                self.assertEqual(protected[str(sidecar.relative_to(root))], digest(sidecar))
                self.assertEqual(inputs[str(path.relative_to(root))], digest(path))
                figure.write_bytes(b'changed prior figure')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs'):
                    read_prior_comparison()

    def test_exclusion_reasons_distinguish_percentage_points_and_relative_changes(self):
        lines = exclusion_lines(dict(excluded_rhos={'0.8': [
            dict(model='monopoly', metric='outside_rate', change=.06),
            dict(model='competition', metric='profit_calendar', change=.07),
            dict(model='conditional_competition', reason='Missing verified coarse/fine comparison')]}))
        self.assertIn('6 puntos porcentuales', lines[0])
        self.assertIn('7 % relativo', lines[1])
        self.assertIn('Missing verified coarse/fine comparison', lines[2])


if __name__ == '__main__':
    unittest.main()
