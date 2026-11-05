"""The finalizer waits for computation and refuses stale or incomplete output."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import high_p0_finish as finish


class HighP0FinishTests(unittest.TestCase):
    def fixture(self, root, status='complete'):
        home = root/'experiment'
        data = home/'data/high_p0_refinement_41'
        data.mkdir(parents=True)
        manifest = dict(status=status, completed=['checkpoint'], numerical_sources={}, input_sha256={},
                        active_workers=0, current_case=None)
        (data/'manifest.json').write_text(json.dumps(manifest))
        return home, data, manifest

    def patches(self, stack, root, home):
        for module, name, value in ((finish, 'ROOT', root), (finish, 'HOME', home)):
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(finish, 'source_snapshot', return_value={}))
        stack.enter_context(patch.object(finish, 'historical_snapshot', return_value={}))
        stack.enter_context(patch.object(finish.io, 'check_hashes'))
        stack.enter_context(patch.object(finish.frozen, 'check_scope', return_value=180))

    def test_waits_for_closed_campaign_and_never_starts_computation(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            home, data, manifest = self.fixture(root, 'running')
            self.patches(stack, root, home)

            def close(_):
                manifest['status'] = 'complete'
                (data/'manifest.json').write_text(json.dumps(manifest))

            sleep = stack.enter_context(patch.object(finish.time, 'sleep', side_effect=close))
            stage = stack.enter_context(patch.object(finish, 'run_stage'))
            stack.enter_context(patch.object(finish, 'validate', return_value=dict(
                status='partial_gridstable', all_metrics_grid_stable=False, figure_files={})))
            stack.enter_context(patch.object(finish.io, 'digest', return_value='closed_sha'))
            self.assertEqual(finish.finish(poll=1), 0)
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual([call.args[2] for call in stage.call_args_list], ['audit', 'report'])
            state = json.loads((data/'finalization_manifest.json').read_text())
            self.assertEqual(state['status'], 'complete_verified')
            self.assertFalse(state['compute_started_by_finalizer'])
            self.assertFalse(state['all_metrics_grid_stable'])

    def test_stopped_campaign_and_wrong_scope_cannot_publish(self):
        for status, count in (('stopped_by_user', 180), ('complete', 179)):
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                home, data, _ = self.fixture(root, status)
                self.patches(stack, root, home)
                stack.enter_context(patch.object(finish.frozen, 'check_scope', return_value=count))
                stage = stack.enter_context(patch.object(finish, 'run_stage'))
                self.assertEqual(finish.finish(), 1)
                stage.assert_not_called()
                self.assertFalse(json.loads((data/'finalization_manifest.json').read_text())['verified'])

    def test_historical_snapshot_excludes_new_outputs_but_includes_old_publications(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root/'experiment'
            old, current = home/'figures/old/plot.png', home/'figures/new/plot.png'
            for path in (old, current):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'figure')
            with patch.object(finish, 'ROOT', root), patch.object(finish, 'HOME', home), patch.object(finish.frozen, 'ROOT', root):
                snapshot = finish.historical_snapshot(current.parent)
            self.assertEqual(set(snapshot), {str(old.relative_to(root))})

    def test_monopoly_references_cannot_include_omitted_cases_or_wrong_grid(self):
        case = dict(displayed_n=41, validation_coarse_n=31)
        point = dict(p0=.8, rho=.6, n=41, validation_coarse_n=31,
                     product2_percent_by_sellers=[75., 74., 73.], monopoly_reference_percent=75.)
        key = finish.frozen.campaign.pair_key(.8, .6)
        report = dict(primary_comparison_references=dict(plotted_cases=[point]),
            metric_retention=dict(product2_conditional=dict(valid_cases=[key])), case_grids={key: case})
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data/'high_p0_policy_diagnostics.json').write_text(json.dumps(dict(
                within_grid_pairs=[{}]*72, refinement_by_model=[{}]*36)))
            with patch.object(finish.frozen, 'validate_report', return_value=report):
                self.assertEqual(finish.validate(data, data/'figures', {}, 180), report)
                point['n'] = 31
                with self.assertRaisesRegex(RuntimeError, 'different case grid'):
                    finish.validate(data, data/'figures', {}, 180)
                point['n'] = 41
                report['metric_retention']['product2_conditional']['valid_cases'] = []
                with self.assertRaisesRegex(RuntimeError, 'retained P2 cases'):
                    finish.validate(data, data/'figures', {}, 180)


if __name__ == '__main__':
    unittest.main()
