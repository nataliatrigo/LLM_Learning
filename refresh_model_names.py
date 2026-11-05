"""Refresh public model names in all three experiments using saved data only."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from oblivious.oe_p0.code.run_experiment import load_bundle, now, write_json
from model_names import MODEL_LABELS, rename_display_text


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('oblivious/01_oe_distribution', 'oblivious/02_oe_p0', 'oblivious/06_equilibrium_comparison')


def protected_files():
    """Keep numerical bundles, source certificates and economic code immutable."""
    files = []
    for experiment in EXPERIMENTS:
        for path in (ROOT / experiment / 'data').rglob('*'):
            if not path.is_file() or 'raw' in path.parts:
                continue
            if (path.suffix == '.npz' or path.name == 'manifest.json'
                    or (path.suffix == '.json' and
                        ('arrays' in path.parts or 'checkpoints' in path.parts
                         or path.name in ('analysis.json', 'input_validation.json')))):
                files.append(path)
    for source in ('oblivious/02_oe_p0/code/primitives.py', 'oblivious/02_oe_p0/code/game.py',
                   'oblivious/02_oe_p0/code/policy.py', 'oblivious/02_oe_p0/code/core.py',
                   'oblivious/01_oe_distribution/code/model.py', 'oblivious/06_equilibrium_comparison/code/simulation.py',
                   'oblivious/06_equilibrium_comparison/code/analysis.py'):
        files.append(ROOT / source)
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def completed_runs(experiment):
    for path in sorted((ROOT / experiment / 'data').glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('status') == 'complete':
            yield path.parent, manifest


def refresh_model_experiment(experiment):
    if experiment == 'oblivious/01_oe_distribution':
        from oblivious.oe_distribution.code.run_experiment import make_figures
    else:
        from oblivious.oe_p0.code.run_experiment import make_figures
    directories = []
    for data, manifest in completed_runs(experiment):
        print(f'Redrawing {experiment}/{data.name} from saved policies and diagnostics', flush=True)
        cases = [load_bundle(data / 'arrays' / tag) for tag in manifest['completed_cases']]
        reports = [load_bundle(data / 'arrays' / tag) for tag in manifest['completed_diagnostics']]
        figures = ROOT / experiment / 'figures' / data.name
        if experiment == 'oblivious/02_oe_p0':
            reports = {(report['rho'], report['n'], int(tag.rsplit('_', 1)[1])): report
                       for tag, report in zip(manifest['completed_diagnostics'], reports)}
        make_figures(cases, reports, figures)
        directories.append(figures)
    return directories


def refresh_comparisons():
    from oblivious.oe_p0.code.run_experiment import case_tag
    from oblivious.equilibrium_comparison.code.cross_rho_figures import make_figures as cross_figures
    from oblivious.equilibrium_comparison.code.inputs import load_pair
    from oblivious.equilibrium_comparison.code.reporting import make_figures, plot_product_usage
    directories = []
    for data, manifest in completed_runs('oblivious/06_equilibrium_comparison'):
        config = manifest['configuration']
        figures = ROOT / 'oblivious/06_equilibrium_comparison/figures' / data.name
        print(f'Redrawing oblivious/06_equilibrium_comparison/{data.name} from saved simulation summaries', flush=True)
        if 'rhos' in config:
            results = [load_bundle(data / 'checkpoints' / case_tag(rho, config['grid']))
                       for rho in config['rhos']]
            cross_figures(results, config, figures)
            # The supplementary figures have their own presentation provenance.
            for module in ('plot_joint_linear', 'plot_joint_combined'):
                subprocess.run([sys.executable, '-B', '-m', 'oblivious.equilibrium_comparison.code.' + module,
                                '--name', data.name], cwd=ROOT, check=True)
        else:
            pair = load_pair(ROOT, rho=config['rho'], n=config['grid'],
                             oe_run=config['oe_run'], p0_run=config['p0_run'],
                             oe_candidate=config.get('oe_candidate'), p0_candidate=config.get('p0_candidate'))
            analysis = load_bundle(data / 'analysis')
            usage = analysis.pop('product_usage', None)
            make_figures(pair, analysis, figures)
            supplement_base = data / 'product2_usage_over_time'
            if supplement_base.with_suffix('.json').exists():
                supplement = load_bundle(supplement_base)
                usage = supplement['product_usage']
            else:
                supplement = {}
            if usage:
                plot_product_usage(usage, figures,
                    stationary_frequencies={key: analysis['predicted'][key]['product2_market'] for key in MODEL_LABELS},
                    early_periods=supplement.get('early_periods', 500), initial=supplement.get('initial', [.5, .5]))
        directories.append(figures)
    return directories


def refresh_report_text():
    def rename_json(value):
        if isinstance(value, dict):
            return {key: rename_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename_json(item) for item in value]
        return rename_display_text(value) if isinstance(value, str) else value

    changed = []
    for experiment in EXPERIMENTS:
        for path in sorted((ROOT / experiment / 'data').glob('*/*')):
            if (not path.is_file() or not (path.suffix in ('.txt', '.csv')
                    or path.name == 'diagnostic_summaries.json' or path.name.endswith('_summary.json'))):
                continue
            original = path.read_text()
            if path.suffix == '.json':
                decoded = json.loads(original)
                renamed = rename_json(decoded)
                updated = original if renamed == decoded else json.dumps(renamed, indent=2, allow_nan=False) + '\n'
            else:
                updated = rename_display_text(original, expand_short_oe=path.suffix == '.txt')
            updated = updated.replace('OE minus OE p0', 'OE distribution minus OE p0')
            if original != updated:
                path.write_text(updated)
                changed.append(str(path.relative_to(ROOT)))
    return changed


def main():
    before = protected_files()
    directories = []
    for experiment in ('oblivious/01_oe_distribution', 'oblivious/02_oe_p0'):
        directories.extend(refresh_model_experiment(experiment))
    directories.extend(refresh_comparisons())
    reports = refresh_report_text()
    after = protected_files()
    if before != after:
        raise RuntimeError('Protected numerical data or economic sources changed during the presentation refresh')
    figure_paths = sorted(path for folder in directories for path in folder.iterdir()
                          if path.suffix in ('.pdf', '.png'))
    write_json(ROOT / 'model_name_refresh.json', dict(
        refreshed_utc=now(), model_display_names=MODEL_LABELS,
        storage_directories_unchanged=list(EXPERIMENTS), no_simulations_or_equilibria_computed=True,
        protected_source_and_input_sha256=before,
        figure_files=[str(path.relative_to(ROOT)) for path in figure_paths],
        renamed_reports=reports,
        presentation_source_sha256={str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__), ROOT / 'model_names.py')}))
    print(f'Renamed {len(figure_paths)} figure files and {len(reports)} reports; '
          f'{len(before)} protected numerical files and sources unchanged.', flush=True)


if __name__ == '__main__':
    main()
