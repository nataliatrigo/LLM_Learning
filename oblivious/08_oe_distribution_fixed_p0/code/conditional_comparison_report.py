"""Add certified conditional OE to the existing fixed-p0 common-grid plots.

The report reads checkpoints, never solves or interpolates decisions, and keeps
the earlier two-model aggregate data unchanged. All requested rho values need
complete certified validation pairs for all three models before publication.
An explicitly requested n101→n151 conditional check never changes the n101
results that are plotted and exported.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from oblivious.oe_p0.code.run_experiment import case_tag, now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle
from oblivious.oe_distribution_fixed_p0.code.common_grid_report import (
    campaign_name, read_cases as read_marginal_cases,
)


RHOS = [.6, .7, .8, .9, .95, .97]
MODELS = (
    ('monopoly', 'Monopoly', '#0072B2', '-', 'o'),
    ('competition', 'Marginal OE', '#D55E00', '--', 's'),
    ('conditional_competition', 'Conditional OE', '#009E73', '-.', '^'),
)
PROBABILITY_METRICS = ('product2_conditional', 'product2_calendar', 'sale_rate', 'outside_rate')
EARNINGS_METRICS = ('profit_calendar', 'value_initial')
SHARED_CONFIGURATION = (
    'p0', 'tolerance', 'dp_tolerance', 'stationary_tolerance',
    'stationary_stability_tolerance', 'strategy', 'market', 'outside_event',
    'forgetting_clock', 'p0_role',
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def comparable_configuration(configuration):
    """The closure changes, but the economics and verification contract do not."""
    return dict(
        parameters={key: value for key, value in configuration['parameters'].items()
                    if key != 'rho'},
        **{key: configuration[key] for key in SHARED_CONFIGURATION})


def validate_conditional_case(case, rho, n, manifest, baseline):
    configuration = case['configuration']
    declared = manifest['configuration']
    if (case['rho'] != rho or case['n'] != n or case['p0'] != .5
            or configuration['n'] != n or configuration['parameters']['rho'] != rho
            or declared['rho'] != rho or declared['p0'] != .5
            or configuration['order'] != declared['order']
            or configuration['tolerance'] != declared['tol']
            or configuration['dp_tolerance'] != declared['dp_tol']
            or {key: value for key, value in configuration['parameters'].items() if key != 'rho'}
               != declared['parameters']):
        raise ValueError(f'Conditional checkpoint/configuration mismatch: rho={rho}, n={n}')
    if comparable_configuration(configuration) != baseline:
        raise ValueError(f'Conditional economics or verification contract differs: rho={rho}, n={n}')
    if 'conditional' not in configuration.get('closure', '').lower():
        raise ValueError(f'Conditional closure is not declared: rho={rho}, n={n}')
    if not case['demand_audit'].get('verified', False):
        raise ValueError(f'Conditional demand audit is unverified: rho={rho}, n={n}')
    solution = case['competition']
    grid = np.linspace(0, 1, n)
    policy = np.asarray(solution['policy'])
    if (not solution.get('verified', False) or policy.shape != (n,)
            or not np.isfinite(policy).all() or np.any((policy < 0) | (policy > 1))
            or not np.array_equal(solution['grid'], grid)):
        raise ValueError(f'Malformed/unverified conditional policy: rho={rho}, n={n}')
    actual_values = np.asarray(solution['actual_values'])
    if actual_values.shape != (2, n, n) or not np.isfinite(actual_values).all():
        raise ValueError(f'Conditional actual-game values are missing: rho={rho}, n={n}')
    outcomes = [row for row in case['outcomes'] if row['model'] == 'conditional_competition']
    policies = sorted((row for row in case['policy_rows'] if row['model'] == 'conditional_competition'),
                      key=lambda row: row['reputation'])
    if (len(outcomes) != 1 or not outcomes[0].get('verified', False) or len(policies) != n
            or any(row['rho'] != rho or row['n'] != n or row['p0'] != .5
                   for row in outcomes + policies)
            or not np.array_equal([row['reputation'] for row in policies], grid)
            or not np.array_equal([row['product2_probability'] for row in policies], policy)):
        raise ValueError(f'Conditional exports differ from the saved solution: rho={rho}, n={n}')
    actual_initial = float(actual_values[:, n//2, n//2].mean())
    if not np.isclose(outcomes[0]['value_initial'], actual_initial, rtol=0, atol=1e-12):
        raise ValueError(f'Conditional reported value is not the actual-game value: rho={rho}, n={n}')
    for metric in PROBABILITY_METRICS + EARNINGS_METRICS:
        if not np.isfinite(outcomes[0][metric]):
            raise ValueError(f'Conditional outcome is nonfinite: rho={rho}, n={n}, metric={metric}')


def read_conditional_cases(name, rhos, coarse_grid, grid, baseline, *, required_grids=None):
    from oblivious.oe_distribution_fixed_p0.code.conditional_run import source_fingerprints

    cases, statuses, inputs = [], {}, {}
    required_grids = list(required_grids if required_grids is not None else (coarse_grid, grid))
    expected_sources = source_fingerprints()
    for rho in rhos:
        directory = HOME/'data'/name/('rho_'+str(rho).replace('.', 'p'))
        path = directory/'manifest.json'
        state = dict(rho=rho, required_grids=required_grids,
                     available_verified_grids=[], source_directory=str(directory.relative_to(ROOT)))
        statuses[str(rho)] = state
        if not path.exists():
            state.update(status='incomplete', reason='Conditional manifest does not exist yet')
            continue
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
        inputs[str(path.relative_to(ROOT))] = sha256(encoded).hexdigest()
        state.update(source_status=manifest['status'], current_grid=manifest.get('current_grid'),
                     quadrature_order=manifest['configuration']['order'],
                     numerical_sources=manifest['numerical_sources'])
        if any(manifest['numerical_sources'].get(source) != expected
               for source, expected in expected_sources.items()):
            raise ValueError(f'Conditional numerical source fingerprints differ: {directory}')
        for source, expected in manifest['numerical_sources'].items():
            if digest(ROOT/source) != expected:
                raise ValueError(f'Conditional numerical source changed: {source}')
            inputs[source] = expected
        for source, expected in manifest.get('reference_input_sha256', {}).items():
            if digest(ROOT/source) != expected:
                raise ValueError(f'Conditional reference input changed: {source}')
            inputs[source] = expected
        for artifact, expected in manifest.get('diagnostic_artifact_sha256', {}).items():
            path = directory/artifact
            if digest(path) != expected:
                raise ValueError(f'Conditional diagnostic artifact changed: {path}')
            inputs[str(path.relative_to(ROOT))] = expected
        for n in required_grids:
            tag = case_tag(rho, n)
            if tag not in manifest['completed']:
                continue
            case = check_bundle(directory, tag, manifest)
            validate_conditional_case(case, rho, n, manifest, baseline)
            for source, expected in case.get('reference_input_sha256', {}).items():
                if (manifest.get('reference_input_sha256', {}).get(source) != expected
                        or digest(ROOT/source) != expected):
                    raise ValueError(f'Conditional case reference provenance differs: {source}')
                inputs[source] = expected
            for suffix in ('.json', '.npz'):
                artifact = directory/'arrays'/(tag+suffix)
                inputs[str(artifact.relative_to(ROOT))] = manifest['checkpoint_sha256'][artifact.name]
            cases.append(case)
            state['available_verified_grids'].append(n)
        state['status'] = ('complete' if state['available_verified_grids'] == required_grids
                           and manifest['status'] != 'running' and manifest.get('current_grid') is None
                           else 'incomplete')
    return cases, statuses, inputs


def sensitivity_selection(rows, rhos, coarse_grid, grid, *, validation_pairs=None):
    """Screen saved results while keeping the displayed grid independent.

    An explicit validation pair may compare the displayed n101 solution with
    n151. The failed historical n51→n101 comparison remains an audit record;
    it is never silently overwritten or relabeled as having passed.
    """
    validation_pairs = validation_pairs or {}
    lookup = {}
    for row in rows:
        key = (float(row['rho']), row['model'], int(row['n']))
        if key in lookup:
            raise ValueError(f'Duplicate saved outcome: {key}')
        lookup[key] = row
    def compare(rho, model, pair, *, selected):
        coarse_n, fine_n = pair
        coarse, fine = (lookup.get((rho, model, n)) for n in pair)
        if coarse is None or fine is None or not coarse['verified'] or not fine['verified']:
            return None, [dict(model=model, coarse_n=coarse_n, fine_n=fine_n,
                               reason='Missing verified coarse/fine comparison')]
        probability, relative, reasons = {}, {}, []
        for metric in PROBABILITY_METRICS + EARNINGS_METRICS:
            first, second = float(coarse[metric]), float(fine[metric])
            if not np.isfinite([first, second]).all():
                reasons.append(dict(model=model, metric=metric, reason='Nonfinite outcome'))
                continue
            difference = abs(second-first)
            if metric in PROBABILITY_METRICS:
                probability[metric] = difference
                change = difference
            else:
                denominator = max(abs(first), abs(second))
                change = difference/denominator if denominator else 0.
                relative[metric] = change
            if change > .05 + 1e-12:
                reasons.append(dict(model=model, metric=metric, change=change, limit=.05))
        passed = (len(probability) == len(PROBABILITY_METRICS) and len(relative) == len(EARNINGS_METRICS)
                  and all(value <= .05 + 1e-12 for value in [*probability.values(), *relative.values()]))
        record = dict(rho=rho, model=model, coarse_n=coarse_n, fine_n=fine_n,
            displayed_n=grid, used_for_selection=selected,
            comparison_role='selected_validation' if selected else 'historical_diagnostic',
            passed=passed, max_probability_change_pp=100*max(probability.values(), default=np.nan),
            max_earnings_relative_change_pct=100*max(relative.values(), default=np.nan),
            **{metric+'_absolute_change': value for metric, value in probability.items()},
            **{metric+'_relative_change': value for metric, value in relative.items()})
        return record, reasons

    comparisons, excluded, included, selected_pairs = [], {}, [], []
    for rho in rhos:
        reasons = []
        for model, *_ in MODELS:
            pair = validation_pairs.get((rho, model), (coarse_grid, grid))
            if grid not in pair:
                raise ValueError(f'Validation must include the displayed grid: rho={rho}, model={model}')
            selected_pairs.append(dict(rho=rho, model=model, displayed_n=grid,
                                       coarse_n=pair[0], fine_n=pair[1],
                                       explicit_finer_check=pair != (coarse_grid, grid)))
            if pair != (coarse_grid, grid):
                historical, _ = compare(rho, model, (coarse_grid, grid), selected=False)
                if historical is not None:
                    comparisons.append(historical)
            selected, failed = compare(rho, model, pair, selected=True)
            if selected is not None:
                comparisons.append(selected)
            reasons.extend(failed)
        if reasons:
            excluded[str(rho)] = reasons
        else:
            included.append(rho)
    return dict(included_rhos=included, excluded_rhos=excluded,
        requested_rhos=rhos, displayed_grid=grid, default_validation_pair=[coarse_grid, grid],
        selected_validation_pairs=selected_pairs,
        probability_change_limit=.05, relative_earnings_change_limit=.05,
        probability_metrics=list(PROBABILITY_METRICS), relative_metrics=list(EARNINGS_METRICS),
        comparisons=comparisons,
        interpretation=('Saved verified comparison pairs, explicitly including the displayed grid; '
                        'finer validation does not replace plotted results; not a convergence proof'))


def coincidence_diagnostics(rows, policies, rhos, grid):
    outcomes = {(float(row['rho']), row['model']): row for row in rows if row['n'] == grid}
    policy_lookup = {}
    for rho in rhos:
        for model, *_ in MODELS:
            ordered = sorted((row for row in policies if row['n'] == grid and row['rho'] == rho
                              and row['model'] == model), key=lambda row: row['reputation'])
            policy_lookup[rho, model] = np.asarray([row['product2_probability'] for row in ordered])
    pairs = []
    for rho in rhos:
        for i, (first, *_unused) in enumerate(MODELS):
            for second, *_unused in MODELS[i+1:]:
                a, b = policy_lookup[rho, first], policy_lookup[rho, second]
                differences = {metric: abs(outcomes[rho, first][metric]-outcomes[rho, second][metric])
                               for metric in PROBABILITY_METRICS + EARNINGS_METRICS}
                pairs.append(dict(rho=rho, first_model=first, second_model=second,
                    policy_exactly_equal=bool(np.array_equal(a, b)),
                    policy_max_absolute_difference=float(np.max(abs(a-b))),
                    outcome_max_absolute_difference=max(differences.values()),
                    outcomes_equal_at_display_precision=all(value <= 5e-12 for value in differences.values()),
                    outcome_absolute_differences=differences))
    return dict(absolute_display_tolerance=5e-12, pairs=pairs)


def style(ax):
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def legend(fig, axes):
    handles = {}
    for ax in np.asarray(axes).ravel():
        artists, labels = ax.get_legend_handles_labels()
        for label, artist in zip(labels, artists):
            handles.setdefault(label, artist)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center',
               ncol=min(4, len(handles)), frameon=False)


def save(fig, directory, stem):
    paths = []
    for suffix in ('png', 'pdf'):
        path = directory/f'{stem}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def displayed_records(rows, policies, n):
    """Select the common plotted/exported resolution, independently of checks."""
    return ([row for row in rows if row['n'] == n],
            [row for row in policies if row['n'] == n])


def validate_market_outside_rows(rows):
    """Distinguish an outside purchase from a purchase from the other seller."""
    for row in rows:
        sellers = 1 if row['model'] == 'monopoly' else 2
        outside, sales = float(row['outside_rate']), float(row['sale_rate'])
        if (not np.isfinite([outside, sales]).all() or not 0 <= outside <= 1
                or not 0 <= sales <= 1 or abs(outside+sellers*sales-1) > 1e-9):
            raise ValueError(f"Outside rate is not a whole-market probability: {row['model']}, rho={row['rho']}")


def render_figures(rows, policies, directory, rhos, *, p0=.5, n=101, validation_note=None):
    """Render the four requested figures, without resampling policies."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fine, fine_policies = displayed_records(rows, policies, n)
    validate_market_outside_rows(fine)
    lookup = {(row['rho'], row['model']): row for row in fine}
    coincidence = coincidence_diagnostics(rows, policies, rhos, n)
    paths = []
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 9}):
        fig, axes = plt.subplots((len(rhos)+1)//2, 2, figsize=(11, 2.8*((len(rhos)+1)//2)),
                                 sharex=True, sharey=True, squeeze=False, layout='constrained')
        for ax, rho in zip(axes.ravel(), rhos):
            for model, label, color, linestyle, marker in MODELS:
                selected = sorted((row for row in fine_policies if row['rho'] == rho
                                   and row['model'] == model), key=lambda row: row['reputation'])
                x = np.asarray([row['reputation'] for row in selected])
                alpha = np.asarray([row['product2_probability'] for row in selected])
                ax.plot(x, alpha, color=color, ls=linestyle, lw=1.5 if model == 'conditional_competition' else 1.9,
                        marker=marker if model == 'conditional_competition' else None,
                        markevery=max(1, n//10), ms=4.5, markerfacecolor='none', label=label)
                mixed = (alpha > 1e-9) & (alpha < 1-1e-9)
                if mixed.any():
                    ax.scatter(x[mixed], alpha[mixed], marker='D', s=28, facecolors='white',
                               edgecolors=color, linewidths=1.2, zorder=5, label='Mixed node')
            identical = [item for item in coincidence['pairs'] if item['rho'] == rho
                         and item['first_model'] == 'competition'
                         and item['second_model'] == 'conditional_competition'
                         and item['policy_exactly_equal']]
            if identical:
                ax.text(.98, .13, 'Marginal = conditional policy', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
            ax.set(title=f'ρ = {rho:g}', xlabel='Own reputation x',
                   ylabel='Pr(product 2 | own sale, x)', xlim=(0, 1), ylim=(-.04, 1.04))
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_yticks([0, .5, 1])
            style(ax)
        for ax in axes.ravel()[len(rhos):]:
            ax.set_visible(False)
        legend(fig, axes)
        fig.suptitle(f'Product policies | fixed outside benchmark p₀ = {p0:g} | n = {n}\n'
                     'Verified grid solutions; diamonds mark mixtures; lines connect saved grid nodes.'
                     + ('\n'+validation_note if validation_note else ''), fontsize=12)
        paths += save(fig, directory, '01_policy_comparison')
        specifications = [
            ('02_product2_usage_comparison', 'Stationary product 2 use', [
                ('product2_conditional', 100, 'Product 2 (% of own sales)', 'Product mix when the seller sells')]),
            ('03_profit_comparison', 'Seller profits', [
                ('profit_calendar', 1, 'Profit / seller / calendar period', 'Stationary profit, including waiting'),
                ('value_initial', 1, 'Expected total discounted profit / seller', 'Actual value from initial reputation x = 0.5')]),
            ('04_outside_option_comparison', 'Outside option choice', [
                ('outside_rate', 100, 'Outside option (% of periods)', 'Buyer chooses outside: no seller makes a sale')]),
        ]
        for stem, title, metrics in specifications:
            width = 7.5 if len(metrics) == 1 else 11
            fig, axes = plt.subplots(1, len(metrics), figsize=(width, 4.8),
                                     squeeze=False, layout='constrained')
            for ax, (metric, scale, ylabel, subtitle) in zip(axes.ravel(), metrics):
                positions = np.arange(len(rhos))
                for model, label, color, linestyle, marker in MODELS:
                    values = scale*np.asarray([lookup[rho, model][metric] for rho in rhos])
                    ax.plot(positions, values, color=color, ls=linestyle, marker=marker,
                            ms=6 if model == 'conditional_competition' else 5, lw=1.7,
                            markerfacecolor=color if model == 'monopoly' else 'none', label=label)
                ax.set(ylabel=ylabel, title=subtitle, xlabel='Persistence ρ (equally spaced cases)')
                ax.set_xticks(positions, [f'{rho:g}' for rho in rhos])
                if len(rhos) > 1:
                    ax.set_xlim(-.2, len(rhos)-.8)
                if metric == 'outside_rate':
                    ax.set_ylim(bottom=0)
                style(ax)
            legend(fig, axes)
            all_overlap = all(abs(lookup[rho, 'competition'][metric]
                                  - lookup[rho, 'conditional_competition'][metric]) <= 5e-12
                              for rho in rhos for metric, *_ in metrics)
            note = '\nMarginal and conditional OE overlap at plotted precision.' if all_overlap else ''
            fig.suptitle(f'{title}\nFixed outside benchmark p₀ = {p0:g} | n = {n} | verified grid solutions'
                         + note + ('\n'+validation_note if validation_note else ''), fontsize=12)
            paths += save(fig, directory, stem)
    return paths, coincidence


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='p0_0p5_conditional_fixed')
    parser.add_argument('--marginal-name', default='p0_0p5_common_grid')
    parser.add_argument('--refinement-name', default='p0_0p5_refinement')
    parser.add_argument('--conditional-refinement-name', default=None, metavar='CAMPAIGN',
                        help='Explicitly validate Conditional rho=.97 n101 against certified n151 in this campaign')
    parser.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    parser.add_argument('--grid', type=int, default=101)
    parser.add_argument('--coarse-grid', type=int, default=51)
    parser.add_argument('--audit', action='append', default=[], metavar='PATH',
                        help='Independent audit artifact inside the repository; may be repeated')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        for name in (args.name, args.marginal_name, args.refinement_name):
            campaign_name(name)
        if args.conditional_refinement_name is not None:
            campaign_name(args.conditional_refinement_name)
        if (sorted(args.rhos) != RHOS or args.coarse_grid != 51 or args.grid != 101):
            raise ValueError('This comparison requires the six retained rho values and common grids51/101')
    except ValueError as exc:
        parser.error(str(exc))
    args.rhos = sorted(args.rhos)
    return args


def main():
    args = arguments()
    marginal, marginal_status, inputs, scientific = read_marginal_cases(
        args.rhos, args.coarse_grid, args.grid, args.marginal_name, args.refinement_name, {})
    if scientific is None:
        raise SystemExit('No certified marginal sources are available; no figures published')
    for state in marginal_status.values():
        inputs.update(state.get('numerical_sources', {}))
    baseline = comparable_configuration(scientific)
    conditional, conditional_status, conditional_inputs = read_conditional_cases(
        args.name, args.rhos, args.coarse_grid, args.grid, baseline)
    inputs.update(conditional_inputs)
    conditional_refinement_status, validation_pairs = {}, {}
    validation_note = None
    if args.conditional_refinement_name is not None:
        refined, conditional_refinement_status, refinement_inputs = read_conditional_cases(
            args.conditional_refinement_name, [.97], args.grid, 151, baseline, required_grids=[151])
        conditional.extend(refined)
        inputs.update(refinement_inputs)
        validation_pairs[.97, 'conditional_competition'] = (101, 151)
        validation_note = 'Conditional OE at ρ = 0.97: n = 101 checked against n = 151.'
    audits = {}
    audit_paths = [Path(value) for value in args.audit]
    default_audit = HOME/'data'/args.name/'independent_review.json'
    if not audit_paths and default_audit.exists():
        audit_paths = [default_audit]
    for path in audit_paths:
        path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        source = str(path.relative_to(ROOT))
        audits[source] = digest(path)
    inputs.update(audits)
    rows = [row for case in marginal for row in case['outcomes']]
    policies = [row for case in marginal for row in case['policy_rows']]
    rows += [row for case in conditional for row in case['outcomes'] if row['model'] == 'conditional_competition']
    policies += [row for case in conditional for row in case['policy_rows'] if row['model'] == 'conditional_competition']
    validate_market_outside_rows(rows)
    selection = sensitivity_selection(rows, args.rhos, args.coarse_grid, args.grid,
                                      validation_pairs=validation_pairs)
    incomplete = dict(marginal={rho: state for rho, state in marginal_status.items() if state['status'] != 'complete'},
                      conditional={rho: state for rho, state in conditional_status.items() if state['status'] != 'complete'},
                      conditional_refinement={rho: state for rho, state in conditional_refinement_status.items()
                                              if state['status'] != 'complete'})
    ready = not any(incomplete.values()) and selection['included_rhos'] == args.rhos
    overview = dict(status='ready' if ready else 'incomplete_or_grid_sensitive',
                    incomplete_sources=incomplete, grid_sensitivity_selection=selection)
    if args.check_only:
        print(json.dumps(overview, indent=2))
        return
    if not ready:
        raise SystemExit('All six rho values need complete, stable, certified three-model comparisons; '
                         'no figures published:\n'+json.dumps(overview, indent=2))
    for source, expected in inputs.items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Source changed while preparing report; retry: {source}')
    data = HOME/'data'/args.name
    figures = HOME/'figures'/args.marginal_name
    data.mkdir(parents=True, exist_ok=True)
    paths, coincidence = render_figures(rows, policies, figures, args.rhos, n=args.grid,
                                       validation_note=validation_note)
    displayed_rows, displayed_policies = displayed_records(rows, policies, args.grid)
    write_csv(data/'comparison_summary_all_grids.csv', rows)
    write_csv(data/'comparison_summary.csv', displayed_rows)
    write_csv(data/'comparison_policies.csv', displayed_policies)
    write_csv(data/'comparison_grid_stability.csv', selection['comparisons'])
    write_json(data/'comparison_grid_sensitivity_selection.json', selection)
    write_json(data/'comparison_coincidence.json', coincidence)
    report = dict(created_utc=now(), status='complete', p0=.5, common_grid=args.grid,
        default_validation_pair=[args.coarse_grid, args.grid], requested_rhos=args.rhos,
        selected_validation_pairs=selection['selected_validation_pairs'],
        finer_validation_campaign=args.conditional_refinement_name,
        finer_validation_reason=('Explicit forward refinement check for Conditional rho=.97; '
                                 'failed historical 51→101 retained; every displayed/exported curve stays n101'
                                 if args.conditional_refinement_name else None),
        models=[dict(model=model, label=label) for model, label, *_ in MODELS],
        shared_scientific_configuration=baseline, source_input_sha256=inputs,
        independent_audit_sha256=audits,
        belief_closures=dict(marginal=scientific['closure'],
                             conditional=conditional[0]['configuration']['closure']),
        conditional_information=conditional[0]['configuration'].get('conditional_event'),
        quadrature_scope=('Same independently audited demand formula; historical marginal n51 keeps its '
                          'original order; each conditional coarse/fine pair shares its declared order'),
        per_rho_provenance=dict(marginal=marginal_status, conditional=conditional_status,
                                conditional_refinement=conditional_refinement_status),
        grid_sensitivity_selection=selection, coincidence_diagnostics=coincidence,
        value_source='Actual two-seller environment; not the perceived oblivious value',
        outside_option_measure='Whole-market share of calendar periods: monopoly 1-sale_rate; symmetric competition 1-2*sale_rate',
        policy_source='Unmodified stored grid policies; lines only connect saved reputation nodes',
        scientific_scope='Observed discretization stability; no continuous convergence or uniqueness proof',
        reporting_source_sha256={str(Path(__file__).relative_to(ROOT)): digest(__file__),
            str((HOME/'code/common_grid_report.py').relative_to(ROOT)): digest(HOME/'code/common_grid_report.py')},
        figure_files={path.name: digest(path) for path in paths})
    write_json(data/'comparison_report_manifest.json', report)
    write_json(figures/'figure_manifest.json', report)
    write_json(figures/'grid_sensitivity_selection.json', selection)
    lines = ['Comparación de monopolio, OE marginal y OE condicional con p0=0.5.',
        'Seis rhos retenidos: '+', '.join(f'{rho:g}' for rho in args.rhos)+'.',
        'Todas las curvas corresponden a soluciones certificadas sobre 101 puntos.',
        'El filtro admite 5 pp en uso/ventas y 5% relativo simétrico en beneficio/valor.',
        ('La comparación usa 51→101 salvo Conditional rho=.97, validado con 101→151; '
         'se conserva el 51→101 fallido como diagnóstico.' if args.conditional_refinement_name
         else 'La comparación usa 51→101 para todos los modelos y rhos.'),
        'Los valores iniciales competitivos provienen del entorno real, no del valor percibido OE.',
        'La opción externa se mide para todo el mercado; una compra al rival no cuenta como outside.',
        'El monopolio y las curvas marginales reutilizan sus checkpoints sin modificarlos.',
        'La estabilidad observada no prueba convergencia al continuo ni unicidad.', '',
        'rho | modelo | uso P2/venta | ventas/calendario | beneficio/calendario | valor inicial | outside/calendario']
    for row in sorted((row for row in rows if row['n'] == args.grid), key=lambda row:(row['rho'],row['model'])):
        lines.append(f"{row['rho']:g} | {row['model']} | {row['product2_conditional']:.8g} | "
                     f"{row['sale_rate']:.8g} | {row['profit_calendar']:.8g} | {row['value_initial']:.8g} | {row['outside_rate']:.8g}")
    (data/'COMPARACION.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
