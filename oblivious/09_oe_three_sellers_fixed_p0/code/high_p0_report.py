"""Publish a separately audited high-p0 grid refinement without solving models.

The frozen expanded reporter supplies the case-specific metric checks and four
per-p0 figures. Plot 05 additionally carries the monopoly references requested
for the previous publication. All outputs belong to a new campaign/directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np

from . import expanded_report as expanded
from . import report as base


HIGH_P0S = (.7, .8, .9)
PAIR_FILE = 'high_p0_policy_equality.csv'
REFINEMENT_FILE = 'high_p0_policy_refinement.csv'
DIAGNOSTIC_FILE = 'high_p0_policy_diagnostics.json'


def action_intervals(grid, alpha):
    """Original-node runs with alpha >= .5, without interpolated thresholds."""
    selected = np.flatnonzero(np.asarray(alpha) >= .5)
    if not len(selected):
        return []
    groups = np.split(selected, np.flatnonzero(np.diff(selected) != 1)+1)
    return [[float(grid[indices[0]]), float(grid[indices[-1]])] for indices in groups]


def policy_diagnostics(rows, policies, p0s, rhos, high_p0s=HIGH_P0S):
    """Separate exact within-grid equality from interpolated refinement changes.

    Product mix uses each model's own sale weights, as stored in summary.csv.
    Between-grid alpha comparisons linearly interpolate the coarse policy only
    for diagnosis; they neither certify thresholds nor filter plotted outcomes.
    """
    outcomes = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows if r['verified']}
    groups = {}
    for row in policies:
        key = (row['p0'], row['rho'], row['n'], row['sellers'], row['closure'])
        groups.setdefault(key, []).append(row)
    vectors = {}
    for key, records in groups.items():
        records = sorted(records, key=lambda row: row['reputation'])
        vectors[key] = (np.asarray([r['reputation'] for r in records]),
                        np.asarray([r['product2_probability'] for r in records]))
    pairs, refinements = [], []
    for p0 in sorted(set(p0s) & set(high_p0s)):
        for rho in rhos:
            available = [{key[2] for key in outcomes if key[:2] == (p0, rho)
                          and key[3:] == (sellers, closure)}
                         for sellers, closure, *_ in base.DEFAULT_MODELS]
            common = sorted(set.intersection(*available))
            selected_grids = common[-2:]
            for n in selected_grids:
                for first, second in ((1, 2), (1, 3), (2, 3)):
                    first_closure = 'monopoly' if first == 1 else 'marginal'
                    key_a, key_b = (p0, rho, n, first, first_closure), (p0, rho, n, second, 'marginal')
                    if key_a not in outcomes or key_b not in outcomes:
                        continue
                    grid_a, alpha_a = vectors[key_a]
                    grid_b, alpha_b = vectors[key_b]
                    base.require(np.array_equal(grid_a, grid_b), 'Policy equality requires matching original nodes')
                    difference = float(np.max(np.abs(alpha_b-alpha_a)))
                    pairs.append(dict(p0=p0, rho=rho, n=n, first_sellers=first, second_sellers=second,
                        exact_policy_equal=bool(np.array_equal(alpha_a, alpha_b)), alpha_max_abs_difference=difference,
                        first_product2_conditional=outcomes[key_a]['product2_conditional'],
                        second_product2_conditional=outcomes[key_b]['product2_conditional'],
                        product2_difference_pp=100*(outcomes[key_b]['product2_conditional']-outcomes[key_a]['product2_conditional']),
                        first_action2_node_intervals=json.dumps(action_intervals(grid_a, alpha_a)),
                        second_action2_node_intervals=json.dumps(action_intervals(grid_b, alpha_b))))
            if len(selected_grids) < 2:
                continue
            coarse_n, fine_n = selected_grids
            for sellers, closure, *_ in base.DEFAULT_MODELS:
                coarse_key, fine_key = (p0, rho, coarse_n, sellers, closure), (p0, rho, fine_n, sellers, closure)
                if coarse_key not in outcomes or fine_key not in outcomes:
                    continue
                coarse_x, coarse_alpha = vectors[coarse_key]
                fine_x, fine_alpha = vectors[fine_key]
                refinements.append(dict(p0=p0, rho=rho, sellers=sellers, closure=closure,
                    coarse_n=coarse_n, fine_n=fine_n,
                    interpolated_alpha_max_abs_change=float(np.max(np.abs(fine_alpha-np.interp(fine_x, coarse_x, coarse_alpha)))),
                    coarse_product2_conditional=outcomes[coarse_key]['product2_conditional'],
                    fine_product2_conditional=outcomes[fine_key]['product2_conditional'],
                    product2_change_pp=100*(outcomes[fine_key]['product2_conditional']-outcomes[coarse_key]['product2_conditional']),
                    coarse_action2_node_intervals=json.dumps(action_intervals(coarse_x, coarse_alpha)),
                    fine_action2_node_intervals=json.dumps(action_intervals(fine_x, fine_alpha))))
    return dict(within_grid_pairs=pairs, refinement_by_model=refinements,
        scope='Exact policy comparisons on each high-p0 case finest two common certified original grids; separately, descriptive refinement changes',
        interval_definition='Inclusive original-node runs where Pr(product 2 | own sale,x) >= 0.5; endpoints bracket no continuous threshold',
        product_mix_definition='Each model uses its own stationary distribution conditional on its own sales',
        interpolation_note='Only the coarse policy is interpolated onto fine nodes for the descriptive alpha change; no pointwise stability claim')


def render_references(rows, figures, p0s, rhos, cases):
    """Same finite-grid cases as the expanded plot, plus horizontal N1 lines."""
    lookup = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows}
    points, omitted, colors = [], {}, {}
    with base.plt.rc_context({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 8}):
        fig, axes = expanded.facets(rhos, panel_height=3.2)
        for ax, rho in zip(axes.ravel(), rhos):
            omitted[str(rho)] = []
            for p0, color in zip(p0s, expanded.P0_COLORS):
                case = cases[expanded.case_key(p0, rho)]
                if not case['metrics']['product2_conditional']['passed']:
                    omitted[str(rho)].append(p0)
                    continue
                n = case['displayed_n']
                values = [100*lookup[p0, rho, n, sellers, closure]['product2_conditional']
                          for sellers, closure, *_ in base.DEFAULT_MODELS]
                ax.plot([1, 3], [values[0], values[0]], color=color, ls=(0, (4, 3)), lw=1.3, alpha=.8, zorder=2)
                ax.plot([1, 2, 3], values, color=color, marker='o', ms=5, lw=1.8, zorder=3)
                colors[p0] = color
                points.append(dict(p0=p0, rho=rho, n=n, validation_coarse_n=case['validation_coarse_n'],
                    product2_percent_by_sellers=values, monopoly_reference_percent=values[0]))
            excluded = ', '.join(f'{p0:g}' for p0 in omitted[str(rho)]) or 'none'
            ax.set(title=f'ρ = {rho:g} | Marginal OE\nOmitted p₀: {excluded}',
                xticks=[1, 2, 3], xlim=(.85, 3.15), ylim=(-2, 102), yticks=np.arange(0, 101, 20),
                xlabel='Number of sellers N', ylabel='Product 2 (% of own sales)')
            ax.tick_params(labelbottom=True)
            base.style(ax)
        for ax in axes.ravel()[len(rhos):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=colors[p0], lw=2, label=f'p₀ = {p0:g}') for p0 in p0s if p0 in colors]
        handles += [Line2D([], [], color='black', lw=1.8, marker='o', label='Outcome at N=1,2,3'),
                    Line2D([], [], color='black', lw=1.3, ls=(0, (4, 3)), label='Monopoly reference (N=1)')]
        fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Product 2 use as the number of sellers changes\n'
            'Dashed: monopoly at the same p₀ and ρ; grids and checks vary by case (see report).', fontsize=11)
        paths = base.save(fig, Path(figures), base.SELLER_COUNT_STEM)
    return paths, dict(plotted_cases=points, omitted_p0_by_rho=omitted,
        reference='Horizontal N=1 outcome for the same p0, rho and displayed grid')


def build_report(data=None, figures=None, *, check_only=False):
    data = Path(data if data is not None else base.HOME/'data/high_p0_refinement_41').resolve()
    figures = Path(figures if figures is not None else base.HOME/'figures/high_p0_refinement_41').resolve()
    if check_only:
        # The frozen diagnostic reader permits a live CSV snapshot, but never publishes it.
        return expanded.build_report(data, figures, check_only=True)
    base.require(not figures.exists(), 'Choose a new output directory to preserve published figures')
    base.require(not (data/'report_manifest.json').exists(), 'Campaign already has a published report')
    manifest, rows, policies, inputs, p0s, rhos, models = base.read_inputs(data)
    base.require(manifest.get('status') == 'complete' and not manifest.get('active_workers')
        and manifest.get('current_case') is None, 'Complete and inactive campaign is required before publication')
    base.require(tuple(models) == base.DEFAULT_MODELS, 'High-p0 refinement is marginal-only')
    base.require(len(p0s) <= len(expanded.P0_COLORS), 'Use at most six distinct p0 values')
    audit, audit_inputs = base.read_optional_audit(data, manifest)
    base.require(audit is not None, 'The complete independent audit is required before publication')
    inputs.update(audit_inputs)
    cases, comparisons = expanded.select_cases(rows, manifest, p0s, rhos)
    retention = expanded.metric_retention(cases, p0s, rhos)
    all_passed = all(case['metrics'][metric]['passed'] for case in cases.values() for metric in expanded.METRICS)
    any_passed = any(case['metrics'][metric]['passed'] for case in cases.values() for metric in expanded.METRICS)
    sources = {base.location(path): base.digest(path) for path in (Path(__file__), Path(base.__file__), Path(expanded.__file__))}
    diagnostics = policy_diagnostics(rows, policies, p0s, rhos)
    base.verify_inputs(inputs)
    base.verify_inputs(sources)
    figures.mkdir(parents=True)
    paths = expanded.render(rows, policies, figures, p0s, rhos, cases)
    # Both renderers write only into the new directory, before any publication hash exists.
    _, references = render_references(rows, figures, p0s, rhos, cases)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='complete_gridstable' if all_passed else 'partial_gridstable' if any_passed else 'no_gridstable_metrics',
        verified=True, diagnostic_only=False, source_campaign_status=manifest['status'],
        requested_p0s=p0s, requested_rhos=rhos, models=[dict(sellers=m[0], closure=m[1], label=m[2]) for m in models],
        case_grids=cases, metric_retention=retention, all_metrics_grid_stable=all_passed,
        probability_change_limit=.05, symmetric_relative_earnings_change_limit=.05,
        relative_change_definition='abs(fine-coarse)/max(abs(fine),abs(coarse)); zero if both zero',
        source_input_sha256=inputs, reporting_source_sha256=sources, independent_audit=audit,
        policy_measure='Original nodes on each case finest common certified grid; no pointwise grid-stability claim',
        product2_measure='Percentage among own sales', outside_measure='Whole-market outside probability',
        value_source='Actual game discounted value at initial reputations 0.5',
        scientific_scope='Independent finite-grid audit with separate metric checks; adjacent grids do not establish continuous convergence',
        primary_comparison_references=references)
    stability, retained = data/expanded.STABILITY_FILE, data/expanded.RETENTION_FILE
    pair_path, refinement_path, diagnostic_path = data/PAIR_FILE, data/REFINEMENT_FILE, data/DIAGNOSTIC_FILE
    base.write_csv(stability, comparisons)
    base.write_json(retained, retention)
    base.write_csv(pair_path, diagnostics['within_grid_pairs'])
    base.write_csv(refinement_path, diagnostics['refinement_by_model'])
    base.write_json(diagnostic_path, diagnostics)
    index = expanded.write_index(figures, data, report)
    relative_data = Path(os.path.relpath(data, figures)).as_posix()
    with index.open('a') as handle:
        handle.write('\nEl plot 05 incluye líneas horizontales discontinuas del monopolio para el mismo p₀ y ρ; '
            'los casos omitidos tampoco muestran referencia. Los p₀ altos usan su grilla refinada; '
            'la tabla anterior muestra la pareja de grillas realmente contrastada en cada caso.\n\n'
            f'Diagnóstico de políticas: [igualdad entre sellers]({relative_data}/{PAIR_FILE}) · '
            f'[cambios entre las dos grillas más finas]({relative_data}/{REFINEMENT_FILE}) · '
            f'[definiciones]({relative_data}/{DIAGNOSTIC_FILE}). '
            'La igualdad exacta se compara entre modelos en los mismos nodos; '
            'los intervalos muestran nodos con probabilidad de producto 2 ≥ 0.5 y no umbrales continuos.\n')
    report['figure_files'] = {base.location(path): base.digest(path) for path in paths}
    report['generated_data_sha256'] = {base.location(path): base.digest(path) for path in
        (stability, retained, pair_path, refinement_path, diagnostic_path, index)}
    report['primary_seller_count_comparison'] = {suffix: base.location(figures/(base.SELLER_COUNT_STEM+'.'+suffix)) for suffix in ('png', 'pdf')}
    base.verify_inputs(inputs)
    base.verify_inputs(sources)
    base.write_json(data/'report_manifest.json', report)
    base.write_json(figures/'report_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--figures', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = build_report(args.data, args.figures, check_only=args.check_only)
    print(json.dumps(result if args.check_only else {key: result[key] for key in
        ('status', 'verified', 'requested_p0s', 'requested_rhos', 'all_metrics_grid_stable')}, indent=2))


if __name__ == '__main__':
    main()
