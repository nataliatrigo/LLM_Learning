"""Audit consecutive grid refinements and render the fixed-p0 comparisons.

This reads the original and per-rho refinement checkpoints without changing
them. It writes an aggregate report in the separate refinement campaign.
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
from scipy.stats import wasserstein_distance

from oblivious.oe_p0.code.run_experiment import case_tag, now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle
from oblivious.oe_distribution_fixed_p0.code.reporting import (
    MODELS, grid_sensitivity_selection, make_figures,
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_cases(name, rhos):
    original = HOME/'data/p0_0p5'
    saved = json.loads((original/'manifest.json').read_text())
    inputs = {str((original/'manifest.json').relative_to(ROOT)): digest(original/'manifest.json')}
    cases = []
    manifests = {}
    for rho in rhos:
        for n in (21, 51):
            tag = case_tag(rho, n)
            cases.append(check_bundle(original, tag, saved))
            for suffix in ('.json', '.npz'):
                path = original/'arrays'/(tag+suffix)
                inputs[str(path.relative_to(ROOT))] = digest(path)
        directory = HOME/'data'/name/('rho_'+str(rho).replace('.', 'p'))
        path = directory/'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest.get('current_grid') is not None or manifest['status'] == 'running':
            raise ValueError(f'Refinement is still running: rho={rho}')
        for source, expected in manifest['numerical_sources'].items():
            if digest(ROOT/source) != expected:
                raise ValueError(f'Refinement numerical source changed: {source}')
        manifests[str(rho)] = manifest
        inputs[str(path.relative_to(ROOT))] = digest(path)
        for tag in manifest['completed']:
            case = check_bundle(directory, tag, manifest)
            if not case['competition']['verified'] or not case['monopoly']['verified']:
                raise ValueError(f'Unverified refinement checkpoint: {tag}')
            cases.append(case)
            for suffix in ('.json', '.npz'):
                path = directory/'arrays'/(tag+suffix)
                inputs[str(path.relative_to(ROOT))] = digest(path)
    return sorted(cases, key=lambda c:(c['rho'], c['n'])), manifests, inputs


def sale_weights(case, model):
    result = case[model]
    if model == 'monopoly':
        return result['sale_stationary']
    # Stationarity of the own-sale chain supplies an independent weighting.
    from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
    from oblivious.oe_p0.code.primitives import Parameters
    params = Parameters(**case['configuration']['parameters'])
    return purchase_chain_statistics(result['grid'], result['policy'], params)['sale_stationary']


def policy_and_distribution_changes(coarse, fine, model):
    """Compare policies and laws on common coordinates without changing them."""
    first, second = coarse[model], fine[model]
    x = np.linspace(0, 1, 10001)
    p_first = np.interp(x, first['grid'], first['policy'])
    p_second = np.interp(x, second['grid'], second['policy'])
    probability_difference = abs(p_second-p_first)
    fine_difference = abs(second['policy']-np.interp(second['grid'], first['grid'], first['policy']))
    mass_first = first['mu'] if model == 'monopoly' else first['mu_a']
    mass_second = second['mu'] if model == 'monopoly' else second['mu_a']

    def cdf(grid, mass):
        indices = np.searchsorted(grid, x, side='right')-1
        cumulative = np.cumsum(mass)
        return np.where(indices < 0, 0, cumulative[np.maximum(indices, 0)])

    cdf_difference = abs(cdf(first['grid'], mass_first)-cdf(second['grid'], mass_second))
    return dict(policy_integrated_l1=float(np.trapezoid(probability_difference, x)),
        policy_sup_difference=float(probability_difference.max()),
        policy_sale_weighted_l1=float(sale_weights(fine, model) @ fine_difference),
        marginal_cdf_sup=float(cdf_difference.max()),
        marginal_wasserstein1=float(wasserstein_distance(first['grid'], second['grid'],
                                                       mass_first, mass_second)),
        marginal_wasserstein1_approx=float(np.trapezoid(cdf_difference, x)),
        policy_comparison='Linear policy extension for this diagnostic only; no reoptimization',
        marginal_comparison='Step CDFs of numerical grid masses; 10001 common diagnostic coordinates')


def consecutive_comparisons(cases):
    records = []
    for rho in sorted({case['rho'] for case in cases}):
        chain = sorted([case for case in cases if case['rho'] == rho], key=lambda c:c['n'])
        for coarse, fine in zip(chain, chain[1:]):
            for model, *_ in MODELS:
                a = next(row for row in coarse['outcomes'] if row['model'] == model)
                b = next(row for row in fine['outcomes'] if row['model'] == model)
                probability = {metric: abs(b[metric]-a[metric]) for metric in
                               ('product2_conditional', 'product2_calendar', 'sale_rate')}
                relative = {metric: abs(b[metric]-a[metric])/max(abs(a[metric]), abs(b[metric]))
                            if a[metric] or b[metric] else 0. for metric in
                            ('profit_calendar', 'value_initial')}
                records.append(dict(rho=rho, model=model, coarse_n=coarse['n'], fine_n=fine['n'],
                    passed=all(value <= .05 for value in [*probability.values(), *relative.values()]),
                    max_probability_change_pp=100*max(probability.values()),
                    max_earnings_relative_change_pct=100*max(relative.values()),
                    **{key+'_difference':value for key,value in probability.items()},
                    **{key+'_relative_difference':value for key,value in relative.items()},
                    **policy_and_distribution_changes(coarse, fine, model)))
    return records


def convergence_figure(cases, directory):
    rhos = sorted({case['rho'] for case in cases})
    fig, axes = plt.subplots(len(rhos), 3, figsize=(13.8, 3.45*len(rhos)),
                             squeeze=False, layout='constrained')
    metrics = [('product2_conditional', 100, 'Product 2 (% of own sales)'),
               ('profit_calendar', 1, 'Stationary profit / seller / period'),
               ('value_initial', 1, 'Actual discounted value from x = 0.5')]
    for row, rho in enumerate(rhos):
        selected = [case for case in cases if case['rho'] == rho]
        for column, (metric, multiplier, label) in enumerate(metrics):
            ax = axes[row, column]
            for model, model_label, color, style, marker in MODELS:
                values = [(case['n'], next(item[metric] for item in case['outcomes']
                                          if item['model'] == model)) for case in selected]
                ax.plot([v[0] for v in values], [multiplier*v[1] for v in values],
                        color=color, ls=style, marker=marker, lw=1.7, ms=4,
                        markerfacecolor='white' if model == 'competition' else color,
                        label=model_label)
            ax.set(xlabel='Grid points per seller', ylabel=label, title=f'ρ = {rho:g}')
            ax.set_xticks(sorted(case['n'] for case in selected))
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Grid refinement at the same fixed outside option p₀ = 0.5\n'
                 'Every point is independently certified on its numerical grid', fontsize=13)
    paths = []
    for extension in ('png','pdf'):
        path = directory/f'04_grid_convergence.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def policy_refinement_figure(cases, directory):
    rhos = sorted({case['rho'] for case in cases})
    grids = sorted({case['n'] for case in cases if case['n'] >= 51})
    colors = plt.cm.viridis(np.linspace(.15,.9,len(grids)))
    fig, axes = plt.subplots(len(rhos), 2, figsize=(11,3.2*len(rhos)),
                             squeeze=False, layout='constrained', sharex=True, sharey=True)
    for row,rho in enumerate(rhos):
        for column,(model,label,*_) in enumerate(MODELS):
            ax=axes[row,column]
            for n,color in zip(grids,colors):
                case=next((case for case in cases if case['rho']==rho and case['n']==n),None)
                if case is None:
                    continue
                solution=case[model]
                ax.plot(solution['grid'],solution['policy'],color=color,lw=1.5,
                        label=f'n = {n}', ls='-' if n==max(grids) else '--')
            ax.set(title=f'{label}, ρ = {rho:g}',xlabel='Own reputation x',
                   ylabel='Pr(product 2 | own sale, x)',xlim=(0,1),ylim=(-.04,1.04))
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=len(grids),frameon=False)
    fig.suptitle('Policy sensitivity to the grid | fixed p₀ = 0.5\n'
                 'Lines connect stored grid decisions; mixtures are retained',fontsize=12)
    paths=[]
    for extension in ('png','pdf'):
        path=directory/f'05_policy_grid_refinement.{extension}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='p0_0p5_refinement')
    parser.add_argument('--rhos',type=float,nargs='+',default=[.95,.97])
    parser.add_argument('--grid',type=int,default=201)
    args=parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('','.','..'):
        parser.error('Use a single campaign name')
    cases,manifests,inputs=read_cases(args.name,args.rhos)
    data,figures=HOME/'data'/args.name,HOME/'figures'/args.name
    review=data/'independent_review.json'
    if review.exists():
        inputs[str(review.relative_to(ROOT))]=digest(review)
    figures.mkdir(parents=True,exist_ok=True)
    rows=[row for case in cases for row in case['outcomes']]
    policies=[row for case in cases for row in case['policy_rows']]
    comparisons=consecutive_comparisons(cases)
    selection=grid_sensitivity_selection(rows,p0=.5,n=args.grid)
    write_csv(data/'summary_all_grids.csv',rows)
    write_csv(data/'policies_all_grids.csv',policies)
    write_csv(data/'consecutive_grid_comparison.csv',comparisons)
    write_csv(data/'summary.csv',[row for row in rows if row['n']==args.grid])
    write_csv(data/'policies.csv',[row for row in policies if row['n']==args.grid])
    paths=[]
    if selection['included_rhos']:
        paths+=make_figures(rows,policies,figures,p0=.5,n=args.grid)
    paths+=convergence_figure(cases,figures)
    paths+=policy_refinement_figure(cases,figures)
    report=dict(created_utc=now(),p0=.5,finest_reported_grid=args.grid,
        source_input_sha256=inputs,grid_sensitivity_selection=selection,
        per_rho_status={rho:manifest['status'] for rho,manifest in manifests.items()},
        consecutive_grid_comparisons=comparisons,
        scientific_scope='Observed discretization stability; no continuous-state convergence, uniqueness or exhaustive equilibrium search proof',
        reporting_source_sha256={str(Path(__file__).relative_to(ROOT)):digest(__file__),
            str((HOME/'code/reporting.py').relative_to(ROOT)):digest(HOME/'code/reporting.py')},
        figure_files={path.name:digest(path) for path in paths})
    write_json(data/'report_manifest.json',report)
    write_json(figures/'figure_manifest.json',report)
    lines=['Refinamiento de grilla: modelo nuevo con p0=0.5',
           'Cada grilla se resolvió y certificó de nuevo; la interpolación solo propone un inicio.',
           'Se conservan los resultados originales y se comparan grillas consecutivas.',
           'Uso/ventas: límite 5 puntos porcentuales. Beneficio/valor: límite relativo 5% simétrico.',
           'La estabilidad observada no demuestra convergencia al continuo ni unicidad.',
           '',f"Rhos que pasan el filtro en n={args.grid}: {selection['included_rhos']}",
           '', 'rho | modelo | grillas | cambio máximo probabilidades (pp) | cambio máximo beneficio/valor (%)']
    for item in comparisons:
        lines.append(f"{item['rho']:g} | {item['model']} | {item['coarse_n']}→{item['fine_n']} | "
                     f"{item['max_probability_change_pp']:.5g} | {item['max_earnings_relative_change_pct']:.5g}")
    (data/'RESULTADOS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
