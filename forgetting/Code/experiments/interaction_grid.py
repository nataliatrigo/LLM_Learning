"""Coarse parameter grid of individual frequencies in Seller-A interaction time."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

from forgetting.Code.experiments import run_experiments as experiments
from forgetting.Code.experiments.interaction_paths import simulate, distribution
from forgetting.Code.model import Primitives
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify(stats: dict, thresholds: dict) -> str:
    zero = thresholds['zero_frequency']
    if stats['mean'] <= zero and stats['approximately_zero_share'] >= thresholds['nearly_all_share']:
        return 'Extinction-like'
    if (stats['p05'] > zero and stats['std'] <= thresholds['max_std']
            and stats['mean_absolute_difference'] <= thresholds['max_mean_absolute_difference']):
        return 'Stable positive frequency'
    return 'Unstable or heterogeneous'


def summarize(choices: np.ndarray, burn_in: int, thresholds: dict) -> tuple[pd.DataFrame, dict]:
    total, paths = choices.shape
    if not 0 <= burn_in < total or (total-burn_in) % 2:
        raise ValueError('Post-burn-in history must split into two nonempty equal windows')
    split = burn_in+(total-burn_in)//2
    first_count = choices[burn_in:split].sum(axis=0)
    second_count = choices[split:].sum(axis=0)
    first = first_count/(split-burn_in)
    second = second_count/(total-split)
    final = (first_count+second_count)/(total-burn_in)
    delta = abs(first-second)
    trajectory = pd.DataFrame({'trajectory': np.arange(1,paths+1),
        'product2_count_after_burn_in': first_count+second_count,
        'post_burn_in_interactions': total-burn_in, 'frequency': final,
        'first_window_product2_count': first_count, 'second_window_product2_count': second_count,
        'first_window_frequency': first, 'second_window_frequency': second,
        'absolute_window_difference': delta})
    stats = {**distribution(final), 'mean_absolute_difference': float(delta.mean()),
        'mean_first_window_frequency': float(first.mean()),
        'mean_second_window_frequency': float(second.mean()),
        'zero_count_share': float(np.mean(final == 0)),
        'approximately_zero_share': float(np.mean(final <= thresholds['zero_frequency'])),
        'first_start': burn_in+1, 'first_end': split, 'second_start': split+1,
        'second_end': total, 'paths': paths}
    stats['classification'] = classify(stats, thresholds)
    return trajectory, stats


def worker(config: dict, p0: float, rho: float) -> tuple[pd.DataFrame, dict]:
    # Parameter-coded seed is invariant to worker scheduling or grid ordering.
    seed_words = [config['experiment']['seed'], int(round(p0*1_000_000)),
                  int(round(rho*1_000_000))]
    seed = int(np.random.SeedSequence(seed_words).generate_state(1, dtype=np.uint64)[0])
    primitives = replace(Primitives.from_mapping(config['seller']), p0=p0)
    result = simulate(primitives, rho, paths=config['experiment']['paths'],
        interactions=config['experiment']['interactions'], seed=seed,
        trace_window=(1,1), **config['solver'])
    trajectories, summary = summarize(result['choices'], config['experiment']['burn_in'],
                                       config['classification'])
    trajectories.insert(0, 'rho', rho)
    trajectories.insert(0, 'p0', p0)
    trajectories['seed'] = str(seed)
    summary.update(p0=p0, rho=rho, seed=str(seed),
                   terminal_observation=result['terminal_observation'])
    print(f'[pair complete] p0={p0:g}, rho={rho:g}, mean={summary["mean"]:.6f}', flush=True)
    return trajectories, summary


def mature_advantages(config: dict, output: Path) -> pd.DataFrame:
    """Reuse matching saved M values; never interpolate its sign between rho nodes."""
    study_config = {'solver': config['mature_solver']}
    cached = pd.DataFrame()
    source = experiments.ROOT/'outputs/policy_and_state_experiments/data/rho_p0_phase_c2_0.65.csv'
    manifest_path = experiments.ROOT/'outputs/policy_and_state_experiments/manifest.json'
    if source.is_file() and manifest_path.is_file():
        saved = experiments.read_config(manifest_path)['config']
        seller = {**saved['seller'], 'c2': .65,
                  'gamma': saved['rho_p0_comparative_statics']['gamma']}
        match = all(seller[k] == v for k,v in config['seller'].items() if k != 'p0')
        match = match and .65 in saved.get('investment_window_sweeps', {}).get('c2', [])
        if match and saved['solver'] == config['mature_solver']:
            cached = pd.read_csv(source)
    rows = []
    for p0 in config['p0_values']:
        for rho in config['rho_values']:
            found = pd.DataFrame()
            if not cached.empty:
                found = cached[np.isclose(cached.p0,p0,atol=1e-14,rtol=0)
                               & np.isclose(cached.rho,rho,atol=1e-14,rtol=0)]
            if len(found) == 1:
                row = found.iloc[0].to_dict()
                row['source'] = str(source.relative_to(experiments.ROOT.parent))
                row['reused'] = True
            else:
                primitives = replace(Primitives.from_mapping(config['seller']),p0=p0)
                row = experiments.solve_summary(study_config, primitives, rho)
                row.update(source='solve_summary: missing exact cached pair', reused=False)
            row.update(p0=p0, rho=rho)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'data/mature_advantage.csv', index=False)
    print(f'[mature advantage] {int(frame.reused.sum())} saved pairs reused; '
          f'{int((~frame.reused).sum())} missing pairs calculated', flush=True)
    return frame


def plots(summary: pd.DataFrame, config: dict, output: Path) -> list[str]:
    experiments.configure_style()
    p0s, rhos = config['p0_values'], config['rho_values']
    figures = output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    stems=[]
    def save(fig, name):
        fig.savefig(figures/f'{name}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
        stems.append(f'figures/{name}.png')
    for metric,title,unit,stem in [
        ('mean','Mean individual product-2 frequency','Percent','mean_frequency'),
        ('std','Across-trajectory standard deviation','Percentage points','frequency_std'),
        ('mean_absolute_difference','Mean absolute difference between late windows',
         'Percentage points','late_window_instability')]:
        values=summary.pivot(index='p0',columns='rho',values=metric).reindex(index=p0s,columns=rhos).to_numpy()*100
        fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
        vmax=100 if metric=='mean' else max(.01,float(values.max()))
        artist=ax.imshow(values,origin='lower',aspect='auto',vmin=0,vmax=vmax,cmap='viridis')
        ax.set_xticks(range(len(rhos)),[f'{r:g}' for r in rhos])
        ax.set_yticks(range(len(p0s)),[f'{p:g}' for p in p0s])
        ax.set(xlabel='Memory persistence rho (sampled values)',ylabel='Outside-option quality p0',title=title)
        ax.grid(False)
        for i in range(len(p0s)):
            for j in range(len(rhos)):
                label = '<0.01' if 0 < values[i,j] < .01 else f'{values[i,j]:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=8,
                        color='black' if values[i,j] > .55*vmax else 'white')
        fig.colorbar(artist,ax=ax,label=unit)
        save(fig,stem)
    fig,axes=plt.subplots(1,len(p0s),figsize=(16,4.6),sharey=False,layout='constrained')
    for ax,p0 in zip(np.atleast_1d(axes),p0s):
        g=summary[summary.p0==p0].sort_values('rho')
        ax.plot(g.rho,100*g['mean'],'o-',lw=1.3,ms=4,color='#2563eb')
        ax.fill_between(g.rho,100*g.p05,100*g.p95,color='#2563eb',alpha=.22)
        peak=100*g.p95.max()
        step=.1 if peak<=1 else 10.
        top=max(step,float(np.ceil(peak/step)*step))
        ax.set(title=f'p0 = {p0:g}',xlabel='Memory persistence rho',ylim=(-.02*top,1.02*top))
    np.atleast_1d(axes)[0].set_ylabel('Individual product-2 frequency (%)')
    fig.suptitle('Mean and 5th–95th trajectory percentiles — y-axis scales differ across panels')
    save(fig,'frequency_vs_rho')
    labels=['M <= 0; frequency ~ 0','M > 0; frequency > tolerance',
            'M > 0; frequency ~ 0','M <= 0; frequency > tolerance']
    codes=summary.assign(code=summary.comparison.map({label:i for i,label in enumerate(labels)}))
    values=codes.pivot(index='p0',columns='rho',values='code').reindex(index=p0s,columns=rhos).to_numpy()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    colors=['#d1d5db','#22c55e','#f59e0b','#ef4444']
    fig,ax=plt.subplots(figsize=(13,5.5))
    ax.imshow(values,origin='lower',aspect='auto',cmap=ListedColormap(colors),
              norm=BoundaryNorm(np.arange(-.5,4.5),4))
    ax.set_xticks(range(len(rhos)),[f'{r:g}' for r in rhos])
    ax.set_yticks(range(len(p0s)),[f'{p:g}' for p in p0s])
    ax.set(xlabel='Memory persistence rho (sampled values)',ylabel='Outside-option quality p0',
           title=f'Mature advantage versus individual frequency (~ 0 means <= {100*config["classification"]["zero_frequency"]:g}%)')
    ax.grid(False)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],
               loc='lower center',ncols=2,fontsize=9)
    fig.tight_layout(rect=(0,.15,1,1))
    save(fig,'mature_advantage_comparison')
    return stems


def report_text(summary: pd.DataFrame, config: dict) -> str:
    lines=['# Individual-path frequency grid', '',
        '100,000 interactions with A per trajectory; 50 independent trajectories per pair. '
        'All histories start at X=Y=0. Fixed primitives: R=1, c1=0.05, c2=0.65, '
        'p1=0.35, p2=0.80, gamma=0.98.', '',
        'Existing interaction_paths.simulate, solve_observation_layers and _policy_and_demand '
        'are reused. Product 2 iff interpolated advantage > 1e-6; otherwise product 1. '
        'The existing terminal-layer approximation is retained. No outside-option draw is made; '
        'the outside option still enters the optimal policy.', '',
        'Per-pair seed: SeedSequence([base_seed, round(p0*1e6), round(rho*1e6)]). '
        'Independent IID outcome draws across trajectories; seed assignment does not depend on worker order.', '',
        '## Explicit descriptive thresholds', '',
        f'Approximately zero: frequency <= {config["classification"]["zero_frequency"]}. '
        f'Extinction-like requires mean frequency at most this threshold and at least {100*config["classification"]["nearly_all_share"]:g}% of individual '
        'frequencies below it. Stable positive requires percentile 5 above that threshold, sample '
        f'standard deviation <= {config["classification"]["max_std"]} and mean absolute late-window difference <= {config["classification"]["max_mean_absolute_difference"]}. '
        'Other cases are labelled unstable or heterogeneous. These labels are descriptive, not proofs.', '',
        'Late windows: 20,001–60,000 and 60,001–100,000; 40,000 interactions each. '
        'Full post-burn-in frequency uses 80,000 interactions. All raw trajectory statistics are saved.', '',
        'M is the maximum advantage on the mature reachable set, not the full boundary. '
        'Matching saved pairs are reused (see data/mature_advantage.csv). Missing pairs are solved '
        'at their exact rho; signs are never interpolated. M > 0 uses the raw numerical sign; '
        'mature_sign_near_tolerance flags |M| <= 1e-6. Simulation policy uses the existing 1e-6 threshold.', '',
        'Heatmap columns are equally spaced sampled categories, not continuous rho intervals. '
        'Curve bands are trajectory percentiles, not confidence intervals or investment regions; '
        'connections between samples are visual guides. Curve panels use different vertical scales. ', '', '## 1. Stable positive pairs', '']
    for p0,g in summary.groupby('p0'):
        selected=g[g.classification=='Stable positive frequency'].rho.tolist()
        lines.append(f'- p0={p0:g}: rho={selected}')
    lines+=['','## 2. Comparison with positive M','']
    for name,g in summary.groupby('comparison'):
        lines.append(f'- {name}: {len(g)} pairs.')
    lines+=['','## 3. Positive M but approximately zero observed frequency','',
            '| p0 | rho | M | Mean frequency | P95 | Zero-count share |','|---|---|---|---|---|---|']
    for _,row in summary[summary.comparison=='M > 0; frequency ~ 0'].iterrows():
        lines.append(f'| {row.p0:g} | {row.rho:g} | {row.reachable_maximum_advantage:.8g} | '
                     f'{row["mean"]:.8g} | {row.p95:.8g} | {row.zero_count_share:.2f} |')
    lines+=['','These cases establish a discrepancy at the chosen frequency tolerance and finite horizon. '
            'They are consistent with rare visits to the product-2 region, but do not establish zero '
            'limiting frequency; rare events, slow transients and numerical resolution remain possible.', '',
            '## 4. Heterogeneity and late-window changes','',
            '| p0 | rho | Mean | Std | First window mean | Second window mean | Mean absolute window difference |','|---|---|---|---|---|---|---|']
    for _,row in summary[summary.classification=='Unstable or heterogeneous'].iterrows():
        lines.append(f'| {row.p0:g} | {row.rho:g} | {row["mean"]:.6f} | {row["std"]:.6f} | {row.mean_first_window_frequency:.6f} | {row.mean_second_window_frequency:.6f} | {row.mean_absolute_difference:.6f} |')
    lines+=['','The flagged pairs show substantial finite-horizon path dependence and/or drift. '
            'This is not evidence sufficient to establish distinct permanent regimes. ', '',
            '## 5. Frequency versus rho','',
            '| p0 | '+ ' | '.join(f'{r:g}' for r in config['rho_values'])+' |',
            '|---|'+'---|'*len(config['rho_values'])]
    for p0,g in summary.groupby('p0'):
        g=g.set_index('rho').reindex(config['rho_values'])
        lines.append(f'| {p0:g} | '+' | '.join(f'{100*v:.5g}%' for v in g['mean'])+' |')
    lines+=['','Numerical stabilization is not a proof of ergodicity or convergence. '
            'Only the requested coarse grid was run; no additional fine sweep is included.','']
    return '\n'.join(lines)


def run(config: dict, output: Path, workers: int = 4) -> None:
    output=output.resolve()
    (output/'data/cases').mkdir(parents=True,exist_ok=True)
    mature=mature_advantages(config,output)
    pairs=[(p,r) for p in config['p0_values'] for r in config['rho_values']]
    summaries=[]
    trajectories=[]
    def collect(result):
        frame,stats=result
        frame.to_csv(output/f'data/cases/p0_{stats["p0"]}_rho_{stats["rho"]}.csv',index=False)
        trajectories.append(frame)
        summaries.append(stats)
        print(f'[grid progress] {len(summaries)}/{len(pairs)} pairs',flush=True)
    if workers==1:
        for p0,rho in pairs:
            collect(worker(config,p0,rho))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(worker,config,p0,rho) for p0,rho in pairs]
            for future in as_completed(futures):
                collect(future.result())
    all_paths=pd.concat(trajectories,ignore_index=True).sort_values(['p0','rho','trajectory'])
    all_paths.to_csv(output/'data/trajectory_summary.csv',index=False)
    summary=pd.DataFrame(summaries).merge(mature,on=['p0','rho'],validate='one_to_one').sort_values(['p0','rho'])
    positive=summary.reachable_maximum_advantage>0
    zero=summary['mean']<=config['classification']['zero_frequency']
    summary['comparison']=np.select([~positive & zero,positive & ~zero,positive & zero],
        ['M <= 0; frequency ~ 0','M > 0; frequency > tolerance','M > 0; frequency ~ 0'],
        default='M <= 0; frequency > tolerance')
    summary['mature_sign_near_tolerance']=abs(summary.reachable_maximum_advantage)<=config['solver']['action_tolerance']
    summary.to_csv(output/'data/parameter_summary.csv',index=False)
    stems=plots(summary,config,output)
    (output/'diagnostic_report.md').write_text(report_text(summary,config))
    (output/'manifest.json').write_text(json.dumps({'config':config,'workers':workers,
        'figures':stems,'trajectory_count':len(all_paths),'parameter_pairs':len(summary)},indent=2)+'\n')
    print(summary.groupby('classification').size().to_string(),flush=True)
    print(f'Results: {output}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=experiments.CONFIG_DIR/'config_interaction_grid.json')
    parser.add_argument('--output-dir',type=Path,default=experiments.ROOT/'outputs/policy_and_state_experiments/interaction_grid')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv)
    if args.workers<1:
        parser.error('--workers must be positive')
    run(experiments.read_config(args.config),args.output_dir,args.workers)


if __name__=='__main__':
    main()
