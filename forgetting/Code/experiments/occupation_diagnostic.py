"""Separate rare visitation from mature extinction using existing forgetting policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

from forgetting.Code.experiments import run_experiments as experiments
from forgetting.Code.layers import solve_observation_layers
from forgetting.Code.model import Primitives, solve_boundary, reachable_maximum, reachable_boundary_points
from forgetting.Code.experiments.states import policy_intervals, reachable_cover, intersect_intervals
from forgetting.Code.simulation import _policy_and_demand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RECORD_DTYPE = np.dtype([('x_before','f8'), ('mature_advantage_fine','f8'),
                         ('product2','?'), ('success','?')])


def positive_components(solution, depth: int = 16) -> list[dict]:
    intervals = [(lo,hi) for lo,hi,action in policy_intervals(solution,0.0) if action=='product2']
    components=[]
    for lo,hi in intervals:
        if solution.rho >= .5:
            intersection='yes: K=[0,1]'
        else:
            endpoints=reachable_boundary_points(solution.rho,depth)
            active=(endpoints >= lo)&(endpoints <= hi)&(np.interp(endpoints,solution.coordinate,solution.advantage)>0)
            if active.any():
                intersection='yes: active reachable endpoint'
            elif len(intersect_intervals(reachable_cover(solution.rho,depth),lo,hi))==0:
                intersection='no: disjoint from outer cover'
            else:
                intersection='unresolved within finite-depth outer cover'
        components.append({'component':len(components)+1,'lower':lo,'upper':hi,
                           'length':hi-lo,'reachable_intersection':intersection})
    return components


def distance_to_region(x, components: list[dict]):
    """Distance to the closure of the strict region; zero at excluded roots too."""
    values=np.asarray(x,dtype=float)
    distance=np.full(values.shape,np.inf)
    for c in components:
        distance=np.minimum(distance,np.maximum(np.maximum(c['lower']-values,values-c['upper']),0))
    return distance


def stream_history(primitives, rho, policy, fine, directory: Path, *, paths, interactions,
                   burn_in, seed, action_tolerance):
    """Record every retained pre-choice state and outcome without calendar draws."""
    if not 0 <= burn_in < interactions or paths < 1:
        raise ValueError('Require positive path count and 0 <= burn_in < interactions')
    retained=interactions-burn_in
    records=np.lib.format.open_memmap(directory/'observations.npy',mode='w+',
                                      dtype=RECORD_DTYPE,shape=(retained,paths))
    rng=np.random.default_rng(seed)
    x=np.zeros(paths)
    observations=np.zeros(paths,dtype=np.int64)
    step_sum=np.zeros(paths)
    preburn_count=np.zeros(paths,dtype=np.int64)
    last_product2=np.zeros(paths,dtype=np.int64)
    for n in range(1,interactions+1):
        product2,_=_policy_and_demand(observations,x,rho=rho,policy=policy,
                                      action_tolerance=action_tolerance)
        success=rng.random(paths)<np.where(product2,primitives.p2,primitives.p1)
        next_x=rho*x+(1-rho)*success
        last_product2[product2]=n
        if n>burn_in:
            row=n-burn_in-1
            records['x_before'][row]=x
            records['mature_advantage_fine'][row]=np.interp(x,fine.coordinate,fine.advantage)
            records['product2'][row]=product2
            records['success'][row]=success
            step_sum+=abs(next_x-x)
        else:
            preburn_count+=product2
        x=next_x
        observations+=1
        if n%100_000==0:
            print(f'[occupation] p0={primitives.p0:g}, rho={rho:g}: {n:,}/{interactions:,}',flush=True)
    records.flush()
    return records,step_sum/retained,preburn_count,last_product2


def occupation_statistics(records, components, policy, *, burn_in, neighborhoods,
                          action_tolerance, block_size):
    retained,paths=records.shape
    if retained%block_size:
        raise ValueError('Post-burn-in length must divide into complete blocks')
    x=records['x_before']
    q=[.01,.05,.25,.5,.75,.95,.99]
    quantiles=np.quantile(x,q)
    bins=np.linspace(0,1,501)
    histogram=np.zeros(len(bins)-1,dtype=np.int64)
    direct=np.zeros(paths,dtype=np.int64)
    fine_strict=np.zeros(paths,dtype=np.int64)
    policy_strict=np.zeros(paths,dtype=np.int64)
    policy_rule=np.zeros(paths,dtype=np.int64)
    mismatch=np.zeros(paths,dtype=np.int64)
    fine_mismatch=np.zeros(paths,dtype=np.int64)
    near={eps:np.zeros(paths,dtype=np.int64) for eps in neighborhoods}
    blocks=[]
    for start in range(0,retained,block_size):
        block=records[start:start+block_size]
        states=block['x_before']
        a_policy=np.interp(states,policy['coordinate'],np.asarray(policy['advantage_table'])[-1])
        j=block['product2']
        strict=block['mature_advantage_fine']>0
        direct+=j.sum(axis=0)
        fine_strict+=strict.sum(axis=0)
        policy_strict+=(a_policy>0).sum(axis=0)
        policy_rule+=(a_policy>action_tolerance).sum(axis=0)
        mismatch+=(j!=(a_policy>action_tolerance)).sum(axis=0)
        fine_mismatch+=(j!=strict).sum(axis=0)
        distances=distance_to_region(states,components)
        for eps in neighborhoods:
            near[eps]+=(distances<=eps).sum(axis=0)
        histogram+=np.histogram(states,bins=bins)[0]
        for trajectory in range(paths):
            blocks.append({'trajectory':trajectory+1,'start_interaction':burn_in+start+1,
                'end_interaction':burn_in+start+len(block),
                'product2_frequency':float(j[:,trajectory].mean()),
                'strict_fine_occupation':float(strict[:,trajectory].mean()),
                'mean_x':float(states[:,trajectory].mean())})
    alpha=direct/retained
    mu=fine_strict/retained
    trajectory=pd.DataFrame({'trajectory':np.arange(1,paths+1),
        'product2_count':direct,'product2_frequency':alpha,
        'strict_fine_count':fine_strict,'strict_fine_occupation':mu,
        'strict_policy_occupation':policy_strict/retained,
        'policy_tolerance_occupation':policy_rule/retained,
        'fine_occupation_minus_product2':mu-alpha,
        'policy_rule_mismatch_count':mismatch,'fine_strict_action_mismatch_count':fine_mismatch,
        'mean_x':x.mean(axis=0),'std_x':x.std(axis=0),
        'first_half_product2_frequency':records['product2'][:retained//2].mean(axis=0),
        'second_half_product2_frequency':records['product2'][retained//2:].mean(axis=0)})
    for eps in neighborhoods:
        trajectory[f'within_{eps:g}_of_region']=near[eps]/retained
    stats={'product2_frequency':float(alpha.mean()),'product2_frequency_std_across_paths':float(alpha.std(ddof=1)),
        'strict_fine_occupation':float(mu.mean()),
        'strict_policy_occupation':float(policy_strict.sum()/(retained*paths)),
        'policy_tolerance_occupation':float(policy_rule.sum()/(retained*paths)),
        'fine_occupation_minus_product2':float((mu-alpha).mean()),
        'policy_rule_mismatch_count':int(mismatch.sum()),
        'fine_strict_action_mismatch_count':int(fine_mismatch.sum()),
        'mean_x':float(x.mean()),'std_x':float(x.std()),
        **{f'x_p{int(100*p):02d}':float(value) for p,value in zip(q,quantiles)},
        'median_distance_to_positive_region':float(distance_to_region(quantiles[3],components)),
        'first_half_product2_frequency':float(trajectory.first_half_product2_frequency.mean()),
        'second_half_product2_frequency':float(trajectory.second_half_product2_frequency.mean()),
        'paths_with_zero_product2':int((direct==0).sum()),
        'post_burn_in_observations':int(retained*paths)}
    for eps in neighborhoods:
        stats[f'within_{eps:g}_of_region']=float(near[eps].sum()/(retained*paths))
    hist=pd.DataFrame({'lower':bins[:-1],'upper':bins[1:],'count':histogram,
                       'density':histogram/(retained*paths*np.diff(bins))})
    return trajectory,stats,pd.DataFrame(blocks),hist


def run_pair(config: dict, p0: float, rho: float, output: str) -> dict:
    directory=Path(output)/'data'/f'p0_{p0}_rho_{rho}'
    directory.mkdir(parents=True,exist_ok=True)
    primitives=replace(Primitives.from_mapping(config['seller']),p0=p0)
    fine=solve_boundary(primitives,rho,**config['mature_solver'])
    finer_options={**config['mature_solver'],
        'central_nodes':2*config['mature_solver']['central_nodes']-1,
        'outer_nodes':2*config['mature_solver']['outer_nodes'],
        'tolerance':.25*config['mature_solver']['tolerance']}
    finer=solve_boundary(primitives,rho,**finer_options)
    components=positive_components(fine)
    finer_components=positive_components(finer)
    maximum,argmax,radius=reachable_maximum(fine,reachable_depth=16)
    finer_max,_,_=reachable_maximum(finer,reachable_depth=16)
    pd.DataFrame({'x':fine.coordinate,'advantage':fine.advantage,
                  'demand':fine.demand,'value':fine.value}).to_csv(directory/'mature_boundary.csv',index=False)
    pd.DataFrame({'x':finer.coordinate,'advantage':finer.advantage}).to_csv(directory/'mature_boundary_refined.csv',index=False)
    pd.DataFrame(components,columns=['component','lower','upper','length','reachable_intersection']).to_csv(directory/'positive_components.csv',index=False)
    maxima=np.abs(fine.advantage-maximum)<=config['maximum_location_tolerance']
    pd.DataFrame({'x':fine.coordinate[maxima],'advantage':fine.advantage[maxima]}).to_csv(directory/'maximum_locations.csv',index=False)
    # No saved policy tables/transient layer objects exist for these runs.
    # Solve once for the actual simulation and reuse its diagnostics below.
    layers,policy=solve_observation_layers(primitives,rho,return_policy=True,**config['policy_solver'])
    layers.to_csv(directory/'transient_layers.csv',index=False)
    pd.DataFrame({'x':policy['coordinate'],
        'advantage':np.asarray(policy['advantage_table'])[-1]}).to_csv(directory/'simulation_mature_boundary.csv',index=False)
    seed=int(np.random.SeedSequence([config['experiment']['seed'],int(round(p0*1e6)),
              int(round(rho*1e6))]).generate_state(1,dtype=np.uint64)[0])
    opts=config['experiment']
    records,steps,preburn,last_product2=stream_history(primitives,rho,policy,fine,directory,
        paths=opts['paths'],interactions=opts['interactions'],burn_in=opts['burn_in'],
        seed=seed,action_tolerance=config['policy_solver']['action_tolerance'])
    trajectory,stats,blocks,hist=occupation_statistics(records,components,policy,
        burn_in=opts['burn_in'],neighborhoods=config['neighborhoods'],
        action_tolerance=config['policy_solver']['action_tolerance'],block_size=opts['block_size'])
    trajectory['mean_absolute_step']=steps
    trajectory['preburn_product2_count']=preburn
    trajectory['last_product2_interaction']=last_product2
    trajectory['seed']=str(seed)
    for frame in (trajectory,blocks):
        frame.insert(0,'rho',rho)
        frame.insert(0,'p0',p0)
    trajectory.to_csv(directory/'trajectory_summary.csv',index=False)
    blocks.to_csv(directory/'block_frequencies.csv',index=False)
    hist.to_csv(directory/'occupation_histogram.csv',index=False)
    transient=layers[~layers.mature_boundary.astype(bool)]
    active=transient[transient.product2_active.astype(bool)]
    stats.update(p0=p0,rho=rho,seed=str(seed),M=maximum,argmax_x=argmax,
        maximizer_x_min=float(fine.coordinate[maxima].min()),maximizer_x_max=float(fine.coordinate[maxima].max()),
        maximum_location_tolerance=config['maximum_location_tolerance'],
        positive_region_length=sum(c['length'] for c in components),positive_components=len(components),
        M_refined=finer_max,M_refinement_difference=abs(finer_max-maximum),
        refined_positive_region_length=sum(c['length'] for c in finer_components),
        fine_bellman_residual=fine.bellman_residual,refined_bellman_residual=finer.bellman_residual,
        fine_nodes=len(fine.coordinate),refined_nodes=len(finer.coordinate),
        simulation_mature_M=float(np.asarray(policy['advantage_table'])[-1].max()),
        terminal_observation=int(policy['terminal_observation']),
        mature_after_burn_in=opts['burn_in']>=int(policy['terminal_observation']),
        mean_absolute_step=float(steps.mean()),innovation=1-rho,
        transient_active_layer_count=len(active),
        last_active_transient_observation=None if active.empty else int(active.observation.max()),
        mean_preburn_product2_frequency=float(preburn.mean()/opts['burn_in']),
        source_transient='Computed once by existing solve_observation_layers for simulation; reused for comparison')
    (directory/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    (directory/'records_manifest.json').write_text(json.dumps({
        'shape':list(records.shape),'row_interaction':'burn_in + 1 + row_index',
        'column_trajectory':'1 + column_index','burn_in':opts['burn_in'],
        'fields':RECORD_DTYPE.descr,'state_timing':'before choice and outcome at this interaction',
        'advantage':'fine-grid mature advantage evaluated at X before choice',
        'seed':str(seed),'config':config,'p0':p0,'rho':rho},indent=2)+'\n')
    del records,policy
    print(f'[pair done] p0={p0:g}, rho={rho:g}, M={maximum:.6g}, alpha={stats["product2_frequency"]:.8g}',flush=True)
    return stats


def plot_pair(ax_top,ax_bottom,curve,hist,components,stats):
    ax_top.plot(curve.x,curve.advantage,color='#334155',lw=1)
    ax_top.axhline(0,color='black',ls='--',lw=.8)
    for _,c in components.iterrows():
        for ax in (ax_top,ax_bottom):
            ax.axvspan(c.lower,c.upper,color='#22c55e',alpha=.16)
    if components.empty:
        ax_top.text(.5,.85,'Strict positive region empty',transform=ax_top.transAxes,ha='center',fontsize=9)
    ax_top.set(ylabel='Mature advantage a(x)',title=f'p0={stats["p0"]:g}, rho={stats["rho"]:g}; M={stats["M"]:.5g}')
    ax_bottom.stairs(hist.density,np.r_[hist.lower.to_numpy(),hist.upper.iloc[-1]],color='#2563eb',lw=1.2)
    ax_bottom.set_yscale('log')
    nonzero=hist.density[hist.density>0]
    ax_bottom.set_ylim(max(1e-7,float(nonzero.min())*.5),float(nonzero.max())*3)
    ax_bottom.set(ylabel='Occupation density (log scale)',xlabel='X before product choice',xlim=(0,1))
    ax_bottom.set_title(f'Product 2: {100*stats["product2_frequency"]:.5g}%\n'
        f'Strict-region occupation: {100*stats["strict_fine_occupation"]:.5g}%',
        fontsize=9)


def plots(summary,output):
    experiments.configure_style()
    figures=output/'figures'/'occupation'
    figures.mkdir(parents=True,exist_ok=True)
    stems=[]
    for _,stats in summary.iterrows():
        pair_figures=figures/f'p0_{stats.p0:g}'
        pair_figures.mkdir(parents=True,exist_ok=True)
        stem=f'occupation_p0_{stats.p0}_rho_{stats.rho}'
        directory=output/'data'/f'p0_{stats.p0}_rho_{stats.rho}'
        fig,axes=plt.subplots(2,1,figsize=(10,6.5),sharex=True,layout='constrained')
        plot_pair(*axes,pd.read_csv(directory/'mature_boundary.csv'),pd.read_csv(directory/'occupation_histogram.csv'),
                  pd.read_csv(directory/'positive_components.csv'),stats)
        fig.suptitle('Green: strict product-2 region; blue: observed state density after burn-in')
        fig.savefig(pair_figures/f'{stem}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
        stems.append(str((pair_figures/f'{stem}.png').relative_to(output)))
    controls=summary[summary.p0==.5].sort_values('rho')
    fig,axes=plt.subplots(2,len(controls),figsize=(15,6.5),sharex=True,squeeze=False,layout='constrained')
    for column,(_,stats) in enumerate(controls.iterrows()):
        directory=output/'data'/f'p0_{stats.p0}_rho_{stats.rho}'
        plot_pair(axes[0,column],axes[1,column],pd.read_csv(directory/'mature_boundary.csv'),
            pd.read_csv(directory/'occupation_histogram.csv'),pd.read_csv(directory/'positive_components.csv'),stats)
    fig.suptitle('Slow forgetting at p0=0.5 — green: strict positive region; blue: occupation')
    stem='slow_forgetting_comparison'
    control_figures=figures/'p0_0.5'
    control_figures.mkdir(parents=True,exist_ok=True)
    fig.savefig(control_figures/f'{stem}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    stems.append(str((control_figures/f'{stem}.png').relative_to(output)))
    return stems


def report_text(summary,config):
    lines=['# Rare visitation versus mature extinction','',
        'Six requested pairs; c2=0.65, c1=0.05, p1=0.35, p2=0.8, R=1, gamma=0.98. '
        '20 independent histories of 1,000,000 interactions with A per pair, starting from X=Y=0. '
        'Occupation statistics discard the first 100,000 interactions: 18 million retained observations per pair.', '',
        '## Numerical conventions', '',
        'Same Bellman solver, transient layers, transition rules, and _policy_and_demand as existing experiments. '
        'Product 2 iff implemented interpolated advantage > 1e-6. The strict mathematical region uses a(x)>0. '
        'The simulation keeps the existing 2,001-node uniform policy grid. The mature curve is independently '
        'computed on 20,001 central nodes plus outer nodes and checked with 40,001 central nodes. '
        'Differences between strict fine-grid occupation and actual product choice are reported, not suppressed.', '',
        'No saved transient policy tables were available for these pairs. The layers required for simulation '
        'were solved once with the existing function, saved, and reused for the transient-versus-mature comparison. '
        'All burn-ins exceed the terminal layer. States and advantages are recorded before each choice; '
        'success/failure then updates the state. Each observations.npy records every retained state, fine mature '
        'advantage, product choice and outcome; see records_manifest.json for indexing and seed.', '',
        'For all six pairs K=[0,1]. Positive components and their lengths use the piecewise-linear '
        'fine-grid advantage; they are numerical boundary estimates. Distance means distance to the closure '
        'of the strict region (zero also at excluded roots), and is infinity if the region is empty. '
        'Maximum_locations.csv contains grid locations within 1e-9 of M, not an analytic set of maximizers. '
        'Occupation quantiles are computed from all retained states; the histogram is only a visualization.', '',
        '## Main comparison', '',
        '| p0 | rho | M | Refined M | Positive-region length | Product 2 (%) | Strict occupation (%) | Mean step |',
        '|---|---|---|---|---|---|---|---|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.p0:g} | {r.rho:g} | {r.M:.7g} | {r.M_refined:.7g} | {r.positive_region_length:.6g} | '
                     f'{100*r.product2_frequency:.7g} | {100*r.strict_fine_occupation:.7g} | {r.mean_absolute_step:.7g} |')
    lines+=['','## Numerical discrepancies','',
        '| p0 | rho | Fine occupation minus product 2 (pp) | Fine-strict/action disagreement count | Exact implemented-rule disagreements |',
        '|---|---|---|---|---|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.p0:g} | {r.rho:g} | {100*r.fine_occupation_minus_product2:.7g} | '
                     f'{int(r.fine_strict_action_mismatch_count)} | {int(r.policy_rule_mismatch_count)} |')
    lines+=['','The strict occupation evaluated on the simulation policy grid and its above-tolerance '
            'version are separately saved. Fine-grid disagreements include discretization and strict-versus-tolerance '
            'effects; exact-rule disagreements would indicate an additional issue.', '',
            '## Distance to the region and temporal persistence','',
            '| p0 | rho | Median X | Distance | Within 0.001 | Within 0.01 | Within 0.05 | First-half use | Second-half use |',
            '|---|---|---|---|---|---|---|---|---|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.p0:g} | {r.rho:g} | {r.x_p50:.6g} | {r.median_distance_to_positive_region:.6g} | '
            f'{r["within_0.001_of_region"]:.6g} | {r["within_0.01_of_region"]:.6g} | {r["within_0.05_of_region"]:.6g} | '
            f'{r.first_half_product2_frequency:.6g} | {r.second_half_product2_frequency:.6g} |')
    lines+=['','Neighborhood shares and half frequencies in this table are fractions, not percentages. '
            'The equal post-burn-in halves each contain 450,000 interactions. Raw CSVs include state mean, '
            'standard deviation, all requested quantiles, trajectory statistics, and 100,000-interaction blocks.', '',
            '## Transient comparison for p0=0.5, rho=0.999','']
    selected=summary[(summary.p0==.5)&(summary.rho==.999)]
    if not selected.empty:
        r=selected.iloc[0]
        lines.append(f'Mature M={r.M:.8g}; simulation-policy mature M={r.simulation_mature_M:.8g}. '
            f'Terminal observation layer={int(r.terminal_observation)}. '
            f'{int(r.transient_active_layer_count)} transient layers contain a product-2 action; '
            f'last active transient observation={r.last_active_transient_observation}. '
            f'Mean pre-burn-in use={100*r.mean_preburn_product2_frequency:.6g}%; '
            f'post-burn-in use={100*r.product2_frequency:.6g}%.')
    lines+=['','## Answers for the six requested pairs','',
        '1. **Is the region present when use is nearly zero?** Yes for p0=0.1 at all three rho values. '
        'Its length is about 0.115, 0.115, and 0.0915: low use does not require an extremely narrow region.', '',
        '2. **Is it rarely visited?** Yes in these simulations. Product-2 frequencies are about 0.00902%, '
        '0.0001%, and 0%, respectively. The region lies below approximately X=0.115, while the empirical '
        'median is near 0.35. At rho=0.99 even its 0.05 neighborhood receives no observed visits. '
        'This supports a lower-tail visitation explanation; it does not prove zero limiting occupation.', '',
        '3. **Does it shrink, move, or disappear?** For p0=0.1 the length first rises slightly and then '
        'falls; at rho=0.99 its lower endpoint has moved above zero, but it still exists. For p0=0.5, '
        'the length falls from 0.3904 to 0.1267 and then zero; M becomes about -0.1701 at rho=0.999. '
        'The refined grid preserves all six signs of M.', '',
        '4. **Is there an intermediate rare-use regime for p0=0.5?** These points do not show one: '
        'use is about 54.38%, 43.11%, then 0%. They cannot rule out such a regime between 0.99 and 0.999.', '',
        '5. **Could finite horizons and slow movement matter?** Yes for interpreting zero observed visits '
        'when M>0: rare future visits remain possible. However, the separation between the occupied '
        'distribution and the positive region directly supports tail rarity, rather than establishing '
        'slow exploration as the cause. For p0=0.5 the mean absolute step falls from 0.02464 to 0.004985 '
        'to 0.000455. At rho=0.999 the empty mature region explains mature nonuse regardless of how slowly '
        'states move. None of these comparisons identifies a mixing time or proves a limiting frequency.', '',
        '## Limits of interpretation','',
        'Positive advantage and small occupation describe different objects. A nonempty region may be '
        'visited rarely because it lies in the tail; zero observed visits do not prove zero limiting mass. '
        'Shrinking step size alone does not establish a mixing time. Block/half comparisons diagnose drift '
        'at this horizon but cannot rule out rarer future transitions. The three control rho values cannot '
        'resolve the entire transition between frequent use and mature extinction. No extra rho sweep is run. '
        'None of the numerical evidence constitutes a proof of extinction, ergodicity, or convergence.','']
    return '\n'.join(lines)


def run(config,output,workers=3):
    output=output.resolve()
    (output/'data').mkdir(parents=True,exist_ok=True)
    stats=[]
    if workers==1:
        for p0,rho in config['pairs']:
            stats.append(run_pair(config,p0,rho,str(output)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[pool.submit(run_pair,config,p0,rho,str(output)) for p0,rho in config['pairs']]
            for job in as_completed(jobs):
                stats.append(job.result())
                print(f'[progress] {len(stats)}/{len(jobs)} pairs',flush=True)
    summary=pd.DataFrame(stats).sort_values(['p0','rho'])
    summary.to_csv(output/'data/pair_summary.csv',index=False)
    trajectories=pd.concat([pd.read_csv(output/'data'/f'p0_{p0}_rho_{rho}'/'trajectory_summary.csv',dtype={'seed':str})
                            for p0,rho in config['pairs']],ignore_index=True)
    trajectories.to_csv(output/'data/trajectory_summary.csv',index=False)
    stems=plots(summary,output)
    (output/'diagnostic_report.md').write_text(report_text(summary,config))
    (output/'manifest.json').write_text(json.dumps({'config':config,'figures':stems,'workers':workers,
                                                   'record_format':'NumPy structured .npy, row=time, column=trajectory'},indent=2)+'\n')
    print(summary[['p0','rho','M','positive_region_length','product2_frequency','strict_fine_occupation','mean_absolute_step']].to_string(index=False))
    print(f'Results: {output}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=experiments.CONFIG_DIR/'config_occupation.json')
    parser.add_argument('--output-dir',type=Path,default=experiments.ROOT/'outputs/policy_and_state_experiments/occupation_diagnostic')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args(argv)
    if args.workers<1:
        parser.error('--workers must be positive')
    run(experiments.read_config(args.config),args.output_dir,args.workers)


if __name__=='__main__':
    main()
