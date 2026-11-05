#!/usr/bin/env python3
"""Explore both raw differences on (0,1), varying feasible baseline levels.

No production solver or previous results are modified. Each plotted difference
pair represents nine scenarios, not a probability or an exhaustive quantifier.
"""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, solve_boundary

OLD = ROOT / 'forgetting/Code/threshold_parameter_map/sweep.py'
spec = importlib.util.spec_from_file_location('prior_threshold_sweep', OLD)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/full_delta_map' / 'data'
RHOS = [.70, .90, .95, .99]
GAPS = np.arange(1, 40, dtype=float) / 40
FRACTIONS = [.1, .5, .9]
KEYS = ['rho', 'delta_p', 'delta_c', 'p1_fraction', 'c1_fraction']


class Solver(prior.Solver):
    """Generalized wrapper of the unchanged compiled Bellman solver.

    Divide rewards and values by R-c1: normalized rewards are 1 and 1-k,
    k=delta_c/(1-c1). Positive scaling leaves the optimal policy invariant.
    """
    def solve(self, ctx, p1, p2, k, v=None, mode=0):
        x, si, sw, fi, fw, w = ctx
        v = np.zeros(len(x)) if v is None else v.copy()
        a = np.empty(len(x)); residual = ctypes.c_double()
        its = self.fun(len(x), si, sw, fi, fw, w, p1, p2, 1., 1.-k,
                       .98, v, a, ctypes.byref(residual), mode, 2e-11, 10000)
        if its < 0:
            raise RuntimeError(f'Convergence failure: {p1=}, {p2=}, {k=}')
        return v, a, residual.value, its


def summarize(knots, a):
    result = prior.summary(knots, a)
    result['detected_regime'] = result['regime']
    return result


def validate(solver):
    cases = [(.70,.025,.975,.1,.9), (.70,.975,.025,.9,.1),
             (.90,.45,.60,.5,.1), (.90,.975,.975,.1,.5),
             (.95,.025,.025,.9,.9), (.95,.45,.60,.1,.5),
             (.99,.975,.975,.9,.9), (.99,.50,.025,.5,.1),
             (.95,.80,.50,.5,.5), (.70,.20,.20,.9,.5)]
    rows = []
    for rho, dp, dc, f, g in cases:
        p1=f*(1-dp); p2=p1+dp; c1=g*(1-dc); c2=c1+dc; scale=1-c1
        ctx=solver.setup(rho,2001)
        v,a,res,its=solver.solve(ctx,p1,p2,dc/scale)
        ref=solve_boundary(Primitives(p1=p1,p2=p2,c1=c1,c2=c2),rho,
                           central_nodes=2001,tolerance=2e-11)
        dv=float(np.max(np.abs(scale*v-ref.value)))
        da=float(np.max(np.abs(scale*a-ref.advantage)))
        knots, advantage=prior.topology_grid(ctx,rho)
        row=dict(rho=rho,delta_p=dp,delta_c=dc,p1_fraction=f,c1_fraction=g,
                 p1=p1,p2=p2,c1=c1,c2=c2,value_max_difference=dv,
                 advantage_max_difference=da,bellman_residual=scale*res,
                 iterations=its,**summarize(knots,advantage(scale*v,dp,dc)))
        rows.append(row)
        print('VALIDATE',rho,dp,dc,row['regime'],f'dV={dv:.3e}',f'dA={da:.3e}',flush=True)
        assert dv<2e-8 and da<2e-8
    (OUT/'reference_validation.json').write_text(json.dumps(rows,indent=2))
    return rows


def sweep_grid(solver,rho,nodes,dps=GAPS,write=True):
    path=OUT/f'grid_rho_{rho:.2f}_nodes_{nodes}.csv'
    if write and path.exists():
        previous=pd.read_csv(path)
        if len(previous)==len(dps)*len(GAPS)*len(FRACTIONS)**2:
            rows=previous.to_dict('records')
            stat=dict(rho=rho,nodes=nodes,seconds=0.,points=len(rows),resumed=True,
                      fixed_policy_shortcuts=int((previous.iterations==0).sum()),
                      bellman_iterations=int(previous.iterations.sum()))
            print('GRID_REUSED',json.dumps(stat),flush=True)
            return rows,stat
    start=time.time();ctx=solver.setup(rho,nodes);rows=[];shortcuts=0;iterations=0
    knots,advantage=prior.topology_grid(ctx,rho)
    for di,dp in enumerate(dps):
        for f in FRACTIONS:
            p1=f*(1-dp);p2=p1+dp
            v1,_,res1,it1=solver.solve(ctx,p1,p2,0.,mode=1)
            v2,_,res2,it2=solver.solve(ctx,p1,p2,0.,mode=2)
            gap1=advantage(v1,1.,0.);gap2=advantage(v2,1.,0.)
            costs=sorted((dc/(1-g*(1-dc)),dc,g) for dc in GAPS for g in FRACTIONS)
            v=None
            for k,dc,g in costs:
                c1=g*(1-dc);c2=c1+dc;scale=1-c1
                a1=-k+dp*gap1;a2=-k+dp*(1-k)*gap2
                if a1.max() < -1e-9:
                    v=v1;res=res1;its=0;ak=scale*a1;shortcuts+=1
                elif a2.min() > 1e-9:
                    v=(1-k)*v2;res=(1-k)*res2;its=0;ak=scale*a2;shortcuts+=1
                else:
                    v,_,res,its=solver.solve(ctx,p1,p2,k,v)
                    ak=advantage(scale*v,dp,dc)
                iterations+=its
                rows.append(dict(rho=rho,R=1.,gamma=.98,p0=.5,
                                 delta_p=float(dp),delta_c=float(dc),p1_fraction=f,c1_fraction=g,
                                 p1=p1,p2=p2,c1=c1,c2=c2,nodes=nodes,
                                 bellman_residual=scale*res,iterations=its,
                                 **summarize(knots,ak)))
        if di%10==0:
            print('PROGRESS',rho,nodes,f'dp={dp:.3f}',f'seconds={time.time()-start:.1f}',flush=True)
    stat=dict(rho=rho,nodes=nodes,seconds=time.time()-start,points=len(rows),
              fixed_policy_shortcuts=shortcuts,bellman_iterations=iterations)
    print('GRID_DONE',json.dumps(stat),flush=True)
    if write:
        pd.DataFrame(rows).to_csv(path,index=False)
    return rows,stat


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true')
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);start=time.time();solver=Solver()
    validation=validate(solver)
    if args.benchmark:
        stats=[]
        for rho in RHOS:
            _,stat=sweep_grid(solver,rho,2001,dps=[.025,.5,.975],write=False)
            stats.append(stat)
        elapsed=sum(s['seconds'] for s in stats)
        result=dict(benchmark_seconds=elapsed,estimated_serial_sweep_seconds=elapsed*13*3,
                    grids=stats)
        (OUT/'benchmark.json').write_text(json.dumps(result,indent=2))
        print('BENCHMARK',json.dumps(result),flush=True);return
    coarse=[];fine=[];stats=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(sweep_grid,solver,rho,nodes) for rho in RHOS for nodes in [2001,4001]]
        for future in as_completed(pending):
            rows,stat=future.result();stats.append(stat)
            (coarse if stat['nodes']==2001 else fine).extend(rows)
    old=pd.DataFrame(coarse).sort_values(KEYS).reset_index(drop=True)
    df=pd.DataFrame(fine).sort_values(KEYS).reset_index(drop=True)
    old.to_csv(OUT/'parameter_map_coarse.csv',index=False)
    df.to_csv(OUT/'parameter_map_fine.csv',index=False)
    joined=df.merge(old,on=KEYS,suffixes=('','_coarse'),validate='one_to_one')
    movement=np.maximum(abs(joined.x_lower-joined.x_lower_coarse).fillna(0),
                        abs(joined.x_upper-joined.x_upper_coarse).fillna(0))
    structural=(joined.regime!=joined.regime_coarse)|(joined.n_components!=joined.n_components_coarse)
    df['root_movement']=movement
    df['advantage_refinement_difference']=np.maximum(abs(joined.a_max-joined.a_max_coarse),
                                                   abs(joined.a_min-joined.a_min_coarse))
    df['coarse_regime']=joined.regime_coarse;df['refinement']='2001_to_4001'
    refine=structural|df.uncertain|(movement>.001)
    extra=0
    for idx in np.flatnonzero(refine):
        row=df.iloc[idx];ctx=solver.setup(row.rho,8001);scale=1-row.c1
        v,_,res,its=solver.solve(ctx,row.p1,row.p2,row.delta_c/scale)
        knots,advantage=prior.topology_grid(ctx,row.rho)
        ss=summarize(knots,advantage(scale*v,row.delta_p,row.delta_c))
        rm=max(abs(ss['x_lower']-row.x_lower) if np.isfinite(ss['x_lower']) and np.isfinite(row.x_lower) else 0,
               abs(ss['x_upper']-row.x_upper) if np.isfinite(ss['x_upper']) and np.isfinite(row.x_upper) else 0)
        for key,value in ss.items():df.at[idx,key]=value
        df.at[idx,'nodes']=8001;df.at[idx,'bellman_residual']=scale*res;df.at[idx,'iterations']=its
        df.at[idx,'root_movement']=rm;df.at[idx,'refinement']='2001_to_4001_to_8001'
        df.at[idx,'uncertain']=ss['uncertain'] or ss['regime']!=row.regime or rm>.001
        extra+=1
    df['detected_regime']=df.regime
    df.loc[df.uncertain,'regime']='uncertain'
    df.to_csv(OUT/'parameter_map.csv',index=False)
    aggregates=[]
    for (rho,dp,dc),group in df.groupby(['rho','delta_p','delta_c'],sort=True):
        counts=group.regime.value_counts().to_dict();n=int(counts.get('two',0));u=int(counts.get('uncertain',0))
        aggregates.append(dict(rho=rho,delta_p=dp,delta_c=dc,n_two=n,n_total=len(group),
                               n_uncertain=u,n_multiple=int(counts.get('multiple',0)),
                               status='all' if n==9 else ('some' if n else ('uncertain' if u else 'none')),
                               patterns=json.dumps(counts,sort_keys=True),
                               **{f'n_{k}':int(counts.get(k,0)) for k in ['all1','all2','one_upper','one_lower']}))
    agg=pd.DataFrame(aggregates);agg.to_csv(OUT/'aggregated_map.csv',index=False)
    manifest=dict(R=1.,gamma=.98,p0=.5,rhos=RHOS,delta_p_grid=GAPS.tolist(),delta_c_grid=GAPS.tolist(),
                  baseline_fractions=FRACTIONS,baseline_rule='p1=f*(1-delta_p), p2=p1+delta_p, c1=g*(1-delta_c), c2=c1+delta_c; independent f,g in {.1,.5,.9}',
                  optional_p0_le_p2_restriction=False,points=len(df),cells=len(agg),scenarios_per_cell=9,
                  normalization='Value and rewards divided by 1-c1 during solution; advantages reported in original profit units.',
                  runtime_seconds=time.time()-start,timing_by_grid=stats,reference_validation=validation,
                  all_points_compared_2001_4001=True,extra_8001_points=extra,
                  structural_changes_2001_to_4001=int(structural.sum()),
                  max_root_movement_2001_to_4001=float(movement.max()),max_final_root_movement=float(df.root_movement.max()),
                  max_bellman_residual=float(df.bellman_residual.max()),uncertain_points=int(df.uncertain.sum()),
                  solver_tolerance=2e-11,classification_extrema_tolerance=1e-6,
                  topology_evaluation='All knots of piecewise linear continuation advantage, merging transformed knots within 1e-12.',
                  counts=df.groupby(['rho','regime']).size().unstack(fill_value=0).to_dict(),
                  aggregate_status_counts=agg.groupby(['rho','status']).size().unstack(fill_value=0).to_dict(),
                  warning='Finite numerical exploration, not a proof or an exhaustive baseline search. Counts among nine scenarios are not probabilities. No certified state discretization error bound.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',json.dumps({k:manifest[k] for k in ['points','cells','runtime_seconds','extra_8001_points',
                  'structural_changes_2001_to_4001','uncertain_points','max_bellman_residual','counts']}),flush=True)


if __name__=='__main__':main()
