#!/usr/bin/env python3
"""Reproducible mature-boundary parameter sweep; no production files modified.

Compiles an in-memory-use C++ Bellman solver to a temporary directory and validates
it against forgetting.Code.model.solve_boundary. All interpolation is piecewise linear
on a uniform state grid; demand is the repository's exact Beta-tail demand.
"""
from __future__ import annotations
import argparse, ctypes, json, sys, tempfile, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, boundary_demand, solve_boundary
OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map' / 'data'
CPP = r'''
#include <cmath>
#include <algorithm>
extern "C" int solve(int n,const int* si,const double* sw,const int* fi,const double* fw,const double* w,double p1,double p2,double r1,double r2,double gamma,double* v,double* adv,double* residual,int mode,double tol,int maxit){
  for(int k=0;k<maxit;k++){
    for(int z=0;z<n;z++){
      int j = (k%2==0)?z:n-1-z;
      double selfS=0,selfF=0,offS=0,offF=0;
      if(si[j]==j)selfS+=1-sw[j];else offS+=(1-sw[j])*v[si[j]];
      if(si[j]+1==j)selfS+=sw[j];else offS+=sw[j]*v[si[j]+1];
      if(fi[j]==j)selfF+=1-fw[j];else offF+=(1-fw[j])*v[fi[j]];
      if(fi[j]+1==j)selfF+=fw[j];else offF+=fw[j]*v[fi[j]+1];
      double a=w[j]*(r1+gamma*(p1*offS+(1-p1)*offF))/(1-w[j]*gamma*(p1*selfS+(1-p1)*selfF));
      double b=w[j]*(r2+gamma*(p2*offS+(1-p2)*offF))/(1-w[j]*gamma*(p2*selfS+(1-p2)*selfF));
      v[j]=(mode==1)?a:((mode==2)?b:std::max(a,b));
    }
    if(k%5==0 || k==maxit-1){
      double err=0;
      for(int j=0;j<n;j++){
        double vs=(1-sw[j])*v[si[j]]+sw[j]*v[si[j]+1];
        double vf=(1-fw[j])*v[fi[j]]+fw[j]*v[fi[j]+1];
        double a=r1+gamma*(p1*vs+(1-p1)*vf);
        double b=r2+gamma*(p2*vs+(1-p2)*vf);
        adv[j]=b-a;
        double target=w[j]*((mode==1)?a:((mode==2)?b:std::max(a,b)));
        err=std::max(err,std::abs(target-v[j]));
      }
      *residual=err;
      if(err<tol)return k+1;
    }
  }
  return -maxit;
}
'''
class Solver:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='threshold-sweep-')
        cpp=Path(self.tmp.name)/'sweep.cpp'; so=Path(self.tmp.name)/'sweep.so'
        cpp.write_text(CPP)
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
        self.lib=ctypes.CDLL(str(so)); self.fun=self.lib.solve
        P=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
        I=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
        self.fun.argtypes=[ctypes.c_int,I,P,I,P,P]+[ctypes.c_double]*5+[P,P,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_double,ctypes.c_int]
        self.fun.restype=ctypes.c_int
    def setup(self,rho,n):
        x=np.linspace(0,1,n); d=boundary_demand(x,rho,.5); w=d/(1-.98+.98*d)
        def ip(t):
            u=t*(n-1); idx=np.minimum(np.floor(u).astype(np.int32),n-2); return idx,np.ascontiguousarray(u-idx)
        si,sw=ip(rho*x+1-rho); fi,fw=ip(rho*x)
        return (x,si,sw,fi,fw,w)
    def solve(self,ctx,dp,dc,v=None,mode=0):
        x,si,sw,fi,fw,w=ctx
        if v is None:v=np.zeros(len(x))
        else:v=v.copy()
        adv=np.empty(len(x)); residual=ctypes.c_double()
        its=self.fun(len(x),si,sw,fi,fw,w,.35,.35+dp,.95,.95-dc,.98,v,adv,ctypes.byref(residual),mode,2e-11,10000)
        if its<0:raise RuntimeError(f'failed convergence {dp=} {dc=} {residual.value=}')
        return v,adv,residual.value,its

def topology_grid(ctx,rho):
    x=ctx[0];n=len(x)
    knots=np.unique(np.r_[0.,1.,(x-(1-rho))/rho,x/rho])
    knots=knots[(knots>=0)&(knots<=1)]
    # Analytically coincident transformed knots can differ by roundoff.
    knots=knots[np.r_[True,np.diff(knots)>1e-12]]
    knots[0]=0.;knots[-1]=1.
    def ip(t):
        u=t*(n-1);idx=np.minimum(np.floor(u).astype(np.int32),n-2);return idx,u-idx
    si,sw=ip(rho*knots+1-rho);fi,fw=ip(rho*knots)
    def advantage(v,dp,dc):
        return -dc+.98*dp*((1-sw)*v[si]+sw*v[si+1]-(1-fw)*v[fi]-fw*v[fi+1])
    return knots,advantage

def summary(x,a):
    active=a>0
    starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
    ends=np.flatnonzero(active & ~np.r_[active[1:],False])
    n=len(starts)
    def zero(j,k):return float(x[j]-a[j]*(x[k]-x[j])/(a[k]-a[j]))
    roots=[]
    for s,e in zip(starts,ends):
        roots.append([0. if s==0 else zero(s-1,s),1. if e==len(x)-1 else zero(e,e+1)])
    if n==0:reg='all1'
    elif n>1:reg='multiple'
    elif active[0] and active[-1]:reg='all2'
    elif active[0]:reg='one_upper'
    elif active[-1]:reg='one_lower'
    else:reg='two'
    # Near-zero extrema/endpoints, not ordinary root-adjacent grid samples.
    ext=np.flatnonzero(((a[1:-1]-a[:-2])*(a[2:]-a[1:-1])<=0))+1
    critical=np.r_[a[0],a[-1],a.max(),a.min(),a[ext]]
    uncertain=bool(np.any(np.abs(critical)<1e-6))
    return dict(regime=reg,n_components=n,x_lower=roots[0][0] if roots else np.nan,x_upper=roots[-1][1] if roots else np.nan,
                a0=float(a[0]),a1=float(a[-1]),a_max=float(a.max()),a_min=float(a.min()),uncertain=uncertain,
                components=json.dumps(roots,separators=(',',':')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); t0=time.time(); solver=Solver(); checks=[]
    points=[(.60,.45,.60),(.70,.15,.94),(.80,.64,.01),(.90,.45,.60),(.95,.45,.60),(.99,.45,.60),(.95,.15,.80),(.99,.64,.20),(.90,.15,.10),(.95,.45,.01),(.70,.61,.29)]
    for rho,dp,dc in points:
        ctx=solver.setup(rho,2001); v,a,res,its=solver.solve(ctx,dp,dc)
        prod=solve_boundary(Primitives(p2=.35+dp,c2=.05+dc),rho,central_nodes=2001,tolerance=2e-11)
        diff=float(np.max(np.abs(a-prod.advantage))); vd=float(np.max(np.abs(v-prod.value)))
        row=dict(rho=rho,delta_p=dp,delta_c=dc,advantage_max_difference=diff,value_max_difference=vd,bellman_residual=res,iterations=its,**summary(ctx[0],a));checks.append(row)
        print('validation',rho,dp,dc,row['regime'],f'da={diff:.2e}',f'dv={vd:.2e}',f'its={its}',flush=True)
        assert diff<2e-8 and vd<2e-8
    benchmark_start=time.time();ctx=solver.setup(.95,2001);v=None
    for dc in np.arange(1,95)/100:v,a,res,its=solver.solve(ctx,.45,float(dc),v)
    elapsed=time.time()-benchmark_start
    print(f'BENCHMARK 94 solves 2001 nodes: {elapsed:.3f}s; naive two-grid total estimate {elapsed*50*6*3/60:.1f}min',flush=True)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    if args.benchmark:return
    rhos=[.60,.70,.80,.90,.95,.99];dps=np.arange(15,65)/100;dcs=np.arange(1,95)/100
    coarse=[];fine=[];stats=[]
    def sweep_grid(rho,nodes):
        start=time.time();ctx=solver.setup(rho,nodes); rows=[]; shortcuts=0
        x,si,sw,fi,fw,w=ctx
        knots,advantage_at_knots=topology_grid(ctx,rho)
        def gap(v):
            return .98*((1-sw)*v[si]+sw*v[si+1]-(1-fw)*v[fi]-fw*v[fi+1])
        v1,a1,res1,it1=solver.solve(ctx,.45,.60,mode=1);g1=gap(v1)
        for di,dp in enumerate(dps):
            v2,a2,res2,it2=solver.solve(ctx,float(dp),.01,mode=2)
            unit2=v2/.94;g2unit=gap(unit2);v=None
            for dc in dcs:
                afree=-dc+dp*g1
                aall2=-dc+dp*(.95-dc)*g2unit
                if afree.max()<=-1e-9:
                    v=v1;a=afree;res=res1;its=0;shortcuts+=1
                elif aall2.min()>=1e-9:
                    v=unit2*(.95-dc);a=aall2;res=res2*(.95-dc)/.94;its=0;shortcuts+=1
                else:
                    v,a,res,its=solver.solve(ctx,float(dp),float(dc),v)
                ak=advantage_at_knots(v,dp,dc)
                rows.append(dict(rho=rho,p0=.5,p1=.35,c1=.05,delta_p=dp,delta_c=dc,nodes=nodes,bellman_residual=res,iterations=its,**summary(knots,ak)))
            if di%10==0:print(f'progress rho={rho} nodes={nodes} dp={dp:.2f} elapsed={time.time()-start:.1f}s',flush=True)
        stat=dict(rho=rho,nodes=nodes,seconds=time.time()-start,fixed_policy_shortcuts=shortcuts)
        print('completed',stat,flush=True)
        pd.DataFrame(rows).to_csv(OUT/f'grid_rho_{rho:.2f}_nodes_{nodes}.csv',index=False)
        return nodes,rows,stat
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(sweep_grid,rho,nodes) for rho in rhos for nodes in [2001,4001]]
        for fut in as_completed(pending):
            nodes,rows,stat=fut.result();stats.append(stat)
            (coarse if nodes==2001 else fine).extend(rows)
    coarse=sorted(coarse,key=lambda d:(d['rho'],d['delta_p'],d['delta_c']))
    fine=sorted(fine,key=lambda d:(d['rho'],d['delta_p'],d['delta_c']))
    pd.DataFrame(coarse).to_csv(OUT/'parameter_map_coarse.csv',index=False)
    pd.DataFrame(fine).to_csv(OUT/'parameter_map_fine.csv',index=False)
    df=pd.DataFrame(fine); old=pd.DataFrame(coarse);keys=['rho','delta_p','delta_c']
    joined=df.merge(old,on=keys,suffixes=('','_coarse'),validate='one_to_one')
    movement=np.maximum(np.abs(joined.x_lower-joined.x_lower_coarse).fillna(0),np.abs(joined.x_upper-joined.x_upper_coarse).fillna(0))
    structural=(joined.regime!=joined.regime_coarse)|(joined.n_components!=joined.n_components_coarse)
    df['root_movement']=movement;df['advantage_refinement_difference']=np.maximum(np.abs(joined.a_max-joined.a_max_coarse),np.abs(joined.a_min-joined.a_min_coarse))
    df['refinement']='2001_to_4001';df['coarse_regime']=joined.regime_coarse
    refine=structural|df.uncertain|(movement>.001)
    extra_count=0
    for idx in np.flatnonzero(refine):
        row=df.iloc[idx];ctx=solver.setup(row.rho,8001);v,a,res,its=solver.solve(ctx,row.delta_p,row.delta_c)
        knots,advantage_at_knots=topology_grid(ctx,row.rho)
        ss=summary(knots,advantage_at_knots(v,row.delta_p,row.delta_c)); prev=row.copy()
        for key,value in ss.items():df.at[idx,key]=value
        df.at[idx,'nodes']=8001;df.at[idx,'bellman_residual']=res;df.at[idx,'iterations']=its
        rm=max(abs(ss['x_lower']-prev.x_lower) if np.isfinite(ss['x_lower']) and np.isfinite(prev.x_lower) else 0, abs(ss['x_upper']-prev.x_upper) if np.isfinite(ss['x_upper']) and np.isfinite(prev.x_upper) else 0)
        df.at[idx,'root_movement']=rm;df.at[idx,'refinement']='2001_to_4001_to_8001'
        df.at[idx,'uncertain']=ss['uncertain'] or (ss['regime']!=prev.regime) or rm>.001
        extra_count+=1
    df['detected_regime']=df.regime
    df.loc[df.uncertain,'regime']='uncertain'
    df.to_csv(OUT/'parameter_map.csv',index=False)
    curves=[];baseline=[]
    for rho in rhos:
        ctx=solver.setup(rho,8001);v,a,res,its=solver.solve(ctx,.45,.60)
        curves.append(pd.DataFrame(dict(rho=rho,x=ctx[0],advantage=a,value=v)))
        knots,advantage_at_knots=topology_grid(ctx,rho)
        baseline.append(dict(rho=rho,delta_p=.45,delta_c=.60,nodes=8001,bellman_residual=res,**summary(knots,advantage_at_knots(v,.45,.60))))
    pd.concat(curves).to_csv(OUT/'baseline_curves.csv',index=False)
    pd.DataFrame(baseline).to_csv(OUT/'baseline_summary.csv',index=False)
    manifest=dict(R=1,gamma=.98,p0=.5,p1=.35,c1=.05,rhos=rhos,delta_p_grid=dps.tolist(),delta_c_grid=dcs.tolist(),points=len(df),
                  validation=checks,timing_by_grid=stats,runtime_seconds=time.time()-t0,solver_tolerance=2e-11,classification_extrema_tolerance=1e-6,
                  topology_evaluation='All knots of the piecewise-linear continuation advantage induced by the interpolated value function',all_points_compared_2001_4001=True,extra_8001_points=extra_count,structural_changes_2001_to_4001=int(structural.sum()),
                  max_root_movement_2001_to_4001=float(movement.max()),max_final_root_movement=float(df.root_movement.max()),
                  uncertain_points=int(df.uncertain.sum()),max_bellman_residual=float(df.bellman_residual.max()),counts=df.groupby(['rho','regime']).size().unstack(fill_value=0).to_dict(),
                  warning='Numerical exploration of the mature boundary with linear state interpolation; no analytical proof or certified discretization error bound.')
    frontier_path=OUT/'fixed_policy_frontier.csv'
    if frontier_path.exists():
        frontier=pd.read_csv(frontier_path)
        checked=df.merge(frontier[['rho','K_rho']],on='rho')
        margin=checked.K_rho*checked.delta_p-checked.delta_c
        consistency=dict(mismatches=int(((checked.detected_regime=='all1')!=(margin<=0)).sum()),minimum_absolute_frontier_margin=float(abs(margin).min()),points=len(checked))
        manifest['frontier_consistency']=consistency
        (OUT/'frontier_consistency.json').write_text(json.dumps(consistency,indent=2))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('DONE',json.dumps({k:manifest[k] for k in ['points','runtime_seconds','extra_8001_points','structural_changes_2001_to_4001','uncertain_points','max_bellman_residual','max_final_root_movement']}),flush=True)
if __name__=='__main__':main()
