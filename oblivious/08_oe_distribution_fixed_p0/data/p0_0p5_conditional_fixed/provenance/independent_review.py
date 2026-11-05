"""Read-only verification of all fixed-p0 conditional results and grid checks.

Run from the repository root with one BLAS thread after every writer closes:
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -u \
oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_conditional_fixed/provenance/independent_review.py

The audit enumerates selection/feedback/interpolation events independently of
Game.transition and verifies actual values, stationarity, own Bellman regret,
conditional rows, economic exports, all provenance hashes and both grid pairs.
"""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_distribution_fixed_p0.code.conditional_run import source_fingerprints
from oblivious.oe_distribution_fixed_p0.code.conditional_refine_case import source_fingerprints as refinement_sources
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_matrix
from oblivious.oe_p0.code.primitives import Parameters
ROOT=next(path for path in [Path.cwd(),*Path(__file__).resolve().parents] if (path/'oblivious/08_oe_distribution_fixed_p0').is_dir())
BASE=ROOT/'oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_conditional_fixed'

def update_indices(grid,coordinates):
    hi=np.searchsorted(grid,coordinates,side='right');hi=np.clip(hi,1,len(grid)-1);lo=hi-1
    w=(coordinates-grid[lo])/(grid[hi]-grid[lo])
    return lo,hi,w

def manual_kernel_checks(grid,params,D,outside,alpha,nu,values):
    n=len(grid); i,j=np.indices((n,n)); success=params.p1+(params.p2-params.p1)*alpha
    events=[]
    for seller in (0,1):
        own=i if seller==0 else j; demand=D if seller==0 else D.T
        for feedback in (0,1):
            coord=params.rho*grid+(1-params.rho)*feedback
            lo,hi,w=update_indices(grid,coord)
            prob=demand*(success[own] if feedback else 1-success[own])
            for target,weight in ((lo,1-w),(hi,w)):
                ii,jj=(target[own],j) if seller==0 else (i,target[own])
                events.append((ii,jj,prob*weight[own]))
    push=nu*outside
    pull=values*outside[None,:,:]
    rows=outside.copy()
    for ii,jj,prob in events:
        np.add.at(push,(ii,jj),nu*prob)
        pull+=values[:,ii,jj]*prob[None,:,:]
        rows+=prob
    rewards=np.stack((D*(params.R-params.c1-(params.c2-params.c1)*alpha[:,None]),D.T*(params.R-params.c1-(params.c2-params.c1)*alpha[None,:])))
    residual=values-rewards-params.gamma*pull
    return dict(manual_stationary_residual_l1=float(abs(push-nu).sum()),
                manual_row_stochasticity=float(abs(rows-1).max()),
                manual_actual_value_residual_max=float(abs(residual).max()))

def validate_stability(directory,manifest):
    info=manifest['grid_stability'];rho=info['rho'];coarse=info['coarse_n'];fine=info['fine_n']
    def outcome(n):
        own_dir=directory if n==fine or n in manifest['configuration']['grids'] else BASE/('rho_'+str(rho).replace('.','p'))
        tag='rho_'+str(rho).replace('.','p')+f'_n{n:03d}'
        return load_bundle(own_dir/'arrays'/tag)['outcomes'][0]
    a,b=outcome(coarse),outcome(fine)
    probs={k:abs(b[k]-a[k]) for k in ('product2_conditional','product2_calendar','sale_rate')}
    rels={k:abs(b[k]-a[k])/max(abs(a[k]),abs(b[k])) if a[k] or b[k] else 0. for k in ('profit_calendar','value_initial')}
    passed=all(v<=.05 for v in [*probs.values(),*rels.values()])
    assert passed==info['passed'],info
    for k,v in probs.items():assert abs(v-info[k+'_absolute_change'])<1e-14,info
    for k,v in rels.items():assert abs(v-info[k+'_relative_change'])<1e-14,info
    assert abs(100*max(probs.values())-info['max_probability_change_pp'])<1e-12,info
    assert abs(100*max(rels.values())-info['max_earnings_relative_change_pct'])<1e-12,info
    return True

def run_audit():
 rows=[]; inputs={}; stability=[]
 directories=sorted(BASE.glob('rho_*'))+[BASE.parent/'p0_0p5_conditional_refinement/rho_0p97']
 for directory in directories:
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest['current_grid'] is None and manifest['status'] in ('grid_stable_under_declared_thresholds','certified_but_grid_sensitive'),directory
    validation=validate_stability(directory,manifest)
    stability.append(dict(manifest['grid_stability'],independently_recomputed=validation))
    inputs[str(manifest_path.relative_to(ROOT))]=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_sources=refinement_sources() if 'p0_0p5_conditional_refinement' in directory.parts else source_fingerprints()
    assert manifest['numerical_sources']==expected_sources,directory
    artifacts={}
    for field in ('reference_input_sha256','positive_proposal_sha256'):
        artifacts.update(manifest.get(field,{}))
    for artifact,expected in manifest.get('diagnostic_artifact_sha256',{}).items():
        artifacts[str((directory/artifact).relative_to(ROOT))]=expected
    for source,expected in artifacts.items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
        inputs[source]=expected
    for tag in manifest['completed']:
        for suffix in ('.json','.npz'):
            path=directory/'arrays'/(tag+suffix)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['checkpoint_sha256'][path.name],path
            inputs[str(path.relative_to(ROOT))]=manifest['checkpoint_sha256'][path.name]
        case=load_bundle(directory/'arrays'/tag)
        for source,expected in case.get('refinement_metadata',{}).get('proposal_provenance_sha256',{}).items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
            inputs[source]=expected
        c=case['competition'];n=case['n'];rho=case['rho'];alpha=c['policy'];grid=c['grid'];nu=c['nu'];p=Parameters(rho=rho)
        assert c['verified'] and case['demand_audit']['verified']
        D,_=outside_demand_matrix(grid,p,.5,256)
        from scipy.special import betainc
        from oblivious.oe_p0.code.primitives import posterior
        a,b=posterior(grid,p);f=betainc(a,b,.5);outside=np.outer(f,f)
        mu=nu.sum(axis=1);conditional=nu/mu[:,None];d=(conditional*D).sum(axis=1)
        row=dict(rho=rho,n=n,**manual_kernel_checks(grid,p,D,outside,alpha,nu,c['actual_values']))
        row['conditional_row_l1_max']=float(abs(conditional-c['conditional_mu']).sum(axis=1).max())
        row['conditional_demand_max']=float(abs(d-c['demand']).max())
        row['independent_positive_conditional_order_l1']=c['conditional_diagnostics']['independent_order_supported_row_l1_max']
        row['dp_regret']=c['dp_diagnostics']['conditional_regret']
        # Independent own-state calendar Bellman using hand interpolation.
        lo_s,hi_s,w_s=update_indices(grid,p.rho*grid+1-p.rho);lo_f,hi_f,w_f=update_indices(grid,p.rho*grid)
        value=c['value'];vs=(1-w_s)*value[lo_s]+w_s*value[hi_s];vf=(1-w_f)*value[lo_f]+w_f*value[hi_f]
        advantage=-(p.c2-p.c1)+p.gamma*(p.p2-p.p1)*(vs-vf)
        quality=p.p1+(p.p2-p.p1)*alpha
        bell=d*(p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(quality*vs+(1-quality)*vf))+p.gamma*(1-d)*value
        row['manual_own_bellman_residual']=float(abs(bell-value).max())
        row['manual_conditional_regret']=float((np.maximum(advantage,0)-alpha*advantage).max())
        row['manual_advantage_difference']=float(abs(advantage-c['advantage']).max())
        outcome=case['outcomes'][0];sold=(nu*D).sum(axis=1);rate=sold.sum();p2=float(sold@alpha/rate);profit=float(sold@(p.R-p.c1-(p.c2-p.c1)*alpha));initial=float(c['actual_values'][:,n//2,n//2].mean())
        row['outcome_discrepancy_max']=max(abs(p2-outcome['product2_conditional']),abs(profit-outcome['profit_calendar']),abs(rate-outcome['sale_rate']),abs(initial-outcome['value_initial']))
        row['joint_nonproduct_l1']=float(abs(nu-np.outer(mu,mu)).sum())
        row['demand_conditional_marginal_difference_max']=float(abs(d-D@mu).max())
        row['minimum_own_mass']=float(mu.min());row['mixed_nodes']=int(np.sum((alpha>0)&(alpha<1)))
        assert row['manual_stationary_residual_l1']<1e-11,row
        assert row['manual_row_stochasticity']<1e-12,row
        assert row['manual_actual_value_residual_max']<=1e-10,row
        assert row['manual_own_bellman_residual']<=1e-10,row
        assert row['manual_conditional_regret']<=1e-10,row
        assert row['conditional_row_l1_max']<=1e-7 and row['independent_positive_conditional_order_l1']<=1e-7,row
        assert row['outcome_discrepancy_max']<1e-11,row
        rows.append(row); print(json.dumps(row),flush=True)
 assert len(rows)==13,len(rows)
 for path,expected in inputs.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
 return dict(verified=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),audit_source=str(Path(__file__).resolve().relative_to(ROOT)),number_cases=len(rows),cases=rows,input_sha256=inputs,numerical_sources=refinement_sources(),grid_stability=stability,reviewer_roles={'rho_0.6_0.7_0.8_0.9':'Independent reviewer; did not solve these cases','rho_0.95_0.97_base':'Operated production solver and positive proposal/attachment; additional manual event/interpolation checks use independent implementation','rho_0.97_n151':'Independent reviewer; another agent operated this refinement'},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Read-only independent event enumeration and interpolation; current numerical-source and input hashes; actual-value residuals; own Bellman regret; positive rowwise closure; reported outcomes')
if __name__=='__main__':
 result=run_audit()
 pending=BASE/'pending_independent_review.json';pending.write_text(json.dumps(result,indent=2)+'\n');pending.replace(BASE/'independent_review.json')
