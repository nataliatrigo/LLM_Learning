from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solve_fluid import solve,OUT,DELTA,R,C1

constant=solve(lambda x: np.full_like(x,.4),.2,n=501)
assert not np.any(constant['policy'])
np.testing.assert_allclose(constant['value'],.4*(R-C1)/DELTA,rtol=2e-12,atol=2e-12)

df=pd.read_csv(OUT/'grid_results.csv')
fine=df[df.n==8001]
checks=json.loads((OUT/'characteristic_checks.json').read_text())
rows=[]
for case in checks:
    row=fine[(fine.demand=='beta') & (fine.p0==.5) & (fine.rho==case['rho'])].iloc[0]
    direct=json.loads(row.thresholds)
    independent=([case['lower']] if case.get('lower') is not None else [])+([case['upper']] if case.get('upper') is not None else [])
    rows.append(dict(rho=case['rho'],upwind_thresholds=direct,characteristic_thresholds=independent,max_difference=max((abs(a-b) for a,b in zip(direct,independent)),default=0),matching_count=len(direct)==len(independent)))
assert all(row['matching_count'] for row in rows)
summary=dict(beta_cases=35,logistic_cases=7,grids=[2001,4001,8001],max_discrete_hjb_residual=float(df.residual.max()),max_4001_to_8001_threshold_change=float(fine.root_change.max()),independent_checks=rows,constant_demand_exact_case_passed=True,numpy=np.__version__,scipy=scipy.__version__,original_note_sha256=hashlib.sha256((Path(__file__).resolve().parents[3] / 'Progress notes/Progress_notes_fluid/Fluid model.tex').read_bytes()).hexdigest())
(OUT/'validation.json').write_text(json.dumps(summary,indent=2))

fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
rhos=[.5,.7,.8,.9,.95,.99,.999]
for ax,kind,title in zip(axs,['beta','logistic'],['Demanda Beta','Aproximación logística']):
    for j,rho in enumerate(rhos):
        row=fine[(fine.demand==kind)&(fine.p0==.5)&(fine.rho==rho)].iloc[0]
        bands=json.loads(row.bands)
        ax.hlines(j,0,1,color='#e4e8ec',lw=15,zorder=0)
        for left,right in bands:
            ax.plot([left,right],[j,j],color='#2b8466',lw=15,solid_capstyle='butt')
        roots=json.loads(row.thresholds)
        for x in roots:
            ax.plot(x,j,'|',color='#153e33',markersize=18,markeredgewidth=1)
        if not bands:
            ax.text(.5,j,'Producto 1 en todo el intervalo',ha='center',va='center',fontsize=8,color='#4f5b66')
    ax.axvline(.35,color='#82939d',lw=.8,ls='--',zorder=0)
    ax.axvline(.8,color='#82939d',lw=.8,ls='--',zorder=0)
    ax.set_title(title,loc='left',fontsize=12)
    ax.set(xlim=(0,1),xlabel='Estado x',yticks=range(len(rhos)),yticklabels=[f'{r:g}' for r in rhos])
    ax.invert_yaxis()
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
axs[0].set_ylabel(r'$\rho$')
fig.suptitle('Modelo fluido: intervalos donde se elige el producto 2',fontsize=14,x=.05,ha='left')
fig.text(.05,-.03,r'$p_0=0.5$, $p_1=0.35$, $p_2=0.80$, $c_1=0.05$, $c_2=0.65$, $R=1$; $\delta=-\log(0.98)$, $\lambda=-\log(\rho)$. Evidencia numérica.',fontsize=9)
fig.savefig(OUT/'policy_intervals.png',dpi=180,bbox_inches='tight')
print(json.dumps(summary,indent=2))
