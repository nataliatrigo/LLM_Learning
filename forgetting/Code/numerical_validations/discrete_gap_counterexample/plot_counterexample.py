from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[4] / 'forgetting/outputs/numerical_validations/discrete_gap_counterexample'
s=np.load(P/'counterexample_value.npz'); x=s['x']; v=s['value']
g=np.interp(.7*x+.3,x,v)-np.interp(.7*x,x,v)
c=json.loads((P/'certificate.json').read_text()); rows=c['points']
xx=np.array([r['x_decimal'] for r in rows]); yy=np.array([r['gap_approx'] for r in rows]); err=c['gap_error_upper']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1,1.35]})
for ax in axes:
 ax.plot(x,g,color='#1d5678',lw=2)
 ax.set_xlabel('$x$');ax.grid(alpha=.17)
axes[0].set_title('Full domain');axes[0].set_xlim(0,1)
axes[0].set_ylabel(r'$V_\rho(\rho x+1-\rho)-V_\rho(\rho x)$')
axes[0].axvspan(.29,.38,color='#de8545',alpha=.15)
axes[1].set_title('Certified valley');axes[1].set_xlim(.29,.38);axes[1].set_ylim(1.279,1.2825)
axes[1].errorbar(xx,yy,yerr=err,fmt='o',color='#bd3e33',capsize=4,ms=5,label='Certified error bounds')
for a,b in zip(xx,yy): axes[1].annotate(f'x = {a:g}',(a,b),xytext=(0,12 if a!=.3375 else -25),textcoords='offset points',ha='center',fontsize=10)
axes[1].ticklabel_format(useOffset=False,axis='y');axes[1].legend(loc='lower left',frameon=False,fontsize=9)
fig.suptitle('The continuation gap need not be unimodal',fontsize=15,y=.98)
fig.text(.5,.015,r'$\rho=0.7,\ p_0=0.7,\ p_1=0.1,\ p_2=0.9,\ c_1=0,\ c_2=0.925,\ R=1,\ \gamma=0.95$',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.055,1,.94))
fig.savefig(P/'counterexample.png',dpi=200)
fig.savefig(P/'counterexample.svg')
