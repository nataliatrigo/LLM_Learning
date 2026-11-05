"""Figures for exact count policies with uniform random replacement."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

PRODUCT1 = '#2563eb'
PRODUCT2 = '#16a34a'


def configure_style():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 180,
                         'axes.grid': True, 'grid.alpha': .2})


def _save(fig, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f'{stem}.png'
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


def plot_interactions(records, solution, output_dir, stem=None, *, trajectory=1):
    """Attached-plot analogue: every pre-choice point of customer 1, unthinned."""
    configure_style()
    n, p0 = solution.n, solution.primitives.p0
    if not 1 <= trajectory <= records.shape[1]:
        raise ValueError('trajectory is one-based and outside the record array')
    stem = stem or f'individual_states_p0_{p0:g}_n_{n}'
    sample = records[:, trajectory-1]
    count = len(sample)
    interactions = np.arange(1, count+1)
    states, action = sample['S_before'], sample['product2']
    files = []
    for limit, suffix in [(count, ''), (min(2000, count), '_first_2000')]:
        if suffix and count <= 2000:
            continue
        fig, ax = plt.subplots(figsize=(13, 4.5), layout='constrained')
        ax.plot(interactions[:limit], states[:limit], color='#94a3b8', lw=.55, zorder=1)
        for mask, color, label in [(~action[:limit], PRODUCT1, 'Seller chose product 1'),
                                   (action[:limit], PRODUCT2, 'Seller chose product 2')]:
            ax.scatter(interactions[:limit][mask], states[:limit][mask],
                       color=color, s=2 if limit>5000 else 8, label=label, zorder=2)
        ax.set(xlabel='Interaction with A', ylabel='Remembered successes S at interaction start',
               title=f'Uniform random replacement · Customer {trajectory} · p₀ = {p0:g}\nMemory capacity n = {n} · Interactions 1–{limit:,}')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=.01, y=.15)
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.18), ncols=2, frameon=False)
        files.append(_save(fig, output_dir, stem+suffix))
    counts1 = np.bincount(states[~action].astype(int), minlength=n+1)
    counts2 = np.bincount(states[action].astype(int), minlength=n+1)
    fig, ax = plt.subplots(figsize=(10, 4.2), layout='constrained')
    occupied = np.flatnonzero(counts1+counts2)
    ax.bar(occupied, counts1[occupied], color=PRODUCT1, label='Seller chose product 1', width=.85)
    ax.bar(occupied, counts2[occupied], bottom=counts1[occupied], color=PRODUCT2,
           label='Seller chose product 2', width=.85)
    ax.set(xlabel='Remembered successes S at interaction start', ylabel='Number of visits',
           title=f'Uniform random replacement · Customer {trajectory} · p₀ = {p0:g}\nMemory capacity n = {n} · {count:,} interactions')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    files.append(_save(fig, output_dir, stem+'_state_distribution'))
    return files


def plot_policy_summary(frame, output_dir, stem='optimal_policies_by_count'):
    """Exact mature policy for every success count, capacity and p0.

    Required columns: n,p0,S,action_min,action_max.  The last two columns use
    0/1 for product 2 and must agree because counts are the complete state.
    """
    configure_style()
    frame = frame.copy()
    aliases = {'s':'S', 'successes':'S', 'min_action':'action_min',
               'max_action':'action_max', 'product2_min':'action_min', 'product2_max':'action_max'}
    frame.rename(columns={k:v for k,v in aliases.items() if k in frame and v not in frame}, inplace=True)
    if np.any(frame.action_min.to_numpy() != frame.action_max.to_numpy()):
        raise ValueError('count-only policies must prescribe one action per count')
    p0s = sorted(frame.p0.unique())
    ns = sorted(frame.n.unique())
    fig, axes = plt.subplots(1,len(p0s),figsize=(max(7,3.1*len(p0s)),max(4,.47*len(ns))),
                             sharey=True,squeeze=False,layout='constrained')
    for ax,p0 in zip(axes.flat,p0s):
        subset = frame[frame.p0==p0]
        for row,n in enumerate(ns):
            data = subset[subset.n==n].sort_values('S')
            if data.empty:
                ax.text(.5,row,'—',color='#94a3b8',ha='center',va='center')
                continue
            classification = (data.action_max.to_numpy()>0).astype(int)
            for kind,color in enumerate((PRODUCT1,PRODUCT2)):
                mask = classification==kind
                ax.scatter(data.S.to_numpy()[mask]/n,np.full(mask.sum(),row),
                           color=color,marker='s',s=max(8,min(110,500/n)),linewidths=0)
        ax.set(title=rf'$p_0={p0:g}$',xlabel='Remembered success fraction S/n',xlim=(-.025,1.025))
        ax.set_yticks(range(len(ns)),[str(n) for n in ns])
        ax.grid(axis='y',visible=False)
    axes[0,0].set_ylabel('Memory capacity n')
    handles=[Line2D([0],[0],marker='s',ls='',color=c,label=l) for c,l in
             [(PRODUCT1,'Product 1'),(PRODUCT2,'Product 2')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,fontsize=9)
    fig.suptitle('Uniform random replacement · Exact optimal mature policies\n— = case outside the experiment')
    return [_save(fig,output_dir,stem)]


def plot_monte_carlo(periods, binned, output_dir):
    """Paper-style calendar MC panels and across-window comparisons with bin CIs."""
    configure_style()
    periods, binned = periods.copy(), binned.copy()
    files=[]
    p0s,ns=sorted(periods.p0.unique()),sorted(periods.n.unique())
    colors=plt.colormaps['viridis'](np.linspace(.06,.9,len(ns)))
    fig,axes=plt.subplots(2,len(p0s),figsize=(3.2*len(p0s),6.6),sharex=True,sharey=True,squeeze=False)
    metrics=('a_market_share','product2_unconditional_rate')
    for col,p0 in enumerate(p0s):
        for n,color in zip(ns,colors):
            data=binned[(binned.p0==p0)&(binned.n==n)].sort_values('bin_center')
            if data.empty:
                continue
            for row,metric in enumerate(metrics):
                x=data.bin_center.to_numpy()+1
                axes[row,col].plot(x,data[metric],lw=1.25,color=color,label=f'n={n}')
                if metric+'_ci_lower' in data:
                    axes[row,col].fill_between(x,data[metric+'_ci_lower'].to_numpy(),
                                              data[metric+'_ci_upper'].to_numpy(),color=color,alpha=.10,lw=0)
        axes[0,col].set_title(rf'$p_0={p0:g}$')
        axes[1,col].set_xlabel('Calendar period')
    axes[0,0].set_ylabel('Seller A market share')
    axes[1,0].set_ylabel('Share of paths using product 2')
    for ax in axes.flat:
        ax.set_ylim(-.02,1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncols=min(8,len(ns)))
    fig.suptitle('Uniform random replacement · Exact count policies\nMonte Carlo: 25-period bins and 95% path-clustered intervals',fontsize=11)
    fig.subplots_adjust(left=.055,right=.995,bottom=.14,top=.84,hspace=.16,wspace=.13)
    files.append(_save(fig,output_dir,'monte_carlo_by_window'))
    pcolors=plt.colormaps['viridis'](np.linspace(.08,.88,len(p0s)))
    for n in ns:
        fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        subset=periods[periods.n==n]
        for p0,color in zip(p0s,pcolors):
            data=subset[subset.p0==p0].sort_values('period')
            for ax,metric in zip(axes,metrics):
                ax.plot(data.period+1,data[metric],color=color,lw=.9,label=rf'$p_0={p0:g}$')
        for ax,label in zip(axes,('Seller A demand (market share)','Share of paths using product 2')):
            ax.set(xlabel='Calendar period',ylabel=label,ylim=(-.02,1.02))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(f'Uniform random replacement · Exact optimal policy · Memory capacity n = {n}')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncols=len(p0s))
        fig.tight_layout(rect=(0,.08,1,.93))
        files.append(_save(fig,output_dir,f'monte_carlo_p0_panels_n_{n}'))
    return files


def plot_long_interaction_diagnostics(blocks,cumulative,stability,output_dir,stem='long_interaction'):
    configure_style()
    files=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    path_columns=[c for c in cumulative if c!='interactions']
    for column in path_columns[::max(1,len(path_columns)//10)]:
        axes[0].plot(cumulative.interactions,cumulative[column],lw=.8,alpha=.8)
    axes[0].set(xlabel='Interactions with A',ylabel='Cumulative product-2 frequency')
    for path in sorted(blocks.trajectory.unique())[::max(1,blocks.trajectory.nunique()//10)]:
        data=blocks[blocks.trajectory==path]
        axes[1].plot(data.end_interaction,data.frequency,'o-',ms=2,lw=.7)
    axes[1].set(xlabel='Block end: interactions with A',ylabel='Product-2 frequency within block')
    axes[2].scatter(stability.first_segment_frequency,stability.second_segment_frequency,s=12,alpha=.7)
    allvalues=np.r_[stability.first_segment_frequency,stability.second_segment_frequency]
    lower,upper=max(0,allvalues.min()-.01),min(1,allvalues.max()+.01)
    axes[2].plot([lower,upper],[lower,upper],ls='--',color='#64748b',lw=1)
    axes[2].set(xlabel='First post-burn-in segment frequency',ylabel='Second segment frequency',
                xlim=(lower,upper),ylim=(lower,upper))
    fig.suptitle('Uniform random replacement · Long interaction diagnostics')
    files.append(_save(fig,output_dir,stem+'_diagnostics'))
    return files


def geometric_reference_comparison(original_records, replacement_records, output_dir,
                                   *, rho=.95, n=20, p0=.5, count=20_000):
    """Compare the saved geometric history with uniform random replacement.

    The reference need not share random numbers with the new simulation.
    Both panels report pre-choice S in evidence units, with identical axes.
    """
    configure_style()
    old = original_records[:, 0] if original_records.ndim == 2 else original_records
    new = replacement_records[:, 0] if replacement_records.ndim == 2 else replacement_records
    count = min(count, len(old), len(new))
    time = np.arange(1,count+1)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    for ax,states,action,title in [
        (axes[0],old['x_before'][:count]/(1-rho),old['product2'][:count],
         f'Geometric forgetting · ρ = {rho:g} · Original saved reference'),
        (axes[1],new['S_before'][:count],new['product2'][:count],
         f'Uniform random replacement · Memory capacity n = {n} · New simulation'),
    ]:
        action=action.astype(bool)
        ax.plot(time,states,color='#94a3b8',lw=.4,zorder=1)
        for mask,color,label in [(~action,PRODUCT1,'Seller chose product 1'),
                                  (action,PRODUCT2,'Seller chose product 2')]:
            ax.scatter(time[mask],states[mask],s=2,color=color,label=label,zorder=2)
        ax.set(title=title,ylabel='State S at interaction start')
        ax.margins(x=.01,y=.1)
    axes[-1].set_xlabel('Interaction with A')
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
    fig.suptitle(f'Customer 1 · p₀ = {p0:g} · Interactions 1–{count:,} · Equal mature evidence')
    return _save(fig,output_dir,f'geometric_vs_random_replacement_p0_{p0:g}_rho_{rho:g}_n_{n}')


def plot_replacement_comparison(old_records, new_records, output_dir,
                                *, n=20, p0=.5, count=20_000):
    """Compare customer 1 under opposite and uniform random replacement.

    The panels are descriptive histories, not a paired statistical estimator.
    """
    configure_style()
    old = old_records[:,0] if old_records.ndim == 2 else old_records
    new = new_records[:,0] if new_records.ndim == 2 else new_records
    count = min(count,len(old),len(new))
    time = np.arange(1,count+1)
    fig,axes = plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True,
                            layout='constrained')
    for ax,sample,title in [
        (axes[0],old,'Opposite-outcome replacement · Previous experiment'),
        (axes[1],new,'Uniform random replacement · New experiment'),
    ]:
        states = sample['S_before'][:count]
        choice = sample['product2'][:count].astype(bool)
        ax.plot(time,states,color='#94a3b8',lw=.4,zorder=1)
        for mask,color,label in [
            (~choice,PRODUCT1,'Seller chose product 1'),
            (choice,PRODUCT2,'Seller chose product 2'),
        ]:
            ax.scatter(time[mask],states[mask],s=2,color=color,label=label,zorder=2)
        ax.set(title=title)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=.01,y=.1)
    fig.supylabel('Remembered successes S at interaction start')
    axes[-1].set_xlabel('Interaction with A')
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
    handles,labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
    fig.suptitle(f'Customer 1 · p₀ = {p0:g} · Memory capacity n = {n}\n'
                 f'Interactions 1–{count:,} · Descriptive comparison, unpaired random streams')
    return _save(fig,output_dir,
                 f'opposite_vs_random_replacement_p0_{p0:g}_n_{n}')


def plot_policy_detail(solution, output_dir, stem=None):
    """Show the exact mature action gap and the policy at every count S."""
    configure_style()
    n, p0 = solution.n, solution.primitives.p0
    s = np.arange(n+1)
    gap = np.asarray(solution.advantages[n])
    if gap.shape != (n+1,):
        raise ValueError('policy detail requires one exact action gap per success count')
    choice = gap > solution.action_tolerance
    active = np.flatnonzero(choice)
    regions = [region for region in np.split(active,np.flatnonzero(np.diff(active)>1)+1) if len(region)]
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True,
                           gridspec_kw={'height_ratios':[3.5,1]},layout='constrained')
    axes[0].plot(s,gap,color='#94a3b8',lw=1,zorder=1)
    axes[0].axhline(0,color='#475569',ls='--',lw=1,label='Indifference')
    for mask,color in ((~choice,PRODUCT1),(choice,PRODUCT2)):
        axes[0].scatter(s[mask],gap[mask],color=color,s=18,zorder=2)
    for region in regions:
        left,right=region[0]-.5,region[-1]+.5
        axes[0].axvspan(left,right,color=PRODUCT2,alpha=.08,zorder=0)
    axes[0].set(ylabel='Product-2 advantage Q₂ − Q₁')
    axes[1].bar(s,np.ones(n+1),width=.98,color=np.where(choice,PRODUCT2,PRODUCT1),linewidth=0)
    axes[1].set(xlabel='Remembered successes S',ylabel='Optimal\nproduct',ylim=(0,1),xlim=(-.5,n+.5))
    axes[1].set_yticks([])
    axes[1].grid(False)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=min(11,n+1),integer=True))
    band_text = '; '.join(f'S = {int(z[0])}' if len(z)==1 else f'S = {int(z[0])}–{int(z[-1])}' for z in regions)
    axes[0].set_title('Product 2: '+band_text if regions else 'Product 1 at every mature count')
    fig.suptitle(f'Uniform random replacement · Exact optimal mature policy\np₀ = {p0:g} · Memory capacity n = {n}')
    handles=[Line2D([0],[0],marker='s',ls='',color=c,label=l) for c,l in
             [(PRODUCT1,'Product 1'),(PRODUCT2,'Product 2')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
    return _save(fig,output_dir,stem or f'exact_policy_detail_p0_{p0:g}_n_{n}')
