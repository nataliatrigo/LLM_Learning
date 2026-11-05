"""Static figures for fixed-window monopoly policies and real Monte Carlo."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

PRODUCT1 = '#2563eb'
PRODUCT2 = '#16a34a'
HISTORY = '#d97706'


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


def _method(solution):
    return str(getattr(solution, 'method', getattr(solution, 'diagnostics', {}).get('method', 'exact_fifo')))


def _approximate(method):
    method = str(method).lower()
    return 'approx' in method or 'project' in method


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
    qualifier = ' · Approximate policy' if _approximate(_method(solution)) else ''
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
               title=f'Customer {trajectory} · p₀ = {p0:g} · n = {n} · Interactions 1–{limit:,}{qualifier}')
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
           title=f'Customer {trajectory} · p₀ = {p0:g} · n = {n} · {count:,} interactions{qualifier}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    files.append(_save(fig, output_dir, stem+'_state_distribution'))
    return files


def plot_policy_summary(frame, output_dir, stem='optimal_policies_by_count'):
    """Count projection of FIFO policy; amber explicitly preserves order effects.

    Required columns: n,p0,S,action_min,action_max. Actions use 0/1 for product2.
    One row per count and n/p0, with min/max over every ordered history available
    in the solved state representation. Optional policy_method marks approximations.
    """
    configure_style()
    frame = frame.copy()
    aliases = {'s':'S', 'successes':'S', 'min_action':'action_min',
               'max_action':'action_max', 'product2_min':'action_min', 'product2_max':'action_max'}
    frame.rename(columns={k:v for k,v in aliases.items() if k in frame and v not in frame}, inplace=True)
    if 'policy_method' not in frame and 'method' in frame:
        frame['policy_method'] = frame['method']
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
            classification = np.where(data.action_min.to_numpy()!=data.action_max.to_numpy(), 2,
                                      np.where(data.action_max.to_numpy()>0,1,0))
            for kind,color in enumerate((PRODUCT1,PRODUCT2,HISTORY)):
                mask = classification==kind
                ax.scatter(data.S.to_numpy()[mask]/n,np.full(mask.sum(),row),
                           color=color,marker='s',s=max(8,min(110,500/n)),linewidths=0)
        ax.set(title=rf'$p_0={p0:g}$',xlabel='Remembered success fraction S/n',xlim=(-.025,1.025))
        ax.set_yticks(range(len(ns)),[str(n) for n in ns])
        ax.grid(axis='y',visible=False)
    axes[0,0].set_ylabel('Window length n')
    handles=[Line2D([0],[0],marker='s',ls='',color=c,label=l) for c,l in
             [(PRODUCT1,'Product 1 for all histories'),(PRODUCT2,'Product 2 for all histories'),
              (HISTORY,'Choice depends on outcome order')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=3,fontsize=9)
    approx = '\nLarge windows: approximate policies · — = case outside the experiment' if 'policy_method' in frame and frame.policy_method.map(_approximate).any() else ''
    fig.suptitle('Fixed-window policy projected by success count'+approx)
    return [_save(fig,output_dir,stem)]


def plot_monte_carlo(periods, binned, output_dir):
    """Paper-style calendar MC panels and across-window comparisons with bin CIs."""
    configure_style()
    periods, binned = periods.copy(), binned.copy()
    for frame in (periods,binned):
        if 'policy_method' not in frame and 'method' in frame:
            frame['policy_method'] = frame['method']
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
            approx='*' if 'policy_method' in data and data.policy_method.map(_approximate).any() else ''
            for row,metric in enumerate(metrics):
                x=data.bin_center.to_numpy()+1
                axes[row,col].plot(x,data[metric],lw=1.25,color=color,label=f'n={n}{approx}')
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
    approx_note = '\n* Approximate policy; all simulated memories follow actual FIFO' if 'policy_method' in periods and periods.policy_method.map(_approximate).any() else ''
    fig.suptitle('Fixed-window Monte Carlo · 25-period bins and 95% path-clustered intervals'+approx_note,fontsize=11)
    fig.tight_layout(rect=(0,.065,1,.92))
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
        approx=' · Approximate policy' if 'policy_method' in subset and subset.policy_method.map(_approximate).any() else ''
        fig.suptitle(f'Fixed-window Monte Carlo · n = {n}{approx}')
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
    files.append(_save(fig,output_dir,stem+'_diagnostics'))
    return files


def geometric_reference_comparison(original_records, fixed_records, output_dir,
                                   *, rho=.95, n=20, p0=.5, count=20_000):
    """Compare an original saved geometric history with the new FIFO history.

    The reference need not share random numbers with the FIFO simulation.
    Both panels report pre-choice S in evidence units, with identical axes.
    """
    configure_style()
    old = original_records[:, 0] if original_records.ndim == 2 else original_records
    new = fixed_records[:, 0] if fixed_records.ndim == 2 else fixed_records
    count = min(count, len(old), len(new))
    time = np.arange(1,count+1)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    for ax,states,action,title in [
        (axes[0],old['x_before'][:count]/(1-rho),old['product2'][:count],
         f'Geometric forgetting · ρ = {rho:g} · Original saved reference'),
        (axes[1],new['S_before'][:count],new['product2'][:count],
         f'Fixed-window forgetting · n = {n} · New simulation'),
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
    return _save(fig,output_dir,f'geometric_vs_fixed_p0_{p0:g}_rho_{rho:g}_n_{n}')


def plot_exact_policy_heatmap(solution, output_dir, stem=None):
    """Show every ordered mature history for an exhaustively solved policy."""
    configure_style()
    n=solution.n
    if n>20 or not hasattr(solution,'advantages'):
        raise ValueError('heatmap requires an exhaustively solved window with n<=20')
    older=n//2
    newer=n-older
    choice=(solution.advantages[n]>solution.action_tolerance).reshape(1<<older,1<<newer)
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    ax.imshow(choice,origin='lower',interpolation='nearest',aspect='auto',
              cmap=ListedColormap([PRODUCT1,PRODUCT2]),vmin=0,vmax=1)
    for axis,bits in ((ax.xaxis,newer),(ax.yaxis,older)):
        positions=np.unique(np.linspace(0,(1<<bits)-1,min(6,1<<bits),dtype=int))
        axis.set_ticks(positions)
        axis.set_ticklabels([format(int(t),f'0{bits}b') for t in positions])
    ax.set(xlabel=f'Newest {newer} outcomes (binary; oldest at left)',
           ylabel=f'Oldest {older} outcomes (binary; oldest at left)',
           title=f'Optimal mature FIFO policy · p₀ = {solution.primitives.p0:g} · n = {n}\nEvery pixel is one complete ordered history; 1 = success')
    ax.grid(False)
    handles=[Line2D([0],[0],marker='s',ls='',color=c,label=l) for c,l in
             [(PRODUCT1,'Product 1'),(PRODUCT2,'Product 2')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
    return _save(fig,output_dir,stem or f'exact_policy_all_histories_p0_{solution.primitives.p0:g}_n_{n}')
