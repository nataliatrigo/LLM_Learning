"""Paper-oriented cross-rho figures from the saved comparison checkpoints."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np

from oblivious.equilibrium_comparison.code.reporting import COLORS
from model_names import MODEL_LABELS


LABELS = MODEL_LABELS


def _save(fig, directory, name):
    for suffix in ('pdf', 'png'):
        fig.savefig(directory / f'{name}.{suffix}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='.9', lw=.6, zorder=0)
    ax.tick_params(length=3, width=.7)


def _rho_label(rho):
    return f'{rho:.3f}' if rho == .999 else f'{rho:.2f}'


def _model_handles():
    return [Line2D([], [], color=COLORS[key], lw=2, label=LABELS[key]) for key in ('oe', 'p0')]


def _method_handles(cdf=False):
    if cdf:
        return [Line2D([], [], color='.3', lw=1.6, label='Simulation (pointwise 95% CI)'),
                Line2D([], [], color='.3', lw=1.4, ls='--', label='Saved stationary prediction')]
    return [Line2D([], [], color='.3', marker='o', lw=1.3, ms=4, label='Simulation (95% MC CI)'),
            Line2D([], [], color='.3', marker='D', mfc='white', lw=1.2, ls='--', ms=5,
                   label='Saved stationary prediction')]


def _note(config):
    return (f"{config['replications']} paired paths; {config['periods']:,} retained periods; "
            f"{config['burn_in']:,} burn-in; start $(0.5,0.5)$. Actual two-seller TS demand.")


def _prediction_note(config):
    return ('Saved-grid stationary predictions.' if config['mode'] == 'grid' else
            'Dashed curves: saved-grid references, not stationary predictions for continuous reputations.')


def _missing(ax):
    ax.text(.5, .5, 'UNVERIFIED / UNAVAILABLE\nSee verification_status.csv',
            transform=ax.transAxes, ha='center', va='center', color='#9b2226', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#9b2226', boxstyle='round,pad=.4'))


def _panel_grid(results):
    cols = 4 if len(results) > 4 else min(4, len(results))
    rows = int(np.ceil(len(results) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.35 * rows),
                            sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.17, wspace=.18, hspace=.34)
    for ax in axs.ravel()[len(results):]:
        ax.set_visible(False)
    return fig, axs


def policy_figure(results, config, directory):
    # Match the original experiments' portrait layout and per-panel styling.
    columns = min(2, len(results))
    rows = int(np.ceil(len(results) / columns))
    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'legend.fontsize': 8, 'axes.linewidth': .8,
        'axes.spines.top': True, 'axes.spines.right': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    }):
        fig, axs = plt.subplots(rows, columns, figsize=(11, 3.7 * rows),
                                squeeze=False, layout='constrained')
        for ax, result in zip(axs.flat, results):
            verified = result['verified']
            nodes = len(result['grid']) if verified else config['grid']
            status = 'verified on grid' if verified else 'UNVERIFIED / UNAVAILABLE'
            ax.set_title(rf'$\rho={result["rho"]:g}$, n={nodes} | {status}')
            has_mixing = False
            if verified:
                for key in ('oe', 'p0'):
                    grid = np.asarray(result['grid'])
                    policy = np.asarray(result['profiles'][key]['policy'])
                    ax.plot(grid, policy, color=COLORS[key], marker='.', ms=3,
                            ls='-' if key == 'oe' else '--', lw=1.3,
                            label=LABELS[key], zorder=3 if key == 'oe' else 4)
                    mixed = (policy > 1e-9) & (policy < 1 - 1e-9)
                    has_mixing |= bool(mixed.any())
                    ax.scatter(grid[mixed], policy[mixed], marker='D', s=24,
                               color=COLORS[key], zorder=6)
                handles, labels = ax.get_legend_handles_labels()
                if has_mixing:
                    handles.append(Line2D([], [], color='.35', marker='D', ls='none',
                                          ms=4.5, label='Mixing at a grid point'))
                    labels.append('Mixing at a grid point')
                ax.legend(handles=handles, labels=labels, loc='best')
            else:
                _missing(ax)
            ax.set(xlabel='Own reputation', ylabel='Probability of product 2',
                   xlim=(-.04, 1.04), ylim=(-.04, 1.04),
                   xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
            ax.grid(alpha=.25)
        for ax in axs.flat[len(results):]:
            ax.set_visible(False)
        fig.suptitle('Product choice: OE distribution and OE p0', fontsize=13)
        _save(fig, directory, '01_optimal_policies_across_rho')


def _rho_axis(ax, results):
    ax.set_xticks(range(len(results)), [_rho_label(r['rho']) for r in results])
    ax.set_xlim(-.3, len(results) - .7)
    ax.set_xlabel(r'Forgetting factor $\rho$ (discrete values)')
    _style(ax)
    for index, result in enumerate(results):
        if not result['verified']:
            ax.axvspan(index - .15, index + .15, color='.95', zorder=0)
            ax.text(index, .5, 'Unverified', rotation=90, ha='center', va='center',
                    transform=ax.get_xaxis_transform(), color='#9b2226', fontsize=8)


def _metric(ax, results, metric):
    for key in ('oe', 'p0'):
        means, lows, highs, predictions = [], [], [], []
        for result in results:
            if not result['verified']:
                means.append(np.nan); lows.append(np.nan); highs.append(np.nan); predictions.append(np.nan)
                continue
            item = result['profiles'][key]
            estimate = item['summary'][metric]
            means.append(estimate['mean']); lows.append(estimate['ci_low']); highs.append(estimate['ci_high'])
            predictions.append(item['predicted'][metric])
        means, lows, highs = map(np.asarray, (means, lows, highs))
        x = np.arange(len(results), dtype=float) + (-.035 if key == 'oe' else .035)
        ax.plot(x, predictions, color=COLORS[key], ls='--', marker='D', mfc='white',
                mew=1.1, ms=6, lw=1.2, zorder=3)
        ax.errorbar(x, means, yerr=[means - lows, highs - means], color=COLORS[key],
                    marker='o', ms=3.5, lw=1.4, elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=4)
    _rho_axis(ax, results)


def usage_figure(results, config, directory):
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.78)
    _metric(ax, results, 'product2_market')
    ax.set(ylim=(-.025, 1.025), ylabel='Product 2 share of all purchases')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(handles=_model_handles(), loc='upper center', bbox_to_anchor=(.53, 1.0), ncol=2, frameon=False)
    fig.legend(handles=_method_handles(), loc='upper center', bbox_to_anchor=(.53, .93), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .095, _note(config), ha='center', fontsize=8)
    fig.text(.5, .055, 'Rates equally spaced; lines guide the eye. MC intervals exclude initialization bias and may be smaller than markers.',
             ha='center', fontsize=7.8)
    fig.text(.5, .015, _prediction_note(config), ha='center', fontsize=7.8)
    _save(fig, directory, '02_product2_usage_across_rho')


def outcomes_figure(results, config, directory):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.subplots_adjust(left=.07, right=.99, bottom=.28, top=.77, wspace=.25)
    for ax, metric, title, label in zip(
        axs, ('seller_profit', 'success_rate'),
        ('(a) Seller profit', '(b) Customer outcomes'),
        ('Mean seller profit / calendar period', 'Customer success rate'),
    ):
        _metric(ax, results, metric)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylabel(label)
    axs[0].set_ylim(0, .5)
    axs[1].set_ylim(0, 1)
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(handles=_model_handles(), loc='upper center', bbox_to_anchor=(.54, 1.0), ncol=2, frameon=False)
    fig.legend(handles=_method_handles(), loc='upper center', bbox_to_anchor=(.54, .93), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .095, _note(config), ha='center', fontsize=8)
    fig.text(.5, .055, 'Profit averages both sellers, including unselected periods. Rates equally spaced; MC intervals exclude initialization bias.',
             ha='center', fontsize=8)
    fig.text(.5, .015, _prediction_note(config), ha='center', fontsize=8)
    _save(fig, directory, '03_profits_customer_outcomes_across_rho')


def cdf_figure(results, config, directory):
    fig, axs = _panel_grid(results)
    fig.subplots_adjust(top=.80, bottom=.22)
    for index, (ax, result) in enumerate(zip(axs.ravel(), results)):
        ax.set_title(rf'({chr(97 + index)}) $\rho={_rho_label(result["rho"])}$', loc='left', fontsize=10)
        if result['verified']:
            grid = result['grid']
            for key in ('oe', 'p0'):
                item, color = result['profiles'][key], COLORS[key]
                d = item['distribution']
                ax.fill_between(grid, d['cdf_low'], d['cdf_high'], color=color, alpha=.16,
                                step='post', linewidth=0, zorder=1)
                ax.step(grid, d['cdf'], where='post', color=color, lw=1.6, zorder=3)
                ax.step(grid, item['stationary_cdf'], where='post', color=color, ls='--', lw=1.4, zorder=4)
        else:
            _missing(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, .5, 1], yticks=[0, .5, 1])
        _style(ax)
    fig.legend(handles=_model_handles(), loc='upper center', bbox_to_anchor=(.54, 1.01), ncol=2, frameon=False)
    fig.legend(handles=_method_handles(cdf=True), loc='upper center', bbox_to_anchor=(.54, .94), ncol=2, frameon=False, fontsize=9)
    fig.supxlabel('Reputation, $x$', y=.13)
    fig.supylabel('Calendar-time reputation CDF, $F(x)$', x=.005)
    fig.text(.5, .07, _note(config), ha='center', fontsize=8)
    fig.text(.5, .025, 'Sellers pooled within each path. Bands are pointwise; common-start simulations may retain initialization bias.',
             ha='center', fontsize=8)
    _save(fig, directory, '04_stationary_reputation_cdfs_across_rho')


def joint_figure(results, config, directory, *, simulated=False, norm=None,
                 name=None, scale_note=None, original_cells=False):
    count = len(results)
    fig, axs = plt.subplots(count, 2, figsize=(6.8, 2.05 * count + 1.1),
                            sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.12, right=.85, top=.95, bottom=.09 if scale_note else .075,
                        hspace=.18, wspace=.10)
    # Match the original single-rho joint heatmaps: purple through green to yellow.
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(cmap(0.))
    cmap.set_under(cmap(0.))
    if norm is None:
        values = [item['stationary_joint'] for r in results for item in r.get('profiles', {}).values()]
        norm = LogNorm(vmin=1e-6, vmax=max(float(a.max()) for a in values) if values else 1.)
    for i, result in enumerate(results):
        for j, key in enumerate(('oe', 'p0')):
            ax = axs[i, j]
            if i == 0:
                ax.set_title(LABELS[key], color=COLORS[key], fontsize=11, pad=10)
            if result['verified']:
                item = result['profiles'][key]
                values = item['distribution']['joint'] if simulated else item['stationary_joint']
                grid = result['grid']
                # Draw boundary nodes at full width so atoms at zero are not
                # hidden underneath the frame. These are masses, not densities.
                edges = (item['distribution']['edges'] if original_cells else
                         np.r_[grid[0] - (grid[1] - grid[0]) / 2,
                               (grid[:-1] + grid[1:]) / 2,
                               grid[-1] + (grid[-1] - grid[-2]) / 2])
                ax.pcolormesh(edges, edges, np.ma.masked_less_equal(values.T, 0), cmap=cmap, norm=norm,
                              shading='flat', rasterized=True)
                boundary = float(values[0, :].sum() + values[:, 0].sum() - values[0, 0])
                if boundary > .5:
                    ax.text(.56, .72, f'{100 * boundary:.3f}% of mass on\n' + r'$x_A=0$ or $x_B=0$',
                            transform=ax.transAxes, ha='center', va='center', fontsize=8, color='white',
                            bbox=dict(facecolor=cmap(0.), edgecolor='none', alpha=.9, pad=2))
            else:
                _missing(ax)
                edges = np.array([-.01, 1.01])
            ax.set(xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]),
                   xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(length=2.5, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(.8)
            if j == 0:
                ax.set_ylabel(rf'$\rho={_rho_label(result["rho"])}$' + '\nSeller B reputation', fontsize=9)
            if i == count - 1:
                ax.set_xlabel('Seller A reputation', fontsize=9)
    cax = fig.add_axes([.89, .29, .025, .42])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    logarithmic = isinstance(norm, LogNorm)
    observed_max = max((float((item['distribution']['joint'] if simulated else item['stationary_joint']).max())
                        for r in results for item in r.get('profiles', {}).values()), default=0.)
    extension = 'min' if logarithmic else ('max' if observed_max > norm.vmax else 'neither')
    bar = fig.colorbar(sm, cax=cax, extend=extension)
    bar.set_label('Joint probability mass per grid cell (' + ('log' if logarithmic else 'linear') + ' scale)',
                  fontsize=9, labelpad=8)
    bar.ax.tick_params(labelsize=8)
    title = 'Common-start simulated joint distributions' if simulated else 'Saved stationary joint distributions'
    fig.suptitle(title, y=.987, fontsize=12)
    footer = (_note(config) + '\nEmpirical calendar-time laws; finite burn-in can leave initialization bias.'
              if simulated else 'Verified saved finite-grid laws under actual two-seller TS demand.\nStationarity is checked independently for every model and rate.')
    display_note = ('Identical axes and mass scale; zero mass is dark purple.' if original_cells else
                    'Identical axes and mass scale; boundary nodes drawn at full cell width; zero mass is dark purple.')
    if scale_note:
        display_note += '\n' + scale_note
    fig.text(.5, .022, footer + '\n' + display_note,
             ha='center', fontsize=7.5, linespacing=1.6)
    if name is None:
        name = 'S01_simulated_joint_distributions_across_rho' if simulated else '05_joint_stationary_distributions_across_rho'
    _save(fig, directory, name)


def make_figures(results, configuration, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        'font.family': 'DejaVu Serif', 'font.size': 9.5,
        'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 9.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7,
    }):
        policy_figure(results, configuration, directory)
        usage_figure(results, configuration, directory)
        outcomes_figure(results, configuration, directory)
        cdf_figure(results, configuration, directory)
        # Share the same scale across both stationary and empirical figures.
        values = [a for r in results for item in r.get('profiles', {}).values()
                  for a in (item['stationary_joint'], item['distribution']['joint'])]
        norm = LogNorm(vmin=1e-6, vmax=max(float(a.max()) for a in values) if values else 1.)
        # Use the same sans-serif type as the original single-rho heatmaps.
        with plt.rc_context({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'}):
            joint_figure(results, configuration, directory, norm=norm)
            joint_figure(results, configuration, directory, simulated=True, norm=norm)
