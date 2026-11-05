"""Show saved and simulated joint laws together, with one shared log scale."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, now, write_json
from oblivious.equilibrium_comparison.code.cross_rho_figures import COLORS, _missing, _note, _rho_label, _save


ROOT = Path(__file__).resolve().parents[3]
NAME = 'S03_joint_stationary_and_simulated_log_scale'
COLUMNS = (
    ('oe', 'stationary_joint', 'OE distribution\nSaved stationary'),
    ('oe', 'simulated_joint', 'OE distribution\nSimulated'),
    ('p0', 'stationary_joint', 'OE p0\nSaved stationary'),
    ('p0', 'simulated_joint', 'OE p0\nSimulated'),
)


def make_combined_figure(results, config, directory):
    values = [array for result in results for item in result.get('profiles', {}).values()
              for array in (item['stationary_joint'], item['distribution']['joint'])]
    norm = LogNorm(vmin=1e-6, vmax=max(float(array.max()) for array in values) if values else 1.)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(cmap(0.))
    cmap.set_under(cmap(0.))
    # The same node-centered cells as 05 and S01 preserve boundary atoms.
    grid = next((result['grid'] for result in results if result['verified']),
                np.linspace(0, 1, config['grid']))
    edges = np.r_[grid[0] - (grid[1] - grid[0]) / 2,
                  (grid[:-1] + grid[1:]) / 2,
                  grid[-1] + (grid[-1] - grid[-2]) / 2]
    panels = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .7}):
        fig, axs = plt.subplots(len(results), 4, figsize=(10.4, 2.05 * len(results) + 1.1),
                                sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=.075, right=.875, top=.94, bottom=.08,
                            hspace=.18, wspace=.14)
        for i, result in enumerate(results):
            for j, (key, kind, title) in enumerate(COLUMNS):
                ax = axs[i, j]
                if i == 0:
                    ax.set_title(title, color=COLORS[key], fontsize=10, pad=11, linespacing=1.5)
                if result['verified']:
                    item = result['profiles'][key]
                    array = item['stationary_joint'] if kind == 'stationary_joint' else item['distribution']['joint']
                    ax.pcolormesh(edges, edges, np.ma.masked_less_equal(array.T, 0),
                                  cmap=cmap, norm=norm, shading='flat', rasterized=True)
                    boundary = float(array[0, :].sum() + array[:, 0].sum() - array[0, 0])
                    if boundary > .5:
                        ax.text(.56, .72, f'{100 * boundary:.3f}% of mass on\n' + r'$x_A=0$ or $x_B=0$',
                                transform=ax.transAxes, ha='center', va='center', fontsize=7,
                                color='white', bbox=dict(facecolor=cmap(0.), edgecolor='none', alpha=.9, pad=2))
                    panels.append(dict(row=i, column=j, rho=result['rho'], model=key, kind=kind,
                                       probability_sum=float(array.sum()),
                                       array_sha256=sha256(np.ascontiguousarray(array).tobytes()).hexdigest()))
                else:
                    _missing(ax)
                    panels.append(dict(row=i, column=j, rho=result['rho'], model=key, kind=kind,
                                       verified=False, reason=result.get('reason', 'Unavailable')))
                ax.set(xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]),
                       xticks=[0, .5, 1], yticks=[0, .5, 1])
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(length=2.5, labelsize=8)
                if j == 0:
                    ax.set_ylabel(rf'$\rho={_rho_label(result["rho"])}$' + '\nSeller B reputation', fontsize=9)
                if i == len(results) - 1:
                    ax.set_xlabel('Seller A reputation', fontsize=8.5)
        cax = fig.add_axes([.91, .29, .017, .42])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='min')
        bar.set_label('Joint probability mass per grid cell (log scale)', fontsize=9, labelpad=8)
        bar.ax.tick_params(labelsize=8)
        fig.suptitle('Joint reputation distributions: saved stationary and simulated', y=.987, fontsize=12)
        fig.text(.5, .022,
                 _note(config) + '\nSaved stationary: verified finite-grid laws. Simulated: empirical calendar-time frequencies; finite burn-in can leave initialization bias.'
                 '\nAll 32 panels use the same axes and log color scale as 05 and S01; boundary nodes are drawn at full cell width.',
                 ha='center', fontsize=7.5, linespacing=1.6)
        _save(fig, Path(directory), NAME)
    return dict(normalization='logarithmic', vmin=norm.vmin, vmax=norm.vmax,
                colormap='viridis', columns=[dict(model=k, kind=t, title=s) for k, t, s in COLUMNS], panels=panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cross_rho_n51')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single folder name')
    data = ROOT / 'oblivious/06_equilibrium_comparison/data' / args.name
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete':
        raise SystemExit('The cross-rho comparison must be complete first')
    config = manifest['configuration']
    bases = [data / 'checkpoints' / case_tag(rho, config['grid']) for rho in config['rhos']]
    results = [load_bundle(base) for base in bases]
    figures = resolve_saved_path(config['figures_output'], project_root=ROOT) / config['name']
    figures.mkdir(parents=True, exist_ok=True)
    metadata = make_combined_figure(results, config, figures)
    source_paths = [manifest_path, Path(__file__), ROOT / 'oblivious/06_equilibrium_comparison/code/cross_rho_figures.py']
    source_paths += [base.with_suffix('.' + suffix) for base in bases for suffix in ('json', 'npz')]
    write_json(figures / (NAME + '.json'), dict(
        created_utc=now(), cross_rho_manifest=str(manifest_path), **metadata,
        source_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in source_paths},
        no_simulations_or_equilibria_computed=True))
    print(f'Added {figures / NAME}.pdf and .png', flush=True)
    print(f'{len(results)} rows x 4 columns, common log scale [{metadata["vmin"]}, {metadata["vmax"]}]', flush=True)


if __name__ == '__main__':
    main()
