"""Add cross-rho simulated heatmaps on the original rho=.95 linear scale."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, now, write_json
from oblivious.equilibrium_comparison.code.cross_rho_figures import joint_figure


ROOT = Path(__file__).resolve().parents[3]
NAME = 'S02_simulated_joint_distributions_original_linear_scale'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cross_rho_n51')
    parser.add_argument('--reference', type=Path,
                        default=ROOT / 'oblivious/06_equilibrium_comparison/data/rho095_n51')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single folder name')
    data = ROOT / 'oblivious/06_equilibrium_comparison/data' / args.name
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise SystemExit('The cross-rho comparison must be complete first')
    config = manifest['configuration']
    reference = load_bundle(args.reference / 'analysis')
    if reference['configuration']['rho'] != .95:
        raise SystemExit('The reference must be an original rho=.95 comparison')
    if reference['configuration']['grid'] != config['grid']:
        raise SystemExit('The reference and cross-rho comparison must use the same grid')
    # Exactly the same maximum over modes and models used in reporting.py.
    vmax = max(float(item['joint'].max()) for mode in reference['distributions'].values()
               for item in mode.values())
    results = [load_bundle(data / 'checkpoints' / case_tag(rho, config['grid']))
               for rho in config['rhos']]
    matching = {}
    for result in results:
        if result['rho'] == .95 and result['verified']:
            matching = {key: bool(np.array_equal(item['distribution']['joint'],
                                                reference['distributions']['grid'][key]['joint']))
                        for key, item in result['profiles'].items()}
    figures = resolve_saved_path(config['figures_output'], project_root=ROOT) / config['name']
    caption = (rf'Linear color scale fixed to the original $\rho=0.95$ comparison: 0 to {vmax:.7f}.'
               '\nThe upper colorbar triangle indicates saturation above this limit.')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .7}):
        joint_figure(results, config, figures, simulated=True,
                     norm=Normalize(vmin=0, vmax=vmax), name=NAME,
                     scale_note=caption, original_cells=True)
    reference_paths = [args.reference / ('analysis.' + ext) for ext in ('json', 'npz')]
    write_json(figures / (NAME + '.json'), dict(
        created_utc=now(), cross_rho_manifest=str(data / 'manifest.json'),
        reference_directory=str(args.reference.resolve()),
        reference_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in reference_paths},
        kind='simulated calendar-time joint frequencies', colormap='viridis',
        normalization='linear', vmin=0, vmax=vmax, clipping='Top color above vmax; colorbar has upper extension',
        reference_rho095_histograms_exactly_equal=matching,
        sources={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), ROOT / 'oblivious/06_equilibrium_comparison/code/cross_rho_figures.py')},
        no_simulations_or_equilibria_computed=True))
    print(f'Added {figures / NAME}.pdf and .png; scale [0, {vmax:.7f}]', flush=True)
    print(f'rho=.95 histograms exactly equal to the reference: {matching}', flush=True)


if __name__ == '__main__':
    main()
