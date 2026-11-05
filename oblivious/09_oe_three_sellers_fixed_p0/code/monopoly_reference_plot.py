"""Add horizontal monopoly references to a new version of published plot 05.

The saved scientific results and the original report remain unchanged.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np

from . import expanded_report as expanded
from . import report as base


def build(data, figures):
    data, figures = Path(data).resolve(), Path(figures).resolve()
    report_path = data/'report_manifest.json'
    source = json.loads(report_path.read_text())
    base.require(source.get('verified') is True and not source.get('diagnostic_only'),
                 'A verified published report is required')
    inputs = {base.location(report_path): base.digest(report_path)}
    for field in ('source_input_sha256', 'reporting_source_sha256',
                  'generated_data_sha256', 'figure_files'):
        base.verify_inputs(source[field])
        for name, digest in source[field].items():
            base.require(name not in inputs or inputs[name] == digest, 'Conflicting input fingerprint')
            inputs[name] = digest
    manifest, rows, _, checked_inputs, p0s, rhos, models = base.read_inputs(data)
    audit, audit_inputs = base.read_optional_audit(data, manifest)
    base.require(manifest['status'] == 'complete' and audit is not None,
                 'A complete, independently audited campaign is required')
    base.require(tuple(models) == base.DEFAULT_MODELS, 'Expected N=1,2,3 marginal comparison')
    for declared in (checked_inputs, audit_inputs):
        for name, digest in declared.items():
            base.require(name not in inputs or inputs[name] == digest, 'Conflicting scientific fingerprint')
            inputs[name] = digest
    cases, _ = expanded.select_cases(rows, manifest, p0s, rhos)
    base.require(cases == source['case_grids'], 'Published grid checks differ from saved results')
    base.require(len(p0s) <= len(expanded.P0_COLORS), 'Not enough distinct p0 colors')
    base.require(not figures.exists(), 'Choose a new output directory to preserve published figures')
    reporting_sources = {base.location(path): base.digest(path)
                         for path in (Path(__file__), Path(base.__file__), Path(expanded.__file__))}
    lookup = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows}
    points, omitted, used_colors = [], {}, {}
    with base.plt.rc_context({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 8}):
        fig, axes = expanded.facets(rhos, panel_height=3.2)
        for ax, rho in zip(axes.ravel(), rhos):
            omitted[str(rho)] = []
            for p0, color in zip(p0s, expanded.P0_COLORS):
                case = cases[expanded.case_key(p0, rho)]
                if not case['metrics']['product2_conditional']['passed']:
                    omitted[str(rho)].append(p0)
                    continue
                grid = case['displayed_n']
                values = [100*lookup[p0, rho, grid, sellers, closure]['product2_conditional']
                          for sellers, closure, *_ in models]
                monopoly = values[0]
                ax.plot([1, 3], [monopoly, monopoly], color=color, ls=(0, (4, 3)),
                        lw=1.3, alpha=.8, zorder=2)
                ax.plot([1, 2, 3], values, color=color, marker='o', ms=5, lw=1.8, zorder=3)
                points.append(dict(p0=p0, rho=rho, n=grid,
                                   product2_percent_by_sellers=values,
                                   monopoly_reference_percent=monopoly))
                used_colors[p0] = color
            excluded = ', '.join(f'{p0:g}' for p0 in omitted[str(rho)]) or 'none'
            ax.set(title=f'ρ = {rho:g} | Marginal OE\nOmitted p₀: {excluded}',
                   xticks=[1, 2, 3], xlim=(.85, 3.15), ylim=(-2, 102),
                   yticks=np.arange(0, 101, 20), xlabel='Number of sellers N',
                   ylabel='Product 2 (% of own sales)')
            ax.tick_params(labelbottom=True)
            base.style(ax)
        for ax in axes.ravel()[len(rhos):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=used_colors[p0], lw=2, label=f'p₀ = {p0:g}')
                   for p0 in p0s if p0 in used_colors]
        handles += [Line2D([], [], color='black', lw=1.8, marker='o', label='Outcome at N=1,2,3'),
                    Line2D([], [], color='black', lw=1.3, ls=(0, (4, 3)), label='Monopoly reference (N=1)')]
        fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Product 2 use as the number of sellers changes\n'
                     'Dashed lines: monopoly at the same p₀ and ρ; overlapping lines indicate equal use.',
                     fontsize=11)
        base.verify_inputs(inputs)
        base.verify_inputs(reporting_sources)
        figures.mkdir(parents=True)
        paths = base.save(fig, figures, base.SELLER_COUNT_STEM)
    index = figures/'README.md'
    index.write_text('\n'.join([
        '# Plot 05 con referencia del monopolio', '',
        f'[PNG]({base.SELLER_COUNT_STEM}.png) · [PDF]({base.SELLER_COUNT_STEM}.pdf)', '',
        'Cada color identifica un p₀. La curva continua compara N=1,2,3; la línea '
        'horizontal discontinua mantiene el uso del producto 2 del monopolio (N=1) '
        'con ese mismo p₀ y ρ. Si ambas coinciden, se superponen. '
        'El eje vertical mide el porcentaje de ventas propias con producto 2.', '',
        'Se conservan exactamente los casos y las grillas del reporte original: '
        '21 → 31, con 22 de 24 pares validados para producto 2. '
        'Se omiten (p₀,ρ)=(0.3,0.8) y (0.8,0.9), incluidas sus referencias.', '',
        'Los datos, el reporte y las figuras anteriores se conservan sin cambios. '
        'No se vuelve a resolver ningún modelo.', '']))
    base.verify_inputs(inputs)
    base.verify_inputs(reporting_sources)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='complete_verified',
                  verified=True, source_input_sha256=inputs, reporting_source_sha256=reporting_sources,
                  requested_p0s=p0s, requested_rhos=rhos,
                  measure='Product 2 as percent of own sales',
                  reference='Horizontal N=1 outcome for the same p0, rho and displayed grid',
                  plotted_cases=points, omitted_p0_by_rho=omitted,
                  figure_files={base.location(path): base.digest(path) for path in paths},
                  generated_data_sha256={base.location(index): base.digest(index)})
    base.write_json(figures/'report_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base.HOME/'data/rho_p0_sweep_31')
    parser.add_argument('--figures', type=Path,
                        default=base.HOME/'figures/rho_p0_sweep_31_monopoly_reference')
    args = parser.parse_args()
    result = build(args.data, args.figures)
    print(json.dumps(dict(status=result['status'], plotted_cases=len(result['plotted_cases']),
                          figure_files=result['figure_files']), indent=2))


if __name__ == '__main__':
    main()
