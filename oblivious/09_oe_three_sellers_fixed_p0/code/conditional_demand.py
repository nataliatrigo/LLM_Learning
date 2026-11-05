"""Plot A's demand against seller count with fixed Beta beliefs and outside.

This is a static buyer-choice calculation. No policy, equilibrium, stationary
distribution, or reputation transition is solved or loaded.
"""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaincc, betaln, roots_jacobi

from oblivious.oe_p0.code import primitives
from oblivious.oe_p0.code.primitives import Parameters, posterior


HOME = Path(__file__).resolve().parents[1]
DEFAULT_P0S = (.1, .3, .5, .7, .8, .9)
DEFAULT_RHOS = (.6, .7, .8, .9)
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')


def _beta_pair(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError('Beta shapes must be a finite positive pair (a, b).')
    return value


def _integrate(own, rival, counts, p0, order):
    a, b = own
    if p0 == 0:
        nodes, weights = roots_jacobi(order, b-1, a-1)
        t = (nodes+1)/2
        weights = weights/weights.sum()
    else:
        nodes, weights = roots_jacobi(order, b-1, 0)
        t = p0+(1-p0)*(nodes+1)/2
        with np.errstate(divide='ignore'):
            weights = np.exp(b*(np.log1p(-p0)-np.log(2))-betaln(a, b)
                             + np.log(weights)+(a-1)*np.log(t))
    # N=1 has an empty product, including when a rival CDF underflows to zero.
    cdf = betainc(*rival, t)
    return np.power(cdf[None, :], counts[:, None]-1) @ weights


def demand_by_seller_count(own_beta, rival_beta, seller_counts, p0=.5, order=128):
    """Return ``(demand, diagnostics)`` for fixed, identical added rivals.

    N counts A plus N-1 rivals. Buyer draws are independent conditional on the
    specified Betas: D_A(N) = integral_p0^1 f_A(t) F_R(t)**(N-1) dt.
    Arbitrary positive Beta shapes are accepted; rho is not an input.
    """
    own, rival = _beta_pair(own_beta), _beta_pair(rival_beta)
    counts = np.asarray(seller_counts, dtype=float)
    if (counts.ndim != 1 or not counts.size or not np.all(np.isfinite(counts))
            or np.any(counts < 1) or np.any(counts != np.floor(counts))):
        raise ValueError('Seller counts must be a nonempty vector of positive integers.')
    if not np.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError('p0 must lie in [0,1].')
    if not np.isfinite(order) or int(order) != order or order < 8:
        raise ValueError('Quadrature order must be an integer >=8.')
    order = int(order)
    if p0 == 1:
        coarse = fine = np.zeros(len(counts))
    else:
        coarse = _integrate(own, rival, counts, p0, order)
        fine = _integrate(own, rival, counts, p0, 2*order)
    change = abs(fine-coarse)
    relative = change/np.maximum(abs(fine), np.finfo(float).tiny)
    diagnostics = dict(order=order, final_order=2*order,
                       order_doubling_max_abs=float(change.max()),
                       order_doubling_max_relative=float(relative.max()))
    if (not np.all(np.isfinite(fine)) or np.any(fine < 0) or np.any(fine > 1+1e-12)
            or (p0 < 1 and np.any(fine == 0))
            or change.max() > 2e-10 or relative.max() > 2e-7):
        raise ArithmeticError(f'Demand quadrature failed; increase order: {diagnostics}')
    if np.any(np.diff(fine[np.argsort(counts)]) > 2e-12):
        raise ArithmeticError('Demand increased when an identical rival was added.')
    return fine, diagnostics


def _reference(own, rival, sellers, p0):
    """Independent adaptive scalar integration for each published point."""
    if p0 == 1:
        return 0.
    a, b = own

    def integrand(t):
        if t <= 0 or t >= 1:
            return 0.
        density = np.exp((a-1)*np.log(t)+(b-1)*np.log1p(-t)-betaln(a, b))
        return density*betainc(*rival, t)**(sellers-1)

    result, error = quad(integrand, p0, 1, epsabs=1e-13, epsrel=1e-11, limit=300)
    if error > 1e-10:
        raise ArithmeticError(f'Independent integration error: {error:g}')
    return result


def _symmetric_demand(beta, counts, p0):
    # Avoid cancelling a small sale probability against a rounded CDF of one.
    with np.errstate(divide='ignore'):
        log_cdf = np.log1p(-betaincc(*beta, p0))
    return -np.expm1(counts*log_cdf)/counts


def _token(value):
    return f'{value:g}'.replace('.', 'p')


def _draw(ax, counts, curves, p0, own_reputations, rival_reputation):
    for index, (x, demand) in enumerate(zip(own_reputations, curves)):
        symmetric = x == rival_reputation
        ax.plot(counts, demand, color=COLORS[index % len(COLORS)],
                ls='--' if symmetric else '-', lw=1.9, marker='o', ms=3,
                label=f'x_A = {x:g}'+(' (symmetric)' if symmetric else ''))
    upper = min(1., 1.08*max(float(curve.max()) for curve in curves)) or 1.
    ax.set(title=f'p₀ = {p0:g}', xlabel='Total sellers N (including A)',
           ylabel='Pr(buyer chooses A)', ylim=(-.015*upper, upper),
           xlim=(counts[0]-.2, counts[-1]+.2))
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(counts[0], counts[-1])
    ax.set_xticks(np.unique(np.r_[1, ticks[(ticks >= 1) & (ticks <= counts[-1])], counts[-1]]))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)


def build(name='conditional_demand', *, max_sellers=20, p0s=DEFAULT_P0S,
          rhos=DEFAULT_RHOS, own_reputations=(.3, .5, .7), rival_reputation=.5,
          order=128):
    """Generate checked CSV, PNG/PDF plots and a reproducible manifest."""
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Use a plain campaign name without path components.')
    if int(max_sellers) != max_sellers or max_sellers < 2:
        raise ValueError('Use an integer max_sellers >=2.')
    counts = np.arange(1, int(max_sellers)+1)
    p0s, rhos, own_reputations = map(list, (p0s, rhos, own_reputations))
    for values in (p0s, rhos, own_reputations):
        if not values or len(values) != len(set(values)):
            raise ValueError('Parameter lists must be nonempty and contain no duplicates.')
    if not np.all(np.isfinite(p0s)) or np.any((np.array(p0s) < 0) | (np.array(p0s) > 1)):
        raise ValueError('p0 values must lie in [0,1].')
    data, figures = HOME/'data'/name, HOME/'figures'/name
    if data.exists() or figures.exists():
        raise FileExistsError('Choose a new --name to preserve existing published results.')

    rows, checks, lookup = [], [], {}
    for rho in rhos:
        params = Parameters(rho=rho)
        rival = np.array(posterior(rival_reputation, params))
        for p0 in p0s:
            for x in own_reputations:
                own = np.array(posterior(x, params))
                demand, check = demand_by_seller_count(own, rival, counts, p0, order)
                reference = np.array([_reference(own, rival, n, p0) for n in counts])
                np.testing.assert_allclose(demand, reference, rtol=2e-8, atol=2e-12)
                if x == rival_reputation:
                    np.testing.assert_allclose(demand, _symmetric_demand(own, counts, p0),
                                               rtol=2e-8, atol=2e-12)
                checks.append(dict(rho=rho, p0=p0, own_reputation=x,
                                   independent_quad_max_abs=float(max(abs(demand-reference))),
                                   **check))
                lookup[rho, p0, x] = demand
                rows.extend(dict(rho=rho, p0=p0, own_reputation=x,
                                 rival_reputation=rival_reputation,
                                 own_beta_a=float(own[0]), own_beta_b=float(own[1]),
                                 rival_beta_a=float(rival[0]), rival_beta_b=float(rival[1]),
                                 sellers=int(n), rivals=int(n-1), demand_A=float(d),
                                 independent_quad=float(ref))
                            for n, d, ref in zip(counts, demand, reference))

    # Monotonicity in the outside threshold is checked on all published curves.
    for rho in rhos:
        for x in own_reputations:
            series = np.array([lookup[rho, p0, x] for p0 in sorted(p0s)])
            if np.any(np.diff(series, axis=0) > 2e-12):
                raise ArithmeticError('Demand increased when p0 increased.')

    data.mkdir(parents=True)
    figures.mkdir(parents=True)
    with (data/'demand.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data/'validation.json').write_text(json.dumps(checks, indent=2, allow_nan=False)+'\n')
    paths = [data/'demand.csv', data/'validation.json']
    pdf_path = figures/'demand_A_by_seller_count.pdf'
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11}), PdfPages(pdf_path) as pdf:
        for rho in rhos:
            columns = min(3, len(p0s))
            panel_rows = (len(p0s)+columns-1)//columns
            fig, axes = plt.subplots(panel_rows, columns, figsize=(5*columns, 3.5*panel_rows),
                                     squeeze=False, layout='constrained')
            for ax, p0 in zip(axes.ravel(), p0s):
                _draw(ax, counts, [lookup[rho, p0, x] for x in own_reputations],
                      p0, own_reputations, rival_reputation)
            for ax in axes.ravel()[len(p0s):]:
                ax.set_visible(False)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center',
                       ncol=min(4, len(own_reputations)), frameon=False)
            fig.suptitle("A's demand as sellers enter | ρ = "+f'{rho:g}\n'
                         f'Fixed reputations: each rival x_R = {rival_reputation:g}; '
                         'independent Beta draws; panel scales vary', fontsize=14)
            png_path = figures/f'demand_A_rho_{_token(rho)}.png'
            fig.savefig(png_path, dpi=180)
            pdf.savefig(fig)
            paths.append(png_path)
            plt.close(fig)
    paths.append(pdf_path)

    index = figures/'README.md'
    lines = ["# Demanda de A al aumentar el número de sellers", '',
             '[PDF con todos los rho](demand_A_by_seller_count.pdf)', '',
             f'N=1..{max_sellers} cuenta a A y sus N-1 rivales. Cada nuevo rival tiene '
             f'reputación fija x_R={rival_reputation:g}; A toma los valores '
             +', '.join(f'{x:g}' for x in own_reputations)+'.', '',
             'El eje vertical es la probabilidad de que el comprador elija A en un período, '
             'incluyendo los períodos en que elige outside. Cada panel fija p0. '
             'Cada panel ajusta su escala vertical para mostrar también las demandas pequeñas. '
             'Las líneas discontinuas indican reputaciones iguales para A y todos sus rivales.', '',
             '```text', 'D_A(N) = integral_p0^1 f_A(t) F_R(t)^(N-1) dt',
             'Beta(x;rho) = Beta(1+x/(1-rho), 1+(1-x)/(1-rho))',
             'Caso simétrico: D_A(N) = [1-F(p0)^N]/N', '```', '',
             'Es demanda condicionada a reputaciones fijas. No usa políticas, distribuciones '
             'estacionarias ni un equilibrio oblivious. rho solo parametriza las Betas. '
             'Para demanda promedio de equilibrio habría que promediar sobre la ley conjunta '
             'de reputaciones correspondiente a cada N.', '',
             'Todos los puntos se contrastan con integración adaptativa independiente; '
             'la cuadratura positiva se duplica en orden. También se comprueban la fórmula '
             'simétrica y que la demanda no aumente al agregar rivales o aumentar p0.', '',
             '| rho | Figura |', '| --- | --- |']
    lines += [f'| {rho:g} | [PNG](demand_A_rho_{_token(rho)}.png) |' for rho in rhos]
    lines += ['', f'[Datos CSV](../../data/{name}/demand.csv) · '
              f'[Validaciones](../../data/{name}/validation.json) · '
              f'[Manifiesto](../../data/{name}/manifest.json)', '',
              'Reproducir desde la raíz con un nombre nuevo:', '', '```bash',
              'MPLCONFIGDIR=/tmp/conditional-demand-mpl .venv/bin/python -B -m '
              'oblivious.oe_three_sellers_fixed_p0.code.conditional_demand '
              f'--name {name}_new --max-sellers {max_sellers} '
              '--p0s '+','.join(map(str, p0s))+' --rhos '+','.join(map(str, rhos))+' '
              '--own-reputations '+','.join(map(str, own_reputations))+' '
              f'--rival-reputation {rival_reputation:g} --order {order}', '```', '']
    index.write_text('\n'.join(lines))
    paths.append(index)
    root = HOME.parents[1]
    sources = [Path(__file__), Path(primitives.__file__)]
    manifest = dict(status='complete_verified', created_utc=datetime.now(timezone.utc).isoformat(),
                    measure='Conditional probability buyer chooses A, including outside periods',
                    configuration=dict(max_sellers=max_sellers, p0s=p0s, rhos=rhos,
                                       own_reputations=own_reputations,
                                       rival_reputation=rival_reputation, order=order),
                    points=len(rows), independent_quad_points=len(rows),
                    max_order_doubling_error=max(check['order_doubling_max_abs'] for check in checks),
                    max_independent_quad_error=max(check['independent_quad_max_abs'] for check in checks),
                    sources={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sources},
                    outputs={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths})
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='conditional_demand')
    parser.add_argument('--max-sellers', type=int, default=20)
    numbers = lambda value: [float(x) for x in value.split(',')]
    parser.add_argument('--p0s', type=numbers, default=DEFAULT_P0S)
    parser.add_argument('--rhos', type=numbers, default=DEFAULT_RHOS)
    parser.add_argument('--own-reputations', type=numbers, default=(.3, .5, .7))
    parser.add_argument('--rival-reputation', type=float, default=.5)
    parser.add_argument('--order', type=int, default=128)
    result = build(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('status', 'points', 'max_independent_quad_error')}, indent=2))


if __name__ == '__main__':
    main()
