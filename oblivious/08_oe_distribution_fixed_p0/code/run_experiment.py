"""Compare a new two-seller marginal OE with monopoly at the same external p0.

Run from the repository root:
    .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.run_experiment
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgetting.Code.model import BoundaryGrid, Primitives, solve_boundary
from oblivious.oe_p0.code.core import _jump_system, _positive_invariant, fixed_demand
from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp, transition_matrix
from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_p0.code.run_experiment import (
    case_tag, load_bundle, now, save_bundle, write_csv, write_json,
)
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from oblivious.oe_distribution_fixed_p0.code.model import (
    FixedOutsideMarginalExperiment, outside_demand_matrix, outside_demand_probability,
)

HOME = Path(__file__).resolve().parents[1]
RHOS = [.6, .7, .8, .9, .95, .97, .99]


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def numerical_sources():
    paths = [Path(__file__), HOME / 'code/model.py', ROOT / 'forgetting/Code/model.py',
             ROOT / 'oblivious/01_oe_distribution/code/model.py',
             ROOT / 'oblivious/03_oe_following_monopoly/code/event_chain.py']
    paths += [ROOT / 'oblivious/02_oe_p0/code' / name for name in
              ('primitives.py', 'policy.py', 'game.py', 'core.py',
               'positive_stationary.py', 'positive_stationary.cpp')]
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in paths}


def solve_monopoly(params, n, p0, dp_tol=1e-10):
    """Solve and independently verify the original mature monopoly problem."""
    grid = np.linspace(0, 1, n)
    demand = fixed_demand(grid, p0, params)
    best = solve_dp(grid, demand, params, tol=dp_tol)
    alpha = best['policy']
    evaluated = evaluate_policy(grid, demand, alpha, params)
    primitives = Primitives(p0=p0, p1=params.p1, p2=params.p2, c1=params.c1,
                            c2=params.c2, revenue=params.R, gamma=params.gamma)
    original = solve_boundary(primitives, params.rho,
        grid=BoundaryGrid(grid, n, 0, 9), tolerance=1e-13)
    regret = float(np.max(np.maximum(evaluated['advantage'], 0)
                          - alpha * evaluated['advantage']))
    value_difference = float(np.max(abs(evaluated['value'] - original.value)))
    event = purchase_chain_statistics(grid, alpha, params)
    eta = event['sale_stationary']
    require(np.all(demand > 0) and np.all(eta > 0),
            'Monopoly has an unrepresentable demand or sale-time probability')
    logs = np.log(eta) - np.log(demand)
    mu = np.exp(logs - logsumexp(logs))
    kernel = transition_matrix(grid, demand, alpha, params)
    jump, rates = _jump_system(kernel)
    independent = _positive_invariant(jump, rates)
    invariant_error = float(np.abs(kernel.T @ mu - mu).sum())
    independent_error = float(np.abs(mu - independent).sum())
    sales = mu * demand
    sale_rate = float(sales.sum())
    sale_law_error = float(np.abs(sales / sale_rate - eta).sum())
    require(regret <= dp_tol and evaluated['residual'] <= dp_tol
            and value_difference < 1e-8 and independent_error < 1e-8
            and invariant_error < 1e-11 and sale_law_error < 1e-9,
            f'Monopoly certificate failed: regret={regret:g}, value={value_difference:g}, '
            f'invariant={invariant_error:g}, independent={independent_error:g}')
    return dict(grid=grid, policy=alpha, value=evaluated['value'], demand=demand,
        advantage=evaluated['advantage'], mu=mu, sale_stationary=eta,
        verified=True, diagnostics=dict(conditional_regret=regret,
            bellman_residual=evaluated['residual'], original_value_max_difference=value_difference,
            original_bellman_residual=original.bellman_residual,
            stationary_residual=invariant_error, independent_stationary_l1=independent_error,
            sale_law_l1=sale_law_error, sale_time=event))


def audit_demand(experiment):
    """Independent adaptive quadrature, order doubling, and event identities."""
    from scipy.special import betainc, betaincc

    game, grid, params = experiment.game, experiment.grid, experiment.params
    doubled, doubled_diagnostics = outside_demand_matrix(
        grid, params, experiment.p0, 2 * experiment.order)
    a, b = posterior(grid, params)
    cdf, sf = betainc(a, b, experiment.p0), betaincc(a, b, experiment.p0)
    outside = np.outer(cdf, cdf)
    partition = float(np.max(abs(game.D + game.D.T + outside - 1)))
    diagonal = float(np.max(abs(game.D.diagonal() - .5 * sf * (1 + cdf))))
    dominance = float(np.max(game.D - sf[:, None]))
    doubling = float(np.max(abs(game.D - doubled)))
    probes = sorted(set((0, len(grid)//4, len(grid)//2, 3*len(grid)//4, len(grid)-1)))
    rows = []
    for i in probes:
        for j in probes:
            exact = outside_demand_probability(float(grid[i]), float(grid[j]),
                                               params, experiment.p0)
            rows.append(dict(i=i, j=j, direct=exact, matrix=float(game.D[i, j]),
                             absolute_error=abs(exact-game.D[i, j])))
    adaptive_error = max(row['absolute_error'] for row in rows)
    require(partition < 1e-11 and diagonal < 1e-11 and dominance < 1e-11
            and doubling < 2e-9 and adaptive_error < 2e-9,
            f'Demand audit failed: partition={partition:g}, doubling={doubling:g}, '
            f'adaptive={adaptive_error:g}')
    return dict(verified=True, partition_max_error=partition, diagonal_max_error=diagonal,
        monopoly_dominance_violation=max(0., dominance), order_doubling_max_error=doubling,
        adaptive_max_error=float(adaptive_error), probes=rows,
        production=game.demand_diagnostics, doubled=doubled_diagnostics)


def solve_competition(experiment, monopoly, seeds, max_evaluations):
    attempts, selected = [], None
    initializations = list(seeds) + [('monopoly_policy', monopoly['policy']),
                                     ('product_1', np.zeros(experiment.n)),
                                     ('half', np.full(experiment.n, .5))]
    seen = set()
    for name, policy in initializations:
        policy = np.asarray(policy, dtype=float)
        signature = policy.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        print(f'  competition start={name}', flush=True)
        try:
            candidate = experiment.solve(initial_policy=policy,
                max_evaluations=max_evaluations, prefer_local=True,
                progress=lambda message: print(f'    {message}', flush=True))
            attempts.append(dict(initialization=name, verified=bool(candidate['verified']),
                diagnostics=candidate['dp_diagnostics'], search=candidate.get('search_metadata', {})))
            if candidate['verified']:
                selected = candidate
                break
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            attempts.append(dict(initialization=name, verified=False, error=str(exc)))
            print(f'    failed: {exc}', flush=True)
    if selected is None:
        raise ArithmeticError('No certified competition solution; attempts=' + json.dumps(attempts))
    # Fresh full joint game evaluation: this is different from perceived OE value.
    alpha, n = selected['policy'], experiment.n
    full = experiment.game.evaluate(np.broadcast_to(alpha[:, None], (n, n)),
                                    np.broadcast_to(alpha[None, :], (n, n)), tol=1e-10)
    selected['actual_values'] = full['values']
    selected['actual_value_diagnostics'] = {key: value for key, value in full.items()
                                           if key != 'values'}
    selected['attempts'] = attempts
    return selected


def economic_outcomes(params, n, p0, monopoly, competition, game):
    """Use actual calendar-time demand for every reported competitive outcome."""
    rows, policies = [], []
    for model, result in (('monopoly', monopoly), ('competition', competition)):
        require(result.get('verified', False), f'Unverified policy cannot be exported as {model}')
        alpha, grid = result['policy'], result['grid']
        if model == 'monopoly':
            sold = result['mu'] * result['demand']
            initial = float(np.interp(.5, grid, result['value']))
            outside_rate = 1 - float(sold.sum())
            perceived = initial
            symmetry = 0.
        else:
            sold = (result['nu'] * game.D).sum(axis=1)
            sold_b = (result['nu'] * game.D.T).sum(axis=0)
            symmetry = float(np.abs(sold - sold_b).sum())
            # n is odd, so the common initial state .5 is an exact grid node.
            initial = float(result['actual_values'][:, n//2, n//2].mean())
            outside_rate = float(np.sum(result['nu'] * game.outside))
            perceived = float(result['value'][n//2])
        rate = float(sold.sum())
        require(rate > 0, f'No representable stationary sales for {model}')
        use = float(sold @ alpha / rate)
        calendar_use = float(sold @ alpha)
        profit = float(sold @ (params.R - params.c1 - (params.c2-params.c1)*alpha))
        quality = params.p1 + (params.p2-params.p1)*use
        reputation = float(sold @ grid / rate)
        event = purchase_chain_statistics(grid, alpha, params)
        event_error = float(np.abs(sold / rate - event['sale_stationary']).sum())
        profit_identity = abs(profit-rate*(params.R-params.c1-(params.c2-params.c1)*use))
        require(abs(reputation-quality) < 1e-8 and event_error < 1e-7
                and profit_identity < 1e-12 and symmetry < 1e-9,
                f'Actual sales weighting failed for {model}: event_l1={event_error:g}, '
                f'reputation-quality={reputation-quality:g}')
        rows.append(dict(rho=params.rho, n=n, p0=p0, model=model, verified=True,
            product2_conditional=use, product2_calendar=calendar_use, sale_rate=rate,
            profit_calendar=profit, value_initial=initial, perceived_value_initial=perceived,
            outside_rate=outside_rate, industry_profit_calendar=profit*(2 if model=='competition' else 1),
            sale_law_l1=event_error, sale_weighted_reputation=reputation,
            sale_weighted_quality=quality, seller_sales_symmetry_l1=symmetry,
            mixed_nodes=int(np.sum((alpha>0) & (alpha<1)))))
        policies.extend(dict(rho=params.rho, n=n, p0=p0, model=model,
            reputation=float(x), product2_probability=float(a)) for x, a in zip(grid, alpha))
    return rows, policies


def check_bundle(directory, tag, manifest):
    for suffix in ('.json', '.npz'):
        path = directory / 'arrays' / (tag + suffix)
        expected = manifest['checkpoint_sha256'][path.name]
        require(sha256(path.read_bytes()).hexdigest() == expected,
                f'Checkpoint changed: {path}')
    return load_bundle(directory / 'arrays' / tag)


def export_run(directory, figures, manifest):
    from oblivious.oe_distribution_fixed_p0.code.reporting import make_figures

    cases = [check_bundle(directory, tag, manifest) for tag in manifest['completed']]
    rows = [row for case in cases for row in case['outcomes']]
    policy_rows = [row for case in cases for row in case['policy_rows']]
    write_csv(directory / 'summary_all_grids.csv', rows)
    write_csv(directory / 'policies_all_grids.csv', policy_rows)
    n = max(manifest['configuration']['grids'])
    fine = [row for row in rows if row['n'] == n]
    fine_policy = [row for row in policy_rows if row['n'] == n]
    write_csv(directory / 'summary.csv', fine)
    write_csv(directory / 'policies.csv', fine_policy)
    refinement = []
    for row in fine:
        for coarse in rows:
            if coarse['rho'] == row['rho'] and coarse['model'] == row['model'] and coarse['n'] != n:
                refinement.append(dict(rho=row['rho'], model=row['model'], coarse_n=coarse['n'], fine_n=n,
                    **{key+'_difference': row[key]-coarse[key] for key in
                       ('product2_conditional', 'product2_calendar', 'profit_calendar', 'value_initial', 'sale_rate')},
                    **{key+'_log10_ratio': float(np.log10(row[key])-np.log10(coarse[key]))
                       if row[key] > 0 and coarse[key] > 0 else None for key in
                       ('product2_calendar', 'profit_calendar', 'sale_rate')}))
    write_csv(directory / 'grid_refinement.csv', refinement)
    paths = make_figures(rows, fine_policy, figures, p0=manifest['configuration']['p0'], n=n)
    lines = ['Nuevo modelo: dos sellers y opción externa fija conocida',
        f"p0={manifest['configuration']['p0']}; n={n}; gamma={manifest['configuration']['parameters']['gamma']}",
        'Optimización OE con distribución marginal; la opción externa no se cierra con un punto fijo.',
        'Predicciones estacionarias deterministas en la grilla; no simulaciones Monte Carlo.',
        'Uso condicional = producto 2 / ventas propias; beneficio calendario = por seller y período.',
        'Valor inicial = beneficio total descontado en el entorno real desde reputación .5.',
        'Monopolio: un seller; competencia: dos sellers. Ambos compiten con el mismo p0.',
        'La certificación discreta no demuestra precisión de grilla, unicidad ni equilibrio de información completa.',
        '', 'rho | modelo | uso P2/venta | ventas/calendario | beneficio/calendario | valor inicial']
    for row in sorted(fine, key=lambda x:(x['rho'], x['model'])):
        lines.append(f"{row['rho']:g} | {row['model']} | {row['product2_conditional']:.8g} | "
            f"{row['sale_rate']:.8g} | {row['profit_calendar']:.8g} | {row['value_initial']:.8g}")
    if manifest.get('failures'):
        lines += ['', 'Casos sin certificado (no se presentan como equilibrios):']
        lines.extend(f"{tag}: {failure}" for tag, failure in manifest['failures'].items())
    (directory / 'RESULTADOS.txt').write_text('\n'.join(lines)+'\n')
    write_json(figures / 'figure_manifest.json', dict(created_utc=now(),
        configuration=manifest['configuration'], source_manifest=str((directory/'manifest.json').relative_to(ROOT)),
        plotting_source_sha256=sha256((HOME/'code/reporting.py').read_bytes()).hexdigest(),
        grid_sensitivity_selection=json.loads((figures/'grid_sensitivity_selection.json').read_text()),
        figure_files={path.name: sha256(path.read_bytes()).hexdigest() for path in paths}))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='p0_0p5')
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--rhos', nargs='+', type=float, default=RHOS)
    parser.add_argument('--grids', nargs='+', type=int, default=[21, 51])
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=1500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ('', '.', '..')
            or not 0 < args.p0 < 1 or any(n < 3 or n % 2 == 0 for n in args.grids)
            or len(set(args.grids)) != len(args.grids) or len(set(args.rhos)) != len(args.rhos)
            or any(not .5 < rho < 1 for rho in args.rhos)
            or args.order < 8 or args.tol <= 0 or args.dp_tol <= 0 or args.max_evaluations < 1):
        parser.error('Require a run name, 0<p0<1, distinct .5<rho<1, odd grids>=3 and positive numerical settings')
    return args


def main():
    args = arguments()
    directory, figures = HOME/'data'/args.name, HOME/'figures'/args.name
    path = directory/'manifest.json'
    config = dict(p0=args.p0, rhos=args.rhos, grids=sorted(args.grids), order=args.order,
        tol=args.tol, dp_tol=args.dp_tol, max_evaluations=args.max_evaluations,
        parameters={key:value for key,value in asdict(Parameters()).items() if key != 'rho'},
        model='New marginal-distribution OE with known fixed external option',
        initial_reputation=.5, stationary_predictions=True, monte_carlo=False)
    if args.plots_only:
        export_run(directory, figures, json.loads(path.read_text()))
        return
    if args.resume:
        manifest = json.loads(path.read_text())
        require(manifest['configuration'] == config, 'Saved configuration differs')
        require(manifest['numerical_sources'] == numerical_sources(), 'Numerical sources changed; use a new name')
    else:
        if directory.exists() or figures.exists():
            raise SystemExit('Existing results protected; use --resume, --plots-only or another --name')
        directory.mkdir(parents=True)
        manifest = dict(status='incomplete', created_utc=now(), configuration=config,
            numerical_sources=numerical_sources(), completed=[], failures={}, checkpoint_sha256={})
        write_json(path, manifest)
    (directory/'arrays').mkdir(exist_ok=True)
    previous_rho = {}
    for rho in args.rhos:
        previous_grid = None
        for n in sorted(args.grids):
            tag = case_tag(rho, n)
            if tag in manifest['completed']:
                case = check_bundle(directory, tag, manifest)
                previous_grid = case['competition']
                previous_rho[n] = case['competition']
                continue
            print(f'[{now()}] new fixed-p0 model: rho={rho:g}, n={n}, p0={args.p0:g}', flush=True)
            started = time.monotonic()
            try:
                params = Parameters(rho=rho, **config['parameters'])
                experiment = FixedOutsideMarginalExperiment(n=n, params=params, p0=args.p0,
                    order=args.order, tol=args.tol, dp_tol=args.dp_tol)
                demand_audit = audit_demand(experiment)
                monopoly = solve_monopoly(params, n, args.p0, args.dp_tol)
                seeds = []
                for name, old in (('previous_grid', previous_grid), ('previous_rho', previous_rho.get(n))):
                    if old is not None:
                        seeds.append((name, np.interp(experiment.grid, old['grid'], old['policy'])))
                competition = solve_competition(experiment, monopoly, seeds, args.max_evaluations)
                outcomes, policy_rows = economic_outcomes(params, n, args.p0, monopoly, competition, experiment.game)
                case = dict(rho=rho, n=n, p0=args.p0, configuration=experiment.configuration(),
                    monopoly=monopoly, competition=competition, demand_audit=demand_audit,
                    outcomes=outcomes, policy_rows=policy_rows, elapsed_seconds=time.monotonic()-started)
                require(manifest['numerical_sources'] == numerical_sources(),
                        'Numerical sources changed during this case; no checkpoint committed')
                save_bundle(directory/'arrays'/tag, case)
                for suffix in ('.json', '.npz'):
                    checkpoint = directory/'arrays'/(tag+suffix)
                    manifest['checkpoint_sha256'][checkpoint.name] = sha256(checkpoint.read_bytes()).hexdigest()
                manifest['completed'].append(tag)
                manifest['failures'].pop(tag, None)
                previous_grid, previous_rho[n] = competition, competition
                print(f'  CERTIFIED in {case["elapsed_seconds"]:.1f}s; '
                      f'usage M={outcomes[0]["product2_conditional"]:.6g}, '
                      f'C={outcomes[1]["product2_conditional"]:.6g}', flush=True)
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                manifest['failures'][tag] = str(exc)
                print(f'  NOT CERTIFIED: {exc}', flush=True)
            write_json(path, manifest)
    manifest.update(status='complete' if not manifest['failures'] else 'complete_with_unverified_cases',
                    finished_utc=now())
    write_json(path, manifest)
    export_run(directory, figures, manifest)
    print(f'Saved {len(manifest["completed"])} certified cases and comparison figures in {figures}', flush=True)


if __name__ == '__main__':
    main()
