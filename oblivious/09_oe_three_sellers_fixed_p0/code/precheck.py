"""Cheap one-seller grid diagnostics before allocating N=3 stationary solves."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import betainc, betaincc

from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_p0.code.policy import solve_dp, evaluate_policy, transition_matrix
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics


HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
P0S, RHOS = (.1, .3, .5, .7, .8, .9), (.6, .7, .8, .9)
GRIDS = (21, 31, 41, 51, 61, 81, 101)
PROBABILITIES = ('product2_conditional', 'outside_rate', 'sale_rate')
EARNINGS = ('profit_calendar', 'value_initial')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), ROOT/'oblivious/03_oe_following_monopoly/code/event_chain.py']
    paths += [ROOT/'oblivious/02_oe_p0/code'/name for name in (
        'primitives.py', 'policy.py', 'core.py', 'positive_stationary.py', 'positive_stationary.cpp')]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def solve_one(p0, rho, n):
    started = time.monotonic()
    params, grid = Parameters(rho=rho), np.linspace(0, 1, n)
    a, b = posterior(grid, params)
    demand, cdf = betaincc(a, b, p0), betainc(a, b, p0)
    best = solve_dp(grid, demand, params, tol=1e-10)
    alpha = best['policy']
    evaluated = evaluate_policy(grid, demand, alpha, params)
    event = purchase_chain_statistics(grid, alpha, params)
    eta = event['sale_stationary']
    holding = eta/demand
    time_per_sale = float(holding.sum())
    mu = holding/time_per_sale
    rate = 1/time_per_sale
    use = float(eta@alpha)
    profit = float(eta@(params.R-params.c1-(params.c2-params.c1)*alpha)/time_per_sale)
    outside = float(mu@cdf)
    regret = float(np.max(np.maximum(evaluated['advantage'], 0)-alpha*evaluated['advantage']))
    kernel = transition_matrix(grid, demand, alpha, params)
    stationary = float(np.abs(kernel.T@mu-mu).sum())
    verified = bool(regret <= 1e-10 and evaluated['residual'] <= 1e-10
        and stationary <= 1e-11 and abs(outside+rate-1) <= 1e-12
        and np.isfinite([profit, use, outside, rate, time_per_sale]).all())
    if not verified:
        raise ArithmeticError(f'Unverified monopoly diagnostic p0={p0},rho={rho},n={n}')
    return dict(p0=p0, rho=rho, n=n, sellers=1, closure='monopoly', verified=verified,
        product2_conditional=use, sale_rate=rate, outside_rate=outside,
        profit_calendar=profit, value_initial=float(evaluated['value'][n//2]),
        bellman_residual=float(evaluated['residual']), conditional_regret=regret,
        stationary_residual=stationary, sale_chain_reverse_l1=event['stability_l1'],
        elapsed_seconds=time.monotonic()-started)


def compare(earlier, later):
    result = dict(p0=later['p0'], rho=later['rho'], coarse_n=earlier['n'], fine_n=later['n'])
    for name in PROBABILITIES+EARNINGS:
        a, b = earlier[name], later[name]
        change = abs(b-a)
        if name in EARNINGS:
            change = change/max(abs(a), abs(b)) if a or b else 0.
        result[name+('_absolute_change' if name in PROBABILITIES else '_relative_change')] = change
    result['passed'] = all(value <= .05+1e-12 for name, value in result.items()
                           if name.endswith(('_absolute_change', '_relative_change')))
    result['passed_without_profit'] = all(value <= .05+1e-12 for name, value in result.items()
        if name.endswith(('_absolute_change', '_relative_change')) and not name.startswith('profit_calendar_'))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    target = HOME/'data/rho_p0_precheck.csv'
    manifest_path = HOME/'data/rho_p0_precheck_manifest.json'
    comparisons_path = HOME/'data/rho_p0_precheck_stability.csv'
    if target.exists() or manifest_path.exists() or comparisons_path.exists():
        raise FileExistsError('Existing precheck outputs are preserved; choose a new campaign explicitly')
    frozen = sources()
    rows, comparisons = [], []
    for p0 in P0S:
        for rho in RHOS:
            grids = (*GRIDS, 151, 201) if p0 >= .7 and rho >= .8 else GRIDS
            earlier = None
            for n in grids:
                row = solve_one(p0, rho, n)
                rows.append(row)
                if earlier is not None:
                    comparisons.append(compare(earlier, row))
                earlier = row
    if frozen != sources():
        raise RuntimeError('Precheck numerical sources changed during the run')
    target.parent.mkdir(parents=True, exist_ok=True)
    write_csv(target, rows)
    write_csv(comparisons_path, comparisons)
    manifest = dict(status='complete_verified_monopoly_precheck', created_utc=datetime.now(timezone.utc).isoformat(),
        p0s=list(P0S), rhos=list(RHOS), base_grids=list(GRIDS),
        supplemental_grids=dict(p0_at_least=.7, rho_at_least=.8, grids=[151, 201]),
        economics={k: v for k, v in asdict(Parameters()).items() if k != 'rho'},
        numerical_sources=frozen, certified_monopoly_grid_cases=len(rows),
        probability_change_limit=.05, symmetric_relative_earnings_change_limit=.05,
        relative_change='abs(fine-coarse)/max(abs(fine),abs(coarse)); zero if both are zero',
        calendar_identity='sale_rate=1/sum(eta/demand); profit=eta@margin/sum(eta/demand)',
        certification='Own-state DP and unweighted regret; sale-chain positive stationary checks in both orders; calendar invariant residual',
        scope='Monopoly diagnostic only. No N=2/N=3 equilibrium solve; no continuous convergence guarantee.',
        output_sha256={str(path.relative_to(ROOT)): digest(path) for path in (target, comparisons_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows=len(rows), comparisons=len(comparisons), output=str(target)), indent=2))


if __name__ == '__main__':
    main()
