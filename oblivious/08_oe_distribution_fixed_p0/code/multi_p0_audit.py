"""Audit saved fixed-p0 checkpoints without solving an equilibrium or a law.

Reconstruct event kernels and rewards, check saved invariant/value equations,
and recompute beliefs and economic exports. Existing positive/reversed-order
solver certificates are required, not rerun or replaced. This audit applies to
the recorded finite grids; it does not establish continuous-state convergence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.special import betainc, betaincc

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import (
    RHOS, campaign_directory, source_fingerprints,
)
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, case_tag, now, write_json,
)

P0S = (.1, .3, .7, .8)
GRIDS = (51, 101, 151)
LIMITS = dict(probability_mass=1e-10, transition=1e-12, stationary=1e-10,
              value_equation=5e-10, action_regret=2e-10, belief=1e-9,
              saved_array=1e-10, outcome=1e-10, sale_law=1e-7)


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def finite(array, shape, label):
    result = np.asarray(array, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), 'Malformed/nonfinite '+label)
    return result


def interpolation(grid, targets):
    """Independent two-node event allocation; no production transition helper."""
    targets = np.asarray(targets)
    lower = np.searchsorted(grid, targets, side='right')-1
    lower = np.minimum(np.maximum(lower, 0), len(grid)-2)
    upper_weight = (targets-grid[lower])/(grid[lower+1]-grid[lower])
    indices = np.column_stack((lower, lower+1)).ravel()
    weights = np.column_stack((1-upper_weight, upper_weight)).ravel()
    return csr_matrix((weights, (np.repeat(np.arange(len(grid)), 2), indices)),
                      shape=(len(grid), len(grid)))


class TargetCache:
    def __init__(self):
        self.values = {}

    def get(self, configuration):
        p0, n = configuration['p0'], configuration['n']
        parameters = configuration['parameters']
        key = (p0, n, configuration['order'], tuple(sorted(parameters.items())))
        if key not in self.values:
            params, grid = Parameters(**parameters), np.linspace(0, 1, n)
            game = OutsideOptionGame(grid, params, p0, configuration['order'])
            success = interpolation(grid, params.rho*grid+1-params.rho)
            failure = interpolation(grid, params.rho*grid)
            identity = eye(n, format='csr')
            a, b = 1+grid/(1-params.rho), 1+(1-grid)/(1-params.rho)
            outside = np.outer(betainc(a, b, p0), betainc(a, b, p0))
            self.values[key] = dict(params=params, grid=grid, D=game.D,
                outside=outside, monopoly_demand=betaincc(a, b, p0),
                S=success, F=failure,
                AS=kron(success, identity, format='csr'),
                AF=kron(failure, identity, format='csr'),
                BS=kron(identity, success, format='csr'),
                BF=kron(identity, failure, format='csr'))
        return self.values[key]


def event_kernel(target, alpha):
    """Assemble own/rival success/failure and outside events independently."""
    n, p, D = len(alpha), target['params'], target['D']
    quality_a = np.broadcast_to((p.p1+(p.p2-p.p1)*alpha)[:, None], (n, n))
    quality_b = quality_a.T
    P = (diags(target['outside'].ravel())
         +diags((D*quality_a).ravel())@target['AS']
         +diags((D*(1-quality_a)).ravel())@target['AF']
         +diags((D.T*quality_b).ravel())@target['BS']
         +diags((D.T*(1-quality_b)).ravel())@target['BF']).tocsr()
    rewards = np.stack((D*(p.R-p.c1-(p.c2-p.c1)*alpha[:, None]),
                        D.T*(p.R-p.c1-(p.c2-p.c1)*alpha[None, :])))
    return P, rewards


def own_equations(target, alpha, demand, value):
    p, S, F = target['params'], target['S'], target['F']
    quality = p.p1+(p.p2-p.p1)*alpha
    K = (diags(1-demand)+diags(demand*quality)@S+diags(demand*(1-quality))@F).tocsr()
    reward = demand*(p.R-p.c1-(p.c2-p.c1)*alpha)
    advantage = -(p.c2-p.c1)+p.gamma*(p.p2-p.p1)*((S-F)@value)
    regret = np.maximum(advantage, 0)-alpha*advantage
    return K, advantage, float(np.max(abs(reward+p.gamma*(K@value)-value))), float(regret.max())


def audit_monopoly(solution, target, metrics):
    n, p = len(target['grid']), target['params']
    alpha = finite(solution['policy'], (n,), 'monopoly policy')
    mu = finite(solution['mu'], (n,), 'monopoly law')
    value = finite(solution['value'], (n,), 'monopoly value')
    require(solution['verified'] and np.all((alpha >= 0)&(alpha <= 1)) and np.all(mu >= 0),
            'Unverified/infeasible monopoly checkpoint')
    demand = target['monopoly_demand']
    metrics['monopoly_law_mass_error'] = abs(float(mu.sum())-1)
    metrics['monopoly_saved_demand_max_error'] = float(np.max(abs(demand-solution['demand'])))
    K, advantage, residual, regret = own_equations(target, alpha, demand, value)
    metrics['monopoly_stationary_l1'] = float(np.abs(K.T@mu-mu).sum())
    metrics['monopoly_value_bellman_max'] = residual
    metrics['monopoly_conditional_regret_max'] = regret
    metrics['monopoly_saved_advantage_max_error'] = float(np.max(abs(advantage-solution['advantage'])))
    sold = mu*demand
    sale_rate = float(sold.sum())
    require(sale_rate > 0, 'Unrepresentable monopoly sales in saved checkpoint')
    eta = finite(solution['sale_stationary'], (n,), 'monopoly sale law')
    metrics['monopoly_sale_law_l1'] = float(np.abs(sold/sale_rate-eta).sum())
    return dict(alpha=alpha, sold=sold, sale_rate=sale_rate,
                outside_rate=1-sale_rate, value_initial=float(value[n//2]))


def raw_outcomes(state, target, sellers):
    alpha, sold, rate = state['alpha'], state['sold'], state['sale_rate']
    p = target['params']
    require(rate > 0, 'Unrepresentable saved stationary sales')
    product2_calendar = float(sold@alpha)
    use = product2_calendar/rate
    profit = float(sold@(p.R-p.c1-(p.c2-p.c1)*alpha))
    return dict(product2_conditional=use, product2_calendar=product2_calendar,
        sale_rate=rate, outside_rate=state['outside_rate'], profit_calendar=profit,
        value_initial=state['value_initial'], industry_profit_calendar=sellers*profit,
        sale_weighted_reputation=float(sold@target['grid']/rate),
        sale_weighted_quality=p.p1+(p.p2-p.p1)*use,
        mixed_nodes=int(np.sum((alpha > 0)&(alpha < 1))))


def audit_monopoly_exclusion(prefix, expected_sources, add_input, audited_bundles):
    """Audit a sufficient one-model grid-failure witness without a joint solve."""
    prefix = Path(prefix)
    for suffix in ('.json','.npz'):
        path = prefix.with_suffix(suffix)
        add_input(path)
        audited_bundles[str(path.relative_to(ROOT))] = digest(path)
    witness = load_bundle(prefix)
    configuration = witness['configuration']
    p0, rho, n = configuration['p0'], configuration['rho'], configuration['n']
    require(p0 in P0S and rho in RHOS and n == 151
            and configuration['parameters'] == asdict(Parameters(rho=rho)),
            'Supplemental monopoly witness calibration differs')
    require(witness['numerical_sources'], 'Supplemental monopoly numerical sources missing')
    for source, expected in witness['numerical_sources'].items():
        require(expected_sources.get(source) == expected, 'Supplemental monopoly numerical source differs')
        add_input(ROOT/source, expected)
    for source, expected in witness['reference_input_sha256'].items():
        add_input(ROOT/source, expected)
    grid, params = np.linspace(0,1,n), Parameters(**configuration['parameters'])
    a, b = 1+grid/(1-rho), 1+(1-grid)/(1-rho)
    # This witness needs only one-dimensional events. No demand matrix,
    # positive invariant solver, dynamic program, or competitive result is made.
    target = dict(grid=grid, params=params, monopoly_demand=betaincc(a,b,p0),
                  S=interpolation(grid,rho*grid+1-rho), F=interpolation(grid,rho*grid))
    require(np.array_equal(witness['monopoly']['grid'],grid), 'Supplemental monopoly grid differs')
    residuals = {}
    state = audit_monopoly(witness['monopoly'],target,residuals)
    fine = raw_outcomes(state,target,1)
    require(len(witness['outcomes']) == 1, 'Supplemental monopoly must export one model only')
    row = witness['outcomes'][0]
    require(row['model'] == 'monopoly' and row['verified'] and row['p0'] == p0
            and row['rho'] == rho and row['n'] == n, 'Supplemental monopoly export differs')
    core = ('product2_conditional','product2_calendar','sale_rate','outside_rate',
            'profit_calendar','value_initial')
    residuals['exported_outcome_max_error'] = max(abs(float(row[k])-fine[k]) for k in core)
    for name,value in residuals.items():
        limit = (LIMITS['action_regret'] if 'regret' in name else
                 LIMITS['value_equation'] if 'bellman' in name else
                 LIMITS['sale_law'] if 'sale_law' in name else
                 LIMITS['belief'] if 'demand' in name else LIMITS['saved_array'])
        require(np.isfinite(value) and value <= limit,
                f'Supplemental monopoly {name}={value:g} exceeds audit limit {limit:g}')
    coarse_prefix = campaign_directory(p0,rho)/'arrays'/(case_tag(rho,101)+'_marginal')
    for suffix in ('.json','.npz'):
        path = coarse_prefix.with_suffix(suffix)
        key = str(path.relative_to(ROOT))
        require(key in witness['reference_input_sha256'], 'Supplemental witness must fingerprint its coarse checkpoint')
        add_input(path,witness['reference_input_sha256'][key])
    coarse_case = load_bundle(coarse_prefix)
    require(coarse_case['p0'] == p0 and coarse_case['rho'] == rho and coarse_case['n'] == 101,
            'Supplemental monopoly coarse calibration differs')
    coarse = next(item for item in coarse_case['outcomes'] if item['model'] == 'monopoly')
    probability = {key:abs(fine[key]-coarse[key]) for key in core[:4]}
    money = {key:abs(fine[key]-coarse[key])/max(abs(fine[key]),abs(coarse[key]))
             if fine[key] or coarse[key] else 0. for key in core[4:]}
    recorded = witness['grid_stability']
    require(recorded['p0'] == p0 and recorded['rho'] == rho and recorded['model'] == 'monopoly'
            and recorded['coarse_n'] == 101 and recorded['fine_n'] == 151
            and recorded['passed'] is False, 'Supplemental witness is not a declared failed grid check')
    differences = {**{k+'_absolute_change':v for k,v in probability.items()},
                   **{k+'_relative_change':v for k,v in money.items()}}
    for name,value in differences.items():
        require(np.isfinite(value) and abs(float(recorded[name])-value) <= 1e-12,
                'Supplemental monopoly grid difference disagrees: '+name)
    require(any(value > .05+1e-12 for value in [*probability.values(),*money.values()]),
            'Supplemental monopoly does not actually fail the declared 5pp/5% criterion')
    return dict(verified=True, p0=p0, rho=rho, model='monopoly', coarse_n=101,fine_n=151,
        checkpoint=str(prefix.relative_to(ROOT)),
        checkpoint_sha256={prefix.with_suffix(s).name:digest(prefix.with_suffix(s)) for s in ('.json','.npz')},
        residuals=residuals, recomputed_grid_stability=dict(passed=False,**differences),
        sufficient_for_common_intersection_exclusion=True,
        scope='Verified monopoly failure suffices for an all-model/all-p0 intersection; no competitive n151 result is inferred')


def audit_case(case, closure, target):
    n, p, grid = case['n'], target['params'], target['grid']
    require(np.array_equal(case['competition']['grid'], grid)
            and np.array_equal(case['monopoly']['grid'], grid), 'Saved grid differs')
    require(case['demand_audit']['verified'], 'Production demand audit is not verified')
    metrics = {}
    projection = case['configuration'].get('demand_endpoint_roundoff_projection')
    if projection is not None:
        raw = target['D']
        projected = np.clip(raw, 0, 1)
        change = float(abs(projected-raw).max())
        valid = (raw >= 0)&(raw <= 1)
        changed = np.argwhere(raw != projected)
        require(closure == 'conditional' and 0 < projection['endpoint_tolerance'] <= 1e-12
                and change <= projection['endpoint_tolerance']
                and projection['positive_probability_floor'] == 0
                and projection['valid_entries_unchanged']
                and np.array_equal(raw[valid], projected[valid])
                and projection['maximum_absolute_change'] == change
                and projection['changed_entry_count'] == len(changed)
                and projection['below_zero_count'] == int(np.sum(raw < 0))
                and projection['above_one_count'] == int(np.sum(raw > 1))
                and projection['raw_matrix_sha256'] == sha256(raw.tobytes()).hexdigest()
                and projection['projected_matrix_sha256'] == sha256(projected.tobytes()).hexdigest(),
                'Endpoint recovery metadata differs from independently reconstructed raw demand')
        for source, expected in projection['helper_input_sha256'].items():
            require(case['reference_input_sha256'].get(source) == expected,
                    'Endpoint recovery search source lacks checked provenance')
        metrics['demand_endpoint_projection_max'] = change
        # Reconstruct the explicitly recorded finite game without modifying the
        # cached raw target reused by the marginal checkpoint.
        target = dict(target, D=projected)
    monopoly = audit_monopoly(case['monopoly'], target, metrics)
    solution = case['competition']
    require(solution['verified'] and solution['dp_diagnostics']['verified']
            and solution['stationary_diagnostics']['verified'], 'Saved competition lacks original certificates')
    alpha = finite(solution['policy'], (n,), 'competition policy')
    nu = finite(solution['nu'], (n,n), 'joint stationary law')
    require(np.all((alpha >= 0)&(alpha <= 1)) and np.all(nu >= 0), 'Infeasible saved policy/law')
    metrics['joint_law_mass_error'] = abs(float(nu.sum())-1)
    metrics['joint_symmetry_l1'] = float(abs(nu-nu.T).sum())
    mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
    metrics['saved_marginal_max_error'] = max(float(abs(mu_a-solution['mu_a']).max()),
                                              float(abs(mu_b-solution['mu_b']).max()))
    D = target['D']
    metrics['market_event_partition_max'] = float(abs(D+D.T+target['outside']-1).max())
    P, reward = event_kernel(target, alpha)
    require(P.data.min(initial=0) >= -1e-14, 'Negative raw event transition')
    metrics['joint_transition_row_sum_max'] = float(abs(np.asarray(P.sum(axis=1)).ravel()-1).max())
    metrics['joint_stationary_l1'] = float(abs(P.T@nu.ravel()-nu.ravel()).sum())
    values = finite(solution['actual_values'], (2,n,n), 'actual joint values')
    value_vectors = values.reshape(2,-1).T
    metrics['actual_value_bellman_max'] = float(abs(reward.reshape(2,-1).T+p.gamma*(P@value_vectors)-value_vectors).max())
    positive = mu_a > 0
    if closure == 'conditional':
        diag = solution['conditional_diagnostics']
        require(diag['verified'] and diag['independent_order_verified'] and diag['positive_mass_floor'] == 0,
                'Original positive/reversed-order conditional certificate missing')
        conditional = np.broadcast_to(mu_b, (n,n)).copy()
        conditional[positive] = nu[positive]/mu_a[positive, None]
        saved = finite(solution['conditional_mu'], (n,n), 'conditional belief')
        require(np.all(saved >= 0), 'Negative saved conditional belief')
        metrics['conditional_belief_row_l1_max'] = float(abs(conditional-saved).sum(axis=1).max())
        metrics['conditional_belief_row_sum_max'] = float(abs(saved.sum(axis=1)-1).max())
        demand = (D*conditional).sum(axis=1)
        metrics['conditional_demand_max_error'] = max(float(abs(demand-solution['conditional_demand']).max()),
                                                       float(abs(demand-solution['actual_conditional_demand']).max()))
    else:
        demand = D@mu_b
    metrics['saved_projected_demand_max_error'] = float(abs(demand-solution['demand']).max())
    projected_value = finite(solution['value'], (n,), 'perceived own-state value')
    _, advantage, residual, regret = own_equations(target, alpha, demand, projected_value)
    metrics['perceived_value_bellman_max'] = residual
    metrics['competition_conditional_regret_max'] = regret
    metrics['competition_saved_advantage_max_error'] = float(abs(advantage-solution['advantage']).max())
    sold = (nu*D).sum(axis=1)
    sold_b = (nu*D.T).sum(axis=0)
    metrics['seller_sales_symmetry_l1'] = float(abs(sold-sold_b).sum())
    rate = float(sold.sum())
    outside = float(np.sum(nu*target['outside']))
    metrics['market_outside_identity_max'] = abs(outside+2*rate-1)
    competition = dict(alpha=alpha, sold=sold, sale_rate=rate, outside_rate=outside,
        value_initial=float(values[:,n//2,n//2].mean()))
    expected = {'monopoly':raw_outcomes(monopoly, target, 1),
                'conditional_competition' if closure == 'conditional' else 'competition':raw_outcomes(competition,target,2)}
    expected_names = {'conditional_competition'} if closure == 'conditional' else {'monopoly','competition'}
    require({row['model'] for row in case['outcomes']} == expected_names
            and len(case['outcomes']) == len(expected_names), 'Malformed exported model rows')
    differences = []
    for row in case['outcomes']:
        require(row['verified'] and row['p0'] == case['p0'] and row['rho'] == case['rho']
                and row['n'] == n, 'Outcome calibration differs')
        for key, value in expected[row['model']].items():
            require(np.isfinite(row[key]), 'Nonfinite exported outcome '+key)
            differences.append(abs(float(row[key])-value))
        perceived = (case['monopoly']['value'][n//2] if row['model'] == 'monopoly'
                     else projected_value[n//2])
        differences.append(abs(float(row['perceived_value_initial'])-float(perceived)))
    for name in expected_names:
        rows = sorted((row for row in case['policy_rows'] if row['model'] == name), key=lambda row:row['reputation'])
        policy = monopoly['alpha'] if name == 'monopoly' else alpha
        require(len(rows) == n and np.array_equal([row['reputation'] for row in rows], grid)
                and np.array_equal([row['product2_probability'] for row in rows], policy)
                and all(row['p0'] == case['p0'] and row['rho'] == case['rho'] and row['n'] == n for row in rows),
                'Policy export differs from saved exact policy')
    metrics['exported_outcome_max_error'] = max(differences, default=0.)
    limits = {}
    for name, value in metrics.items():
        if 'endpoint_projection' in name:
            limit = 1e-12
        elif 'regret' in name:
            limit = LIMITS['action_regret']
        elif 'bellman' in name:
            limit = LIMITS['value_equation']
        elif 'sale_law' in name:
            limit = LIMITS['sale_law']
        elif 'stationary' in name:
            limit = LIMITS['stationary']
        elif 'conditional_belief' in name or 'demand' in name:
            limit = LIMITS['belief']
        elif 'transition' in name or 'partition' in name:
            limit = LIMITS['transition']
        elif 'outcome' in name:
            limit = LIMITS['outcome']
        else:
            limit = LIMITS['saved_array']
        limits[name] = limit
        require(np.isfinite(value) and value <= limit, f'{name}={value:g} exceeds audit limit {limit:g}')
    return dict(p0=case['p0'], rho=case['rho'], n=n, closure=closure,
        verified=True, positive_own_rows=int(positive.sum()), zero_own_rows=int((~positive).sum()),
        minimum_positive_own_mass=float(mu_a[positive].min()) if np.any(positive) else None,
        residuals=metrics, audit_limits=limits)


def run_audit(p0s=P0S, rhos=RHOS, output=None):
    cache, records, statuses, inputs, audited_bundles = TargetCache(), [], [], {}, {}
    expected_sources = source_fingerprints()

    def add_input(path, expected=None):
        path = Path(path)
        key, actual = str(path.relative_to(ROOT)), digest(path)
        require(expected is None or actual == expected, 'Input/source hash differs: '+key)
        require(key not in inputs or inputs[key] == actual, 'Input changed during audit: '+key)
        inputs[key] = actual

    add_input(Path(__file__).resolve())
    for p0 in p0s:
        for rho in rhos:
            directory = campaign_directory(p0, rho)
            path = directory/'manifest.json'
            require(path.exists(), 'Current campaign manifest missing: '+str(path))
            manifest = json.loads(path.read_text())
            require(manifest.get('current_grid') is None and manifest.get('current_model') is None
                    and manifest['status'] != 'running', 'Audit requires closed numerical campaigns: '+str(path))
            require(manifest['numerical_sources'] == expected_sources, 'Immutable numerical source set differs')
            add_input(path)
            for source, expected in manifest['numerical_sources'].items():
                add_input(ROOT/source, expected)
            for source, expected in manifest['reference_input_sha256'].items():
                add_input(ROOT/source, expected)
            configuration = manifest['configuration']
            require(configuration['p0'] == p0 and configuration['rho'] == rho
                    and configuration['parameters'] == {k:v for k,v in asdict(Parameters()).items() if k != 'rho'},
                    'Campaign calibration differs')
            statuses.append(dict(p0=p0, rho=rho, status=manifest['status'],
                completed=list(manifest['completed']), failures=manifest.get('failures', {}),
                missing_base_checkpoints=[case_tag(rho,n)+'_'+closure for n in (51,101)
                    for closure in ('marginal','conditional')
                    if case_tag(rho,n)+'_'+closure not in manifest['completed']]))
            for tag in manifest['completed']:
                closure = 'conditional' if tag.endswith('_conditional') else 'marginal'
                require(tag.endswith('_'+closure), 'Unknown completed model tag: '+tag)
                for suffix in ('.json','.npz'):
                    file = directory/'arrays'/(tag+suffix)
                    expected = manifest['checkpoint_sha256'][file.name]
                    add_input(file, expected)
                    audited_bundles[str(file.relative_to(ROOT))] = expected
                case = load_bundle(directory/'arrays'/tag)
                n, config = case['n'], case['configuration']
                require(n in GRIDS and tag == case_tag(rho,n)+'_'+closure
                        and case['rho'] == rho and case['p0'] == p0
                        and config['n'] == n and config['p0'] == p0
                        and config['parameters'] == asdict(Parameters(rho=rho))
                        and config['order'] == configuration['order']
                        and config['tolerance'] == configuration['tol']
                        and config['dp_tolerance'] == configuration['dp_tol']
                        and config.get('closure','').lower().startswith('conditional') == (closure == 'conditional'),
                        'Checkpoint calibration/tag differs: '+tag)
                for source, expected in case.get('reference_input_sha256', {}).items():
                    require(manifest['reference_input_sha256'].get(source) == expected,
                            'Checkpoint proposal provenance differs from manifest')
                    add_input(ROOT/source, expected)
                result = audit_case(case, closure, cache.get(config))
                result['checkpoint'] = str((directory/'arrays'/tag).relative_to(ROOT))
                result['checkpoint_sha256'] = {tag+s:manifest['checkpoint_sha256'][tag+s] for s in ('.json','.npz')}
                records.append(result)
                print(f'AUDITED p0={p0:g} rho={rho:g} n={n} {closure}', flush=True)
    require(records, 'No completed checkpoints were available for audit')
    supplemental = []
    witness_prefix = HOME/'data/p0_sweep/provenance/monopoly_only_p0_0p8_rho_0p9_n151'
    if .8 in p0s and .9 in rhos and witness_prefix.with_suffix('.json').exists():
        supplemental.append(audit_monopoly_exclusion(witness_prefix,expected_sources,add_input,audited_bundles))
        print('AUDITED supplemental monopoly exclusion p0=.8 rho=.9 n101→151',flush=True)
    for source, expected in inputs.items():
        require(digest(ROOT/source) == expected, 'Audited input changed before publication: '+source)
    maxima = {}
    for record in records:
        for name, value in record['residuals'].items():
            maxima[name] = max(maxima.get(name, 0.), value)
    review = dict(created_utc=now(), verified=True,
        scope='Independent saved finite-grid event/economic audit; original positive and reversed-order certificates required, no fresh GTH/equilibrium/DP solves',
        limitations='Does not prove uniqueness, continuous-state convergence, or full-information Markov-perfect equilibrium; grid sensitivity is reported separately',
        p0s=list(p0s), rhos=list(rhos), audited_bundle_count=len(records),
        audited_supplemental_monopoly_count=len(supplemental),
        audited_total_bundle_count=len(records)+len(supplemental),
        audited_file_count=len(audited_bundles), audited_input_count=len(inputs),
        cached_target_game_count=len(cache.values), audit_source_sha256=digest(Path(__file__)),
        audit_source=str(Path(__file__).resolve().relative_to(ROOT)),
        campaign_statuses=statuses, residual_maxima=maxima, cases=records,
        supplemental_monopoly_exclusions=supplemental,
        audited_bundle_sha256=audited_bundles, audited_input_sha256=inputs, thresholds=LIMITS)
    destination = Path(output) if output is not None else HOME/'data/p0_sweep/independent_review.json'
    write_json(destination, review)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0s', type=float, nargs='+', default=list(P0S))
    parser.add_argument('--rhos', type=float, nargs='+', default=list(RHOS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (len(set(args.p0s)) != len(args.p0s) or any(p0 not in P0S for p0 in args.p0s)
            or len(set(args.rhos)) != len(args.rhos) or any(rho not in RHOS for rho in args.rhos)):
        parser.error('Use distinct current p0 values .1,.3,.7,.8 and retained rho cases')
    review = run_audit(args.p0s, args.rhos, args.output)
    print(json.dumps(dict(verified=review['verified'], audited_bundle_count=review['audited_bundle_count'],
                          residual_maxima=review['residual_maxima'])), flush=True)


if __name__ == '__main__':
    main()
