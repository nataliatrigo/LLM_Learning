"""Independent equations and provenance audit for the four-seller extension.

Only stored policies are evaluated.  In particular this module never searches
for an equilibrium or resumes computation.  The N1--N3 controls reuse the
immutable, independently audited source bundles, verified byte for byte.
"""

from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.special import betainc, betaln

from oblivious.oe_p0.code.primitives import Parameters, interpolation_matrix, posterior
from oblivious.oe_p0.code.positive_stationary import positive_invariant
from . import audit as frozen


HOME, ROOT = frozen.HOME, frozen.ROOT
BASELINE = HOME/'data/high_p0_refinement_41'
require, close = frozen.require, frozen.close


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_map():
    references = frozen.source_map()
    path = Path(__file__).resolve()
    references[str(path.relative_to(ROOT))] = _digest(path)
    return references


def orbit_observables(representatives, law, demand, policy, params):
    """Reconstruct tagged marginal/sales and symmetric reward without helpers.

    A stationary orbit has four equally likely choices of tagged seller,
    including repeated coordinates.  ``law`` is total mass of each orbit;
    multiplying it by the orbit size here would count its permutations twice.
    """
    representatives = np.asarray(representatives, dtype=int)
    law, demand = np.asarray(law), np.asarray(demand)
    n = len(policy)
    weights = np.broadcast_to(law[:, None]/4, representatives.shape)
    marginal = np.bincount(representatives.ravel(), weights=weights.ravel(), minlength=n)
    sales = np.bincount(representatives.ravel(),
                        weights=(weights*demand).ravel(), minlength=n)
    margins = params.R-params.c1-(params.c2-params.c1)*np.asarray(policy)
    reward_mean = np.sum(demand*margins[representatives], axis=1)/4
    return marginal, sales, reward_mean


def independent_marginal_beliefs(game, law):
    """Average the actual three-rival joint law, with focal state replaced.

    The rivals in one orbit remain jointly distributed.  No product of
    single-rival marginals appears.  This deliberately does not use the
    production kernel's marginal-demand operator.
    """
    representatives = np.asarray(game.representatives, dtype=int)
    demand = np.asarray(game.demand)
    result = np.zeros(game.n)
    for x in range(game.n):
        contribution = np.zeros(len(law))
        for removed in range(4):
            rivals = np.delete(representatives, removed, axis=1)
            state = np.sort(np.column_stack((np.full(len(law), x), rivals)), axis=1)
            focal_position = np.sum(rivals < x, axis=1)
            orbit = game.orbit_index(state)
            contribution += demand[orbit, focal_position]/4
        result[x] = law@contribution
    return result


def _adaptive_quadrature(game):
    """Independent scalar adaptive integrals at extreme and unequal states."""
    n = game.n
    states = {(0,)*4, (n//2,)*4, (n-1,)*4,
              (0,0,n-1,n-1), (0,n//3,2*n//3,n-1)}
    a, b = posterior(game.grid, game.params)
    maximum, relative = 0., 0.
    tested = 0
    for state in sorted(states):
        for seller in range(4):
            own = state[seller]
            rivals = state[:seller]+state[seller+1:]

            def integrand(t):
                if t <= 0 or t >= 1:
                    return 0.
                density = np.exp((a[own]-1)*np.log(t)+(b[own]-1)*np.log1p(-t)
                                 -betaln(a[own], b[own]))
                return density*np.prod([betainc(a[j], b[j], t) for j in rivals])

            value, error_estimate = quad(integrand, game.p0, 1,
                                         epsabs=1e-15, epsrel=1e-11, limit=250)
            position = sum(j < own for j in state)
            stored = float(game.demand[game.orbit_index(np.asarray(state)), position])
            error = abs(stored-value)
            require(error <= max(2e-10, 20*error_estimate),
                    f'Independent N4 demand integral failed at {state}, seller {seller}: {error:g}')
            maximum = max(maximum, error)
            relative = max(relative, error/max(abs(value), np.finfo(float).tiny))
            tested += 1
    return dict(states_checked=len(states), seller_integrals_checked=tested,
                maximum_absolute_error=maximum, maximum_relative_error=relative)


def _load_checkpoint(data, stem, fingerprints):
    paths = [data/(stem+suffix) for suffix in ('.json', '.npz')]
    expected = {}
    for path in paths:
        key = str(path.relative_to(ROOT))
        require(key in fingerprints, f'Checkpoint absent from frozen fingerprints: {key}')
        expected[key] = fingerprints[key]
    frozen.verify_hashes(expected)
    record = json.loads(paths[0].read_text())
    require(_digest(paths[1]) == record['checkpoint_sha256'], f'NPZ internal hash mismatch: {stem}')
    with np.load(paths[1], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays, expected


def audit_checkpoint(data, stem, fingerprints):
    """Fresh N4 equations audit; do not rerun equilibrium or stationary search."""
    from .four_seller_kernel import FourSellerKernel

    data = Path(data)
    record, arrays, expected = _load_checkpoint(data, stem, fingerprints)
    config, summary = record['configuration'], record['summary']
    n = config['n']
    require(config['sellers'] == 4 and config['closure'] == 'marginal', 'Expected marginal N4 case')
    require(stem == frozen.case_stem(config['p0'], config['parameters']['rho'], n, 4, 'marginal'),
            f'N4 checkpoint stem/configuration mismatch: {stem}')
    for name in ['p0','n','sellers','closure']:
        require(config[name] == summary[name], f'Summary/config mismatch for {name}: {stem}')
    require(config['parameters']['rho'] == summary['rho'], 'Summary/configuration rho mismatch')
    require(summary['verified'] and record['diagnostics']['stationary']['verified'],
            'Stored N4 finite-grid certificate is not verified')
    require('quotient' in config.get('joint_representation','').lower()
            and 'mean' in config.get('actual_value_scope','').lower(),
            'N4 checkpoint must explicitly describe quotient mass and mean seller value')
    stationary = record['diagnostics']['stationary']
    require(stationary['tolerance'] <= 1e-11 and stationary['stability_tolerance'] <= 1e-7
            and stationary['stability_l1'] <= stationary['stability_tolerance']
            and stationary.get('positive_iteration') is not None,
            'N4 stationary certificate omitted or weakened an independent stability check')
    for name in ['grid','policy','value','demand','advantage']:
        require(arrays[name].shape == (n,) and np.isfinite(arrays[name]).all(),
                f'Invalid N4 {name}: {stem}')
    close(arrays['grid'], np.linspace(0,1,n), 0, 'N4 stored grid')
    alpha = arrays['policy']
    require(np.all((alpha >= 0)&(alpha <= 1)), 'Invalid N4 policy probabilities')
    params = Parameters(**config['parameters'])
    game = FourSellerKernel(arrays['grid'], params, p0=config['p0'], order=config['order'])
    representatives = np.asarray(game.representatives)
    require(np.array_equal(arrays['representatives'], representatives), 'N4 orbit ordering mismatch')
    if 'orbit_sizes' in arrays:
        require(np.array_equal(arrays['orbit_sizes'], game.orbit_sizes), 'N4 orbit multiplicity mismatch')
    require(int(np.sum(game.orbit_sizes)) == n**4, 'N4 orbit multiplicities do not cover the full chain')
    law, actual = arrays['joint'], arrays['actual_value']
    for name, values in [('joint',law), ('actual_value',actual)]:
        require(values.shape == (len(representatives),) and np.isfinite(values).all(),
                f'Invalid N4 quotient {name}')
    require(np.all(law > 0) and abs(law.sum()-1) <= 1e-12, 'Invalid N4 quotient probability law')
    calendar = game.transition(alpha)
    jump, rates = game.jump(alpha)
    residual = float(np.sum(abs(calendar.T@law-law)))
    flow = law*rates
    flow /= flow.sum()
    flow_residual = float(np.sum(abs(jump.T@flow-flow)))
    require(residual <= 1e-11 and flow_residual <= 1e-11,
            f'N4 invariant law failed: calendar={residual:g}, normalized flow={flow_residual:g}')
    marginal, sales, reward_mean = orbit_observables(representatives, law, game.demand, alpha, params)
    require(abs(marginal.sum()-1) <= 1e-12 and np.all(marginal > 0), 'Invalid N4 tagged marginal')
    sale_rate = float(sales.sum())
    sale_law = sales/sale_rate
    q = params.p1+(params.p2-params.p1)*alpha
    success = interpolation_matrix(arrays['grid'], params.rho*arrays['grid']+1-params.rho)
    failure = interpolation_matrix(arrays['grid'], params.rho*arrays['grid'])
    selected = (diags(q)@success+diags(1-q)@failure).tocsr()
    reference_sale = positive_invariant(selected, np.ones(n), np.arange(n)[::-1])
    sale_error = float(np.sum(abs(reference_sale-sale_law)))
    require(sale_error <= 1e-7, f'N4 own-sales embedded invariant mismatch: {sale_error:g}')
    use = float(sale_law@alpha)
    outside = float(law@game.outside)
    partition = abs(outside+4*sale_rate-1)
    require(partition <= 1e-10, f'N4 market accounting failed: {partition:g}')
    initial_rows = np.flatnonzero(np.all(representatives == n//2, axis=1))
    require(len(initial_rows) == 1, 'N4 symmetric initial orbit is missing or duplicated')
    initial = int(initial_rows[0])
    expected_summary = dict(product2_conditional=use, sale_rate=sale_rate,
                            outside_rate=outside, product2_calendar=sale_rate*use,
                            profit_calendar=float(law@reward_mean),
                            value_initial=float(actual[initial]),
                            perceived_value_initial=float(arrays['value'][n//2]))
    summary_errors = {name:close(value,summary[name],1e-10,'N4 summary '+name)
                      for name,value in expected_summary.items()}
    # This operator represents the mean payoff of all four labeled sellers.
    # At the all-equal initial state exchangeability makes that mean equal to
    # each seller's payoff.  It is not a tagged seller value at unequal states.
    actual_residual = float(np.max(abs(actual-reward_mean-params.gamma*(calendar@actual))))
    require(actual_residual <= 2e-11, f'N4 mean discounted Bellman equation failed: {actual_residual:g}')
    beliefs = independent_marginal_beliefs(game, law)
    belief_error = close(beliefs, arrays['demand'], 1e-10, 'N4 joint-rival marginal beliefs')
    own_kernel = diags(1-beliefs)+diags(beliefs)@selected
    own_reward = beliefs*(params.R-params.c1-(params.c2-params.c1)*alpha)
    bellman = float(np.max(abs(arrays['value']-own_reward-params.gamma*(own_kernel@arrays['value']))))
    advantage = -(params.c2-params.c1)+params.gamma*(params.p2-params.p1)*((success-failure)@arrays['value'])
    advantage_error = close(advantage, arrays['advantage'], 1e-10, 'N4 action advantage')
    regret = float(np.max(np.maximum(advantage,0)-alpha*advantage))
    require(bellman <= 1e-10 and regret <= config['tolerance']+1e-12,
            f'N4 perceived policy Bellman/complementarity failed: {bellman:g}, {regret:g}')
    quadrature = _adaptive_quadrature(game)
    frozen.verify_hashes(expected)
    return dict(verified=True,input_sha256=expected,p0=config['p0'],rho=params.rho,n=n,
                sellers=4,closure='marginal',probability_sum=float(law.sum()),
                minimum_quotient_probability=float(law.min()),
                full_states=n**4,solved_states=len(law),
                calendar_invariant_l1=residual,rate_weighted_invariant_l1=flow_residual,
                own_sale_embedded_l1=sale_error,market_partition_error=partition,
                summary_max_abs=max(summary_errors.values()),belief_max_abs=belief_error,
                perceived_bellman_residual=bellman,action_advantage_max_abs=advantage_error,
                conditional_regret=regret,actual_discounted_residual=actual_residual,
                actual_discounted_error_bound=actual_residual/(1-params.gamma),
                initial_quotient_index=initial,initial_reputations=[.5]*4,
                actual_value_scope='Mean of four seller payoffs on each permutation orbit; '
                                   'equals each seller payoff at the all-equal initial state only',
                adaptive_demand_quadrature=quadrature,
                scope='Fresh finite-grid equations/accounting audit; independent stationary stability '
                      'checks remain in the frozen checkpoint certificate; no grid convergence or uniqueness proof')


def _imported_case(data, stem, fingerprints, baseline_manifest, baseline_audit):
    record, _, expected = _load_checkpoint(data, stem, fingerprints)
    require(stem in baseline_manifest['completed'] and stem in baseline_audit['cases'],
            f'N1--N3 baseline case was not independently audited: {stem}')
    source_record = baseline_audit['cases'][stem]
    require(source_record['verified'], f'Unverified baseline audit: {stem}')
    for suffix in ('.json','.npz'):
        source, target = BASELINE/(stem+suffix), data/(stem+suffix)
        source_key = str(source.relative_to(ROOT))
        expected_source = baseline_manifest['input_sha256'][source_key]
        require(source_record['input_sha256'][source_key] == expected_source,
                f'Baseline case audit fingerprint mismatch: {stem+suffix}')
        require(_digest(source) == expected_source == _digest(target),
                f'Imported baseline bytes differ: {stem+suffix}')
    summary = record['summary']
    require(summary['sellers'] in (1,2,3), 'Only N1--N3 cases can reuse baseline proofs')
    return dict(source_record, input_sha256=expected,
                reused_immutable_independent_audit=True,
                source_audit=str((BASELINE/'independent_audit.json').relative_to(ROOT)),
                source_checkpoint=str((BASELINE/(stem+'.json')).relative_to(ROOT)))


def audit_campaign(data):
    """Verify the complete 16-case campaign, write and return its audit record."""
    data = Path(data).resolve()
    manifest_path = data/'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    audit_sources = _source_map()
    require(manifest['status'] == 'complete', 'N4 audit requires a closed complete campaign')
    config = manifest['configuration']
    require(config['p0s'] == [.1,.3] and config['rhos'] == [.6] and config['grids'] == [21,31],
            'N4 campaign parameter/grid scope changed')
    models = {(int(count),closure) for count,closure in config['models']}
    require(models == {(1,'monopoly'),(2,'marginal'),(3,'marginal'),(4,'marginal')},
            'N4 campaign model scope changed')
    stems = {frozen.case_stem(p0,rho,n,count,closure)
             for p0,rho,n in product(config['p0s'],config['rhos'],config['grids'])
             for count,closure in models}
    require(len(manifest['completed']) == len(stems) and set(manifest['completed']) == stems,
            'N4 campaign is missing or duplicating a configured checkpoint')
    frozen.verify_hashes(manifest['numerical_sources'])
    frozen.verify_hashes(manifest['input_sha256'])
    baseline_manifest = json.loads((BASELINE/'manifest.json').read_text())
    baseline_audit = json.loads((BASELINE/'independent_audit.json').read_text())
    require(baseline_manifest['status'] == 'complete' and baseline_audit['status'] == 'complete_verified'
            and baseline_audit['verified'], 'Source baseline campaign is not complete and independently verified')
    require(baseline_audit['manifest_snapshot_sha256'] == _digest(BASELINE/'manifest.json'),
            'Source baseline manifest changed after its independent audit')
    frozen.verify_hashes(baseline_audit['audit_sources'])
    frozen.verify_hashes(baseline_manifest['numerical_sources'])
    cases = {}
    for stem in sorted(stems):
        record = json.loads((data/(stem+'.json')).read_text())
        cases[stem] = (audit_checkpoint(data,stem,manifest['input_sha256'])
                       if record['summary']['sellers'] == 4 else
                       _imported_case(data,stem,manifest['input_sha256'],baseline_manifest,baseline_audit))
    require(manifest_path.read_bytes() == manifest_bytes, 'Closed N4 manifest changed during audit')
    frozen.verify_hashes(manifest['numerical_sources'])
    frozen.verify_hashes(manifest['input_sha256'])
    frozen.verify_hashes(audit_sources)
    result = dict(status='complete_verified',verified=True,audited_utc=frozen.now(),
                  expected_model_grid_cases=len(stems),audited_model_grid_cases=len(cases),
                  fresh_N4_cases=4,reused_immutable_N1_N2_N3_cases=12,
                  manifest_snapshot_sha256=sha256(manifest_bytes).hexdigest(),
                  audit_sources=audit_sources,numerical_sources=manifest['numerical_sources'],
                  baseline_provenance={str((BASELINE/name).relative_to(ROOT)):_digest(BASELINE/name)
                                       for name in ['manifest.json','independent_audit.json']},
                  cases=cases,scope='Stored finite-grid policies, fresh N4 equations/accounting, '
                  'and byte-identical independently audited controls; no grid stability or uniqueness proof')
    frozen.write_json(data/'independent_audit.json',result)
    return result


__all__ = ['audit_checkpoint','audit_campaign','orbit_observables','independent_marginal_beliefs']
