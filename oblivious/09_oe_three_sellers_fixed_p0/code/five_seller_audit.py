"""Independent equations and provenance audit for the five-seller extension.

Only stored policies are evaluated.  In particular this module never searches
for an equilibrium or resumes computation.  The N1--N4 controls reuse the
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
BASELINE = HOME/'data/four_sellers_low_p0'
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

    A stationary orbit has five equally likely choices of tagged seller,
    including repeated coordinates.  ``law`` is total mass of each orbit;
    multiplying it by the orbit size here would count its permutations twice.
    """
    representatives = np.asarray(representatives, dtype=int)
    law, demand = np.asarray(law), np.asarray(demand)
    n = len(policy)
    weights = np.broadcast_to(law[:, None]/5, representatives.shape)
    marginal = np.bincount(representatives.ravel(), weights=weights.ravel(), minlength=n)
    sales = np.bincount(representatives.ravel(),
                        weights=(weights*demand).ravel(), minlength=n)
    margins = params.R-params.c1-(params.c2-params.c1)*np.asarray(policy)
    reward_mean = np.sum(demand*margins[representatives], axis=1)/5
    return marginal, sales, reward_mean


def independent_marginal_beliefs(game, law):
    """Average the actual four-rival joint law, with focal state replaced.

    The rivals in one orbit remain jointly distributed.  No product of
    single-rival marginals appears.  This deliberately does not use the
    production kernel's marginal-demand operator.
    """
    representatives = np.asarray(game.representatives, dtype=int)
    demand = np.asarray(game.demand)
    # Encode sorted four-rival tuples independently of the production kernel's
    # combinatorial indices.  This array has n**4 entries (923521 at n=31),
    # rather than the n**5 labeled market.  Its nonzero entries are total masses
    # of rival permutation orbits, so repeated coordinates retain multiplicity.
    rival_law = np.zeros(game.n**4)
    for removed in range(5):
        rivals = np.delete(representatives, removed, axis=1)
        indices = np.ravel_multi_index(rivals.T, (game.n,)*4)
        rival_law += np.bincount(indices, weights=np.asarray(law)/5,
                                 minlength=len(rival_law))
    occupied = np.flatnonzero(rival_law)
    rivals = np.column_stack(np.unravel_index(occupied, (game.n,)*4))
    weights = rival_law[occupied]
    result = np.zeros(game.n)
    for x in range(game.n):
        state = np.column_stack((np.full(len(rivals), x), rivals))
        focal_position = np.sum(rivals < x, axis=1)
        orbit = game.orbit_index(state)
        result[x] = weights@demand[orbit, focal_position]
    return result


def _stationary_certificate(stationary):
    """Check recorded independent methods without rerunning the large chain."""
    require(stationary.get('verified') is True, 'Unverified N5 stationary certificate')
    invariant, stability = stationary['tolerance'], stationary['stability_tolerance']
    require(0 < invariant <= 1e-11 and 0 < stability <= 1e-7,
            'N5 stationary certification tolerances were weakened')
    for name in ['residual','rate_weighted_residual','row_sum_error']:
        value = stationary[name]
        require(np.isfinite(value) and 0 <= value <= invariant,
                f'N5 recorded invariant check failed: {name}')
    for name in ['stability_l1','independent_calendar_l1','independent_jump_l1']:
        value = stationary[name]
        require(np.isfinite(value) and 0 <= value <= stability,
                f'N5 recorded independent stability check failed: {name}')
    positive = stationary.get('positive_iteration')
    require(isinstance(positive, dict) and positive.get('iterations', 0) > 0,
            'N5 stationary certificate omitted positive two-start iteration')
    for name in ['discrepancy','calendar_discrepancy','jump_discrepancy']:
        value = positive[name]
        require(np.isfinite(value) and 0 <= value <= stability,
                f'N5 positive two-start stability check failed: {name}')
    methods = ' '.join(stationary.get('methods', [])).lower()
    require(all(word in methods for word in ['original','reversed','uniform','concentrated']),
            'N5 certificate omitted independent orders/references or two positive starts')


def _adaptive_quadrature(game):
    """Independent scalar adaptive integrals at extreme and unequal states."""
    n = game.n
    states = {(0,)*5, (n//2,)*5, (n-1,)*5,
              (0,0,n//2,n-1,n-1), (0,n//3,2*n//3,n-1,n-1)}
    a, b = posterior(game.grid, game.params)
    maximum, relative = 0., 0.
    tested = 0
    for state in sorted(states):
        for seller in range(5):
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
                    f'Independent N5 demand integral failed at {state}, seller {seller}: {error:g}')
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
    """Fresh N5 equations audit; do not rerun equilibrium or stationary search."""
    from .five_seller_kernel import FiveSellerKernel

    data = Path(data)
    record, arrays, expected = _load_checkpoint(data, stem, fingerprints)
    config, summary = record['configuration'], record['summary']
    n = config['n']
    require(config['sellers'] == 5 and config['closure'] == 'marginal', 'Expected marginal N5 case')
    require(stem == frozen.case_stem(config['p0'], config['parameters']['rho'], n, 5, 'marginal'),
            f'N5 checkpoint stem/configuration mismatch: {stem}')
    for name in ['p0','n','sellers','closure']:
        require(config[name] == summary[name], f'Summary/config mismatch for {name}: {stem}')
    require(config['parameters']['rho'] == summary['rho'], 'Summary/configuration rho mismatch')
    require(summary['verified'] and record['diagnostics']['stationary']['verified'],
            'Stored N5 finite-grid certificate is not verified')
    require('quotient' in config.get('joint_representation','').lower()
            and 'mean' in config.get('actual_value_scope','').lower(),
            'N5 checkpoint must explicitly describe quotient mass and mean seller value')
    stationary = record['diagnostics']['stationary']
    _stationary_certificate(stationary)
    require(0 < config['tolerance'] <= 1e-7, 'N5 conditional regret tolerance was weakened')
    require(config.get('initial_reputations') == .5,
            'N5 actual initial value must refer to the exchangeable all-.5 state')
    for name in ['grid','policy','value','demand','advantage']:
        require(arrays[name].shape == (n,) and np.isfinite(arrays[name]).all(),
                f'Invalid N5 {name}: {stem}')
    close(arrays['grid'], np.linspace(0,1,n), 0, 'N5 stored grid')
    alpha = arrays['policy']
    require(np.all((alpha >= 0)&(alpha <= 1)), 'Invalid N5 policy probabilities')
    params = Parameters(**config['parameters'])
    game = FiveSellerKernel(arrays['grid'], params, p0=config['p0'], order=config['order'])
    representatives = np.asarray(game.representatives)
    require(np.array_equal(arrays['representatives'], representatives), 'N5 orbit ordering mismatch')
    require('orbit_sizes' in arrays and np.array_equal(arrays['orbit_sizes'], game.orbit_sizes),
            'N5 orbit multiplicity mismatch or omitted multiplicities')
    require(int(np.sum(game.orbit_sizes)) == n**5, 'N5 orbit multiplicities do not cover the full chain')
    require(stationary['solved_states'] == len(representatives)
            and stationary['full_states'] == n**5
            and stationary.get('symmetric_lumping') is True,
            'N5 certificate state count or exact quotient description is inconsistent')
    law, actual = arrays['joint'], arrays['actual_value']
    for name, values in [('joint',law), ('actual_value',actual)]:
        require(values.shape == (len(representatives),) and np.isfinite(values).all(),
                f'Invalid N5 quotient {name}')
    require(np.all(law > 0) and abs(law.sum()-1) <= 1e-12, 'Invalid N5 quotient probability law')
    calendar = game.transition(alpha)
    jump, rates = game.jump(alpha)
    require(np.isfinite(rates).all() and np.all(rates > 0), 'N5 exit rates must be strictly positive')
    row_error = float(np.max(abs(np.asarray(calendar.sum(axis=1)).ravel()-1)))
    jump_row_error = float(np.max(abs(np.asarray(jump.sum(axis=1)).ravel()-1)))
    require(max(row_error,jump_row_error) <= 1e-11, 'N5 calendar or jump rows do not sum to one')
    residual = float(np.sum(abs(calendar.T@law-law)))
    flow = law*rates
    flow /= flow.sum()
    flow_residual = float(np.sum(abs(jump.T@flow-flow)))
    require(residual <= 1e-11 and flow_residual <= 1e-11,
            f'N5 invariant law failed: calendar={residual:g}, normalized flow={flow_residual:g}')
    marginal, sales, reward_mean = orbit_observables(representatives, law, game.demand, alpha, params)
    require(abs(marginal.sum()-1) <= 1e-12 and np.all(marginal > 0), 'Invalid N5 tagged marginal')
    sale_rate = float(sales.sum())
    sale_law = sales/sale_rate
    q = params.p1+(params.p2-params.p1)*alpha
    success = interpolation_matrix(arrays['grid'], params.rho*arrays['grid']+1-params.rho)
    failure = interpolation_matrix(arrays['grid'], params.rho*arrays['grid'])
    selected = (diags(q)@success+diags(1-q)@failure).tocsr()
    reference_sale = positive_invariant(selected, np.ones(n), np.arange(n)[::-1])
    sale_error = float(np.sum(abs(reference_sale-sale_law)))
    require(sale_error <= 1e-7, f'N5 own-sales embedded invariant mismatch: {sale_error:g}')
    use = float(sale_law@alpha)
    outside = float(law@game.outside)
    partition = abs(outside+5*sale_rate-1)
    require(partition <= 1e-10, f'N5 market accounting failed: {partition:g}')
    initial_rows = np.flatnonzero(np.all(representatives == n//2, axis=1))
    require(len(initial_rows) == 1, 'N5 symmetric initial orbit is missing or duplicated')
    initial = int(initial_rows[0])
    expected_summary = dict(product2_conditional=use, sale_rate=sale_rate,
                            outside_rate=outside, product2_calendar=sale_rate*use,
                            profit_calendar=float(law@reward_mean),
                            value_initial=float(actual[initial]),
                            perceived_value_initial=float(arrays['value'][n//2]))
    summary_errors = {name:close(value,summary[name],1e-10,'N5 summary '+name)
                      for name,value in expected_summary.items()}
    # This operator represents the mean payoff of all five labeled sellers.
    # At the all-equal initial state exchangeability makes that mean equal to
    # each seller's payoff.  It is not a tagged seller value at unequal states.
    actual_residual = float(np.max(abs(actual-reward_mean-params.gamma*(calendar@actual))))
    require(actual_residual <= 2e-11, f'N5 mean discounted Bellman equation failed: {actual_residual:g}')
    beliefs = independent_marginal_beliefs(game, law)
    belief_error = close(beliefs, arrays['demand'], 1e-10, 'N5 joint-rival marginal beliefs')
    own_kernel = diags(1-beliefs)+diags(beliefs)@selected
    own_reward = beliefs*(params.R-params.c1-(params.c2-params.c1)*alpha)
    bellman = float(np.max(abs(arrays['value']-own_reward-params.gamma*(own_kernel@arrays['value']))))
    advantage = -(params.c2-params.c1)+params.gamma*(params.p2-params.p1)*((success-failure)@arrays['value'])
    advantage_error = close(advantage, arrays['advantage'], 1e-10, 'N5 action advantage')
    regret = float(np.max(np.maximum(advantage,0)-alpha*advantage))
    require(bellman <= 1e-10 and regret <= config['tolerance']+1e-12,
            f'N5 perceived policy Bellman/complementarity failed: {bellman:g}, {regret:g}')
    quadrature = _adaptive_quadrature(game)
    frozen.verify_hashes(expected)
    return dict(verified=True,input_sha256=expected,p0=config['p0'],rho=params.rho,n=n,
                sellers=5,closure='marginal',probability_sum=float(law.sum()),
                minimum_quotient_probability=float(law.min()),
                full_states=n**5,solved_states=len(law),
                calendar_row_sum_error=row_error,jump_row_sum_error=jump_row_error,
                calendar_invariant_l1=residual,rate_weighted_invariant_l1=flow_residual,
                own_sale_embedded_l1=sale_error,market_partition_error=partition,
                summary_max_abs=max(summary_errors.values()),belief_max_abs=belief_error,
                perceived_bellman_residual=bellman,action_advantage_max_abs=advantage_error,
                conditional_regret=regret,actual_discounted_residual=actual_residual,
                actual_discounted_error_bound=actual_residual/(1-params.gamma),
                initial_quotient_index=initial,initial_reputations=[.5]*5,
                actual_value_scope='Mean of five seller payoffs on each permutation orbit; '
                                   'equals each seller payoff at the all-equal initial state only',
                adaptive_demand_quadrature=quadrature,
                scope='Fresh finite-grid equations/accounting audit; independent stationary stability '
                      'checks remain in the frozen checkpoint certificate; no grid convergence or uniqueness proof')


def _imported_case(data, stem, fingerprints, baseline_manifest, baseline_audit):
    record, _, expected = _load_checkpoint(data, stem, fingerprints)
    require(stem in baseline_manifest['completed'] and stem in baseline_audit['cases'],
            f'N1--N4 baseline case was not independently audited: {stem}')
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
    require(summary['sellers'] in (1,2,3,4), 'Only N1--N4 cases can reuse baseline proofs')
    return dict(source_record, input_sha256=expected,
                reused_immutable_independent_audit=True,
                source_audit=str((BASELINE/'independent_audit.json').relative_to(ROOT)),
                source_checkpoint=str((BASELINE/(stem+'.json')).relative_to(ROOT)))


def audit_campaign(data):
    """Verify the complete 20-case campaign, write and return its audit record."""
    data = Path(data).resolve()
    manifest_path = data/'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    audit_sources = _source_map()
    require(manifest['status'] == 'complete', 'N5 audit requires a closed complete campaign')
    config = manifest['configuration']
    require(config['p0s'] == [.1,.3] and config['rhos'] == [.6] and config['grids'] == [21,31],
            'N5 campaign parameter/grid scope changed')
    models = {(int(count),closure) for count,closure in config['models']}
    require(len(config['models']) == 5
            and models == {(1,'monopoly'),(2,'marginal'),(3,'marginal'),(4,'marginal'),(5,'marginal')},
            'N5 campaign model scope changed')
    stems = {frozen.case_stem(p0,rho,n,count,closure)
             for p0,rho,n in product(config['p0s'],config['rhos'],config['grids'])
             for count,closure in models}
    require(len(manifest['completed']) == len(stems) and set(manifest['completed']) == stems,
            'N5 campaign is missing or duplicating a configured checkpoint')
    frozen.verify_hashes(manifest['numerical_sources'])
    frozen.verify_hashes(manifest['input_sha256'])
    baseline_bytes = {name:(BASELINE/name).read_bytes()
                      for name in ['manifest.json','independent_audit.json']}
    baseline_manifest = json.loads(baseline_bytes['manifest.json'])
    baseline_audit = json.loads(baseline_bytes['independent_audit.json'])
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
                       if record['summary']['sellers'] == 5 else
                       _imported_case(data,stem,manifest['input_sha256'],baseline_manifest,baseline_audit))
    require(manifest_path.read_bytes() == manifest_bytes, 'Closed N5 manifest changed during audit')
    frozen.verify_hashes(manifest['numerical_sources'])
    frozen.verify_hashes(manifest['input_sha256'])
    frozen.verify_hashes(audit_sources)
    require(all((BASELINE/name).read_bytes() == contents for name,contents in baseline_bytes.items()),
            'Frozen N1--N4 source campaign or audit changed during N5 audit')
    frozen.verify_hashes(baseline_audit['audit_sources'])
    frozen.verify_hashes(baseline_manifest['numerical_sources'])
    result = dict(status='complete_verified',verified=True,audited_utc=frozen.now(),
                  expected_model_grid_cases=len(stems),audited_model_grid_cases=len(cases),
                  fresh_N5_cases=4,reused_immutable_N1_N2_N3_N4_cases=16,
                  manifest_snapshot_sha256=sha256(manifest_bytes).hexdigest(),
                  audit_sources=audit_sources,numerical_sources=manifest['numerical_sources'],
                  baseline_provenance={str((BASELINE/name).relative_to(ROOT)):sha256(contents).hexdigest()
                                       for name,contents in baseline_bytes.items()},
                  cases=cases,scope='Stored finite-grid policies, fresh N5 equations/accounting, '
                  'and byte-identical independently audited controls; no grid stability or uniqueness proof')
    frozen.write_json(data/'independent_audit.json',result)
    return result


__all__ = ['audit_checkpoint','audit_campaign','orbit_observables','independent_marginal_beliefs']
