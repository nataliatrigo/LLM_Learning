"""Independent checkpoint audit for the isolated three-seller experiment.

This module evaluates stored policies; it never searches for an equilibrium or
resumes either campaign.  During an active run, only immutable completed bundles
are audited.  Mutable CSV fingerprints are checked once the campaign closes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.special import betainc, betaln

from oblivious.oe_p0.code.primitives import Parameters, interpolation_matrix, posterior
from oblivious.oe_p0.code.positive_stationary import positive_invariant
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from .kernel import FixedOutsideKernel


HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
OUTPUT = 'independent_audit.json'


class AuditFailure(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def write_json(path, record):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
        temporary = handle.name
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_map():
    paths = [Path(__file__), HOME/'code/kernel.py']
    paths += [ROOT/'oblivious'/folder/'code'/name for folder, name in [
        ('08_oe_distribution_fixed_p0', 'model.py'),
        ('08_oe_distribution_fixed_p0', 'conditional_model.py'),
        ('01_oe_distribution', 'model.py'),
        ('04_oe_conditional_distribution', 'model.py'),
        ('02_oe_p0', 'game.py'), ('02_oe_p0', 'primitives.py'),
        ('02_oe_p0', 'policy.py'), ('02_oe_p0', 'core.py'),
        ('02_oe_p0', 'positive_stationary.py'), ('02_oe_p0', 'positive_stationary.cpp')]]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_hashes(mapping):
    for name, expected in mapping.items():
        require(digest(ROOT/name) == expected, f'Fingerprint mismatch: {name}')


def close(first, second, tolerance, message):
    difference = float(np.max(np.abs(np.asarray(first)-np.asarray(second))))
    require(difference <= tolerance, f'{message}: max abs difference={difference:g}')
    return difference


def _independent_beliefs(joint, selection, closure):
    sellers, n = joint.ndim, joint.shape[0]
    if sellers == 1:
        return selection.copy()
    own = joint.reshape(n, -1).sum(axis=1)
    if closure == 'conditional':
        return (joint*selection).reshape(n, -1).sum(axis=1)/own
    rivals = joint.sum(axis=0)
    return selection.reshape(n, -1)@rivals.ravel()


def _quadrature_spots(game):
    """Independent adaptive scalar integrals at central and extreme states."""
    n, sellers = game.n, game.sellers
    states = {(0,)*sellers, (n//2,)*sellers, (n-1,)*sellers,
              (0,)+(n-1,)*(sellers-1), (n-1,)+(0,)*(sellers-1)}
    a, b = posterior(game.grid, game.params)
    errors, relative = [], []
    for state in sorted(states):
        own = state[0]
        def integrand(t):
            if t <= 0 or t >= 1:
                return 0.
            log_density = ((a[own]-1)*np.log(t)+(b[own]-1)*np.log1p(-t)
                           -betaln(a[own], b[own]))
            rivals = np.prod([betainc(a[index], b[index], t) for index in state[1:]])
            return np.exp(log_density)*rivals
        value, quadrature_error = quad(integrand, game.p0, 1, epsabs=1e-15,
                                       epsrel=1e-11, limit=250)
        stored = float(game.demand[(0,)+state])
        error = abs(stored-value)
        errors.append(error)
        relative.append(error/max(abs(value), np.finfo(float).tiny))
        require(error <= max(2e-10, 20*quadrature_error),
                f'Independent adaptive demand integral failed at {state}: {error:g}')
    return dict(states_checked=len(states), maximum_absolute_error=max(errors),
                maximum_relative_error=max(relative))


def _old_two_seller_check(config, arrays, game, actual_residual):
    # Deliberately evaluate only preselected samples, never solve an equilibrium.
    if (config['sellers'] != 2 or config['closure'] != 'marginal'
            or (config['p0'], config['parameters']['rho']) not in {(.5,.6), (.8,.8)}):
        return None
    experiment_class = (FixedOutsideConditionalExperiment if config['closure'] == 'conditional'
                        else FixedOutsideMarginalExperiment)
    old = experiment_class(n=config['n'], params=Parameters(**config['parameters']),
                           p0=config['p0'], order=config['order'], tol=config['tolerance'])
    alpha = arrays['policy']
    evaluated = old.evaluate(alpha)
    lhs = old.game.transition(np.broadcast_to(alpha[:, None], (config['n'],)*2),
                              np.broadcast_to(alpha[None, :], (config['n'],)*2))
    rhs = game.transition(alpha)
    delta = lhs-rhs
    kernel_difference = float(np.max(abs(delta.data), initial=0))
    demand_difference = close(old.game.D, game.demand[0], 2e-9, 'N2 historical demand')
    joint_l1 = float(np.sum(abs(evaluated['nu']-arrays['joint'])))
    require(joint_l1 <= 1e-7, f'N2 historical joint-law mismatch: {joint_l1:g}')
    require(kernel_difference <= 2e-9, f'N2 historical calendar kernel mismatch: {kernel_difference:g}')
    belief_difference = close(evaluated['demand'], arrays['demand'], 1e-7, 'N2 historical belief demand')
    value_difference = close(evaluated['value'], arrays['value'], 1e-6, 'N2 historical perceived value')
    advantage_difference = close(evaluated['advantage'], arrays['advantage'], 1e-7, 'N2 historical action advantage')
    p = game.params
    reward = old.game.D*(p.R-p.c1-(p.c2-p.c1)*alpha[:, None])
    actual = arrays['actual_value'].ravel()
    old_discount_residual = float(np.max(abs(actual-reward.ravel()-p.gamma*(lhs@actual))))
    require(old_discount_residual <= actual_residual+5e-10,
            f'N2 historical actual discounted equation mismatch: {old_discount_residual:g}')
    return dict(kernel_max_abs=kernel_difference, demand_max_abs=demand_difference,
                joint_l1=joint_l1, belief_max_abs=belief_difference,
                perceived_value_max_abs=value_difference, advantage_max_abs=advantage_difference,
                actual_discounted_equation_residual=old_discount_residual,
                historical_stricter_verified=bool(evaluated['verified']),
                historical_conditional_regret=float(evaluated['dp_diagnostics']['conditional_regret']),
                method='Fresh historical 08 evaluate(stored policy), no equilibrium search')


def audit_checkpoint(data, stem, fingerprints):
    json_path, npz_path = data/(stem+'.json'), data/(stem+'.npz')
    expected = {str(path.relative_to(ROOT)): fingerprints[str(path.relative_to(ROOT))]
                for path in (json_path, npz_path)}
    verify_hashes(expected)
    record = json.loads(json_path.read_text())
    require(digest(npz_path) == record['checkpoint_sha256'], f'NPZ internal hash mismatch: {stem}')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    config, summary = record['configuration'], record['summary']
    n, sellers, closure = config['n'], config['sellers'], config['closure']
    for name in ['p0', 'n', 'sellers', 'closure']:
        require(config[name] == summary[name], f'Summary/configuration mismatch for {name}: {stem}')
    require(config['parameters']['rho'] == summary['rho'], f'Summary/configuration rho mismatch: {stem}')
    require(stem == case_stem(config['p0'], config['parameters']['rho'], n, sellers, closure),
            f'Checkpoint stem/configuration mismatch: {stem}')
    shape = (n,)*sellers
    for name in ['grid', 'policy', 'value', 'demand', 'advantage']:
        require(arrays[name].shape == (n,) and np.isfinite(arrays[name]).all(), f'Invalid {name}: {stem}')
    for name in ['joint', 'actual_value']:
        require(arrays[name].shape == shape and np.isfinite(arrays[name]).all(), f'Invalid {name}: {stem}')
    alpha, joint = arrays['policy'], arrays['joint']
    require(np.all((alpha >= 0)&(alpha <= 1)), f'Invalid policy probabilities: {stem}')
    require(np.all(joint > 0) and abs(joint.sum()-1) <= 1e-12, f'Invalid joint probability law: {stem}')
    close(arrays['grid'], np.linspace(0,1,n), 0, 'Stored grid')
    require(bool(summary['verified']), f'Checkpoint was not certified: {stem}')
    require(record['diagnostics']['stationary']['verified'], f'Stored stationary certificate is unverified: {stem}')
    params = Parameters(**config['parameters'])
    game = FixedOutsideKernel(arrays['grid'], params, p0=config['p0'], sellers=sellers, order=config['order'])
    calendar = game.transition(alpha)
    jump, rates = game.jump(alpha)
    law = joint.ravel()
    residual = float(np.sum(abs(calendar.T@law-law)))
    flow = law*rates
    normalized_flow = flow/flow.sum()
    flow_residual = float(np.sum(abs(jump.T@normalized_flow-normalized_flow)))
    require(residual <= 1e-11 and flow_residual <= 1e-11,
            f'Full invariant law failed: calendar={residual:g}, normalized flow={flow_residual:g}')
    marginals = np.stack([joint.sum(axis=tuple(axis for axis in range(sellers) if axis != seller))
                          for seller in range(sellers)])
    marginal_l1 = float(np.max(np.sum(abs(marginals-marginals[0]), axis=1)))
    require(marginal_l1 <= 1e-10, f'Seller marginal symmetry failed: {stem}')
    sales = (joint*game.demand[0]).reshape(n,-1).sum(axis=1)
    sale_rate = float(sales.sum())
    sale_law = sales/sale_rate
    q = params.p1+(params.p2-params.p1)*alpha
    success = interpolation_matrix(arrays['grid'], params.rho*arrays['grid']+1-params.rho)
    failure = interpolation_matrix(arrays['grid'], params.rho*arrays['grid'])
    selected = (diags(q)@success+diags(1-q)@failure).tocsr()
    reference_sale = positive_invariant(selected, np.ones(n), np.arange(n)[::-1])
    sale_l1 = float(np.sum(abs(reference_sale-sale_law)))
    require(sale_l1 <= 1e-7, f'Own-sales embedded invariant mismatch: {sale_l1:g}')
    use = float(sale_law@alpha)
    outside = float(np.sum(joint*game.outside))
    expected_summary = dict(product2_conditional=use, sale_rate=sale_rate,
                            outside_rate=outside, product2_calendar=sale_rate*use,
                            profit_calendar=sale_rate*(params.R-params.c1-(params.c2-params.c1)*use),
                            value_initial=float(arrays['actual_value'][(n//2,)*sellers]),
                            perceived_value_initial=float(arrays['value'][n//2]))
    summary_errors = {name: close(value, summary[name], 1e-10, 'Summary '+name)
                      for name,value in expected_summary.items()}
    market_partition = abs(outside+sellers*sale_rate-1)
    require(market_partition <= 1e-10, f'Market accounting failed: {market_partition:g}')
    beliefs = _independent_beliefs(joint, game.demand[0], closure)
    belief_error = close(beliefs, arrays['demand'], 1e-10, 'Belief demand')
    own_kernel = diags(1-beliefs)+diags(beliefs)@selected
    own_reward = beliefs*(params.R-params.c1-(params.c2-params.c1)*alpha)
    bellman = float(np.max(abs(arrays['value']-own_reward-params.gamma*(own_kernel@arrays['value']))))
    advantage = -(params.c2-params.c1)+params.gamma*(params.p2-params.p1)*((success-failure)@arrays['value'])
    advantage_error = close(advantage, arrays['advantage'], 1e-10, 'Action advantage')
    regret = float(np.max(np.maximum(advantage,0)-alpha*advantage))
    require(bellman <= 1e-10 and regret <= config['tolerance']+1e-12,
            f'Perceived policy Bellman/complementarity failed: {bellman:g}, {regret:g}')
    own_alpha = alpha.reshape((n,)+(1,)*(sellers-1))
    reward = game.demand[0]*(params.R-params.c1-(params.c2-params.c1)*own_alpha)
    actual = arrays['actual_value'].ravel()
    actual_residual = float(np.max(abs(actual-reward.ravel()-params.gamma*(calendar@actual))))
    actual_bound = actual_residual/(1-params.gamma)
    require(actual_residual <= 2e-11, f'Actual discounted-value equation failed: {actual_residual:g}')
    old_comparison = _old_two_seller_check(config, arrays, game, actual_residual)
    adaptive_quadrature = _quadrature_spots(game)
    verify_hashes(expected)
    return dict(verified=True, input_sha256=expected, p0=config['p0'], rho=params.rho,
                n=n, sellers=sellers, closure=closure, probability_sum=float(joint.sum()),
                minimum_joint_probability=float(joint.min()), calendar_invariant_l1=residual,
                rate_weighted_invariant_l1=flow_residual, marginal_symmetry_l1=marginal_l1,
                own_sale_embedded_l1=sale_l1, market_partition_error=market_partition,
                summary_max_abs=max(summary_errors.values()), belief_max_abs=belief_error,
                perceived_bellman_residual=bellman, action_advantage_max_abs=advantage_error,
                conditional_regret=regret, actual_discounted_residual=actual_residual,
                actual_discounted_error_bound=actual_bound,
                initial_value_index=[n//2]*sellers, initial_reputations=[.5]*sellers,
                historical_N2_comparison=old_comparison, adaptive_demand_quadrature=adaptive_quadrature)


def case_stem(p0, rho, n, sellers, closure):
    def label(value):
        return f'{value:g}'.replace('.', 'p')
    return f'p0_{label(p0)}/rho_{label(rho)}/n{n}_N{sellers}_{closure}'


def verify_import_provenance(data, manifest):
    """Check every imported byte against its immutable source manifest."""
    provenance = manifest.get('import_provenance')
    if provenance is None:
        return None
    source_manifest = ROOT/provenance['source_manifest']
    snapshot = ROOT/provenance['source_manifest_snapshot']
    expected_manifest_sha = provenance['source_manifest_sha256']
    require(digest(source_manifest) == expected_manifest_sha,
            'Imported source manifest no longer matches its recorded fingerprint')
    require(digest(snapshot) == expected_manifest_sha,
            'Imported source manifest snapshot differs from the original')
    require(manifest['input_sha256'].get(str(snapshot.relative_to(ROOT))) == expected_manifest_sha,
            'Imported source manifest snapshot is absent from campaign fingerprints')
    source = json.loads(snapshot.read_text())
    require(source['status'] != 'running', 'Cannot certify imports from an active source campaign')
    imported = provenance['imported_stems']
    require(len(imported) == len(set(imported)), 'Duplicate imported checkpoint stems')
    require(set(imported) <= set(source['completed']), 'Imported bundle was not completed in the source campaign')
    require(set(imported) <= set(manifest['completed']), 'Imported bundle is missing from the target campaign')
    source_inputs = provenance['source_input_sha256']
    require(len(source_inputs) == 2*len(imported), 'Imported source bundle fingerprint count is incorrect')
    verify_hashes(source_inputs)
    for stem in imported:
        for suffix in ('.json','.npz'):
            original = source_manifest.parent/(stem+suffix)
            target = data/(stem+suffix)
            original_name, target_name = str(original.relative_to(ROOT)), str(target.relative_to(ROOT))
            expected = source['input_sha256'].get(original_name)
            require(expected is not None and source_inputs.get(original_name) == expected,
                    f'Imported bundle does not match the source manifest: {stem+suffix}')
            require(manifest['input_sha256'].get(original_name) == expected,
                    f'Original imported bundle absent from target fingerprints: {stem+suffix}')
            require(manifest['input_sha256'].get(target_name) == expected and digest(target) == expected,
                    f'Imported checkpoint bytes differ from the source: {stem+suffix}')
    for path, expected in source['numerical_sources'].items():
        require(manifest['numerical_sources'].get(path) == expected,
                f'Original numerical source changed during marginal import: {path}')
    phases = manifest.get('execution_entrypoint_history', [])
    require(len(phases) >= 2, 'Imported campaign does not preserve execution phase history')
    for phase in phases:
        verify_hashes(phase['numerical_sources'])
    require(phases[-1]['numerical_sources'] == manifest['numerical_sources'],
            'Current execution phase source fingerprints differ from the campaign')
    require(set(phases[0]['completed_stems']) == set(imported),
            'Imported execution phase has the wrong checkpoint scope')
    return dict(verified=True, source_manifest=provenance['source_manifest'],
                source_manifest_sha256=expected_manifest_sha,
                source_manifest_snapshot=provenance['source_manifest_snapshot'],
                imported_stems=imported, imported_model_grid_cases=len(imported),
                identical_checkpoint_files=2*len(imported), source_input_sha256=source_inputs,
                source_campaign_status=source['status'], source_failures=source.get('failures', {}),
                scope=provenance.get('scope'),
                execution_entrypoint_history=phases)


def audit_pass(data, previous=None):
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    config = manifest['configuration']
    expected_stems = {case_stem(p0,rho,n,sellers,closure)
                      for p0,rho,n,(sellers,closure) in
                      product(config['p0s'],config['rhos'],config['grids'],config['models'])}
    require(len(manifest['completed']) == len(set(manifest['completed'])), 'Duplicate completed checkpoint stems')
    require(set(manifest['completed']) <= expected_stems, 'Completed checkpoint outside the configured experiment scope')
    audit_sources = source_map()
    verify_hashes(manifest['numerical_sources'])
    imported = verify_import_provenance(data, manifest)
    if previous:
        require(previous['audit_sources'] == audit_sources, 'Independent audit sources changed during one audit campaign')
    records = dict(previous.get('cases', {})) if previous else {}
    for stem in manifest['completed']:
        if stem in records:
            verify_hashes(records[stem]['input_sha256'])
        else:
            records[stem] = audit_checkpoint(data, stem, manifest['input_sha256'])
    expected_count = len(expected_stems)
    completed = manifest['status'] == 'complete'
    if completed:
        require(len(manifest['completed']) == expected_count, 'Closed campaign has an incomplete checkpoint count')
        require(set(manifest['completed']) == expected_stems, 'Closed campaign has the wrong checkpoint scope')
        verify_hashes(manifest['input_sha256'])
    verify_hashes(manifest['numerical_sources'])
    verify_hashes(audit_sources)
    current = json.loads(manifest_path.read_text())
    if completed:
        require(digest(manifest_path) == manifest_sha, 'Closed manifest changed during audit')
    partial_status = ('running_partial_verified' if manifest['status'] in {'created','running'}
                      else 'partial_verified_inputs_closed')
    return dict(status='complete_verified' if completed else partial_status,
                verified=completed, audited_utc=now(), manifest_status=manifest['status'],
                current_manifest_status=current['status'], manifest_snapshot_sha256=manifest_sha,
                expected_model_grid_cases=expected_count, audited_model_grid_cases=len(records),
                audit_sources=audit_sources, numerical_sources=manifest['numerical_sources'],
                campaign_input_sha256=manifest['input_sha256'] if completed else None,
                import_provenance=imported,
                mutable_csv_fingerprints_checked=completed, cases=records,
                scope='Stored finite-grid policies and independent equations/accounting audit; no grid stability, equilibrium uniqueness, or mixing-time guarantee')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', '--data', dest='name', default='marginal_pilot',
                        help='Campaign folder inside this experiment data directory')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args(argv)
    require('/' not in args.name and '..' not in args.name, 'Use a simple campaign name')
    require(1 <= args.interval <= 60, 'Watch interval must be 1..60 seconds')
    data = HOME/'data'/args.name
    target = data/OUTPUT
    previous = json.loads(target.read_text()) if target.exists() else None
    while True:
        result = audit_pass(data, previous)
        write_json(target, result)
        print(f"{now()} {result['status']} {result['audited_model_grid_cases']}/{result['expected_model_grid_cases']}", flush=True)
        if result['verified'] or not args.watch or result['current_manifest_status'] not in {'created','running'}:
            return 0 if result['verified'] or not args.watch else 1
        previous = result
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
