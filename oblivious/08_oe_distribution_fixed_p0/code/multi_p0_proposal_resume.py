"""Resume a campaign with an externally searched policy as a proposal only.

Preserve the closed proposal and its search provenance inside the campaign,
fingerprint them and this wrapper, and freshly certify only its policy in the
target experiment. No imported stationary law, value, or certificate is reused.
Actual joint values and atomic checkpoints remain the immutable driver's work.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code import multi_p0_endpoint_recovery as endpoint


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def wrapper_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(runner.ROOT)): digest(path)}


def preserve_proposal(prefix, provenance, args):
    """Copy stable input bytes before the original driver is allowed to run."""
    prefix = Path(prefix).resolve()
    if prefix.suffix in ('.json', '.npz'):
        prefix = prefix.with_suffix('')
    source_files = [prefix.with_suffix(s) for s in ('.json','.npz')]
    source_files += [Path(path).resolve() for path in provenance]
    runner.require(all(path.is_file() for path in source_files), 'Proposal or provenance input is missing')
    source_hashes = {path:digest(path) for path in source_files}
    bundle = load_bundle(prefix)
    proposer_script_hash = bundle.get('proposal_script_sha256')
    if isinstance(proposer_script_hash, str):
        runner.require(proposer_script_hash in source_hashes.values(),
                       'Include the fingerprinted proposer script in --proposal-provenance')
    elif isinstance(proposer_script_hash, dict):
        runner.require(all(expected in source_hashes.values() for expected in proposer_script_hash.values()),
                       'Include every fingerprinted proposer script in --proposal-provenance')
    configuration = bundle['configuration']
    n, parameters = configuration['n'], configuration['parameters']
    runner.require(configuration['p0'] == args.p0 and parameters['rho'] == args.rho,
                   'Proposal p0/rho differs from the requested campaign')
    alpha = np.asarray(bundle['competition']['policy'], dtype=float)
    runner.require(alpha.shape == (n,) and np.isfinite(alpha).all()
                   and np.all((alpha >= 0)&(alpha <= 1))
                   and np.array_equal(bundle['competition']['grid'], np.linspace(0,1,n)),
                   'Proposal policy/grid is malformed')
    current_sources = runner.source_fingerprints()
    provided_sources = bundle.get('source_fingerprints', {})
    for source, expected in provided_sources.items():
        runner.require(current_sources.get(source) == expected and digest(runner.ROOT/source) == expected,
                       'External proposer used different numerical sources: '+source)
    directory = runner.campaign_directory(args.p0, args.rho)
    runner.require(args.resume and (directory/'manifest.json').exists(),
                   'External proposals require --resume of an existing exclusively owned campaign')
    identifier = runner.case_tag(args.rho,n)+'_'+source_hashes[prefix.with_suffix('.json')][:12]
    destination = directory/'provenance/external_policy_proposals'/identifier
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for source in source_files:
        target = destination/source.name
        expected = source_hashes[source]
        if target.exists():
            runner.require(digest(target) == expected, 'Protected preserved proposal input differs: '+str(target))
        else:
            temporary = target.with_name('_pending_'+target.name)
            shutil.copyfile(source, temporary)
            runner.require(digest(temporary) == expected, 'Proposal input changed during preservation')
            os.replace(temporary, target)
        runner.require(digest(source) == expected and digest(target) == expected,
                       'Proposal input changed while it was being preserved')
        inputs[str(target.relative_to(runner.ROOT))] = expected
    preserved_prefix = destination/prefix.name
    inputs.update(wrapper_reference())
    return dict(configuration=configuration, policy=alpha.copy(), input_sha256=inputs,
        preserved_prefix=str(preserved_prefix.relative_to(runner.ROOT)),
        proposer_numerical_source_sha256=provided_sources,
        proposer_script_sha256=proposer_script_hash,
        original_proposal_verified=bool(bundle['competition'].get('verified', False)))


def install(proposal):
    endpoint.install()
    baseline = runner.baseline_seed
    regular_solve = runner.solve_from_proposals

    def baseline_with_proposal_provenance(rho, n, closure):
        seed, inputs = baseline(rho,n,closure)
        return seed, dict(inputs, **proposal['input_sha256'])

    def solve_with_fresh_external_policy(experiment, proposals, max_evaluations, log):
        matching = experiment.configuration() == proposal['configuration']
        external_attempt = dict(initialization='preserved_external_policy_proposal',
                                stage='fresh_target_certificate', target_configuration_matches=matching)
        if matching:
            for source, expected in proposal['input_sha256'].items():
                runner.require(digest(runner.ROOT/source) == expected, 'External proposal provenance changed: '+source)
            log('  freshly certify preserved external policy proposal (no imported laws or values)')
            try:
                fresh = experiment.evaluate(proposal['policy'])
                verified = (fresh.get('verified', False) and fresh['dp_diagnostics']['verified']
                            and fresh['stationary_diagnostics']['verified'])
                if 'conditional_diagnostics' in fresh:
                    verified = (verified and fresh['conditional_diagnostics']['verified']
                                and fresh['conditional_diagnostics']['independent_order_verified'])
                external_attempt.update(verified=bool(verified), dp=fresh['dp_diagnostics'],
                    conditional=fresh.get('conditional_diagnostics', {}))
                if verified:
                    fresh['attempts'] = [external_attempt]
                    fresh['search_metadata'] = dict(
                        method='Fresh original target-game certificate of preserved external policy proposal',
                        seed_is_proposal_only=True, imported_stationary_or_value_reuse=False,
                        external_proposal_configuration=proposal['configuration'],
                        external_proposal_input_sha256=proposal['input_sha256'],
                        preserved_external_proposal=proposal['preserved_prefix'],
                        proposer_numerical_source_sha256=proposal['proposer_numerical_source_sha256'],
                        proposer_script_sha256=proposal['proposer_script_sha256'],
                        fresh_full_certificate=True, unique=False, exhaustive=False)
                    return fresh
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                external_attempt.update(verified=False, error=str(error))
                log('    external proposal did not certify: '+str(error))
        else:
            external_attempt.update(verified=False, reason='External policy targets a different model/grid; use normal proposals')
        result = regular_solve(experiment, proposals, max_evaluations, log)
        result['attempts'] = [external_attempt]+result.get('attempts', [])
        result.setdefault('search_metadata', {})['external_proposal_attempt'] = external_attempt
        return result

    runner.baseline_seed = baseline_with_proposal_provenance
    runner.solve_from_proposals = solve_with_fresh_external_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--proposal-provenance', type=Path, nargs='*', default=[])
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = runner.arguments()
    proposal = preserve_proposal(options.proposal, options.proposal_provenance, args)
    install(proposal)
    runner.main()


if __name__ == '__main__':
    main()
