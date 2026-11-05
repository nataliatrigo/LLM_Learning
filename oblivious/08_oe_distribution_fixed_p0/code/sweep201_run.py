"""Fresh 201/251 comparison, preserving all historical fixed-p0 results.

The frozen numerical driver and its certificates are unchanged. Historical
same-p0 policies are checked proposals only; demand, equilibrium, monopoly and
actual discounted values are recalculated on every target grid.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code import multi_p0_endpoint_recovery as endpoint

P0S = (.1, .3, .5, .7, .8, .9)
RHOS = (.6, .7, .8, .9)
GRIDS = (201, 251)
CAMPAIGN = runner.HOME/'data/p0_sweep_201'


def campaign_directory(p0, rho):
    return CAMPAIGN/('p0_'+runner.number_tag(p0))/('rho_'+runner.number_tag(rho))


def helper_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(runner.ROOT)): sha256(path.read_bytes()).hexdigest()}


def checked_same_p0_seed(args, n, closure):
    """Check old artifacts and calibration before returning only a policy seed."""
    if args.p0 == .5:
        seed, inputs = runner._sweep201_original_baseline_seed(args.rho, n, closure)
        return {key:seed[key].copy() for key in ('grid','policy')}, dict(inputs, **helper_reference())
    directory = runner.HOME/'data/p0_sweep'/('p0_'+runner.number_tag(args.p0))/('rho_'+runner.number_tag(args.rho))
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    expected = dict(p0=args.p0, rho=args.rho, order=args.order, tol=args.tol,
                    dp_tol=args.dp_tol,
                    parameters={k:v for k,v in asdict(Parameters()).items() if k != 'rho'})
    runner.require(manifest['configuration'] == expected,
                   'Same-p0 proposal calibration differs')
    runner.require(manifest['numerical_sources'] == runner.source_fingerprints(),
                   'Same-p0 proposal numerical sources changed')
    available = []
    for tag in manifest['completed']:
        if tag.endswith('_'+closure):
            case = runner.check_bundle(directory, tag, manifest)
            if case['n'] < n:
                available.append((case['n'], tag, case))
    runner.require(bool(available), 'No certified smaller same-p0 policy proposal')
    _, tag, case = max(available, key=lambda item:item[0])
    runner.require(case['p0'] == args.p0 and case['rho'] == args.rho
                   and case['configuration']['parameters'] == asdict(Parameters(rho=args.rho))
                   and case['competition']['verified'],
                   'Same-p0 proposal target or equilibrium certificate differs')
    if closure == 'conditional':
        runner.require(case['competition']['conditional_diagnostics']['independent_order_verified'],
                       'Same-p0 conditional proposal lacks reversed-order certificate')
    inputs = dict(runner.input_hashes(directory, tag), **helper_reference())
    inputs[str(path.relative_to(runner.ROOT))] = sha256(path.read_bytes()).hexdigest()
    for source, expected_hash in manifest.get('reference_input_sha256', {}).items():
        runner.require(sha256((runner.ROOT/source).read_bytes()).hexdigest() == expected_hash,
                       'Same-p0 proposal reference changed: '+source)
        inputs[source] = expected_hash
    return {key:case['competition'][key].copy() for key in ('grid','policy')}, inputs


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0', type=float, required=True, choices=P0S)
    parser.add_argument('--rho', type=float, required=True, choices=RHOS)
    parser.add_argument('--grids', type=int, nargs='+', default=list(GRIDS))
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=250)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (len(set(args.grids)) != len(args.grids) or any(n not in GRIDS for n in args.grids)
            or args.order < 8 or args.tol <= 0 or args.dp_tol <= 0 or args.max_evaluations < 1):
        parser.error('Use distinct grids 201/251 and positive numerical settings')
    return args


def install(args):
    if getattr(runner, '_sweep201_installed', False):
        raise RuntimeError('A 201/251 worker may own only one calibration per process')
    runner._sweep201_original_baseline_seed = runner.baseline_seed
    runner.baseline_seed = lambda rho,n,closure: checked_same_p0_seed(args, n, closure)
    runner.campaign_directory = campaign_directory
    runner.arguments = lambda: args
    endpoint.install()
    runner._sweep201_installed = True


def main():
    args = arguments()
    install(args)
    try:
        runner.main()
    except KeyboardInterrupt:
        path = campaign_directory(args.p0,args.rho)/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.setdefault('interruptions',[]).append(dict(
                interrupted_utc=runner.now(), grid=manifest.get('current_grid'),
                model=manifest.get('current_model'),
                reason='Worker interrupted; completed atomic certified checkpoints preserved'))
            manifest.update(status='interrupted_preserving_certified_checkpoints',
                            current_grid=None,current_model=None,updated_utc=runner.now())
            runner.export_completed(path.parent,manifest)
            runner.write_json(path,manifest)
        raise


if __name__ == '__main__':
    main()
