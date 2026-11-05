"""Unpublished one-dimensional monopoly diagnostics; never competitive results.

Uses the unchanged original monopoly solver and its complete certificates.
It does not write to a case manifest or qualify a grid for publication.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, check_bundle, load_bundle, now, save_bundle, solve_monopoly,
    write_json,
)
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import (
    source_fingerprints, stability,
)


DEST = Path(__file__).resolve().parent
SOURCES = source_fingerprints()
SOURCES[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__).read_bytes()).hexdigest()


def hashes(prefix):
    return {str(prefix.with_suffix(suffix).relative_to(ROOT)):
            sha256(prefix.with_suffix(suffix).read_bytes()).hexdigest()
            for suffix in ('.json', '.npz')}


def row(params, p0, n, result):
    sold = result['mu'] * result['demand']
    rate = float(sold.sum())
    use = float(sold @ result['policy'] / rate)
    return dict(p0=p0, rho=params.rho, n=n, model='monopoly', verified=True,
        sale_rate=rate, outside_rate=1-rate,
        product2_calendar=float(sold @ result['policy']),
        product2_conditional=use,
        profit_calendar=float(sold @ (params.R-params.c1
            -(params.c2-params.c1)*result['policy'])),
        value_initial=float(np.interp(.5, result['grid'], result['value'])))


def main():
    params = Parameters(rho=.9)
    summaries = []
    for p0, grids in ((.8, (201, 251)), (.9, (151, 201, 251))):
        tag = str(p0).replace('.', 'p')
        case_dir = HOME/'data/p0_sweep'/('p0_'+tag)/'rho_0p9'
        manifest = json.loads((case_dir/'manifest.json').read_text())
        coarse_tag = 'rho_0p9_n101_marginal'
        coarse = check_bundle(case_dir, coarse_tag, manifest)
        inputs = hashes(case_dir/'arrays'/coarse_tag)
        references = {101: dict(p0=p0, rho=.9, n=101,
                               outcomes=[coarse['outcomes'][0]])}
        fine151_tag = 'rho_0p9_n151_marginal'
        if fine151_tag in manifest['completed']:
            fine151 = check_bundle(case_dir, fine151_tag, manifest)
            references[151] = dict(p0=p0, rho=.9, n=151,
                                  outcomes=[fine151['outcomes'][0]])
            inputs.update(hashes(case_dir/'arrays'/fine151_tag))
        for n in grids:
            prefix = DEST/f'monopoly_only_p0_{tag}_rho_0p9_n{n}'
            started = time.monotonic()
            print(f'[{now()}] unpublished diagnostic p0={p0}, rho=.9, n={n}', flush=True)
            result = solve_monopoly(params, n, p0)
            outcome = row(params, p0, n, result)
            current = dict(p0=p0, rho=.9, n=n, outcomes=[outcome])
            comparisons = [stability(ref, current)[0]
                           for g, ref in sorted(references.items()) if g < n]
            bundle = dict(status='verified_monopoly_unpublished_diagnostic',
                diagnostic_only=True, publishable_comparison=False,
                configuration=dict(p0=p0, rho=.9, n=n,
                                   parameters=asdict(params), dp_tol=1e-10),
                monopoly=result, outcomes=[outcome], grid_stability=comparisons,
                numerical_sources=SOURCES, reference_input_sha256=dict(inputs),
                created_utc=now(), elapsed_seconds=time.monotonic()-started,
                scope=('Unpublished one-dimensional monopoly grid diagnostic only. '
                       'No competitive equilibrium or conditional equilibrium is inferred. '
                       'It does not certify any displayed comparison grid.'))
            save_bundle(prefix, bundle)
            loaded = load_bundle(prefix)
            assert loaded['monopoly']['verified'] and loaded['outcomes'] == bundle['outcomes']
            references[n] = current
            inputs.update(hashes(prefix))
            summaries.append(dict(p0=p0, rho=.9, n=n, outcome=outcome,
                grid_stability=comparisons, diagnostic_bundle_sha256=hashes(prefix)))
            write_json(DEST/'diagnostic_summary.json', dict(
                status='unpublished_monopoly_diagnostics', diagnostic_only=True,
                generated_utc=now(), results=summaries, numerical_sources=SOURCES))
            print(json.dumps(dict(p0=p0, n=n, profit=outcome['profit_calendar'],
                                  comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    main()
