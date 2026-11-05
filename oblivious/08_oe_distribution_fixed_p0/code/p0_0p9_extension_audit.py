"""Audit the added p0=.9 comparison using the unchanged raw-checkpoint audit.

The added p0 has its own review inside the sweep and records this entrypoint
as an input. The previous multi-p0 audit source and review are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code.multi_p0_audit import digest, require, run_audit
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import campaign_directory
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, case_tag, write_json


P0S = (.9,)
RHOS = (.6, .7, .8)
OUTPUT = HOME/'data/p0_sweep/p0_0p9/independent_review.json'


def main():
    for rho in RHOS:
        manifest = json.loads((campaign_directory(.9, rho)/'manifest.json').read_text())
        required = {case_tag(rho, n)+'_'+closure
                    for n in (51, 101) for closure in ('marginal', 'conditional')}
        require(required.issubset(manifest['completed']),
                f'p0=.9 rho={rho:g} lacks required n51/n101 checkpoints')
    # Protect both historical artifacts across this new audit. No numerical
    # solve is invoked by run_audit: it verifies the saved finite-grid equations.
    preserved = (
        HOME/'code/multi_p0_audit.py',
        HOME/'data/p0_sweep/independent_review.json',
    )
    preserved_sha = {str(path.relative_to(ROOT)): digest(path) for path in preserved}
    entrypoint = Path(__file__).resolve()
    entrypoint_key = str(entrypoint.relative_to(ROOT))
    entrypoint_sha = digest(entrypoint)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    review = run_audit(p0s=P0S, rhos=RHOS, output=OUTPUT)
    require(all(not case['missing_base_checkpoints'] for case in review['campaign_statuses']),
            'All selected p0=.9 cases must have complete n51/n101 checkpoints')
    require(digest(entrypoint) == entrypoint_sha, 'Audit entrypoint changed during review')
    for source, expected in preserved_sha.items():
        require(digest(ROOT/source) == expected, 'Historical audit artifact changed: '+source)
    review['audit_entrypoint'] = entrypoint_key
    review['audit_entrypoint_sha256'] = entrypoint_sha
    review['audited_input_sha256'][entrypoint_key] = entrypoint_sha
    review['audited_input_count'] = len(review['audited_input_sha256'])
    review['historical_audit_preserved_sha256'] = preserved_sha
    review['publication_requirement_base_checkpoints_complete'] = True
    write_json(OUTPUT, review)
    print(json.dumps(dict(
        verified=review['verified'],
        audited_bundle_count=review['audited_bundle_count'],
        audited_input_count=review['audited_input_count'],
        output=str(OUTPUT.relative_to(ROOT)),
        residual_maxima=review['residual_maxima'],
    )), flush=True)


if __name__ == '__main__':
    main()
