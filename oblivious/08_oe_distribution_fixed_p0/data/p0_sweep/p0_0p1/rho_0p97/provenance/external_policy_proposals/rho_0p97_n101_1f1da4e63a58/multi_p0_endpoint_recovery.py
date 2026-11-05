"""Recover strict conditional-demand validation from probability endpoint dust.

The unchanged outside-demand quadrature can return 1+2.22e-16 at a saturated
entry. The conditional belief constructor correctly rejects D>1. This explicit
wrapper projects only invalid endpoint entries within 1e-12, rejects larger
errors, and leaves every valid tiny positive probability untouched. All final
positive/reversed-order/optimality certificates are the original methods.

Use the same arguments as multi_p0_run, only in an exclusively owned campaign.
The wrapper and its projection metadata are fingerprinted in each new case.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code import multi_p0_fast_run as fast
from oblivious.oe_distribution_fixed_p0.code.conditional_model import (
    FixedOutsideConditionalExperiment as OriginalConditionalExperiment,
)

ENDPOINT_TOLERANCE = 1e-12


def helper_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(runner.ROOT)): sha256(path.read_bytes()).hexdigest()}


def project_demand_endpoints(raw):
    raw = np.asarray(raw, dtype=float)
    if (raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.isfinite(raw).all()
            or np.any(raw < -ENDPOINT_TOLERANCE) or np.any(raw > 1+ENDPOINT_TOLERANCE)):
        raise ArithmeticError('Demand error exceeds explicit probability-endpoint roundoff tolerance')
    projected = np.clip(raw, 0, 1)
    valid = (raw >= 0)&(raw <= 1)
    if not np.array_equal(projected[valid], raw[valid]):
        raise ArithmeticError('Endpoint recovery changed a valid positive demand probability')
    changed = np.argwhere(raw != projected)
    metadata = dict(
        method='Project only invalid probability endpoint roundoff; original quadrature unchanged',
        endpoint_tolerance=ENDPOINT_TOLERANCE,
        maximum_absolute_change=float(np.max(abs(projected-raw), initial=0)),
        changed_entry_count=len(changed), below_zero_count=int(np.sum(raw < 0)),
        above_one_count=int(np.sum(raw > 1)), valid_entries_unchanged=True,
        positive_probability_floor=0., raw_minimum=float(raw.min()), raw_maximum=float(raw.max()),
        raw_matrix_sha256=sha256(raw.tobytes()).hexdigest(),
        projected_matrix_sha256=sha256(projected.tobytes()).hexdigest(),
        changed_entries=[dict(i=int(i), j=int(j), before=float(raw[i,j]), after=float(projected[i,j]))
                         for i,j in changed],
        helper_input_sha256=helper_reference(),
    )
    return projected, metadata


class EndpointSafeConditionalExperiment(OriginalConditionalExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game.D, self.endpoint_projection = project_demand_endpoints(self.game.D)
        self.game.selection = np.stack((self.game.D, self.game.D.T))
        self.game.demand_diagnostics = dict(self.game.demand_diagnostics,
                                          endpoint_roundoff_projection=self.endpoint_projection)

    def configuration(self):
        result = super().configuration()
        result['demand_endpoint_roundoff_projection'] = self.endpoint_projection
        return result


def install():
    fast.install()
    if getattr(runner, '_endpoint_recovery_helper_installed', False):
        return
    original_seed_loader = runner.baseline_seed

    def baseline_seed_with_recovery_provenance(rho, n, closure):
        seed, inputs = original_seed_loader(rho, n, closure)
        return seed, dict(inputs, **helper_reference())

    runner.baseline_seed = baseline_seed_with_recovery_provenance
    runner.FixedOutsideConditionalExperiment = EndpointSafeConditionalExperiment
    runner._endpoint_recovery_helper_installed = True


def main():
    install()
    runner.main()


if __name__ == '__main__':
    main()
