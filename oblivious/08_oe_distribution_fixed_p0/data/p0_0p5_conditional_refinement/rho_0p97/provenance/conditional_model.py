"""Conditional-distribution OE with two sellers and a fixed outside option.

This experiment changes only the belief closure of the fixed-p0 marginal OE.
At every positive-mass own state x, the rival belief is nu[x, :]/mu_A[x]
from the actual joint calendar-time law. The seller still chooses a symmetric
own-state policy; it does not observe or condition its action on the realized
rival reputation. An outside purchase leaves both reputations unchanged.

The established conditional solver is game-agnostic: its transition matrix,
demand entries and policy sensitivities all come from ``self.game``. Its fresh
certificate uses positive joint stationary elimination, and independently
reverses the elimination order to check conditional laws on every positive row.
No probability floor is introduced. Exactly zero own-mass rows use the rival
calendar marginal as their explicit off-path completion.

Only numerical source reuse is intended. Historical model modules and their
outputs remain unchanged.
"""

from oblivious.oe_conditional_distribution.code.model import (
    ConditionalExperiment,
    _ConditionalResponse,
    conditional_from_joint,
)
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_p0.code.primitives import Parameters


class FixedOutsideConditionalExperiment(ConditionalExperiment, FixedOutsideMarginalExperiment):
    """Conditional OE sharing demand and economic timing with the marginal OE.

    Inherited ``solve`` and ``evaluate`` are the original conditional methods.
    Explicit construction through the fixed-p0 experiment installs its outside
    option game, rather than the original two-option conditional game.
    """

    def __init__(self, n=21, params=Parameters(), p0=.5, order=128,
                 tol=1e-7, dp_tol=1e-10):
        FixedOutsideMarginalExperiment.__init__(
            self, n=n, params=params, p0=p0, order=order, tol=tol, dp_tol=dp_tol)

    def configuration(self):
        # The MRO passes ConditionalExperiment.configuration through the
        # fixed-p0 configuration before adding the conditional closure.
        result = super().configuration()
        result.update(
            model="Conditional OE with fixed outside option",
            conditional_event="Calendar-time own reputation, before selection; not conditional on a sale",
            certificate="Fresh positive joint stationary law and reversed-order rowwise conditional check",
        )
        return result


__all__ = ["FixedOutsideConditionalExperiment", "conditional_from_joint"]
