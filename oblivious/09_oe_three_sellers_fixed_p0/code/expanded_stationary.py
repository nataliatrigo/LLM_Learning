"""Process-local positive stationary reference for the expanded marginal sweep.

The original positive GTH arithmetic and all original acceptance checks remain
in use.  Its tested dense-storage implementation replaces sparse-map storage,
and a larger state-count budget enables both positive elimination orders on
the exact exchangeable-seller quotient through n=51.  No campaign is launched
by importing or installing this module.
"""

from hashlib import sha256
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_gth as dense
from oblivious.oe_p0.code import positive_stationary
from . import model
from .run_experiment import ROOT


def source_reference():
    paths = [Path(__file__).resolve(), Path(positive_stationary.__file__).resolve(),
             Path(positive_stationary.__file__).with_suffix('.cpp').resolve()]
    return dict(dense.helper_reference(), **{
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in paths})


def install(reference_max_states=25000, threads=2):
    """Install final checks in this process and return an undo callback.

    Dense GTH allocations use the existing cross-process memory gate.  The
    stationary module and exploratory search are not replaced: only the model
    module's final symmetric certificate receives the larger reference budget.
    Both sparse candidates, both positive orders, and every original tolerance
    remain in the existing stationary implementation.
    """
    if (isinstance(reference_max_states, bool) or int(reference_max_states) != reference_max_states
            or reference_max_states < 6000):
        raise ValueError('Use an integer positive-reference budget >=6000 states')
    if threads != 2:
        raise ValueError('The checked dense positive GTH backend uses exactly two threads')
    reference_max_states = int(reference_max_states)
    previous_backend = dense.install(threads=threads)
    previous_check = model.checked_symmetric_stationary

    def expanded_check(*args, **kwargs):
        kwargs['reference_max_states'] = max(kwargs.get('reference_max_states', 2000), reference_max_states)
        kwargs['positive_max_iterations'] = max(kwargs.get('positive_max_iterations', 100_000), 1_000_000)
        prior_time_limit = kwargs.get('positive_time_limit', 120)
        kwargs['positive_time_limit'] = (None if prior_time_limit is None
                                         else max(prior_time_limit, 240))
        return previous_check(*args, **kwargs)

    model.checked_symmetric_stationary = expanded_check

    def restore():
        model.checked_symmetric_stationary = previous_check
        positive_stationary._kernel = previous_backend

    return restore


__all__ = ['install', 'source_reference']
