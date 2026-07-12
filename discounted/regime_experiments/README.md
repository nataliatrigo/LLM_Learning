# Discounted regime experiments

This project studies the stationary discounted policy approximation across the
three orderings of `p0`, `p1`, and `p2`. It does not use the fluid model and it
does not modify the paper or the core DP code.

Run the complete configured design from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python discounted/regime_experiments/run_all.py
```

Run a six-configuration validation first:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python discounted/regime_experiments/run_all.py --smoke
```

Outputs are isolated under `discounted/regime_experiments/outputs/`.

## Solver terminology

The reused routine `discounted.DP.exact_dp.solve_discounted_finite_horizon`
performs exact backward induction for a finite terminal truncation. The new
analysis uses a long discounted tail and compares two truncations on a common
interior. It calls the resulting policy a stationary approximation only when
that comparison is stable.

- `analysis_outer_diagonal`: largest history length included in a snapshot.
- `reliable_interior_grid`: common interior used for policy diagnostics.
- `remaining_tail`: discounted calendar periods between the snapshot and the
  zero terminal layer.

The legacy label `t=50` is a calendar-period policy slice in the truncated
problem. It is neither 50 periods remaining nor, without a tail-stability
check, an exact stationary policy.

## Output tables

- `parameter_summary.csv`: configuration-level economics and convergence.
- `diagonal_boundaries.csv`: policy geometry by history length.
- `state_policy.csv`: selected state-level policies and reach probabilities.
- `regime_summary.csv`: descriptive regime aggregates.
- `primary_convergence.csv`: three-grid confirmation for baseline and primary
  representative cases.
- `extended_tail_stability.csv`: long-tail stability on the enlarged
  `n<=180` interior for representative Regime I and II cases.

The centered-state identities are checked deterministically. All occupancy,
reachability, and value calculations are recursive DP calculations; no Monte
Carlo estimates are used.

`figure7_reach_weighted_investment_center.png` reports the exact state-level
reachability-weighted center of product-2 states in standardized frontier
coordinates. Its observed signs are reported as numerical evidence rather
than forced to match an ex ante narrative.
