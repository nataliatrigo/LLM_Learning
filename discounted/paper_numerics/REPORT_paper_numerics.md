# Paper Numerical Illustration

## Numerical inventory and method

The baseline paper computation uses backward recursion on the rearranged stationary Bellman equation, with zero terminal value on diagonal `N_outer+1`. It is a truncated-state approximation, not an exact stationary solution. The older `discounted/DP/exact_dp.py` instead performs finite-calendar-horizon backward induction and is not used for the main figures.

Baseline: `p0=0.5`, `p1=0.35`, `p2=0.8`, `c1=0.05`, `c2=0.65`, `R=1.0`, `gamma=0.98`. These values agree with the current repository defaults.

## Convergence

| smaller_outer | larger_outer | N_report | maximum_abs_value_difference | maximum_abs_gap_difference | policy_disagreements | policy_disagreement_fraction | maximum_bellman_residual | last_active_diagonal | maximum_lower_endpoint_change | maximum_upper_endpoint_change | distance_to_outer_boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 800 | 1200 | 600 | 0.818466 | 0.0255829 | 0 | 0 | 7.10543e-15 | 435 | 0 | 0 | 765 |
| 1200 | 1600 | 600 | 0.000253181 | 5.89736e-06 | 0 | 0 | 7.10543e-15 | 435 | 0 | 0 | 1165 |

The reliable reported interior is `n<=600`. The largest solve has outer diagonal `1600`. The empirical last active diagonal is `435`, leaving `1165` diagonals to the terminal boundary.

## Interval diagnostics

- Tested diagonals: `601`.
- Active diagonals: `436`.
- Active diagonals with one robust interval: `436`.
- Diagonals containing numerical ties: `0`.
- Robust interval violations: `0`.
- Active diagonals containing gaps: `0`.
- Empirical last active diagonal: `435`.

The action tolerance is `max(1e-10, 1e-8 max(1, max|advantage|))` separately on each diagonal. A violation requires two robust product-2 components separated by a robust product-1 state.

The continuation-gap panels use diagonals `[25, 100, 114, 250, 410, 435, 460, 600]` and plot the unsmoothed computed sequences.

## Existing robustness evidence

The prior reproducible search in `discounted/DP/analysis/test_gap_unimodality.py` examined 1175 configurations across all three quality regimes and found no robust within-diagonal interval violation. However, 78 first-pass cases had policy changes between the small outer grids, so the broad search is supporting evidence rather than a claim that all 1175 policies were fully converged. The baseline and every candidate violation received stronger outer-grid checks.

## Interpretation

- **Analytical:** the paper proves localization in an outer shrinking collar and eventual extinction.
- **Numerical approximation:** the stationary truncated-state recursion produces the plotted baseline policy and continuation gaps.
- **Empirical regularity:** every stable active baseline diagonal is a single product-2 interval.
- **Open:** a general analytical proof of the interval property is not provided.
