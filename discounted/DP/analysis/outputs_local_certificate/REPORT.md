# Local Extinction Certificate Report

## Parameters

```text
p1=0.35, p2=0.8, c1=0.05, c2=0.65, R=1.0, gamma=0.98, p0=0.5
Delta_p=0.45000000000000007, Delta_c=0.6, Delta_c/Delta_p=1.33333333333
```

## Demand Identity Check

Verified `D(S+1,F)-D(S,F+1)=P(Bin(S+F+2,p0)=S+1)` through diagonal n=1000.

- Maximum absolute error: `5.360e-14`
- Maximum relative error: `3.697e+00`
- Maximum relative error where the binomial mass is at least `1e-12`: `1.012e-04`

## Correctness Checks

Demand is computed with the Beta survival function `betaincc(S+1,F+1,p0) = P(Beta(S+1,F+1) >= p0)`.

- Checked demand states: `501501`
- Max absolute error versus Beta survival: `0.000e+00`
- Max distance from lower-tail CDF: `1.000e+00`

The local recursion uses `A=(R-c1)*(D(S+1,F)-D(S,F+1))/(ell(D(S+1,F))*ell(D(S,F+1)))` and `B=gamma*D(S,F+1)/ell(D(S,F+1))`.

| N | min_B | max_B | B_lower_violations | B_upper_violations | B_range_pass |
| --- | --- | --- | --- | --- | --- |
| 800 | 1.83713e-240 | 0.98 | 0 | 0 | True |
| 1200 | 0 | 0.98 | 0 | 0 | True |
| 1600 | 0 | 0.98 | 0 | 0 | True |
| 2400 | 0 | 0.98 | 0 | 0 | True |

## Empirical Bellman Comparison

The empirical comparison used the existing finite-horizon DP solver on `S+F <= 800`.

- Empirical product-2 states: `5113`
- Empirical largest investment diagonal `n_emp`: `435`

Validation against the certificate:

| terminal_kind | compared_states | empirical_product2_states | violations | min_gamma_U_minus_threshold_on_empirical_product2 | max_gamma_U_minus_threshold_on_empirical_product2 |
| --- | --- | --- | --- | --- | --- |
| selected_N800 | 321201 | 5113 | 0 | 2.02203 | 31.2948 |
| selected_N1200 | 321201 | 5113 | 0 | 2.02203 | 31.2948 |
| selected_N1600 | 321201 | 5113 | 0 | 2.02203 | 31.2948 |
| selected_N2400 | 321201 | 5113 | 0 | 2.02203 | 31.2948 |

The validation checks the requested inequality directly: every empirical product-2 state must have `gamma*U(S,F)-Delta_c/Delta_p >= 0`. If violations were positive, the script would stop.

## Certified Extinction

The table below uses only valid terminal upper bounds. `certified_extinction_diagonal` is blank when the artificial terminal boundary leaves at least one non-certified state on the last grid diagonal.

| N | terminal_kind | terminal_bound | certified_share | largest_noncertified_diagonal | certified_extinction_diagonal |
| --- | --- | --- | --- | --- | --- |
| 800 | crude / unrolled | 47.5 | 0.705493 | 800 |  |
| 800 | lemma2 | 10902.6 | 0.513398 | 800 |  |
| 1200 | crude / unrolled | 47.5 | 0.788246 | 1200 |  |
| 1200 | lemma2 | 8907.45 | 0.632724 | 1200 |  |
| 1600 | crude / unrolled | 47.5 | 0.834009 | 1600 |  |
| 1600 | lemma2 | 7716.48 | 0.708812 | 1600 |  |
| 2400 | crude / unrolled | 47.5 | 0.883428 | 2400 |  |
| 2400 | lemma2 | 6302.45 | 0.795775 | 2400 |  |

## Interpretation

No valid terminal condition produced a full certified extinction tail inside the computed grids. In every valid run, the last artificial grid diagonal still contains at least one non-certified state, so the certified extinction diagonal is blank.

At the largest grid, `N=2400`, the selected valid certificate rules out product 2 on `88.343%` of states, but its largest non-certified diagonal is still `2400` because of the terminal-bound boundary layer.

Thus this truncated-grid implementation improves the old theorem in a state-by-state sense inside the grid, and it passes the empirical no-overlap check against `n_emp=435`. It does not yet provide a valid finite-grid extinction cutoff close to the empirical diagonal `435` under these diagonal-wide terminal bounds.

## Old Theorem 1 Bound

- `kappa0(p0) = 2.60167188056`
- `bar_n = 5.156404e+10`
- The theorem certifies product 1 once `S+F+2 > bar_n`, i.e. beyond approximately diagonal `5.156404e+10`.

## Terminal-Bound Sensitivity

| N | crude / unrolled | lemma2 | optimistic_zero | selected |
| --- | --- | --- | --- | --- |
| 800 | 47.5 | 10902.6 | 0 | 47.5 |
| 1200 | 47.5 | 8907.45 | 0 | 47.5 |
| 1600 | 47.5 | 7716.48 | 0 | 47.5 |
| 2400 | 47.5 | 6302.45 | 0 | 47.5 |

Boundary sensitivity across diagnostic grids:

| N | reference_N | common_diagonals | max_abs_m_low_change | mean_abs_m_low_change | max_abs_m_high_change | mean_abs_m_high_change | substantial_upper_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 800 | 2400 | 801 | 0.00997506 | 1.55724e-05 | 0.48005 | 0.10477 | True |
| 1200 | 2400 | 1201 | 0.00915141 | 1.03936e-05 | 0.488353 | 0.0712308 | True |
| 1600 | 2400 | 1601 | 0.00873908 | 7.79982e-06 | 0.492509 | 0.0539154 | True |
| 2400 | 2400 | 2401 | 0 | 0 | 0 | 0 | False |

Identical valid terminal choices are combined in the displayed tables and plots; the raw CSV files still keep each terminal condition separately.

The `optimistic_zero` terminal is included in the CSV outputs only as a non-certified diagnostic. It is not used as a certificate.

## Output Files

- `demand_identity_check.csv`
- `grid_sensitivity.csv`
- `terminal_bounds.csv`
- `correctness_checks.csv`
- `boundary_sensitivity.csv`
- `empirical_comparison_N800.csv.gz`
- `certificate_states_N800.csv.gz`
- Per-grid diagnostics: `diagnostics_N800/`, `diagnostics_N1200/`, `diagnostics_N1600/`, `diagnostics_N2400/`
- Root copies for `N=800`:
  - `plot1_empirical_product2_region_N800.png`
  - `plot2_noncertified_region_N800.png`
  - `plot3_empirical_vs_noncertified_N800.png`
  - `plot4_noncertified_boundaries_N800.png`
  - `plot5_gamma_U_margin_heatmap_N800.png`
- `grid_sensitivity_extinction_diagonal.png`
- `terminal_condition_comparison.png`

