# Gamma Sweep: Investment Extinction Scaling

## Fits

All completed gammas: beta=2.27609, SE=0.0654717, R^2=0.996701, n=6.
Excluding gamma=0.90: beta=2.20827, SE=0.0447513, R^2=0.998769, n=5.

## Predictions

- P1 beta in [1.6, 2.4]: **PASS**.
- P2 n_star_max at gamma=0.90 approximately 30-40: **FAIL** (n_star_max(0.90)=9).
- P3 containment constant c roughly stable across gamma: **FAIL** (min_c_all range [1.34, 2.89], CV=0.296).

## Per-Gamma Table

| gamma | n_max_used | horizon | complete | n_star_max | investment_states | min_c_all | n_bar_over_n_star | band_violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 200 | 333 | True | 9 | 19 | 1.34164 | 3.092551e+05 | 0 |
| 0.92 | 200 | 367 | True | 18 | 47 | 1.34164 | 6.163649e+05 | 0 |
| 0.94 | 200 | 425 | True | 38 | 139 | 1.66667 | 1.712535e+06 | 0 |
| 0.96 | 200 | 540 | True | 97 | 550 | 2.12132 | 7.970500e+06 | 0 |
| 0.98 | 1000 | 1685 | True | 435 | 5113 | 2.52982 | 1.185380e+08 | 0 |
| 0.99 | 4000 | 5376 | True | 1831 | 44364 | 2.88675 | 1.839318e+09 | 0 |

## Attempt Log

| gamma | doubling | n_max | horizon | n_star_max | investment states | interior? |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.90 | 0 | 200 | 333 | 9 | 19 | yes |
| 0.92 | 0 | 200 | 367 | 18 | 47 | yes |
| 0.94 | 0 | 200 | 425 | 38 | 139 | yes |
| 0.96 | 0 | 200 | 540 | 97 | 550 | yes |
| 0.98 | 0 | 500 | 1185 | 435 | 5113 | NO |
| 0.98 | 1 | 1000 | 1685 | 435 | 5113 | yes |
| 0.99 | 0 | 2000 | 3376 | 1831 | 44364 | NO |
| 0.99 | 1 | 4000 | 5376 | 1831 | 44364 | yes |

## Flags

- Single-interval check: no violations in completed solves.
- Tolerance/residual: exact_dp.py uses exact finite-horizon backward induction, not value iteration; iterations and residual are not applicable.

## Takeaway

1. The empirical extinction exponent is about 2.27609 on the completed sweep.
2. The theoretical bound remains very loose: n_bar/n_star ranges from 3.092551e+05 to 1.839318e+09 on completed runs.
3. Collar localization is fairly stable only if the reported c-range/CV is accepted: min_c_all range [1.34, 2.89], CV=0.296.

## Outputs

- `gamma_sweep.csv`
- `gamma_sweep_attempts.csv`
- `scaling_plot.png`
- `gap_plot.png`
