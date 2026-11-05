# Experiment 12 results: long-run average profit

Completed 24 parameter combinations. The active grids were read from `oblivious/10_fluid_oblivious/config.json`: p0 = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]; rho = [0.6, 0.7, 0.8, 0.9].

Economic parameters: p1=0.35, p2=0.8, c1=0.05, c2=0.65, R=1. The source gamma=0.98 is recorded for provenance and excluded from the objective.

Both markets are reoptimized over all constant proportions u in [0,1]. The objective is long-run sales per unit time times margin: D(q(u))*m(u). Demand retains the rho-dependent Beta posterior and the outside option. Initial conditions, transient earnings and discounting do not enter this criterion.

- Monopoly: 24 detected verified solutions; 0 cases with multiple solutions; 0 cases without a verified solution; 13 interior mixtures.
- Competition: 24 detected verified solutions; 0 cases with multiple solutions; 0 cases without a verified solution; 13 interior mixtures.

## Product choice, seller demand, and total use

Every detected monopoly/competition pair is retained in `data/comparisons.csv`; no solutions are averaged.

- Competition product-2 fraction among sales: higher in 17 pairs, lower in 0, equal within 2e-06 in 7.
- Competition demand per seller: higher in 0 pairs, lower in 24, equal within 2e-06 in 0.
- Competition product-2 use per seller per unit time: higher in 9 pairs, lower in 15, equal within 2e-06 in 0.
- Competition total market product-2 use per unit time: higher in 24 pairs, lower in 0, equal within 2e-06 in 0.

A higher fraction u is a change in product choice. The rate D*u also changes when demand changes. The table records an exact decomposition of the per-seller rate difference into choice and demand contributions. The market measure uses D_M*u_M for monopoly and 2*D_C*u_C for competition.

## All parameter cases

Each row is one verified solution; multiple rows would display multiplicity explicitly.

| p0 | rho | Market | u | q(u) | Demand / seller | Average profit / seller | Product 2 / seller / time | Product 2 / market / time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.6 | competition | 0.101093 | 0.395492 | 0.499177 | 0.443940 | 0.050463 | 0.100927 |
| 0.1 | 0.6 | monopoly | 0.000000 | 0.350000 | 0.946610 | 0.899280 | 0.000000 | 0.000000 |
| 0.1 | 0.7 | competition | 0.333471 | 0.500062 | 0.499920 | 0.374899 | 0.166709 | 0.333418 |
| 0.1 | 0.7 | monopoly | 0.000000 | 0.350000 | 0.957330 | 0.909464 | 0.000000 | 0.000000 |
| 0.1 | 0.8 | competition | 0.630288 | 0.633630 | 0.500000 | 0.285913 | 0.315144 | 0.630288 |
| 0.1 | 0.8 | monopoly | 0.000000 | 0.350000 | 0.972673 | 0.924040 | 0.000000 | 0.000000 |
| 0.1 | 0.9 | competition | 1.000000 | 0.800000 | 0.500000 | 0.175000 | 0.500000 | 1.000000 |
| 0.1 | 0.9 | monopoly | 0.000000 | 0.350000 | 0.992564 | 0.942935 | 0.000000 | 0.000000 |
| 0.3 | 0.6 | competition | 0.272158 | 0.472471 | 0.475441 | 0.374032 | 0.129395 | 0.258790 |
| 0.3 | 0.6 | monopoly | 0.016249 | 0.357312 | 0.677728 | 0.637235 | 0.011012 | 0.011012 |
| 0.3 | 0.7 | competition | 0.420750 | 0.539338 | 0.489060 | 0.341144 | 0.205772 | 0.411544 |
| 0.3 | 0.7 | monopoly | 0.100177 | 0.395080 | 0.721247 | 0.641833 | 0.072252 | 0.072252 |
| 0.3 | 0.8 | competition | 0.642706 | 0.639218 | 0.498678 | 0.281442 | 0.320504 | 0.641007 |
| 0.3 | 0.8 | monopoly | 0.170260 | 0.426617 | 0.778330 | 0.659902 | 0.132519 | 0.132519 |
| 0.3 | 0.9 | competition | 1.000000 | 0.800000 | 0.500000 | 0.175000 | 0.500000 | 1.000000 |
| 0.3 | 0.9 | monopoly | 0.205600 | 0.442520 | 0.857023 | 0.708450 | 0.176204 | 0.176204 |
| 0.5 | 0.6 | competition | 0.532306 | 0.589538 | 0.416658 | 0.262751 | 0.221789 | 0.443579 |
| 0.5 | 0.6 | monopoly | 0.417854 | 0.538034 | 0.539214 | 0.377065 | 0.225313 | 0.225313 |
| 0.5 | 0.7 | competition | 0.639855 | 0.637935 | 0.444641 | 0.251706 | 0.284506 | 0.569011 |
| 0.5 | 0.7 | monopoly | 0.500093 | 0.575042 | 0.592759 | 0.385260 | 0.296435 | 0.296435 |
| 0.5 | 0.8 | competition | 0.769953 | 0.696479 | 0.477279 | 0.232926 | 0.367482 | 0.734965 |
| 0.5 | 0.8 | monopoly | 0.572911 | 0.607810 | 0.667791 | 0.404851 | 0.382585 | 0.382585 |
| 0.5 | 0.9 | competition | 1.000000 | 0.800000 | 0.499465 | 0.174813 | 0.499465 | 0.998930 |
| 0.5 | 0.9 | monopoly | 0.618656 | 0.628395 | 0.779685 | 0.451287 | 0.482357 | 0.482357 |
| 0.7 | 0.6 | competition | 0.813996 | 0.716298 | 0.317914 | 0.146750 | 0.258781 | 0.517562 |
| 0.7 | 0.6 | monopoly | 0.771674 | 0.697253 | 0.376687 | 0.183445 | 0.290680 | 0.290680 |
| 0.7 | 0.7 | competition | 0.901950 | 0.755877 | 0.356292 | 0.145663 | 0.321357 | 0.642715 |
| 0.7 | 0.7 | monopoly | 0.850578 | 0.732760 | 0.433209 | 0.190462 | 0.368478 | 0.368478 |
| 0.7 | 0.8 | competition | 0.996212 | 0.798295 | 0.410461 | 0.144595 | 0.408906 | 0.817813 |
| 0.7 | 0.8 | monopoly | 0.925258 | 0.766366 | 0.520578 | 0.205548 | 0.481669 | 0.481669 |
| 0.7 | 0.9 | competition | 1.000000 | 0.800000 | 0.451097 | 0.157884 | 0.451097 | 0.902193 |
| 0.7 | 0.9 | monopoly | 0.983270 | 0.792472 | 0.668774 | 0.240784 | 0.657586 | 0.657586 |
| 0.8 | 0.6 | competition | 0.964303 | 0.783936 | 0.246517 | 0.091561 | 0.237717 | 0.475433 |
| 0.8 | 0.6 | monopoly | 0.943431 | 0.774544 | 0.278800 | 0.107043 | 0.263029 | 0.263029 |
| 0.8 | 0.7 | competition | 1.000000 | 0.800000 | 0.269374 | 0.094281 | 0.269374 | 0.538747 |
| 0.8 | 0.7 | monopoly | 1.000000 | 0.800000 | 0.320844 | 0.112295 | 0.320844 | 0.320844 |
| 0.8 | 0.8 | competition | 1.000000 | 0.800000 | 0.285252 | 0.099838 | 0.285252 | 0.570503 |
| 0.8 | 0.8 | monopoly | 1.000000 | 0.800000 | 0.344640 | 0.120624 | 0.344640 | 0.344640 |
| 0.8 | 0.9 | competition | 1.000000 | 0.800000 | 0.309408 | 0.108293 | 0.309408 | 0.618815 |
| 0.8 | 0.9 | monopoly | 1.000000 | 0.800000 | 0.382598 | 0.133909 | 0.382598 | 0.382598 |
| 0.9 | 0.6 | competition | 1.000000 | 0.800000 | 0.114857 | 0.040200 | 0.114857 | 0.229714 |
| 0.9 | 0.6 | monopoly | 1.000000 | 0.800000 | 0.122341 | 0.042819 | 0.122341 | 0.122341 |
| 0.9 | 0.7 | competition | 1.000000 | 0.800000 | 0.113476 | 0.039717 | 0.113476 | 0.226952 |
| 0.9 | 0.7 | monopoly | 1.000000 | 0.800000 | 0.120769 | 0.042269 | 0.120769 | 0.120769 |
| 0.9 | 0.8 | competition | 1.000000 | 0.800000 | 0.107737 | 0.037708 | 0.107737 | 0.215474 |
| 0.9 | 0.8 | monopoly | 1.000000 | 0.800000 | 0.114265 | 0.039993 | 0.114265 | 0.114265 |
| 0.9 | 0.9 | competition | 1.000000 | 0.800000 | 0.085551 | 0.029943 | 0.085551 | 0.171102 |
| 0.9 | 0.9 | monopoly | 1.000000 | 0.800000 | 0.089562 | 0.031347 | 0.089562 | 0.089562 |

## Focal diagnostic: p0=0.8, rho=0.9

- Monopoly: u=1.00000000, q=0.80000000, D=0.3825984512, average profit=0.1339094579, per-seller use=0.3825984512, market use=0.3825984512, maximum deviation gain=0 (tolerance 1.35e-10).
- Competition: u=1.00000000, q=0.80000000, D=0.3094076638, average profit=0.1082926823, per-seller use=0.3094076638, market use=0.6188153275, maximum deviation gain=0 (tolerance 1.09e-10).

The fixed-rival payoff curves and the full best-response correspondence appear in the diagnostic figure. At this case both markets choose product 2 exclusively, so the per-seller use difference comes from demand. The two competing sellers together serve more product 2 than the monopolist.

Experiment 10 reported two discounted stationary states at this parameter case. Here the average-profit objectives are optimized anew: u=0 is rejected by a profitable constant-mixture deviation. The criteria and solution concepts differ, so the earlier stationary branches are not imported.

## Numerical verification

Each global maximum search scans the whole interval, refines every sampled local peak, and compares both endpoints. The competition search scans and refines minima of the nonnegative unilateral deviation gain, including zeros without sign changes, and supplements them with partial-own-derivative candidates. Rival actions remain fixed in every best response.

Acceptance tolerance is `1e-12 + 1e-9 * profit_scale`, with profit_scale the maximum absolute payoff in the relevant full best-response search. Distinct candidates are deduplicated within 1e-6 in u; no candidate averaging is performed. The saved solution rows include actual scales and tolerances.

The coarse/fine runs use 501/1001 own-action nodes, 241/481 rival scan nodes, and quadrature orders 128/256. Maximum solution change in u: 1.02e-07. Maximum independent adaptive-demand error: 5.15e-13. Maximum accelerated-scan profit error against shared Jacobi quadrature: 1.36e-13. All required numerical checks passed: True.

These are global exploration, refinement and independent quadrature checks, not a mathematical proof that an arbitrarily narrow additional optimum cannot exist. All detected verified solutions are preserved. If no equilibrium is detected, the best residual and its location remain in `data/case_summary.csv` and `data/search_results.json`.

## Files

- [Three comparisons and diagnostic, combined PDF](figures/overview/average_profit_monopoly_vs_competition.pdf)
- [All solutions](data/solutions.csv), [all branch pairs](data/comparisons.csv), [case diagnostics](data/case_summary.csv)
- [Configuration used](data/configuration_used.json), [validation](data/validation.json), [manifest](data/manifest.json)
- [Independent demand audits](data/independent_validation.csv), [refinement](data/refinement.csv)
