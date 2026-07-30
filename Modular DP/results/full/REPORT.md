# Modular discounted-DP experiment

Profile: `full`. All quantities below are computed outputs, not smoothed
curves. The original paper solver and its outputs were left unchanged.

## 1. Repository files inspected

- `discounted/paper_numerics/stationary_solver.py`: stationary discounted
  benchmark and exact idle-self-loop rearrangement.
- `discounted/paper_numerics/run_paper_numerics.py`: baseline outer-grid
  comparisons, policy classification, tables, and plots.
- `discounted/paper_numerics/outputs/`: benchmark figures and tables.
- `discounted/DP/exact_dp.py`: finite-calendar-horizon solver; inspected but
  not substituted for the stationary benchmark.
- `../LLM_Learning_archived_experiments/discounted_DP/experiments/epsilon_greedy_paths.py`
  and `verify_epsilon_nonextinction.py`: archived prior experiments used only
  as conceptual checks. Their directed-exploration default is not used here.
- `pyproject.toml` and `.gitignore`: dependencies and generated-data policy.

## 2. Learning rules

For count state `(S,F)`, standard Thompson sampling uses
`Beta(1+S,1+F)`. Scaled updates use `Beta(1+eta*S,1+eta*F)`.

Mean-preserving temperature samples from
`Beta((1+S)/T,(1+F)/T)`: it preserves the standard posterior mean while
changing concentration. Uniform two-seller epsilon-greedy assigns demand
`1-epsilon/2`, `1/2`, or `epsilon/2` according as the posterior mean is above,
at, or below `p0`.

For discrete trajectories, effective sample size means sampling-Beta
concentration minus its value at `(S,F)=(0,0)`: it is `n` for standard TS and
epsilon-greedy, `eta*n` for scaled updates, and `n/T` for temperature.
The raw observation count and full Beta concentration are saved separately.

For forgetting, normalized counts are `x=(1-rho)(a-1)` and
`y=(1-rho)(b-1)`, hence `a=1+x/(1-rho)` and
`b=1+y/(1-rho)`. At state `(x,y)`, both forgetting models use the Thompson
demand

`D(x,y) = Pr[Beta(a,b) >= p0] = 1 - I_{p0}(a,b)`,

where `I` is the regularized incomplete beta function. Their effective sample
size is `a+b-2=(x+y)/(1-rho)`, while Beta concentration is `a+b`. Success maps
to `(rho*x+1-rho,rho*y)` and failure to
`(rho*x,rho*y+1-rho)`. Observation-time idle is the identity. Calendar-time
idle maps to `(rho*x,rho*y)`.

## 3. Relationships between methods

Scaled updates, mean-preserving temperature, epsilon-greedy, and each
forgetting clock are distinct demand or state-transition rules. `eta=1` and
`T=1` are verified aliases of standard Thompson sampling.

## 4. Solvers

Integer models use backward recursion on triangular diagonals with zero value
at the outer boundary, exactly matching the benchmark approximation. Because
idle is an exact self-loop, the solver uses
`V=D*M/[1-gamma*(1-D)]`.

Forgetting models use regular triangular grids in normalized `(x,y)`
coordinates and piecewise-linear barycentric interpolation. The base linear
resolutions are `[201, 401]`; selected `rho=[0.95, 0.99]` cases additionally use resolution `801`.
Here `grid_resolution=N` means `N` equally spaced coordinates on each edge;
`grid_node_count=N(N+1)/2` is the number of triangular nodes actually stored.
Observation-time forgetting uses the rearranged self-loop equation.
Calendar-time forgetting always uses the generic Bellman equation, including
at the origin where its idle transition happens to fix one state.

## 5. Validation and convergence

Every invocation of `scripts/run_experiments.py` reruns the validation gate; it
does not trust a prior passing file. The gate runs the stdlib unit suite,
compares every reported baseline array against the existing solver, checks
`eta=1`, `T=1`, Bellman residuals, transition
invariants, interpolation at nodes, and one case from each continuous family.
The experiment's `run_manifest.json` records both the experiment configuration
hash and `validation_configuration_hash`, so the gate configuration is
auditable. See `results/validation/VALIDATION.md` for the check-level record.

Discrete outer-grid comparisons:

| method | parameter_value | comparison_scope | coarse_outer_diagonal | fine_outer_diagonal | common_report_diagonal | maximum_value_difference | maximum_gap_difference | raw_action_changes | action_changes | distance_to_outer_boundary |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_ts | NA | base | 800 | 1200 | 600 | 0.818466 | 0.0255829 | 0 | 0 | 765 |
| standard_ts | NA | base | 1200 | 1600 | 600 | 0.000253181 | 5.89736e-06 | 0 | 0 | 1165 |
| scaled_updates | 0.25 | base | 800 | 1200 | 600 | 0.818466 | 0.0166245 | 0 | 0 | 1104 |
| scaled_updates | 0.25 | base | 1200 | 1600 | 600 | 0.000253181 | 4.20377e-06 | 0 | 0 | 1504 |
| scaled_updates | 0.5 | base | 800 | 1200 | 600 | 0.818466 | 0.0211369 | 0 | 0 | 990 |
| scaled_updates | 0.5 | base | 1200 | 1600 | 600 | 0.000253181 | 5.09309e-06 | 0 | 0 | 1390 |
| scaled_updates | 1 | base | 800 | 1200 | 600 | 0.818466 | 0.0255829 | 0 | 0 | 765 |
| scaled_updates | 1 | base | 1200 | 1600 | 600 | 0.000253181 | 5.89736e-06 | 0 | 0 | 1165 |
| scaled_updates | 2 | base | 800 | 1200 | 600 | 0.818466 | 0.0288036 | 2 | 2 | 600 |
| scaled_updates | 2 | base | 1200 | 1600 | 600 | 0.000253181 | 6.52585e-06 | 0 | 0 | 1000 |
| scaled_updates | 2 | extended_boundary | 3000 | 3600 | 2200 | 4.45428e-06 | 9.32026e-08 | 0 | 0 | 2715 |
| scaled_updates | 4 | base | 800 | 1200 | 600 | 0.818466 | 0.0262022 | 0 | 0 | 600 |
| scaled_updates | 4 | base | 1200 | 1600 | 600 | 0.000253181 | 6.34323e-06 | 0 | 0 | 1000 |
| scaled_updates | 4 | extended_boundary | 3000 | 3600 | 2200 | 4.45428e-06 | 1.02788e-07 | 0 | 0 | 1816 |
| mean_preserving_temperature | 0.25 | base | 800 | 1200 | 600 | 0.818466 | 0.0262133 | 0 | 0 | 600 |
| mean_preserving_temperature | 0.25 | base | 1200 | 1600 | 600 | 0.000253181 | 6.34479e-06 | 0 | 0 | 1000 |
| mean_preserving_temperature | 0.25 | extended_boundary | 3000 | 3600 | 2200 | 4.45428e-06 | 1.02783e-07 | 0 | 0 | 1816 |
| mean_preserving_temperature | 0.5 | base | 800 | 1200 | 600 | 0.818466 | 0.0288049 | 1 | 1 | 600 |
| mean_preserving_temperature | 0.5 | base | 1200 | 1600 | 600 | 0.000253181 | 6.5252e-06 | 0 | 0 | 1000 |
| mean_preserving_temperature | 0.5 | extended_boundary | 3000 | 3600 | 2200 | 4.45428e-06 | 9.31977e-08 | 0 | 0 | 2716 |
| mean_preserving_temperature | 1 | base | 800 | 1200 | 600 | 0.818466 | 0.0255829 | 0 | 0 | 765 |
| mean_preserving_temperature | 1 | base | 1200 | 1600 | 600 | 0.000253181 | 5.89736e-06 | 0 | 0 | 1165 |
| mean_preserving_temperature | 2 | base | 800 | 1200 | 600 | 0.818466 | 0.0211532 | 0 | 0 | 988 |
| mean_preserving_temperature | 2 | base | 1200 | 1600 | 600 | 0.000253181 | 5.09513e-06 | 0 | 0 | 1388 |
| mean_preserving_temperature | 4 | base | 800 | 1200 | 600 | 0.818466 | 0.0166706 | 0 | 0 | 1098 |
| mean_preserving_temperature | 4 | base | 1200 | 1600 | 600 | 0.000253181 | 4.20991e-06 | 0 | 0 | 1498 |
| epsilon_greedy | 0 | base | 800 | 1200 | 600 | 0.818466 | 0.528797 | 0 | 0 | 600 |
| epsilon_greedy | 0 | base | 1200 | 1600 | 600 | 0.000253181 | 0.000153583 | 0 | 0 | 1000 |
| epsilon_greedy | 0.02 | base | 800 | 1200 | 600 | 0.77806 | 0.379221 | 0 | 0 | 600 |
| epsilon_greedy | 0.02 | base | 1200 | 1600 | 600 | 0.000222 | 8.82683e-05 | 0 | 0 | 1000 |
| epsilon_greedy | 0.05 | base | 800 | 1200 | 600 | 0.719906 | 0.306871 | 0 | 0 | 600 |
| epsilon_greedy | 0.05 | base | 1200 | 1600 | 600 | 0.000181402 | 6.57657e-05 | 0 | 0 | 1000 |
| epsilon_greedy | 0.1 | base | 800 | 1200 | 600 | 0.629406 | 0.217351 | 0 | 0 | 600 |
| epsilon_greedy | 0.1 | base | 1200 | 1600 | 600 | 0.000127819 | 3.89934e-05 | 0 | 0 | 1000 |
| epsilon_greedy | 0.2 | base | 800 | 1200 | 600 | 0.471577 | 0.110516 | 0 | 0 | 600 |
| epsilon_greedy | 0.2 | base | 1200 | 1600 | 600 | 6.00304e-05 | 1.28996e-05 | 0 | 0 | 1000 |
| epsilon_greedy | 0.5 | base | 800 | 1200 | 600 | 0.161497 | 0.0384282 | 0 | 0 | 600 |
| epsilon_greedy | 0.5 | base | 1200 | 1600 | 600 | 3.50202e-06 | 7.0014e-07 | 0 | 0 | 1000 |
| epsilon_greedy | 1 | base | 800 | 1200 | 600 | 0.00764663 | 0 | 0 | 0 | NA |
| epsilon_greedy | 1 | base | 1200 | 1600 | 600 | 8.5868e-10 | 0 | 0 | 0 | NA |

Continuous grid comparisons:

| method | parameter_value | coarse_grid_size | fine_grid_size | maximum_value_difference | maximum_advantage_difference | raw_action_changes | action_changes | strong_opposite_action_changes | fine_bellman_residual | cells_per_transition_fine |
|---|---|---|---|---|---|---|---|---|---|---|
| observation_forgetting_ts | 0.8 | 201 | 401 | 0.000784084 | 0.000492183 | 1 | 1 | 1 | 9.39977e-11 | 80 |
| observation_forgetting_ts | 0.9 | 201 | 401 | 0.00263181 | 0.00236933 | 1 | 1 | 1 | 9.69109e-11 | 40 |
| observation_forgetting_ts | 0.95 | 201 | 401 | 0.00828066 | 0.00650589 | 1 | 1 | 1 | 9.7252e-11 | 20 |
| observation_forgetting_ts | 0.95 | 401 | 801 | 0.00266827 | 0.00203125 | 4 | 4 | 4 | 9.49711e-11 | 40 |
| observation_forgetting_ts | 0.98 | 201 | 401 | 0.0514347 | 0.024173 | 4 | 4 | 4 | 9.75007e-11 | 8 |
| observation_forgetting_ts | 0.99 | 201 | 401 | 0.125564 | 0.0350713 | 11 | 11 | 11 | 9.54614e-11 | 4 |
| observation_forgetting_ts | 0.99 | 401 | 801 | 0.0433406 | 0.0170527 | 7 | 7 | 7 | 9.59517e-11 | 8 |
| calendar_forgetting_ts | 0.8 | 201 | 401 | 0.000227384 | 5.12853e-05 | 1 | 1 | 1 | 9.75611e-11 | 80 |
| calendar_forgetting_ts | 0.9 | 201 | 401 | 0.00197319 | 0.00034142 | 4 | 4 | 4 | 9.6783e-11 | 40 |
| calendar_forgetting_ts | 0.95 | 201 | 401 | 0.00991408 | 0.00235358 | 5 | 5 | 5 | 9.73515e-11 | 20 |
| calendar_forgetting_ts | 0.95 | 401 | 801 | 0.00302201 | 0.000773023 | 7 | 7 | 7 | 9.73586e-11 | 40 |
| calendar_forgetting_ts | 0.98 | 201 | 401 | 0.0926532 | 0.0349714 | 20 | 20 | 20 | 9.7522e-11 | 8 |
| calendar_forgetting_ts | 0.99 | 201 | 401 | 0.315723 | 0.0702007 | 48 | 48 | 48 | 9.77067e-11 | 4 |
| calendar_forgetting_ts | 0.99 | 401 | 801 | 0.126363 | 0.0435382 | 51 | 51 | 51 | 9.76996e-11 | 8 |

A configuration is not marked converged merely because its node residual is
small. Discrete convergence also requires zero robust action changes on the
common reported interior. Continuous convergence likewise requires exactly
zero tolerance-filtered (`action_changes`) policy changes on common nested-grid
nodes. `strong_opposite_action_changes` is an additional diagnostic, not a
substitute for that deliberately strict flag; raw boundary-node changes also
remain visible. For a discontinuous action boundary, value convergence and the
share of policy disagreements must also be inspected rather than treating a
handful of node changes as proof of nonconvergence.

### 5.1 Observation-forgetting refinement

The `rho=.95` observation-time policy was additionally solved on nested grids
through resolution 1601 for every dynamic `p0`, and through 3201 for the most
sensitive case `p0=.5`. The RMS value error has observed order
`1.85--2.02`. The maximum robust disagreement share is
`0.0050%` on common grid nodes and `0.0396%` on the
fixed simulated-state panels; every fine-grid Bellman residual is at most
`9.642e-11`. For `p0=.5`, the RMS difference falls from `8.317e-04` to `5.240e-05`, while disagreement on the common simulated-state panel falls from `0.0396%` to `0.0024%`.

| p0 | coarse_resolution | fine_resolution | root_mean_square_value_difference | robust_action_changes | robust_action_change_share | visited_robust_action_disagreement_share | fine_bellman_residual | strict_policy_converged | stable_for_panels |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 401 | 801 | 0.000810579 | 2 | 2.48136e-05 | 4e-06 | 9.63638e-11 | no | yes |
| 0.1 | 801 | 1601 | 0.00020548 | 4 | 1.24533e-05 | 0 | 9.6378e-11 | no | yes |
| 0.3 | 401 | 801 | 0.00132299 | 0 | 0 | 0.000212 | 9.64206e-11 | yes | yes |
| 0.3 | 801 | 1601 | 0.00032666 | 2 | 6.22663e-06 | 4.4e-05 | 9.57527e-11 | no | yes |
| 0.5 | 401 | 801 | 0.000831697 | 4 | 4.96272e-05 | 0.000396 | 9.49711e-11 | no | yes |
| 0.5 | 801 | 1601 | 0.000208797 | 3 | 9.33995e-06 | 9.6e-05 | 9.60085e-11 | no | yes |
| 0.5 | 1601 | 3201 | 5.23975e-05 | 8 | 6.2383e-06 | 2.4e-05 | 9.54188e-11 | no | yes |
| 0.7 | 401 | 801 | 0.000685318 | 4 | 4.96272e-05 | 0.000108 | 9.38591e-11 | no | yes |
| 0.7 | 801 | 1601 | 0.000172127 | 1 | 3.11332e-06 | 8e-06 | 9.56284e-11 | no | yes |
| 0.9 | 401 | 801 | 5.51512e-05 | 4 | 4.96272e-05 | 5.2e-05 | 3.90976e-12 | no | yes |
| 0.9 | 801 | 1601 | 1.52581e-05 | 3 | 9.33995e-06 | 0 | 3.87468e-12 | no | yes |

Exact zero action changes remains visible as the conservative
`strict_policy_converged` diagnostic. It is not an appropriate sole criterion
for a discontinuous argmax boundary, because refinement quadruples the number
of common nodes and can move a convergent boundary across a handful of them.
The decreasing value error and disagreement shares establish numerical
convergence in measure at the scale of the plotted Monte Carlo panels.


## 6. Demand sensitivity

The regression is OLS of `log(g_n)` on `log(n)` over the configured inclusive
large-n range. A missing slope means `g_n` is zero (not an artificial slope of
zero).

| method | parameter_name | parameter_value | estimated_sensitivity_slope | max_demand_sensitivity | fit_n_min | fit_n_max | fit_observations |
|---|---|---|---|---|---|---|---|
| standard_ts | NA | NA | -0.49853 | 0.5 | 500 | 2000 | 1501 |
| scaled_updates | eta | 0.25 | -0.497124 | 0.159104 | 500 | 2000 | 1501 |
| scaled_updates | eta | 0.5 | -0.498162 | 0.292893 | 500 | 2000 | 1501 |
| scaled_updates | eta | 1 | -0.49853 | 0.5 | 500 | 2000 | 1501 |
| scaled_updates | eta | 2 | -0.498408 | 0.75 | 500 | 2000 | 1501 |
| scaled_updates | eta | 4 | -0.497737 | 0.9375 | 500 | 2000 | 1501 |
| mean_preserving_temperature | temperature | 0.25 | -0.497007 | 0.773438 | 500 | 2000 | 1501 |
| mean_preserving_temperature | temperature | 0.5 | -0.49792 | 0.625 | 500 | 2000 | 1501 |
| mean_preserving_temperature | temperature | 1 | -0.49853 | 0.5 | 500 | 2000 | 1501 |
| mean_preserving_temperature | temperature | 2 | -0.49914 | 0.414214 | 500 | 2000 | 1501 |
| mean_preserving_temperature | temperature | 4 | -0.500058 | 0.366226 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0 | -0.000245777 | 1 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0.02 | -0.000245777 | 0.98 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0.05 | -0.000245777 | 0.95 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0.1 | -0.000245777 | 0.9 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0.2 | -0.000245777 | 0.8 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 0.5 | -0.000245777 | 0.5 | 500 | 2000 | 1501 |
| epsilon_greedy | epsilon | 1 | NA | 0 | 500 | 2000 | 0 |

Forgetting sensitivity is evaluated over the continuous triangular nodes. The
maximizer's normalized coordinates, posterior mean, effective sample size, and
sampling-Beta concentration are reported explicitly:

| method | parameter_name | parameter_value | grid_resolution | grid_node_count | max_demand_sensitivity | argmax_x | argmax_y | argmax_posterior_mean | argmax_effective_sample_size | argmax_effective_concentration |
|---|---|---|---|---|---|---|---|---|---|---|
| observation_forgetting_ts | rho | 0.8 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| observation_forgetting_ts | rho | 0.9 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| observation_forgetting_ts | rho | 0.95 | 801 | 321201 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| observation_forgetting_ts | rho | 0.98 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| observation_forgetting_ts | rho | 0.99 | 801 | 321201 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| calendar_forgetting_ts | rho | 0.8 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| calendar_forgetting_ts | rho | 0.9 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| calendar_forgetting_ts | rho | 0.95 | 801 | 321201 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| calendar_forgetting_ts | rho | 0.98 | 401 | 80601 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |
| calendar_forgetting_ts | rho | 0.99 | 801 | 321201 | 0.5 | 0 | 0 | 0.5 | 0 | 2 |

Standard TS, scaled updates, and fixed mean-preserving temperatures should be
read against the `-1/2` reference. Epsilon-greedy has a discrete jump near the
greedy threshold; its maximal sensitivity need not vanish. At `p0=.5`, the
epsilon-greedy sequence alternates by diagonal parity between the full jump
and a half jump created by the exact tie state, which is why both horizontal
bands are shown rather than smoothed away.

## 7. Optimal-policy results

| method | parameter_name | parameter_value | solver_type | grid_node_count | grid_resolution | comparison_scope | bellman_residual | action_changes | strong_opposite_action_changes | last_active_diagonal | last_active_censored | simulation_policy_last_active_diagonal | simulation_policy_last_active_censored | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_ts | NA | NA | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 435 | no | 435 | no | yes |
| scaled_updates | eta | 0.25 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 96 | no | 96 | no | yes |
| scaled_updates | eta | 0.5 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 210 | no | 210 | no | yes |
| scaled_updates | eta | 1 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 435 | no | 435 | no | yes |
| scaled_updates | eta | 2 | discrete_triangular_truncation | 2423301 | NA | extended_boundary | 7.10543e-15 | 0 | NA | 885 | no | 885 | no | yes |
| scaled_updates | eta | 4 | discrete_triangular_truncation | 2423301 | NA | extended_boundary | 7.10543e-15 | 0 | NA | 1784 | no | 1784 | no | yes |
| mean_preserving_temperature | temperature | 0.25 | discrete_triangular_truncation | 2423301 | NA | extended_boundary | 7.10543e-15 | 0 | NA | 1784 | no | 1784 | no | yes |
| mean_preserving_temperature | temperature | 0.5 | discrete_triangular_truncation | 2423301 | NA | extended_boundary | 7.10543e-15 | 0 | NA | 884 | no | 884 | no | yes |
| mean_preserving_temperature | temperature | 1 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 435 | no | 435 | no | yes |
| mean_preserving_temperature | temperature | 2 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 212 | no | 212 | no | yes |
| mean_preserving_temperature | temperature | 4 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 102 | no | 102 | no | yes |
| epsilon_greedy | epsilon | 0 | discrete_triangular_truncation | 180901 | NA | base | 3.55271e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 0.02 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 0.05 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 0.1 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 0.2 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 0.5 | discrete_triangular_truncation | 180901 | NA | base | 7.10543e-15 | 0 | NA | 600 | yes | NA | NA | yes |
| epsilon_greedy | epsilon | 1 | discrete_triangular_truncation | 180901 | NA | base | 3.55271e-15 | 0 | NA | NA | no | NA | NA | yes |
| observation_forgetting_ts | rho | 0.8 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.39977e-11 | 1 | 1 | NA | NA | NA | NA | no |
| observation_forgetting_ts | rho | 0.9 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.69109e-11 | 1 | 1 | NA | NA | NA | NA | no |
| observation_forgetting_ts | rho | 0.95 | continuous_triangular_interpolation | 321201 | 801 | continuous_nested_grid | 9.49711e-11 | 4 | 4 | NA | NA | NA | NA | no |
| observation_forgetting_ts | rho | 0.98 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.75007e-11 | 4 | 4 | NA | NA | NA | NA | no |
| observation_forgetting_ts | rho | 0.99 | continuous_triangular_interpolation | 321201 | 801 | continuous_nested_grid | 9.59517e-11 | 7 | 7 | NA | NA | NA | NA | no |
| calendar_forgetting_ts | rho | 0.8 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.75611e-11 | 1 | 1 | NA | NA | NA | NA | no |
| calendar_forgetting_ts | rho | 0.9 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.6783e-11 | 4 | 4 | NA | NA | NA | NA | no |
| calendar_forgetting_ts | rho | 0.95 | continuous_triangular_interpolation | 321201 | 801 | continuous_nested_grid | 9.73586e-11 | 7 | 7 | NA | NA | NA | NA | no |
| calendar_forgetting_ts | rho | 0.98 | continuous_triangular_interpolation | 80601 | 401 | continuous_nested_grid | 9.7522e-11 | 20 | 20 | NA | NA | NA | NA | no |
| calendar_forgetting_ts | rho | 0.99 | continuous_triangular_interpolation | 321201 | 801 | continuous_nested_grid | 9.76996e-11 | 51 | 51 | NA | NA | NA | NA | no |

Discrete endpoint data and each diagonal's maximum continuation gap are saved
in `discrete_diagonals.csv`. The policy-point CSV retains every raw product-2
point and flags `robust_action2` and `numerical_tie` separately. Figures use
the robust, tolerance-filtered classification, whose computed advantage
exceeds the configured action tolerance.
Forgetting's last active diagonal is `NaN` by construction because its state is
not indexed by cumulative observation count.

When the base discrete common interior censors a boundary, the full profile
solves outer diagonals `[3000, 3600]`
and compares them through `n<=2200`.
Those extended comparisons and policy/diagonal points are exported, with
`comparison_scope=extended_boundary`. The `simulation_policy_*` and
`simulation_outer_diagonal` fields separately document any still-larger policy
solve needed to cover the simulated observation counts; these are not the same
as the reported-grid convergence comparison.

## 8. Simulations and extinction versus persistence

Every method starts from the same `Beta(1,1)` prior and reuses seed
`20260720`. The profile uses `1000` paths of
`5000` calendar periods. Simulations stream
path-level accumulators and retain only a small fixed set of exact
representative trajectories. Both calendar-time and observation-time
aggregates are saved for nonforgetting models; calendar time is always saved
for forgetting.

| method | parameter_name | parameter_value | mean_discounted_profit | mean_calendar_profit | mean_demand | product2_share_calendar | product2_share_conditional_on_A | late_product2_share_conditional_on_A | maximum_observation_count |
|---|---|---|---|---|---|---|---|---|---|
| standard_ts | NA | NA | 27.0271 | 0.107297 | 0.132214 | 0.03985 | 0.230288 | 0 | 935 |
| scaled_updates | eta | 0.25 | 22.3396 | 0.0729087 | 0.0820581 | 0.0110806 | 0.0983424 | 0 | 653 |
| scaled_updates | eta | 0.5 | 25.0055 | 0.0790646 | 0.0929967 | 0.0202482 | 0.165621 | 0 | 671 |
| scaled_updates | eta | 1 | 27.0271 | 0.107297 | 0.132214 | 0.03985 | 0.230288 | 0 | 935 |
| scaled_updates | eta | 2 | 28.2372 | 0.171025 | 0.218025 | 0.0794438 | 0.275819 | 0 | 1276 |
| scaled_updates | eta | 4 | 27.94 | 0.301402 | 0.392781 | 0.165069 | 0.304821 | 0.0113377 | 2155 |
| mean_preserving_temperature | temperature | 0.25 | 29.2719 | 0.30204 | 0.393631 | 0.159699 | 0.304953 | 0 | 2153 |
| mean_preserving_temperature | temperature | 0.5 | 28.5339 | 0.171018 | 0.218023 | 0.0788966 | 0.275882 | 0 | 1289 |
| mean_preserving_temperature | temperature | 1 | 27.0271 | 0.107297 | 0.132214 | 0.03985 | 0.230288 | 0 | 935 |
| mean_preserving_temperature | temperature | 2 | 25.3158 | 0.0793397 | 0.0934149 | 0.0204022 | 0.166808 | 0 | 669 |
| mean_preserving_temperature | temperature | 4 | 23.4445 | 0.0738723 | 0.0832563 | 0.0111238 | 0.100642 | 0 | 659 |
| epsilon_greedy | epsilon | 0 | 25.6801 | 0.39585 | 0.527861 | 0.648176 | 0.333506 | 0.331209 | 5000 |
| epsilon_greedy | epsilon | 0.02 | 27.1573 | 0.649196 | 0.866447 | 0.409371 | 0.334525 | 0.335122 | 4961 |
| epsilon_greedy | epsilon | 0.05 | 28.1329 | 0.688462 | 0.918772 | 0.367872 | 0.334376 | 0.334946 | 4905 |
| epsilon_greedy | epsilon | 0.1 | 28.7689 | 0.690638 | 0.921837 | 0.353311 | 0.334541 | 0.334488 | 4774 |
| epsilon_greedy | epsilon | 0.2 | 28.1455 | 0.664268 | 0.886653 | 0.344635 | 0.334703 | 0.335214 | 4543 |
| epsilon_greedy | epsilon | 0.5 | 24.8044 | 0.551284 | 0.736224 | 0.347525 | 0.335218 | 0.335139 | 3782 |
| epsilon_greedy | epsilon | 1 | 23.7087 | 0.474829 | 0.5 | 0 | 0 | 0 | 2606 |
| observation_forgetting_ts | rho | 0.8 | 18.277 | 0.370894 | 0.598098 | 0.64767 | 0.549726 | 0.549913 | 3134 |
| observation_forgetting_ts | rho | 0.9 | 20.5192 | 0.428925 | 0.705299 | 0.630057 | 0.569771 | 0.569668 | 3661 |
| observation_forgetting_ts | rho | 0.95 | 22.6103 | 0.484663 | 0.778083 | 0.583499 | 0.545236 | 0.544166 | 3998 |
| observation_forgetting_ts | rho | 0.98 | 24.6726 | 0.539443 | 0.816628 | 0.505975 | 0.482504 | 0.481897 | 4191 |
| observation_forgetting_ts | rho | 0.99 | 25.6671 | 0.556555 | 0.806774 | 0.451864 | 0.433542 | 0.432133 | 4141 |
| calendar_forgetting_ts | rho | 0.8 | 18.6863 | 0.369781 | 0.476097 | 0.352389 | 0.288536 | 0.288612 | 2485 |
| calendar_forgetting_ts | rho | 0.9 | 20.102 | 0.414684 | 0.686421 | 0.644226 | 0.576484 | 0.575546 | 3552 |
| calendar_forgetting_ts | rho | 0.95 | 22.3371 | 0.474828 | 0.782728 | 0.612474 | 0.572329 | 0.571543 | 4012 |
| calendar_forgetting_ts | rho | 0.98 | 24.5641 | 0.536655 | 0.829024 | 0.528091 | 0.504327 | 0.503663 | 4234 |
| calendar_forgetting_ts | rho | 0.99 | 25.6236 | 0.55987 | 0.824218 | 0.469574 | 0.451191 | 0.450216 | 4210 |

`product2_share_calendar` means that the optimal policy prescribed product 2
at the realized state, divided by all simulated path-periods.
`product2_share_conditional_on_A` pools actual product-2 uses and divides by
all realized A interactions. `late_product2_share_conditional_on_A` is the
same pooled ratio restricted to one-indexed calendar periods
`t=4000,...,5000`.
For each configured threshold, `probability_product2_used_at_or_after_*` is the
fraction of paths with at least one actual use at or after that inclusive
calendar period, but still within the finite horizon. `mean_observed_last_*`
only averages the latest *observed* use among paths with a use; it cannot show
that a path has a finite last use. No figure reports the uninformative share of
all grid states prescribing product 2.

For `p0=.5`, long epsilon-greedy simulations use the exact stationary reduction
to balance `k=S-F`, independently checked against the triangular policy.
Nonforgetting simulations only extend a policy with product 1 after a stable,
long inactive tail; otherwise they solve a larger triangular grid rather than
silently extrapolate.

## 9. Numerical limitations

- A stationary triangular solve is still an outer-boundary approximation; the
  action-stability and distance-to-boundary columns quantify that limitation.
- Continuous results discretize a genuinely continuous state. Raw switches at
  a moving action boundary can remain. The strict `converged` flag requires
  exactly zero tolerance-filtered action changes; refinement studies also
  report value convergence and disagreement shares on common nodes and visited
  states.
- Late simulated use is evidence only inside the finite simulated horizon. An
  observed latest use is right-censored by the horizon and is not evidence of
  a finite last-use time.
- Common random numbers reduce comparison noise but do not turn simulation
  evidence into a theorem.

## 10. Recommended paper figures

Figures 1--11 cover the core learning-rule diagnostics:
raw and log-log demand sensitivity; policy regions for scaled updates,
temperature, and epsilon-greedy; last active diagonal against `eta` and `T`;
forgetting policy maps; late realized product-2 use against `rho` and
`epsilon`; and representative paths. Figure identifier 12 is intentionally
unused so later output identifiers remain stable. Figures 13--22 add the
complete and focused method-by-p0 comparisons documented in Section 13.
Figure 23 reports the dedicated observation-forgetting nested-grid
refinement. Figure 24 re-expresses the observation-forgetting product-2 policy
in the same coordinates as the discrete policy figures: effective sample size
against posterior mean, with one panel per `rho`. Its points are continuous
interpolation-grid nodes, not a claim that every point is an exactly reachable
finite history. Figure 25 gives the separate parameter-specific
observation-forgetting non-extinction certificate: a strict boundary action
interval and a finite outcome word whose image lies inside it.
Recommended candidates for the paper are the log-log sensitivity figure, one
compact policy-region comparison, the two late-use parameter plots, and a
selected subset of the dynamic panels. The paper itself was not modified.

## 11. Main economic interpretation

Rules whose marginal demand response shrinks with accumulated evidence tend to
localize investment and eventually remove product 2 from the computed count
policy. A nonvanishing demand jump (including the greedy cutoff) or bounded
effective memory can keep the value of influencing future demand away from
zero and can therefore generate recurrent investment. For observation-time forgetting at `rho=.95`, the dedicated nested-grid study supports numerical stability of the plotted persistence result. Other forgetting specifications with `converged=no` remain preliminary unless they receive the same refinement audit.
The exact outcome remains parameter dependent: in
particular, `epsilon=1` makes demand constant at one half and removes the
reputational return to quality.

## 12. Bottom line for this numerical profile

- **Fixed scaled updates:** every tested configuration has a finite, uncensored last active observation diagonal (maximum 1784), so the numerical policy evidence supports eventual extinction. A late-window calendar share as high as 1.134% remains because some paths reach that observation boundary slowly; this is delayed extinction, not a reappearance beyond the boundary.
- **Fixed mean-preserving temperature:** every tested configuration has a finite, uncensored last active observation diagonal (maximum 1784), so the numerical policy evidence supports eventual extinction. No late-window product-2 use was observed.
- **Forgetting:** recurrent late use is present for observation_forgetting_ts, rho=0.8 (54.991%), observation_forgetting_ts, rho=0.9 (56.967%), observation_forgetting_ts, rho=0.95 (54.417%), observation_forgetting_ts, rho=0.98 (48.190%), observation_forgetting_ts, rho=0.99 (43.213%), calendar_forgetting_ts, rho=0.8 (28.861%), calendar_forgetting_ts, rho=0.9 (57.555%), calendar_forgetting_ts, rho=0.95 (57.154%), calendar_forgetting_ts, rho=0.98 (50.366%), calendar_forgetting_ts, rho=0.99 (45.022%); the evidence does not support extinction for those configurations. The strict exact-node flag remains `converged=no` for some continuous policies. The dedicated observation-time `rho=.95` refinement nevertheless shows second-order value convergence and policy convergence in measure; configurations not covered by that refinement remain grid-sensitive.
- **Uniform epsilon-greedy:** recurrent late use is present for epsilon_greedy, epsilon=0 (33.121%), epsilon_greedy, epsilon=0.02 (33.512%), epsilon_greedy, epsilon=0.05 (33.495%), epsilon_greedy, epsilon=0.1 (33.449%), epsilon_greedy, epsilon=0.2 (33.521%), epsilon_greedy, epsilon=0.5 (33.514%); the evidence does not support extinction for those configurations. The endpoint `epsilon=1` is the explicit exception: demand is constant at one half, so product 2 has no reputational return and is never used.

These are statements about numerical evidence on the reported grids,
simulation horizon, and parameter set. They are **not theorem claims**.


### 12.1 Parameter-specific observation-forgetting result

A separate hybrid analytic/computer-assisted argument addresses the asymptotic
question for observation-time forgetting at `rho=0.95` and the
base primitives. The exact one-dimensional boundary reduction, together with a
padded `50,000`-cell Bellman enclosure, gives product 2 a
strict advantage on `x in [0.33, 0.42]`; the certified margin
is `0.828069`. The outcome word
`(0^19 1^6)^2` maps every sufficiently mature belief into that interval.
Because this word has uniformly positive conditional probability under either
adaptive product choice, a martingale recurrence argument implies infinitely
many product-2 interactions almost surely, and even a strictly positive but
very conservative asymptotic frequency lower bound.

This is stronger than the finite-horizon simulation statement, but it is still
parameter-specific. The recurrence argument is analytic; the Bellman-sign
lemma uses float64 arithmetic and explicitly padded SciPy beta tails rather
than a directed-rounding interval special-function library. Full details and
the numerical-scope qualification are in
`results/full/OBSERVATION_FORGETTING_NONEXTINCTION.md`; Figure 25 visualizes the
two ingredients.


## 13. Dynamic simulation matrix across p0

### 13.1 Design and audit of the legacy figure

The dynamic experiment is a `10 x 5` matrix:
10 representative demand specifications, each evaluated at
`p0 in {0.1, 0.3, 0.5, 0.7, 0.9}`, for `50` method-p0 series. Figures 13--17
preserve the complete original eight-specification comparison for each p0.
Figures 18--22 provide a focused view with standard TS, scaled updates at
`eta in {0.5,2}`, epsilon-greedy at `epsilon=0.1`, and observation-time
forgetting at `rho=0.95`. The focused set omits the
mean-preserving-temperature and calendar-forgetting curves, while the complete
set remains available. Both sets use the same fixed color for a specification
at every p0. This holds the outside option fixed within a plot and avoids
placing all `50` series in one unreadable figure.

The product-2 panel includes a dashed quality-maintenance benchmark
`a_maint=clip((p0-p1)/(p2-p1),0,1)`, the smallest stationary product-2 share
whose expected quality `p1+a(p2-p1)` reaches `p0`. This is a sustainability
benchmark, not the dynamically optimal policy. No line is drawn when `p0>p2`,
because even product 2 cannot reach the outside-option quality. The summary
CSV records the benchmark's feasibility, its value, and each method's overall
and late conditional-use gap relative to it.

The script behind the legacy title *Exact discounted simulated paths* did not
plot one exact or representative path. Each plotted point was a
contemporaneous cross-sectional Monte Carlo statistic at one calendar period;
there was no rolling or cumulative averaging. The legacy defaults were 400
independent paths, a 250-period simulation, and a policy from a finite-horizon
DP with horizon 700. The new experiment preserves the useful four outcomes but
uses the modular study's stationary discounted policies, subject to the
reported triangular truncation or continuous interpolation checks.

The plotted metrics are defined exactly as follows:

- **Seller A demand / market share:** the realized fraction of paths selecting
  A in that calendar period. The CSV also saves the cross-path mean of the
  model's instantaneous demand probability; it is distinct from the plotted
  realized share.
- **Product-2 use conditional on A:** actual product-2 uses divided by actual A
  selections in the period. Paths selecting B are excluded, never silently
  coded as product 1. The CSV separately saves product-2 use over all calendar
  path-periods (`chosen_A & action2`) and the diagnostic share of simulated
  paths whose current policy action is product 2. The plotted curve is the
  conditional actual use rate, not that prescription diagnostic.
- **Calendar-period profit:** mean realized Seller A profit across all paths,
  including zero whenever B is selected. It is not conditional on interacting
  with A and is not formatted as a percentage.
- **Posterior mean:** mean belief used by the demand rule at the **start** of
  the period, before that period's choice and outcome. For standard TS this is
  `(S+1)/(S+F+2)`; scaled updates and forgetting use their corresponding
  method-specific Beta parameters. Only valid path-level values enter the
  cross-sectional mean.

This profile uses `1,000` paths, `250` periods, seed `20260630`, and a
`95%` confidence level. The exact same choice and outcome uniforms
are reused across all method-p0 configurations (common random numbers).
`rolling_window=0` means no rolling or cumulative average. The
time-series CSV contains a confidence interval for every contemporaneous
estimate: Wilson score intervals for binary proportions (realized A share and
conditional/calendar/prescribed product-2 shares), and Student-t intervals for
cross-path means (demand probability, calendar profit, posterior mean,
effective sample size, and Beta concentration). The conditional product-2
denominator contains only realized A selections. Summary columns prefixed
`late_` use the inclusive calendar window `t=201,...,250`.
Confidence intervals are saved in CSV rather than overplotting 40 bands.


### 13.2 Method-by-p0 Monte Carlo summaries

#### Standard Thompson sampling

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.99668 | 1 | 0.00344343 | 0 | 0.003432 | 0 | 0.944787 | 0.95 | 0.364859 | 0.354076 |
| 0.3 | 0.96028 | 0.96944 | 0.0455117 | 0.00371348 | 0.043704 | 0.0036 | 0.886044 | 0.918808 | 0.41533 | 0.37473 |
| 0.5 | 0.868912 | 0.82952 | 0.398742 | 0.336918 | 0.346472 | 0.27948 | 0.617583 | 0.620356 | 0.572906 | 0.533603 |
| 0.7 | 0.683564 | 0.67308 | 0.796988 | 0.661318 | 0.544792 | 0.44512 | 0.322511 | 0.372354 | 0.708341 | 0.709182 |
| 0.9 | 0.057204 | 0.00836 | 0.758548 | 0.0239234 | 0.043392 | 0.0002 | 0.0283086 | 0.007822 | 0.580823 | 0.558683 |

#### Scaled updates, eta=0.5

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.994148 | 1 | 0.0017422 | 0 | 0.001732 | 0 | 0.943401 | 0.95 | 0.364878 | 0.354441 |
| 0.3 | 0.928972 | 0.93404 | 0.0525269 | 0.00274078 | 0.048796 | 0.00256 | 0.853246 | 0.885802 | 0.425603 | 0.37898 |
| 0.5 | 0.791356 | 0.69748 | 0.376872 | 0.255807 | 0.29824 | 0.17842 | 0.572844 | 0.555554 | 0.575473 | 0.526571 |
| 0.7 | 0.48164 | 0.20548 | 0.638618 | 0.110181 | 0.307584 | 0.02264 | 0.273008 | 0.181622 | 0.686933 | 0.64432 |
| 0.9 | 0.049212 | 0.01114 | 0.53556 | 0 | 0.026356 | 0 | 0.0309378 | 0.010583 | 0.539628 | 0.512004 |

#### Scaled updates, eta=2

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.998292 | 1 | 0.00524095 | 0 | 0.005232 | 0 | 0.945238 | 0.95 | 0.367421 | 0.354256 |
| 0.3 | 0.976292 | 0.9862 | 0.039144 | 0.00429933 | 0.038216 | 0.00424 | 0.904548 | 0.934346 | 0.405446 | 0.370613 |
| 0.5 | 0.899776 | 0.88664 | 0.394983 | 0.355432 | 0.355396 | 0.31514 | 0.64155 | 0.653224 | 0.559302 | 0.528936 |
| 0.7 | 0.702388 | 0.74786 | 0.819519 | 0.782927 | 0.57562 | 0.58552 | 0.321897 | 0.359155 | 0.683674 | 0.693121 |
| 0.9 | 0.052884 | 0.0057 | 0.857651 | 0.105263 | 0.045356 | 0.0006 | 0.0230262 | 0.005055 | 0.578006 | 0.559932 |

#### Mean-preserving temperature, T=0.5

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.99814 | 1 | 0.00155489 | 0 | 0.001552 | 0 | 0.947302 | 0.95 | 0.360618 | 0.35303 |
| 0.3 | 0.978176 | 0.98654 | 0.0375801 | 0.00397348 | 0.03676 | 0.00392 | 0.907211 | 0.934861 | 0.403735 | 0.370368 |
| 0.5 | 0.90688 | 0.88914 | 0.394619 | 0.356794 | 0.357872 | 0.31724 | 0.646813 | 0.654339 | 0.561049 | 0.529915 |
| 0.7 | 0.675516 | 0.73066 | 0.820966 | 0.785098 | 0.554576 | 0.57364 | 0.308995 | 0.349943 | 0.680782 | 0.691528 |
| 0.9 | 0.040264 | 0.00844 | 0.857143 | 0.353081 | 0.034512 | 0.00298 | 0.0175436 | 0.00623 | 0.581479 | 0.576028 |

#### Mean-preserving temperature, T=2

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.993432 | 1 | 0.00619267 | 0 | 0.006152 | 0 | 0.940069 | 0.95 | 0.370425 | 0.355386 |
| 0.3 | 0.928528 | 0.93482 | 0.0557226 | 0.00297383 | 0.05174 | 0.00278 | 0.851058 | 0.886411 | 0.429924 | 0.37949 |
| 0.5 | 0.794608 | 0.69996 | 0.376925 | 0.256243 | 0.299508 | 0.17936 | 0.575173 | 0.557346 | 0.579221 | 0.527023 |
| 0.7 | 0.494492 | 0.18906 | 0.629826 | 0.0770126 | 0.311444 | 0.01456 | 0.282901 | 0.170871 | 0.693363 | 0.641299 |
| 0.9 | 0.08108 | 0.01368 | 0.649087 | 0 | 0.052628 | 0 | 0.0454492 | 0.012996 | 0.60601 | 0.574681 |

#### Epsilon-greedy, epsilon=0.05

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.975792 | 0.97614 | 0.000467313 | 0 | 0.000456 | 0 | 0.926729 | 0.927333 | 0.358827 | 0.352203 |
| 0.3 | 0.963128 | 0.97484 | 0.0217998 | 0.00190801 | 0.020996 | 0.00186 | 0.902374 | 0.924982 | 0.385103 | 0.36273 |
| 0.5 | 0.859992 | 0.90802 | 0.353161 | 0.335565 | 0.303716 | 0.3047 | 0.634763 | 0.679799 | 0.518942 | 0.508602 |
| 0.7 | 0.2791 | 0.40872 | 0.799957 | 0.771139 | 0.223268 | 0.31518 | 0.131184 | 0.199176 | 0.594282 | 0.611635 |
| 0.9 | 0.024804 | 0.0248 | 0 | 0 | 0 | 0 | 0.0235638 | 0.02356 | 0.42729 | 0.396482 |

#### Epsilon-greedy, epsilon=0.1

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.950944 | 0.95116 | 0.000475317 | 0 | 0.000452 | 0 | 0.903126 | 0.903602 | 0.359201 | 0.352508 |
| 0.3 | 0.944724 | 0.95042 | 0.0224701 | 0.00178868 | 0.021228 | 0.0017 | 0.884751 | 0.901879 | 0.386641 | 0.36325 |
| 0.5 | 0.881144 | 0.91882 | 0.352891 | 0.335343 | 0.310948 | 0.30812 | 0.650518 | 0.688007 | 0.52611 | 0.510606 |
| 0.7 | 0.405392 | 0.50486 | 0.779858 | 0.74282 | 0.316148 | 0.37502 | 0.195434 | 0.254605 | 0.613082 | 0.616017 |
| 0.9 | 0.049924 | 0.04964 | 8.01218e-05 | 0.000402901 | 4e-06 | 2e-05 | 0.0474254 | 0.047146 | 0.407394 | 0.378979 |

#### Epsilon-greedy, epsilon=0.2

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.900304 | 0.90298 | 0.000319892 | 0 | 0.000288 | 0 | 0.855116 | 0.857831 | 0.358956 | 0.352625 |
| 0.3 | 0.896676 | 0.90296 | 0.0226481 | 0.00203774 | 0.020308 | 0.00184 | 0.839657 | 0.856708 | 0.385997 | 0.36365 |
| 0.5 | 0.865716 | 0.88978 | 0.353876 | 0.336308 | 0.306356 | 0.29924 | 0.638617 | 0.665747 | 0.531413 | 0.511004 |
| 0.7 | 0.529832 | 0.5937 | 0.765684 | 0.731211 | 0.405684 | 0.43412 | 0.25993 | 0.303543 | 0.645461 | 0.637207 |
| 0.9 | 0.10042 | 0.09898 | 0 | 0 | 0 | 0 | 0.095399 | 0.094031 | 0.388746 | 0.369155 |

#### Observation-time forgetting, rho=0.95

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.989648 | 0.99508 | 0.00270399 | 0.000100494 | 0.002676 | 0.0001 | 0.93856 | 0.945266 | 0.368613 | 0.36276 |
| 0.3 | 0.880116 | 0.87874 | 0.182667 | 0.170836 | 0.160768 | 0.15012 | 0.739649 | 0.744731 | 0.442865 | 0.428703 |
| 0.5 | 0.779952 | 0.7828 | 0.557009 | 0.545171 | 0.43444 | 0.42676 | 0.48029 | 0.487604 | 0.5902 | 0.582959 |
| 0.7 | 0.54028 | 0.562 | 0.890398 | 0.882064 | 0.481064 | 0.49572 | 0.224628 | 0.236468 | 0.689524 | 0.704603 |
| 0.9 | 0.050936 | 0.00784 | 0.763546 | 0.0484694 | 0.038892 | 0.00038 | 0.025054 | 0.00722 | 0.556031 | 0.524116 |

#### Calendar-time forgetting, rho=0.95

| p0 | overall_realized_A_share | late_realized_A_share | overall_product2_share_conditional_on_A | late_product2_share_conditional_on_A | overall_product2_share_calendar | late_product2_share_calendar | overall_calendar_profit | late_calendar_profit | overall_posterior_mean | late_posterior_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.98954 | 0.9949 | 0.00262748 | 0.000100513 | 0.0026 | 0.0001 | 0.938503 | 0.945095 | 0.369255 | 0.363195 |
| 0.3 | 0.88152 | 0.88086 | 0.194115 | 0.183071 | 0.171116 | 0.16126 | 0.734774 | 0.740061 | 0.450861 | 0.438348 |
| 0.5 | 0.784076 | 0.78642 | 0.58101 | 0.568195 | 0.455556 | 0.44684 | 0.471539 | 0.478995 | 0.600303 | 0.5949 |
| 0.7 | 0.559664 | 0.57626 | 0.900948 | 0.893104 | 0.504228 | 0.51466 | 0.229144 | 0.238651 | 0.697914 | 0.708334 |
| 0.9 | 0.0541 | 0.04732 | 0.124658 | 0.0257819 | 0.006744 | 0.00122 | 0.0473486 | 0.044222 | 0.458125 | 0.449494 |

For forgetting, `effective_sample_size=a+b-2` is the normalized effective
count used by the implementation and `effective_concentration=a+b` includes
the prior. Both are reported because forgetting bounds effective memory even
though calendar time continues:

| method_spec_label | p0 | overall_effective_sample_size | late_effective_sample_size | overall_effective_concentration | late_effective_concentration |
|---|---|---|---|---|---|
| Observation-time forgetting, rho=0.95 | 0.1 | 18.327 | 19.9997 | 20.327 | 21.9997 |
| Observation-time forgetting, rho=0.95 | 0.3 | 18.1802 | 19.999 | 20.1802 | 21.999 |
| Observation-time forgetting, rho=0.95 | 0.5 | 17.8477 | 19.9967 | 19.8477 | 21.9967 |
| Observation-time forgetting, rho=0.95 | 0.7 | 16.256 | 19.6325 | 18.256 | 21.6325 |
| Observation-time forgetting, rho=0.95 | 0.9 | 6.59311 | 8.19559 | 8.59311 | 10.1956 |
| Calendar-time forgetting, rho=0.95 | 0.1 | 18.1986 | 19.8861 | 20.1986 | 21.8861 |
| Calendar-time forgetting, rho=0.95 | 0.3 | 16.222 | 17.5967 | 18.222 | 19.5967 |
| Calendar-time forgetting, rho=0.95 | 0.5 | 14.4223 | 15.724 | 16.4223 | 17.724 |
| Calendar-time forgetting, rho=0.95 | 0.7 | 10.2676 | 11.4521 | 12.2676 | 13.4521 |
| Calendar-time forgetting, rho=0.95 | 0.9 | 1.0073 | 0.960992 | 3.0073 | 2.96099 |

### 13.3 Economic-regime diagnostics

| method | regime | tested_p0 | late_A_share | late_conditional_P2 | late_calendar_profit |
|---|---|---|---|---|---|
| Standard Thompson sampling | p0 < p1 | 0.1, 0.3 | 96.9%--100.0% | 0.0%--0.4% | 0.9188--0.95 |
| Standard Thompson sampling | p1 < p0 < p2 | 0.5, 0.7 | 67.3%--83.0% | 33.7%--66.1% | 0.3724--0.6204 |
| Standard Thompson sampling | p0 > p2 | 0.9 | 0.8% | 2.4% | 0.007822 |
| Scaled updates, eta=0.5 | p0 < p1 | 0.1, 0.3 | 93.4%--100.0% | 0.0%--0.3% | 0.8858--0.95 |
| Scaled updates, eta=0.5 | p1 < p0 < p2 | 0.5, 0.7 | 20.5%--69.7% | 11.0%--25.6% | 0.1816--0.5556 |
| Scaled updates, eta=0.5 | p0 > p2 | 0.9 | 1.1% | 0.0% | 0.01058 |
| Scaled updates, eta=2 | p0 < p1 | 0.1, 0.3 | 98.6%--100.0% | 0.0%--0.4% | 0.9343--0.95 |
| Scaled updates, eta=2 | p1 < p0 < p2 | 0.5, 0.7 | 74.8%--88.7% | 35.5%--78.3% | 0.3592--0.6532 |
| Scaled updates, eta=2 | p0 > p2 | 0.9 | 0.6% | 10.5% | 0.005055 |
| Mean-preserving temperature, T=0.5 | p0 < p1 | 0.1, 0.3 | 98.7%--100.0% | 0.0%--0.4% | 0.9349--0.95 |
| Mean-preserving temperature, T=0.5 | p1 < p0 < p2 | 0.5, 0.7 | 73.1%--88.9% | 35.7%--78.5% | 0.3499--0.6543 |
| Mean-preserving temperature, T=0.5 | p0 > p2 | 0.9 | 0.8% | 35.3% | 0.00623 |
| Mean-preserving temperature, T=2 | p0 < p1 | 0.1, 0.3 | 93.5%--100.0% | 0.0%--0.3% | 0.8864--0.95 |
| Mean-preserving temperature, T=2 | p1 < p0 < p2 | 0.5, 0.7 | 18.9%--70.0% | 7.7%--25.6% | 0.1709--0.5573 |
| Mean-preserving temperature, T=2 | p0 > p2 | 0.9 | 1.4% | 0.0% | 0.013 |
| Epsilon-greedy, epsilon=0.05 | p0 < p1 | 0.1, 0.3 | 97.5%--97.6% | 0.0%--0.2% | 0.925--0.9273 |
| Epsilon-greedy, epsilon=0.05 | p1 < p0 < p2 | 0.5, 0.7 | 40.9%--90.8% | 33.6%--77.1% | 0.1992--0.6798 |
| Epsilon-greedy, epsilon=0.05 | p0 > p2 | 0.9 | 2.5% | 0.0% | 0.02356 |
| Epsilon-greedy, epsilon=0.1 | p0 < p1 | 0.1, 0.3 | 95.0%--95.1% | 0.0%--0.2% | 0.9019--0.9036 |
| Epsilon-greedy, epsilon=0.1 | p1 < p0 < p2 | 0.5, 0.7 | 50.5%--91.9% | 33.5%--74.3% | 0.2546--0.688 |
| Epsilon-greedy, epsilon=0.1 | p0 > p2 | 0.9 | 5.0% | 0.0% | 0.04715 |
| Epsilon-greedy, epsilon=0.2 | p0 < p1 | 0.1, 0.3 | 90.3%--90.3% | 0.0%--0.2% | 0.8567--0.8578 |
| Epsilon-greedy, epsilon=0.2 | p1 < p0 < p2 | 0.5, 0.7 | 59.4%--89.0% | 33.6%--73.1% | 0.3035--0.6657 |
| Epsilon-greedy, epsilon=0.2 | p0 > p2 | 0.9 | 9.9% | 0.0% | 0.09403 |
| Observation-time forgetting, rho=0.95 | p0 < p1 | 0.1, 0.3 | 87.9%--99.5% | 0.0%--17.1% | 0.7447--0.9453 |
| Observation-time forgetting, rho=0.95 | p1 < p0 < p2 | 0.5, 0.7 | 56.2%--78.3% | 54.5%--88.2% | 0.2365--0.4876 |
| Observation-time forgetting, rho=0.95 | p0 > p2 | 0.9 | 0.8% | 4.8% | 0.00722 |
| Calendar-time forgetting, rho=0.95 | p0 < p1 | 0.1, 0.3 | 88.1%--99.5% | 0.0%--18.3% | 0.7401--0.9451 |
| Calendar-time forgetting, rho=0.95 | p1 < p0 < p2 | 0.5, 0.7 | 57.6%--78.6% | 56.8%--89.3% | 0.2387--0.479 |
| Calendar-time forgetting, rho=0.95 | p0 > p2 | 0.9 | 4.7% | 2.6% | 0.04422 |

For `p0<p1`, small late conditional product-2 rates are consistent with early
investment followed by harvesting with product 1. For `p1<p0<p2`, positive
late use is consistent with continued or recurrent reputation investment
because product 1 pulls reputation below the outside option while product 2
pushes it upward. For `p0>p2`, a small late A share is consistent with demand
collapse even if product 2 is still used on some scarce A interactions. The
table reports the actual ranges for every method, so deviations from these
qualitative patterns remain visible rather than being promoted to a theorem.

Two qualifications matter. Every positive epsilon-greedy value gives A an
exploration floor, so "demand collapse" means approach toward that positive
floor rather than literal zero; the exact balance reduction is available only
at `p0=0.5`, while the other p0 curves use the general triangular DP.
Forgetting has bounded effective memory and a continuous interpolated state,
so recurrent use may reflect a genuinely persistent incentive but its size
must be assessed together with the grid checks below.

### 13.4 Dynamic-policy solver checks

| method_spec_label | method | parameter_name | parameter_value | p0 | solver_type | coarse_outer_diagonal | fine_outer_diagonal | coarse_grid_size | fine_grid_size | action_changes | action_change_share | visited_state_robust_action_disagreement_share | bellman_residual | fine_bellman_residual | balance_policy_disagreements | policy_grid_stable_for_panels | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Standard Thompson sampling | standard_ts | NA | NA | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Standard Thompson sampling | standard_ts | NA | NA | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Standard Thompson sampling | standard_ts | NA | NA | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Standard Thompson sampling | standard_ts | NA | NA | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Standard Thompson sampling | standard_ts | NA | NA | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=0.5 | scaled_updates | eta | 0.5 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=0.5 | scaled_updates | eta | 0.5 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=0.5 | scaled_updates | eta | 0.5 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=0.5 | scaled_updates | eta | 0.5 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=0.5 | scaled_updates | eta | 0.5 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=2 | scaled_updates | eta | 2 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=2 | scaled_updates | eta | 2 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=2 | scaled_updates | eta | 2 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=2 | scaled_updates | eta | 2 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Scaled updates, eta=2 | scaled_updates | eta | 2 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 3.55271e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=0.5 | mean_preserving_temperature | temperature | 0.5 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=0.5 | mean_preserving_temperature | temperature | 0.5 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=0.5 | mean_preserving_temperature | temperature | 0.5 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=0.5 | mean_preserving_temperature | temperature | 0.5 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=0.5 | mean_preserving_temperature | temperature | 0.5 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 3.55271e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=2 | mean_preserving_temperature | temperature | 2 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=2 | mean_preserving_temperature | temperature | 2 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=2 | mean_preserving_temperature | temperature | 2 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=2 | mean_preserving_temperature | temperature | 2 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Mean-preserving temperature, T=2 | mean_preserving_temperature | temperature | 2 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.05 | epsilon_greedy | epsilon | 0.05 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.05 | epsilon_greedy | epsilon | 0.05 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.05 | epsilon_greedy | epsilon | 0.05 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | 0 | NA | yes |
| Epsilon-greedy, epsilon=0.05 | epsilon_greedy | epsilon | 0.05 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.05 | epsilon_greedy | epsilon | 0.05 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 3.55271e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.1 | epsilon_greedy | epsilon | 0.1 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.1 | epsilon_greedy | epsilon | 0.1 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.1 | epsilon_greedy | epsilon | 0.1 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | 0 | NA | yes |
| Epsilon-greedy, epsilon=0.1 | epsilon_greedy | epsilon | 0.1 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.1 | epsilon_greedy | epsilon | 0.1 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 1.77636e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.2 | epsilon_greedy | epsilon | 0.2 | 0.1 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.2 | epsilon_greedy | epsilon | 0.2 | 0.3 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.2 | epsilon_greedy | epsilon | 0.2 | 0.5 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | 0 | NA | yes |
| Epsilon-greedy, epsilon=0.2 | epsilon_greedy | epsilon | 0.2 | 0.7 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 7.10543e-15 | NA | NA | NA | yes |
| Epsilon-greedy, epsilon=0.2 | epsilon_greedy | epsilon | 0.2 | 0.9 | stationary_discrete_triangular | 1200 | 1600 | NA | NA | 0 | NA | NA | 3.55271e-15 | NA | NA | NA | yes |
| Observation-time forgetting, rho=0.95 | observation_forgetting_ts | rho | 0.95 | 0.1 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 2 | 2.48136e-05 | 4e-06 | 9.63638e-11 | 9.63638e-11 | NA | yes | no |
| Observation-time forgetting, rho=0.95 | observation_forgetting_ts | rho | 0.95 | 0.3 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 0 | 0 | 0.000212 | 9.64206e-11 | 9.64206e-11 | NA | yes | yes |
| Observation-time forgetting, rho=0.95 | observation_forgetting_ts | rho | 0.95 | 0.5 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 4 | 4.96272e-05 | 0.000396 | 9.49711e-11 | 9.49711e-11 | NA | yes | no |
| Observation-time forgetting, rho=0.95 | observation_forgetting_ts | rho | 0.95 | 0.7 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 4 | 4.96272e-05 | 0.000108 | 9.38591e-11 | 9.38591e-11 | NA | yes | no |
| Observation-time forgetting, rho=0.95 | observation_forgetting_ts | rho | 0.95 | 0.9 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 4 | 4.96272e-05 | 5.2e-05 | 3.90976e-12 | 3.90976e-12 | NA | yes | no |
| Calendar-time forgetting, rho=0.95 | calendar_forgetting_ts | rho | 0.95 | 0.1 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 3 | 3.72204e-05 | 1.2e-05 | 9.78488e-11 | 9.78488e-11 | NA | yes | no |
| Calendar-time forgetting, rho=0.95 | calendar_forgetting_ts | rho | 0.95 | 0.3 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 4 | 4.96272e-05 | 0.000108 | 9.72094e-11 | 9.72094e-11 | NA | yes | no |
| Calendar-time forgetting, rho=0.95 | calendar_forgetting_ts | rho | 0.95 | 0.5 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 7 | 8.68476e-05 | 0.000168 | 9.73586e-11 | 9.73586e-11 | NA | yes | no |
| Calendar-time forgetting, rho=0.95 | calendar_forgetting_ts | rho | 0.95 | 0.7 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 6 | 7.44408e-05 | 2.8e-05 | 9.75078e-11 | 9.75078e-11 | NA | yes | no |
| Calendar-time forgetting, rho=0.95 | calendar_forgetting_ts | rho | 0.95 | 0.9 | stationary_continuous_triangular_interpolation | NA | NA | 401 | 801 | 4 | 4.96272e-05 | 0.000628 | 9.62386e-11 | 9.62386e-11 | NA | yes | no |

Common random numbers reduce Monte Carlo noise in cross-configuration
comparisons; they do not eliminate sampling uncertainty, truncation error, or
continuous-grid error. The strict zero-change criterion still fails for `9` forgetting rows, but every forgetting row passes the separate panel-stability threshold: the maximum common-node robust action-change share is 0.0087% and the maximum coarse-versus-fine disagreement share on simulated states is 0.0628%. These differences are small at the scale of the plotted Monte Carlo curves, but the interpolated boundaries are not exact. All dynamic conclusions are numerical,
finite-horizon evidence and **not theorem claims**.
