# Discounted numerical inventory

This inventory records the numerical work inspected before preparing the paper
illustration. It distinguishes the paper's stationary count-state model from
finite-calendar-horizon computations and from separate binary-belief or fluid
models.

## Solvers

| Location | Actual method | Role |
|---|---|---|
| `discounted/DP/exact_dp.py` | Finite-calendar-horizon backward induction with a zero terminal time layer; the self-loop moves to the next calendar period. | Legacy simulations and an independent finite-tail cross-check only. It is not an exact stationary solver. |
| `discounted/DP/analysis/test_gap_unimodality.py` | Backward triangular recursion for the rearranged stationary Bellman equation, with zero value on outer state diagonal `N_outer+1`. | Existing baseline gap, policy-region, interval, and broad parameter-search evidence. Its solver logic is reused in the paper-specific implementation. |
| `discounted/paper_numerics/stationary_solver.py` | Paper-specific stationary truncated-state recursion, retaining a common reliable interior. | Main paper figures and convergence diagnostics. |
| `discounted/DP/analysis/local_extinction_certificate.py` and `certificate_boundary_study.py` | Backward recursion for a valid upper-bound/certificate operator, not for the value function itself. | State-dependent sufficient certificate and outer-bound diagnostics. |
| `discounted/Fluid/fluid_model.py` | Numerical solution of the separate deterministic fluid approximation. | Not used in the discrete baseline illustration. |
| `discounted/BinaryBelief*/` | Value iteration or simulation for separate binary-belief/logit models. | Different models; not used in the paper illustration. |

The scripts in `discounted/regime_experiments/` call
`solve_discounted_finite_horizon` from `exact_dp.py`. Their reports correctly
describe the result as a stationary approximation only after tail comparisons;
they are not the source of the main baseline figures here.

## Existing baseline and gap outputs

- `discounted/DP/analysis/outputs/investment_extinction_map.png` and
  `extinction_summary.md`: earlier policy-region and empirical extinction check,
  reporting last active diagonal 435.
- `discounted/DP/analysis/outputs_gap_shape/figure1_baseline_continuation_gap.png`:
  earlier continuation-gap panels.
- `discounted/DP/analysis/outputs_gap_shape/figure2_baseline_policy_region.png`:
  earlier stationary-truncation policy map.
- `discounted/DP/outputs*/plots/best_response_value_difference_*.png` and
  `continuation_value_gap.md`: outputs from the finite-calendar-horizon pipeline,
  not used as stationary policy evidence.

The paper figures were regenerated because the earlier plots did not provide
the requested convergence table, publication captions, statistical-reference
labeling, and direct threshold visualization in one reproducible baseline run.

## Interval and parameter searches

- `discounted/DP/analysis/test_gap_unimodality.py` and
  `outputs_gap_shape/REPORT_unimodality_and_band_search.md` implement the broad
  reproducible search: 175 structured plus 1,000 fixed-seed random cases across
  all three quality regimes. No robust within-diagonal interval violation was
  found. Seventy-eight cases changed actions between the small first-pass outer
  grids, so the full set is supporting evidence rather than 1,175 fully
  converged policy estimates.
- The same study reports baseline outer-grid stability, no baseline interval or
  unimodality violation, and empirical extinction at 435.
- `discounted/regime_experiments/policy_diagnostics.py` checks connected
  components and interval structure for the separate regime experiment.

## Convergence and certificate reports

- `discounted/DP/analysis/outputs_gap_shape/REPORT_unimodality_and_band_search.md`:
  stationary-truncation baseline and broad shape search.
- `discounted/DP/analysis/outputs_local_certificate/REPORT.md`: terminal-bound
  sensitivity and state-dependent certificate diagnostics.
- `discounted/DP/analysis/outputs_certificate_boundary/REPORT_certificate_boundary.md`:
  outer-grid stability of the certificate operator through outer diagonal 2400.
- `discounted/DP/analysis/outputs_gamma_sweep/SUMMARY.md`: gamma sweep using
  finite-tail backward recursion.
- `discounted/regime_experiments/REPORT_regime_experiments.md`: three-regime
  finite-tail study with convergence flags and Bellman checks.
- `discounted/Fluid/outputs_gamma_0999/convergence_study/numerical_convergence_report.md`:
  convergence of the separate fluid approximation.

Raw parameter-search and diagnostic CSV files are generated beside these
reports but are globally ignored by Git. The paper-specific run writes its own
local CSV tables under `discounted/paper_numerics/outputs/tables/`.
