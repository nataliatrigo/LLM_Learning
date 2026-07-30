# Validation

Overall result: **PASS**.

| Check | Pass | Metric | Requirement | Notes |
|---|---:|---:|---:|---|
| legacy_baseline_value | True | 0 | <= 5e-13 |  |
| legacy_baseline_demand | True | 0 | <= 5e-13 |  |
| legacy_baseline_gap | True | 0 | <= 5e-13 |  |
| legacy_baseline_advantage | True | 0 | <= 5e-13 |  |
| legacy_baseline_bellman_residual | True | 7.10543e-15 | <= 1e-11 |  |
| baseline_policy_stability_outer_grids | True | 0 | == 0 | common interior n<=80 |
| eta_one_value | True | 0 | <= 5e-13 |  |
| eta_one_actions | True | 0 | == 0 |  |
| temperature_one_value | True | 0 | <= 5e-13 |  |
| temperature_one_actions | True | 0 | == 0 |  |
| observation_forgetting_ts_grid_41_bellman_residual | True | 9.67198e-10 | <= 5e-08 |  |
| observation_forgetting_ts_bellman_dispatch | True | 0 | == 0 | exact_idle_self_loop_rearranged |
| observation_forgetting_ts_grid_81_bellman_residual | True | 9.45327e-10 | <= 5e-08 |  |
| observation_forgetting_ts_bellman_dispatch | True | 0 | == 0 | exact_idle_self_loop_rearranged |
| observation_forgetting_ts_robust_policy_grid_stability | True | 0.00464576 | <= 0.02 | grid 41->81 |
| calendar_forgetting_ts_grid_41_bellman_residual | True | 9.66036e-10 | <= 5e-08 |  |
| calendar_forgetting_ts_bellman_dispatch | True | 0 | == 0 | generic_idle_transition |
| calendar_forgetting_ts_grid_81_bellman_residual | True | 9.67638e-10 | <= 5e-08 |  |
| calendar_forgetting_ts_bellman_dispatch | True | 0 | == 0 | generic_idle_transition |
| calendar_forgetting_ts_robust_policy_grid_stability | True | 0.00232288 | <= 0.02 | grid 41->81 |
