# Average-Cost Reputation Study

This folder is a separate project for the average per-stage version of the
dynamic reputation model. It does not write to the `other_experiments/discounted/outputs/`
folder used by the discounted-cost study.

## Run

From the repository root:

```bash
uv run python other_experiments/average_cost/main.py
```

The default run solves Seller A's finite-state average-reward best response on
a 25-observation rolling-window approximation across 9 values of `p_0`, simulates
histories under that policy, and writes only inside:

- `other_experiments/average_cost/outputs/data/`
- `other_experiments/average_cost/outputs/plots/`

For a quick smoke test:

```bash
uv run python other_experiments/average_cost/main.py --p0-count 3 --max-observations 25 --T 40 --n-rep 20 --outputs-dir /tmp/llm_learning_average_cost_smoke
```

To run the solver without simulations:

```bash
uv run python other_experiments/average_cost/main.py --skip-simulation
```

To run the projection-to-boundary finite approximation instead of the
rolling-window benchmark:

```bash
uv run python other_experiments/average_cost/main.py --method projection
```

The default simulation also creates five-path diagnostic plots for `p_0=0.5`
and `p_0=0.7`. To change those values:

```bash
uv run python other_experiments/average_cost/main.py --sample-path-p0-grid 0.4,0.6 --sample-path-count 5
```

For a denser, slower numerical sweep:

```bash
uv run python other_experiments/average_cost/main.py --max-observations 50 --p0-count 17
```

## Numerical Methods

The code supports four numerical methods.

1. `rolling_window`: finite average-reward MDP with bounded memory. For
   `S+F<N`, successes and failures update the Beta sufficient statistics
   normally. At `S+F=N`, the new observation is added and one old observation is
   randomly deleted. This is the benchmark method.

2. `projection`: finite average-reward MDP with bounded precision. For `S+F<N`,
   the transition is the usual Bayesian count update. At `S+F=N`, the true
   `(N+1)`-observation next state is projected back to `S+F=N` by preserving the
   success share as closely as possible.

3. `stochastic_projection`: finite average-reward MDP with bounded precision.
   It projects the boundary transition to the two neighboring boundary states so
   the projected success share is preserved in expectation.

4. `finite_horizon`: exact Bayesian finite-horizon dynamic program. It uses
   backward induction over all states with `S+F<=T`, has no artificial forgetting,
   and produces a nonstationary policy that depends on time remaining.

The first three methods are finite approximations of the infinite average-reward
problem. The finite-horizon method solves a different exact finite-horizon
problem. None of these methods is a closed-form solution to the original unbounded
average-reward Bayesian MDP; the goal is to check whether the qualitative policy
is robust across numerical approaches.

## Robustness Experiments

Run the robustness experiment from the repository root:

```bash
uv run python other_experiments/average_cost/main.py --run-robustness
```

By default this runs `p_0 in {0.1,0.3,0.5,0.7,0.9}`,
`N in {25,50,100,200}` for `rolling_window`, `projection`, and
`stochastic_projection`, and
`T in {25,50,100,200}` for `finite_horizon`. The robustness average-reward solves
use a solver-only demand floor of `1e-8` for conditioning; reported `rho` values
and simulations still use the unfloored Beta tail. To change the grid:

```bash
uv run python other_experiments/average_cost/main.py \
  --run-robustness \
  --robustness-p0-grid 0.3,0.5,0.7 \
  --robustness-n-grid 25,50 \
  --finite-horizon-t-grid 25,50 \
  --robustness-demand-floor 1e-8
```

Robustness outputs are written under `other_experiments/average_cost/outputs/robustness/`, for example:

- `other_experiments/average_cost/outputs/robustness/data/method_comparison.csv`
- `other_experiments/average_cost/outputs/robustness/data/path_weighted_comparison.csv`
- `other_experiments/average_cost/outputs/robustness/data/finite_horizon_by_time_remaining.csv`
- `other_experiments/average_cost/outputs/robustness/data/policy_agreement_summary.csv`
- `other_experiments/average_cost/outputs/robustness/data/bias_threshold_diagnostic.csv`
- `other_experiments/average_cost/outputs/robustness/data/boundary_drift_diagnostic.csv`
- `other_experiments/average_cost/outputs/robustness/diagnostic_report.md`
- `other_experiments/average_cost/outputs/robustness/plots/method_comparison_dashboard_extended.png`
- `other_experiments/average_cost/outputs/robustness/plots/product2_region_comparison_extended_N=25.png`
- `other_experiments/average_cost/outputs/robustness/plots/bias_threshold_diagnostic_extended.png`
- `other_experiments/average_cost/outputs/robustness/plots/boundary_drift_diagnostic_extended_N=25.png`
- `other_experiments/average_cost/outputs/robustness/finite_horizon_T=25/plots/finite_horizon_product2_usage_all_p0.png`
- `other_experiments/average_cost/outputs/robustness/finite_horizon_T=25/plots/finite_horizon_time_remaining_extended.png`
- `other_experiments/average_cost/outputs/robustness/rolling_window_N=25/data/policy_by_state.csv`
- `other_experiments/average_cost/outputs/robustness/projection_N=25/data/policy_by_state.csv`
- `other_experiments/average_cost/outputs/robustness/stochastic_projection_N=25/data/policy_by_state.csv`
- `other_experiments/average_cost/outputs/robustness/finite_horizon_T=25/data/finite_horizon_policy.csv`

For stationary policies, simulations keep true Bayesian counts growing. The
finite policy is applied by mapping true counts back to the finite grid using the
configured mapping rule. The exact finite-horizon policy is used directly up to
horizon `T`.

The path-weighted diagnostics use `--path-diagnostic-rep 1000` by default. To
change the replication count or mapping rule:

```bash
uv run python other_experiments/average_cost/main.py \
  --run-robustness \
  --path-diagnostic-rep 500 \
  --simulation-state-mapping cap_total_count_preserve_share
```

## Outputs

Data files are written to `other_experiments/average_cost/outputs/data/`:

- `average_reward_summary.csv`
- `average_reward_policy_by_state.csv`
- `average_reward_policy_iteration.csv`
- `average_reward_diagnostics.csv`
- `simulation_replications.csv`
- `simulation_timeseries.csv`
- `path_diagnostics.csv`

Plots are written to `other_experiments/average_cost/outputs/plots/`:

- `average_reward_by_p0.png`
- `average_reward_policy_heatmaps.png`
- `average_reward_policy_posterior_state_space.png`
- `average_reward_bias_gap_posterior_state_space.png`
- `average_reward_policy_iteration.png`
- `simulation_paths_by_p0.png`
- `sample_paths_p0_0_50.png`
- `sample_paths_p0_0_70.png`

## Paper

The model writeup for this version is in `other_experiments/average_cost/Paper/main.tex`.
Compile it from the paper folder:

```bash
cd other_experiments/average_cost/Paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
