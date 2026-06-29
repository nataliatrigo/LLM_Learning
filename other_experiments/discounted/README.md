# LLM Learning

This repo contains a single-file Python simulation of a dynamic reputation
model with two competing LLM providers.

## Run

```bash
uv run python other_experiments/discounted/main.py
```

The default run solves Seller A's dynamic-programming best response across
values of `p_0` and simulates histories under that policy. It saves CSV data in
`other_experiments/discounted/outputs/data/` and plots in `other_experiments/discounted/outputs/plots/`.

For a quick smoke test:

```bash
uv run python other_experiments/discounted/main.py --T 30 --n-rep 20 --outputs-dir /tmp/llm_learning_smoke
```

To choose the focused path diagnostic value:

```bash
uv run python other_experiments/discounted/main.py --diagnostic-p0 0.70
```

To also run the forgetting experiment:

```bash
uv run python other_experiments/discounted/main.py --run-forgetting-experiment --forgetting-decays 0.95
```

For decay values very close to one, use a coarser pseudo-count grid:

```bash
uv run python other_experiments/discounted/main.py --run-forgetting-experiment --forgetting-decays 0.995 --forgetting-grid-step 2.0
```

## Outputs

Data files are written to `other_experiments/discounted/outputs/data/`:

- `best_response_summary.csv`
- `best_response_policy_by_state.csv`
- `value_iteration_convergence.csv`
- `simulation_replications.csv`
- `simulation_timeseries.csv`
- `path_diagnostics.csv`
- `truncation_robustness.csv`
- `forgetting_experiment_summary.csv`
- `forgetting_experiment_policy_by_state.csv`
- `forgetting_experiment_policy_iteration.csv`
- `forgetting_experiment_replications.csv`
- `forgetting_experiment_timeseries.csv`
- `forgetting_experiment_paths.csv`

Plots are written to `other_experiments/discounted/outputs/plots/`:

- `discounted_initial_summary.png`
- `best_response_by_p0.png`
- `best_response_policy_heatmaps.png`
- `best_response_policy_posterior_state_space.png`
- `best_response_value_difference_posterior_state_space.png`
- `simulation_paths_by_p0.png`
- `simulation_user_belief_paths_by_p0.png`
- `discounted_forgetting_simulation_by_period.png`
- `discounted_forgetting_effective_observations.png`

More specific diagnostic plots are written under `other_experiments/discounted/outputs/plots/details/`:

- `interior/best_response_policy_posterior_state_space_interior.png`
- `interior/best_response_value_difference_posterior_state_space_interior.png`
- `p0_specific/optimal_vs_always_product2_value_p0_<value>.png`
- `p0_specific/user_belief_optimal_vs_always_product2_p0_<value>.png`
- `diagnostics/simulation_sample_paths_p0_<value>.png`
- `diagnostics/path_diagnostic_p0_<value>.png`
