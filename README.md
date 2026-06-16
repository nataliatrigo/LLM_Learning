# LLM Learning

This repo contains a single-file Python simulation of a dynamic reputation
model with two competing LLM providers.

## Run

```bash
uv run python main.py
```

The default run solves Seller A's dynamic-programming best response across
values of `p_0` and simulates histories under that policy. It saves CSV data in
`outputs/data/` and plots in `outputs/plots/`.

For a quick smoke test:

```bash
uv run python main.py --T 30 --n-rep 20 --outputs-dir /tmp/llm_learning_smoke
```

To choose the focused path diagnostic value:

```bash
uv run python main.py --diagnostic-p0 0.70
```

## Outputs

Data files are written to `outputs/data/`:

- `best_response_summary.csv`
- `best_response_policy_by_state.csv`
- `value_iteration_convergence.csv`
- `simulation_replications.csv`
- `simulation_timeseries.csv`
- `path_diagnostics.csv`
- `truncation_robustness.csv`

Plots are written to `outputs/plots/`:

- `best_response_by_p0.png`
- `best_response_policy_heatmaps.png`
- `best_response_policy_posterior_state_space.png`
- `best_response_policy_posterior_state_space_interior.png`
- `best_response_value_difference_posterior_state_space.png`
- `best_response_value_difference_posterior_state_space_interior.png`
- `simulation_paths_by_p0.png`
- `simulation_user_belief_paths_by_p0.png`
- `path_diagnostic_p0_<value>.png`
