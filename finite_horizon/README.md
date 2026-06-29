# Finite Horizon

This folder contains the exact finite-horizon study for the Bernoulli/Beta
reputation model, focused on the beginning of a long horizon.

## Folder structure

- `main.py`: exact dynamic program and simulation.
- `paper/`: INFORMS-style model formulation and fluid approximation.
- `outputs/plots/`: main finite-horizon figures.
- `outputs/plots_forgetting/`: figures from the optional forgetting experiment.

Generated CSV files are written locally to `outputs/data/` and excluded from
Git because they are large and fully reproducible.

## Reproducing the results

The default run solves the exact dynamic program with final period `T=1000`,
stores policy snapshots for the first 200 periods, and simulates those early
periods from the prior:

```bash
uv run python finite_horizon/main.py
```

For a faster smoke test:

```bash
uv run python finite_horizon/main.py --T 80 --early-periods 20 --snapshot-periods 1,2,5,10,20 --p0-grid 0.5,0.9 --n-rep 20 --outputs-dir /tmp/llm_learning_finite_horizon_smoke
```

To repeat the long-horizon run with `T=2000`, retain the early-period policy
summary through period 500, simulate periods 1--1000, and add a standalone
policy-share plot for periods 1--1000:

```bash
uv run python finite_horizon/main.py --T 2000 --early-periods 500 --simulation-periods 1000 --snapshot-periods 1,2,5,10,25,50,100,200,500 --policy-plot-periods 1000 --outputs-dir finite_horizon/outputs
```

The extended policy plot uses the exact dynamic-program summaries, while the
four-panel simulated-path plot uses all 1000 simulated periods.

To run the same finite-horizon experiment with a UCB user instead of Thompson
sampling:

```bash
uv run python finite_horizon/main.py --user-policy ucb
```

UCB outputs are written to `finite_horizon/outputs_ucb/` by default. The UCB
index is
`posterior_mean + sqrt(alpha * log(t+1)/(S+F+2))`, with `--ucb-alpha 2.0` by
default.

To run a myopic Bayesian user who updates the Beta posterior but chooses only by
posterior mean:

```bash
uv run python finite_horizon/main.py --user-policy posterior_mean
```

Posterior-mean outputs are written to
`finite_horizon/outputs_posterior_mean/` by default. This user chooses A when
`(S+1)/(S+F+2) >= p0`.

To also run the forgetting or limited-memory experiment:

```bash
uv run python finite_horizon/main.py --run-forgetting-experiment --forgetting-decays 0.95 --forgetting-grid-step 0.5
```

The forgetting experiment uses discounted pseudo-counts
`S' = lambda S + y` and `F' = lambda F + (1-y)` when Seller A is chosen.
When A is not chosen, both pseudo-counts decay by `lambda`. Because these
states are continuous, this is an approximate dynamic program on a pseudo-count
grid rather than the exact integer-count solver used by the baseline model.

Outputs are written to `finite_horizon/outputs/` by default.
The summary plot includes the product-2 continuation threshold
`(c2-c1)/(p2-p1)`. A separate product-2 path plot overlays the clipped
asymptotic mixing reference `(p0-p1)/(p2-p1)`.

Data files:

- `finite_horizon/outputs/data/finite_horizon_summary.csv`
- `finite_horizon/outputs/data/finite_horizon_product2_by_time.csv`
- `finite_horizon/outputs/data/finite_horizon_product2_by_posterior_mean.csv`
- `finite_horizon/outputs/data/finite_horizon_policy_snapshots.csv`
- `finite_horizon/outputs/data/simulation_paths_early.csv`
- `finite_horizon/outputs/data/simulation_timeseries_early.csv`
- `finite_horizon/outputs/data/forgetting_experiment_summary.csv`
- `finite_horizon/outputs/data/forgetting_experiment_policy_by_time.csv`
- `finite_horizon/outputs/data/forgetting_experiment_paths.csv`
- `finite_horizon/outputs/data/forgetting_experiment_timeseries.csv`

Plots:

- `finite_horizon/outputs/plots/finite_horizon_initial_summary.png`
- `finite_horizon/outputs/plots/finite_horizon_product2_usage_all_p0.png`
- `finite_horizon/outputs/plots/finite_horizon_policy_heatmaps_extended.png`
- `finite_horizon/outputs/plots/finite_horizon_policy_posterior_state_space.png`
- `finite_horizon/outputs/plots/finite_horizon_simulation_by_period.png`
- `finite_horizon/outputs/plots/finite_horizon_product2_paths_with_asymptotic.png`
- `finite_horizon/outputs/plots/finite_horizon_reputation_diagnostic_paths.png`
- `finite_horizon/outputs/plots/finite_horizon_posterior_density_evolution.png`
- `finite_horizon/outputs/plots_forgetting/finite_horizon_forgetting_simulation_by_period.png`
- `finite_horizon/outputs/plots_forgetting/finite_horizon_forgetting_effective_observations.png`
- `finite_horizon/outputs/plots_forgetting/finite_horizon_forgetting_representative_path.png`
- `finite_horizon/outputs/plots_forgetting/finite_horizon_forgetting_posterior_density_evolution.png`

The script does not export the full `T x state` policy table. For `T=1000`, that
object is too large and not needed for early-period diagnostics.

`simulation_paths_early.csv` also carries path-level reputation diagnostics:
`marginal_reputation_value = M_t(S_t,F_t)`, `rho = D(S_t,F_t)`,
`total_count = S_t+F_t`, `posterior_mean = (S_t+1)/(S_t+F_t+2)`,
and the product-2 net benefit `(p2-p1)M_t - (c2-c1)`.
