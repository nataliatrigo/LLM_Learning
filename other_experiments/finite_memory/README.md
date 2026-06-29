# Finite Memory

This folder contains a finite-horizon experiment where the user learns from at
most `N` remembered observations. The default is `N=20`.

```bash
uv run python other_experiments/finite_memory/main.py
```

The default run uses:

- horizon `T=1000`
- memory size `N=20`
- Thompson-sampling user demand
- p0 grid `0.1,0.3,0.5,0.7,0.9`

Outputs are written to `other_experiments/finite_memory/outputs/` by default.

Important modeling convention:

- The state is the unordered memory count `(S,F)` with `S+F<=N`.
- If memory is full and a new Seller A outcome is observed, one old remembered
  observation is forgotten uniformly at random and the new outcome is added.
- Under this convention, the dynamic program is exact and finite-state.

If instead the user literally keeps the last `N` chronological observations,
then `(S,F)` alone is not Markov; the exact state would need the ordered binary
sequence of remembered outcomes.

Data:

- `other_experiments/finite_memory/outputs/data/finite_memory_summary.csv`
- `other_experiments/finite_memory/outputs/data/finite_memory_product2_by_time.csv`
- `other_experiments/finite_memory/outputs/data/finite_memory_policy_snapshots.csv`
- `other_experiments/finite_memory/outputs/data/simulation_paths.csv`
- `other_experiments/finite_memory/outputs/data/simulation_timeseries.csv`

Plots:

- `other_experiments/finite_memory/outputs/plots/finite_memory_product2_by_time.png`
- `other_experiments/finite_memory/outputs/plots/finite_memory_policy_heatmaps.png`
- `other_experiments/finite_memory/outputs/plots/finite_memory_simulation_by_period.png`
