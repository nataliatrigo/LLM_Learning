# Figure catalog

The numbered figures form one ordered narrative for the modular-DP study and
are grouped by subject inside each profile:

- `demand/`: `01`--`02`, demand sensitivity;
- `policy/`: `03`--`08`, policy regions and extinction diagnostics; `24`, the
  observation-forgetting policy in effective-sample-size coordinates;
- `simulation/`: `09`--`11`, simulated use and representative trajectories;
- `diagnostics/`: `23`, observation-forgetting value and policy convergence;
  `25`, the parameter-specific
  observation-forgetting non-extinction certificate;
- `dynamic/`: `13`--`17`, the complete eight-specification comparison for
  each `p0`;
- `dynamic/focused/`: `18`--`22`, the readable comparison using standard TS,
  scaled updates, epsilon-greedy at `epsilon=.1`, and observation forgetting.

Figure identifier `12` is intentionally unused so the filenames of later
curated outputs remain stable.

Only `full/` is a curated, versioned output. The `quick/` and `smoke/`
directories are generated locally to check the pipeline and are ignored by
Git. Their reproducible configuration files remain under `configs/`.

Run the complete profile from the repository root with:

```bash
uv run python "Modular DP/scripts/run_experiments.py" --config "Modular DP/configs/full.json"
```
