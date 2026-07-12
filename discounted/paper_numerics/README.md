# Paper numerics

Run the baseline stationary truncated-state approximation from the repository
root with:

```bash
uv run python discounted/paper_numerics/run_paper_numerics.py
```

The script solves three outer grids, writes convergence and interval diagnostics,
and regenerates all figures used by the numerical illustration. CSV outputs are
kept locally under `outputs/tables/` and are excluded from Git by the repository's
global data-output rule.
