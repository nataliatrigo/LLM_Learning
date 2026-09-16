# Paper experiments

Minimal reproduction code for the numerical section of
`discounted/OverleafPaper/main.tex`: cumulative learning and observation-time
forgetting. The local manuscript is an independent Git repository and remains
at its existing path; it is not required to run the experiments.

## Run

Use Python 3.12 or later and the dependencies pinned in `uv.lock`:

```bash
uv sync --locked
uv run python -B -m extinction.run_experiments
uv run python -B -m forgetting.run_experiments
```

Both commands solve from scratch, run the paper's Monte Carlo simulations,
and write PDF/PNG figures, CSV data, and the effective configuration in
`<study>/outputs/`. These generated files are ignored by Git. The full forgetting
sweep solves 2,257 parameter pairs plus threshold refinements and validation;
it can take several minutes.

Each runner accepts:

- `--config PATH`: use a different JSON configuration.
- `--output-dir PATH`: write results to another directory.
- `--skip-simulation`: run deterministic calculations only; remove stale Monte
  Carlo files from that output directory.
- `--sync-paper-assets`: copy this run's figure PDFs into the local manuscript's
  `figures/` directory.

## What is reproduced

| Paper result | Runner | Output |
| --- | --- | --- |
| Cumulative-learning policy regions | `extinction.run_experiments` | `figures/cumulative_learning_overlay.pdf` |
| Cumulative-learning simulations | `extinction.run_experiments` | `figures/monte_carlo_cumulative_learning_p0_panels.pdf` |
| Appendix cutoff table and truncation checks | `extinction.run_experiments` | `data/paper_cutoffs.csv`, `data/p0_policy_diagnostics.csv` |
| Investment incentives across forgetting rates | `forgetting.run_experiments` | `figures/investment_incentives_across_forgetting_rates.pdf` |
| Positive-investment window by outside-option quality | `forgetting.run_experiments` | `figures/positive_investment_window_by_p0.pdf` |
| Forgetting simulations | `forgetting.run_experiments` | `figures/monte_carlo_forgetting_p0_panels.pdf` |

Both experiments use the manuscript calibration: revenue 1, costs 0.05/0.65,
product success probabilities 0.35/0.80, discount factor 0.98, and outside-option
qualities 0.1/0.3/0.5/0.7/0.9. The persistence-window plot uses a denser grid of
37 outside-option qualities. Grid sizes, tolerances, simulation sizes, and
random seeds are specified in each study's `config.json`.

The cumulative-learning runner compares truncations at 4,000 and 4,500
observations and checks stability on the retained region through 3,500. The
paper's last-active counts are 3,165, 991, 435, 186, and 69. The forgetting
runner preserves the reachable-boundary calculation (including the Cantor-like
set for rho below 0.5), root refinement, and coarse/fine solver diagnostics.
The incentive figure displays the raw action-value difference `Q2 - Q1`.
`positive_window_by_p0.csv` records one row per detected positive interval
(and an empty row for a p0 with none). `window_index` and `window_count` identify
disconnected windows. Censored bounds mark edges of the scanned rho range,
not verified zero crossings. When windows are disconnected, the plot displays
separate vertical intervals at each sampled p0 instead of shading over gaps.
A finite rho grid can miss narrower sign changes; refine it around features
of interest.

These are numerical estimates; Monte Carlo paths illustrate the solved policy.

## Code and checks

Each study contains `model.py`, `simulation.py`, `run_experiments.py`,
`config.json`, and focused tests. Forgetting also uses `layers.py` to solve the
transient policies needed by its simulation. The extinction tests retain the
full-value solver as an independent check of the compact policy solver.

```bash
uv run python -B -m unittest discover -s extinction/tests -v
uv run python -B -m unittest discover -s forgetting/tests -v
```

The exploratory modular/discounted studies, alternative plotting scripts,
additional calibrations, generated reports, and stored experiment outputs have
been removed. Only the plots used by the current manuscript are generated.

## Inspect reachable states

To distinguish gaps in the mature reachable set from product-choice regions:

```bash
uv run python -B -m forgetting.plot_reachable_states \
  --config forgetting/config.json --p0 0.9 \
  --rhos 0.35 0.43 0.45 0.47 0.49 0.5 0.7 \
  --output-dir forgetting/outputs/reachable_states
```

`--c2 0.65` overrides the cost without changing the configuration. `--depth 10`
sets the outer-cover resolution; `--focus-rho 0.45` selects the separate
construction diagram. Outputs include PDF/PNG figures and CSV interval and
policy diagnostics. These are limiting-boundary calculations, not finite-time
reachable states or probabilities of visiting states.

The first figure separates the reachable-set cover, the policy on the entire
boundary, and their intersection. Gray means product 1, green means product 2,
and amber marks near-indifference. Tiny cylinders are shown as thin location
marks so rasterization does not hide them. For rho below 0.5, each depth-d
cylinder has width rho**d and contains unresolved smaller gaps. For rho at
least 0.5 the reachable set is exactly [0,1]. The summary uses the solver's
separately configured reachable depth, not the drawing depth.
