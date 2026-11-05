# Paper experiments

Minimal reproduction code for the numerical section of
`OverleafPaper/main.tex`: cumulative learning and observation-time
forgetting. The local manuscript is an independent Git repository and is not
required to run the experiments.

## Quick guide

Todos los experimentos **oblivious** y sus comparaciones están agrupados en
[oblivious/](oblivious/README.md), en este orden:

| Orden | Experimento | Carpeta y guía |
| --- | --- | --- |
| 1 | OE distribution | [01_oe_distribution](oblivious/01_oe_distribution/README.md) |
| 2 | OE p0 | [02_oe_p0](oblivious/02_oe_p0/README.md) |
| 3 | OE following monopoly | [03_oe_following_monopoly](oblivious/03_oe_following_monopoly/README.md) |
| 4 | OE conditional distribution | [04_oe_conditional_distribution](oblivious/04_oe_conditional_distribution/README.md) |
| 5 | OE p0 temporal | [05_oe_p0_temporal](oblivious/05_oe_p0_temporal/README.md) |
| 6 | Equilibrium comparison | [06_equilibrium_comparison](oblivious/06_equilibrium_comparison/code/README.md) |
| 7 | Full comparison | [07_full_comparison](oblivious/07_full_comparison/code/README.md) |
| 8 | Nuevo OE distribution con opción externa fija | [08_oe_distribution_fixed_p0](oblivious/08_oe_distribution_fixed_p0/README.md) |

Las funciones numéricas compartidas están en `oblivious/02_oe_p0/code/`.
Los comandos usan alias de módulos sin números, por ejemplo
`python -m oblivious.oe_p0.code.run_experiment`. Cada guía documenta sus
comandos y resultados; la comparación de monopolio y competencia está dentro de
`oblivious/07_full_comparison/Monopoly_vs_competition/`.
Los nombres públicos de los modelos se definen en `model_names.py`.
Se conservan los identificadores de datos y los certificados históricos de
fuentes. Tras el traslado, usa otro `--name` para nuevas ejecuciones: la
reanudación de campañas parciales sigue exigiendo las huellas originales del
código y los resultados.

Para actualizar figuras e informes desde los datos guardados:

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B -m refresh_model_names
```

The [forgetting folder guide](forgetting/README.md) describes the Code folders,
result locations, and renamed directories. The main forgetting experiment workflows use one entry point:

```bash
uv run python -B -m forgetting --help
uv run python -B -m forgetting initial --help
```

| Command after `python -m forgetting` | Purpose |
| --- | --- |
| `initial` | First interactions and single-customer visit-count histograms |
| `individual` | Plot a saved segment, including interactions 100001–101000 |
| `generate` | Simulate individual segments missing from saved results |
| `overlap` | Compare visited states with the mature policy |
| `reachable` | Plot the limiting reachable set and policy regions |
| `paper` | Full paper experiments and investment windows |
| `monte-carlo` | Calendar-time simulations across rho values |
| `paths` | Individual interaction histories |
| `grid` | Compare interaction histories across parameter pairs |
| `occupation` | Long-run state-visitation diagnostics |

To refresh the saved histograms with green shading for the long-run product-2
policy region (original S units, using the simulation's action tolerance):

```bash
uv run python -B -m forgetting distributions --root forgetting/outputs/policy_and_state_experiments/first_2000
```

The bars count customer 1's visits, including the initial transient. Shading
represents the long-run policy, which can differ from the initial-period policy.
Each pair saves the shaded intervals in `data/<pair>/histogram_policy_regions.json`.

For the first **2,000 interactions**, reusing saved initial records:

```bash
uv run python -B -m forgetting initial \
  --source forgetting/outputs/policy_and_state_experiments/first_20000 --reuse-records \
  --count 2000 --output forgetting/outputs/policy_and_state_experiments/first_2000
```

The first-2,000-interaction experiments also include p0 = 0.3, 0.4, 0.7,
0.8, and 0.9, each at rho = 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, and 0.999.
Each case retains 20 trajectories; the plots and histograms show customer 1.
For a fixed rho, seeds and random draw ordering match the p0=0.5 baseline,
so comparisons across p0 use common outcome uniforms. All other parameters
and solver settings are inherited from that baseline.

To add new p0 values (existing experiments are protected from overwriting):

```bash
uv run python -B -m forgetting extend-initial --p0-values 0.3 0.4 0.7 0.8 0.9
```

The default for `initial` remains 20,000. The generated `reachable_states`,
`occupation_diagnostic` and `interaction_grid` experiment folders were removed
at the user's request. Their commands below document how to regenerate them.
The `interaction_paths` frequency-and-block experiment has been restored with
its original configuration and seed. For existing
initial histories, use the saved `first_20000` or `first_50000` source explicitly.
Monte Carlo rho folders are grouped under `monte_carlo/data/rho_<value>/`.

### Where to edit

| File | Responsibility |
| --- | --- |
| `forgetting/Code/model.py` | Primitives, boundary Bellman solver, reachable maxima |
| `forgetting/Code/layers.py` | Transient policies before reaching the boundary approximation |
| `forgetting/Code/simulation.py` | Calendar-time simulation and policy evaluation |
| `forgetting/Code/experiments/interaction_paths.py` | Interaction-time simulation and path diagnostics |
| `forgetting/Code/experiments/states.py` | All state plots, prefix replay, and reachable-set geometry |
| `forgetting/Code/experiments/run_experiments.py` | Paper figures and investment-window sweeps |
| `forgetting/Code/experiments/run_monte_carlo.py` | Monte Carlo comparisons across rho values |
| `forgetting/Code/experiments/interaction_grid.py` | Parameter-grid experiments |
| `forgetting/Code/experiments/occupation_diagnostic.py` | Long-run visitation statistics |
| `forgetting/Code/experiments/policy_summary.py` | Policy cutoffs, thresholds, and late-state extrema |
| `forgetting/Code/cli.py` | Command selection only |
| `forgetting/Code/experiments/config*.json` | Experiment settings, sizes, seeds, and tolerances |
| `Progress notes/` | Mathematical notes, including the standalone persistence proof |
| `forgetting/outputs/` | Generated simulations, tables, and figures |
| `forgetting/outputs/numerical_validations/` | Numerical certificates and verification results |

The five former state scripts (`plot_individual_states.py`,
`run_individual_states.py`, `plot_initial_states.py`, `plot_state_overlap.py`,
`plot_reachable_states.py`) are consolidated in `forgetting/Code/experiments/states.py`.
Experiment scripts and JSON configurations live together in
`forgetting/Code/experiments/`. The `python -m forgetting` commands above are unchanged;
direct module invocations use `forgetting.Code.experiments.<module>`. All results are grouped under `forgetting/outputs/`; the former `test` directory
is now `policy_and_state_experiments`.

## Detailed experiment reference

## Run

Use Python 3.12 or later and the dependencies pinned in `uv.lock`:

```bash
uv sync --locked
uv run python -B -m extinction.run_experiments
uv run python -B -m forgetting paper
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
| Investment incentives across forgetting rates | `forgetting paper` | `figures/investment_incentives_across_forgetting_rates.pdf` |
| Positive-investment window by outside-option quality | `forgetting paper` | `figures/positive_investment_window_by_p0.pdf` |
| Forgetting simulations | `forgetting paper` | `figures/monte_carlo_forgetting_p0_panels.pdf` |

Both experiments use the manuscript calibration: revenue 1, costs 0.05/0.65,
product success probabilities 0.35/0.80, discount factor 0.98, and outside-option
qualities 0.1/0.3/0.5/0.7/0.9. The persistence-window plot uses a denser grid of
37 outside-option qualities. Grid sizes, tolerances, simulation sizes, and
random seeds are specified in `extinction/config.json` and
`forgetting/Code/experiments/config_paper.json`.

The cumulative-learning runner compares truncations at 4,000 and 4,500
observations and checks stability on the retained region through 3,500. The
paper's last-active counts are 3,165, 991, 435, 186, and 69. The forgetting
runner preserves the reachable-boundary calculation (including the Cantor-like
set for rho below 0.5), root refinement, and coarse/fine solver diagnostics.
The incentive figure displays the raw action-value difference `Q2 - Q1`.
`positive_window_by_p0.csv` records one row per detected positive interval
(and an empty row for a p0 with none). `window_index` and `window_count` identify
disconnected windows. Censored bounds mark edges of the scanned rho range,
not verified zero crossings. The plot displays separate vertical intervals at
each sampled p0, with no area fill or interpolation between p0 samples. Open
endpoints indicate censored bounds; solid endpoints indicate detected roots.
A finite rho grid can miss narrower sign changes; refine it around features
of interest.

These are numerical estimates; Monte Carlo paths illustrate the solved policy.

## Code and checks

All forgetting source code is grouped in `forgetting/Code/`. Experiment runners
and their configurations are in `forgetting/Code/experiments/`.
The extinction runner and configuration remain in `extinction/`. Forgetting
uses `Code/layers.py` to solve transient policies and `Code/experiments/states.py`
for all state plots. These experiment workflows are selected through `python -m forgetting`.
The extinction tests retain the full-value solver as an independent check of
the compact policy solver.

```bash
uv run python -B -m unittest discover -s extinction/tests -v
uv run python -B -m unittest discover -s forgetting/Code/model_and_simulation_validation -v
```

## Repository layout

- `oblivious/`: seven numbered oblivious experiments and comparisons; see its [folder guide](oblivious/README.md).
- `extinction/`: cumulative-learning model, runner, and configuration.
- `forgetting/Code/`: all forgetting scripts, grouped by purpose and study.
- `forgetting/Code/experiments/`: experiment runners and configurations.
- `forgetting/Code/visualizations/`: plotting and comparison scripts.
- `forgetting/Code/model_and_simulation_validation/`: automated checks for the forgetting model, simulations, and experiment commands.
- `extinction/tests/`: automated checks for the extinction study.
- `Progress notes/`: mathematical notes and their PDFs.
- `forgetting/outputs/`: all forgetting results, including the former `output/research` studies.
- `forgetting/outputs/numerical_validations/`: numerical certificates retained for later review.
- `forgetting/Code/experiments/config_policy_comparisons.json`: exploratory investment-window comparisons.
- `forgetting/outputs/policy_and_state_experiments/`: comparison results (`figures/`, `data/`, and
  `manifest.json`).
- `OverleafPaper/`: independent manuscript repository.

Generated forgetting results stay under `forgetting/outputs/`; scripts that
reproduce them are grouped under `forgetting/Code/`. Results are ignored by Git,
while source code and directory guides remain trackable.

## Compare investment windows across costs and success probabilities

```bash
uv run python -B -m forgetting paper \
  --config forgetting/Code/experiments/config_policy_comparisons.json \
  --output-dir forgetting/outputs/policy_and_state_experiments --skip-simulation
```

Edit `investment_window_sweeps` in `forgetting/Code/experiments/config_policy_comparisons.json` to choose values:

```json
"investment_window_sweeps": {
  "c2": [0.50, 0.60, 0.65, 0.6665, 0.68, 0.80],
  "p2": [0.75, 0.8, 0.85]
}
```

The cost comparison holds `p2 = 0.8` fixed; the success-probability comparison
holds `c2 = 0.6665` fixed, using the configuration's seller baseline. All other
parameters remain fixed. Each value gets its own panel with the same axes.
The test configuration disables Monte Carlo simulations.
The investment-window plots use 41 values of `p0` from 0.1 to 0.9, in steps
of 0.02, to show changes between the five original comparison values.

The additional figures are `positive_investment_window_by_p0_vary_c2` and
`positive_investment_window_by_p0_vary_p2`, each saved as PNG and PDF under
`forgetting/outputs/policy_and_state_experiments/figures/`. Corresponding CSVs under `data/` record
every detected interval, the scenario's `c2` and `p2`, and censored endpoints.
The investment criterion is strictly positive advantage on the reachable set.
No area is filled between samples or across disconnected windows. Detection
still uses a finite rho grid: increase `rho_linear.points` to check for narrow
windows or gaps. Increase `rho_p0_comparative_statics.p0_points` for more p0 samples.
The paper configuration does not enable these additional parameter sweeps.

## Monte Carlo for several rho values

Run only Monte Carlo, without recalculating investment windows:

```bash
uv run python -B -m forgetting monte-carlo \
  --config forgetting/Code/experiments/config_policy_comparisons.json \
  --rhos 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 \
  --output-dir forgetting/outputs/policy_and_state_experiments/monte_carlo
```

Omit `--rhos` to use `simulation.rhos` from the configuration (or the single
`simulation.rho` if no list is present). This dedicated command explicitly
runs simulations even when `simulation.enabled` is false in the full runner.
The full `forgetting paper` command continues to use a single
`simulation.rho`.

Monte Carlo uses `p0_comparison.values`, also including the seller's baseline
`p0`, rather than the dense investment-window p0 grid. Adjust `simulation.paths`,
`periods`, and `seed` to control sample size, duration, and reproducibility.
Costs, success probabilities, and revenue come from `seller`; gamma comes from
`rho_p0_comparative_statics.gamma`. The cost and success-probability sweep lists
are not used by this command. The same seed is reused for each p0 across rho
cases, so the scenarios use common random draws and are not independent samples.

The test configuration uses 20,000 paths and 3,000 periods for each combination.
All Monte Carlo figures are PNG only and are saved together in `figures/`.
The combined figure is `figures/monte_carlo_forgetting_by_rho.png`.
Each column holds p0 fixed; colored curves compare rho. The top row shows seller
A's market share and the bottom row the share of all paths using product 2 in
each period. These are simulated frequencies, not maxima of Q2 minus Q1.
Individual figures include the fixed rho in their filename and title.
Each `data/rho_<value>/` subfolder contains per-period and binned
CSV summaries, path summaries, and effective configuration. Summary CSVs include
uncertainty estimates; the comparison figure shows point estimates only.
Using the same output directory replaces files for the requested cases; older
rho subfolders may remain, while the top-level manifest lists the current run.

## Individual histories in interaction time

```bash
uv run python -B -m forgetting paths \
  --config forgetting/Code/experiments/config_interaction.json \
  --output-dir forgetting/outputs/policy_and_state_experiments/interaction_paths
```

This experiment counts only interactions where A is selected. It reuses
`solve_observation_layers` and `_policy_and_demand` from the calendar simulation;
the demand function remains part of the Bellman policy, but there is no
outside-option selection draw. Every step generates one outcome with the
chosen product's success probability and updates the existing normalized state.
Product 2 is chosen only when its interpolated advantage exceeds the configured
action tolerance (1e-6); otherwise the existing implementation chooses product 1.

The separate configuration fixes p0=0.5, rho=0.95, c2=0.65, p1=0.35, p2=0.8,
c1=0.05, revenue=1, and gamma=0.98. It runs 100 independent simulated histories
of 200,000 A interactions, from zero observations, with seed 20260809.
The first 20,000 interactions are discarded for post-burn-in diagnostics.
There are 18 non-overlapping blocks of 10,000 interactions per trajectory.
Interaction n uses the state after n-1 observations, before product choice.
The existing terminal-layer approximation is used after the configured
remaining evidence mass falls below 1e-5.

All six diagnostic figures are PNGs together in `figures/`. `data/` contains
trajectory summaries, block frequencies, sampled individual cumulative curves,
late state dynamics (interactions 50,000–55,000), simulation-length statistics,
and a JSON statistical summary. Ten evenly spaced trajectory IDs are selected
before inspecting outcomes; the state plots use the first three of these.
The length check uses the same path prefixes at 50,000, 100,000, and 200,000,
reporting both full-history F_j(N) and post-burn-in frequencies. The scatter
compares interactions 20,001–100,000 against 100,001–200,000 using each segment's
own denominator. Numerical stabilization is not a proof of ergodicity, and the
baseline experiment does not perform an additional parameter grid.

## Coarse grid of individual-path frequencies

```bash
uv run python -B -m forgetting grid \
  --config forgetting/Code/experiments/config_interaction_grid.json \
  --output-dir forgetting/outputs/policy_and_state_experiments/interaction_grid --workers 4
```

This extends the individual-history experiment to five p0 values and twelve
rho values, keeping c2=0.65 and the other baseline primitives fixed. Each pair
uses 50 independent paths of 100,000 interactions, discards 20,000 interactions,
and compares the equal windows 20,001–60,000 and 60,001–100,000. This differs
from the unequal windows of the earlier 200,000-interaction diagnostic.

`classification` in the separate configuration records the descriptive
thresholds: approximately zero is at most 0.001; extinction-like also requires
at least 95% of paths below that threshold. Stable positive requires percentile
5 above 0.001, sample standard deviation at most 0.01, and mean absolute
late-window difference at most 0.01. Everything else is labelled unstable or
heterogeneous. These are changeable numerical criteria, not model theorems.

The runner reuses the existing interaction simulator and policy. Reproducible
seeds are derived from the base seed and numerical parameter values, independently
of worker scheduling. Matching saved mature-reachable advantage values are
reused after calibration and solver checks; only missing exact pairs are solved.
No interpolation is used to infer the sign of the mature advantage.

Results include `data/trajectory_summary.csv`, `data/parameter_summary.csv`,
`data/mature_advantage.csv` with reuse provenance, `diagnostic_report.md`, and
five PNG figures together in `figures/`. Heatmaps show sampled grid cells;
frequency-curve bands are 5th–95th trajectory percentiles, not confidence
intervals or shaded investment windows. No additional fine grid is run.

## Rare visitation versus mature extinction

```bash
uv run python -B -m forgetting occupation \
  --config forgetting/Code/experiments/config_occupation.json \
  --output-dir forgetting/outputs/policy_and_state_experiments/occupation_diagnostic --workers 3
```

The six targeted pairs use p0=0.1 with rho=0.95/0.97/0.99 and p0=0.5 with
rho=0.95/0.99/0.999. Each pair has 20 independent histories of 1,000,000
interactions with A, with the first 100,000 discarded for occupation statistics.
All other primitives use the original c2=0.65 baseline.

The existing policy solver and interpolation are reused at the existing
2,001-node uniform policy resolution. Mature advantages are separately solved
with 20,001 central nodes and checked with 40,001 central nodes. Strict positive
components are estimated from linear interpolation of the fine advantage.
Statistics distinguish fine-grid strict occupation, policy-grid strict
occupation, and the implemented rule (advantage above 1e-6). Any discrepancies
between occupation and actual product-2 choices are explicitly counted.

Occupation PNGs are grouped by p0 under `figures/occupation/p0_<value>/`.
Individual-state PNGs are grouped under `figures/individual_states/p0_<value>/`.
`data/pair_summary.csv` contains region
geometry, maxima, grid checks, occupation frequencies, state moments and
quantiles, neighborhood shares, step sizes, and transient comparisons.
`data/trajectory_summary.csv` preserves per-history results. Per-pair folders
also contain boundary curves, positive components, maximizer locations,
transient-layer summaries, block frequencies, and occupation histograms.

Every post-burn-in observation is retained in a structured `observations.npy`:
rows are consecutive interactions after burn-in, columns are trajectories,
and fields are `x_before`, `mature_advantage_fine`, `product2`, and `success`.
The matching `records_manifest.json` records the seed, configuration, and
indexing convention. Read without loading the whole array using
`np.load(path, mmap_mode="r")`. The state is recorded before product choice;
the recorded outcome generates the next state using the existing transition.
Occupation distances are to the closure of the strict positive region and
are infinite for an empty region. Zero observed visits do not prove zero
limiting occupation, and the experiment does not establish a mixing time.

To see the actual states of a single saved customer, with product-2 decisions
highlighted in green, run:

```bash
uv run python -B -m forgetting individual \
  --p0 0.5 --rho 0.95 --trajectory 1 --start 100001 --count 1000
```

Each point is one pre-choice state, with no averaging or thinning. The colors
use the recorded simulation policy (including its numerical action tolerance),
not a reclassification on the independent fine grid. Change `--trajectory`
(1–20), `--start`, or `--count` to inspect another segment. Only interactions
100001–1000000 were retained in the default occupation experiment. PNGs are
saved under `occupation_diagnostic/figures/individual_states/p0_<value>/`.

To generate the same single-customer plot for additional rho values at p0=0.5:

```bash
uv run python -B -m forgetting generate --p0 0.5 --rhos 0.3 0.5 0.7 0.8 0.9
```

This reuses saved segments when available. Otherwise it runs the existing
interaction simulator from X=Y=0 through interaction 101000, retaining the
last 1000 states of one customer. New runs share an outcome seed across rho;
all baseline primitives and policy solver settings come from
`config_occupation.json`. These extra segments are for visualization, not
additional million-interaction occupation experiments.

To compare all saved states of one customer with the mature product-2 region:

```bash
uv run python -B -m forgetting overlap --p0 0.5 --rho 0.95 --trajectory 1
```

This uses the saved simulation boundary and verifies every recorded choice.
The upper band shows the policy region; the lower histogram shows percentages
of visits, colored by actual product choice. Bins are split at policy cutoffs
so colors do not extend beyond their decision region. The PNG is saved under
`figures/individual_states/p0_<value>/`.

## First 20,000 interactions and state-visit distributions

```bash
uv run python -B -m forgetting initial
```

This reproduces customer 1's first 20,000 interactions for every parameter
combination with a saved occupation manifest, starting from X=Y=0. Original
seeds, solver settings, and random draw ordering are preserved. Single-customer
runs are extended to 20 trajectories using a separate random stream for the
additional customers; the original customer's history is unchanged.

Outputs are in `forgetting/outputs/policy_and_state_experiments/first_20000/`: individual histories in
`figures/individual_states/`, single-customer histograms in `figures/state_distributions/`,
and a comparison in `figures/state_distributions/all_state_distributions.png`.
Histograms use only customer 1, matching the individual history plot, and count
its visits in interactions 1–20,000, including the initial transient. The other
saved trajectories are not included. Plots show the original state S=X/(1-rho),
with English titles, axis labels,
and legends. For each rho, 100 bins span S=0 to S=1/(1-rho); heights are
raw visit counts, summing to 20,000 per histogram. Each bar groups an interval
of states rather than a single exact state. Saved x_before records remain
normalized; histogram CSV bin edges are in original S units.
Records, provenance manifests, and histogram counts are saved under `data/`.
The original post-burn-in outputs are preserved. Plots contain only titles,
axis labels, and product legends where relevant.

To reuse the previously saved 50,000-interaction records without resimulation:

```bash
uv run python -B -m forgetting initial \
  --source forgetting/outputs/policy_and_state_experiments/first_50000 --reuse-records
```

## Inspect reachable states

To distinguish gaps in the mature reachable set from product-choice regions:

```bash
uv run python -B -m forgetting reachable \
  --config forgetting/Code/experiments/config_paper.json --p0 0.9 \
  --rhos 0.35 0.43 0.45 0.47 0.49 0.5 0.7 \
  --output-dir forgetting/outputs/reachable_states
```

For the test calibration, use `--config forgetting/Code/experiments/config_policy_comparisons.json` and
`--output-dir forgetting/outputs/policy_and_state_experiments/reachable_states` to keep the exploratory
results together.

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

## Policy and long-run state summary

```bash
uv run python -B -m forgetting policy-summary
```

The table covers every saved pair under `forgetting/outputs/policy_and_state_experiments/first_2000/data`.
It also includes saved comparison cases under `policy_summary_cases` in that
same root. These are kept separate from the plot datasets so different costs
cannot be confused with an existing `(p0, rho)` pair. The table identifies
`c2`; the CSV records all seller primitives and each row's source manifest.
It is saved as `tables/policy_summary.csv` and `tables/policy_summary.md`.
For each pair it reports the numerical cutoff K (completed A purchases),
the first interaction using the limiting-policy approximation (K+1), boundary
policy thresholds in original S units with switching directions, and the
minimum/maximum state observed for customer 1 at purchases 100001–200000.
The simulator continues the saved customer's random stream, retaining the
original draw width even though the table only measures customer 1.
The compact JSON cache allows interrupted runs to resume without recomputing
completed pairs; inputs and sample window are fingerprinted.

The added comparison uses `p0=0.9, c2=0.6665` at rho
`0.43, 0.44, 0.445, 0.45, 0.455, 0.46, 0.47`. The middle three points have
negative reachable advantage between positive flanks. The original histories
use `c2=0.65`, which does not show this gap at those rho values. Additional
cases use new 2,000-purchase prefixes with the reference seed and customer-1
draw ordering, then the same 100,001–200,000 extrema window as the other rows.

K is a numerical approximation cutoff, not an exact convergence time.
Thresholds use the same grid and action tolerance as the plots. The observed
extrema are finite-sample statistics, not support bounds. For rho below 0.5,
the product-2 region is displayed as its intersection with `C_rho^S`, the
limiting reachable Cantor set in original S units. It displays `None` when
the sampled reachable advantage does not exceed the action tolerance, even
if product 2 is optimal in an unreachable boundary gap. For rho at least
0.5, the entire boundary is reachable. This describes the limiting policy,
not all transient choices. These remain numerical estimates. The CSV retains
the full-boundary intervals in `product2_intervals_S` and
`product2_full_boundary_region_S`; `product2_region_S` is the reachable-region
description and `product2_reachable` records its estimated nonemptiness.
The CSV also records the interactions at which the observed extrema occurred.
