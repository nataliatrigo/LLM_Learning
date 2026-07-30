# Modular DP

This directory is an isolated, reproducible experiment for comparing demand and
learning rules in the discounted hidden-product reputation model. It does not
modify the paper solver or any existing output. The benchmark remains
`discounted/paper_numerics/stationary_solver.py`.

## Model and conventions

The shared seller primitives are

```text
p0=.50, p1=.35, p2=.80, c1=.05, c2=.65, R=1, gamma=.98.
```

The main learning-rule sweep holds `p0=.50`. The optional dynamic matrix keeps
all other primitives fixed and replaces only `p0` with
`{.10,.30,.50,.70,.90}` for every representative demand specification.

The package implements standard Thompson sampling, scaled updates,
mean-preserving sampling temperature, uniform two-seller epsilon-greedy
demand, observation-time forgetting, and calendar-time forgetting.
Integer-count models use a triangular stationary
truncation. Forgetting models use normalized effective counts on a triangular
grid with barycentric interpolation.

For forgetting, `x=(1-rho)(a-1)` and `y=(1-rho)(b-1)`, so the sampling
distribution at `(x,y)` is
`Beta(a,b)=Beta(1+x/(1-rho), 1+y/(1-rho))`. Thompson demand is the Beta tail
`Pr[theta>=p0]=1-I_{p0}(a,b)`. Effective sample size is
`a+b-2=(x+y)/(1-rho)`; sampling concentration is `a+b`.

The two forgetting clocks deliberately use different Bellman operators:

- observation-time forgetting has an exact idle self-loop and uses the
  algebraic rearrangement;
- calendar-time forgetting moves the belief when B is selected and uses the
  generic Bellman equation.

Figure 08 retains the full forgetting diagnostic in normalized `(x,y)` and
belief-concentration coordinates. Figure 24 provides the direct analogue of
the discrete policy-region plots for the preferred observation-time clock:
one panel per `rho`, effective sample size `n_eff=a+b-2` on the horizontal axis,
posterior mean on the vertical axis, and the `p0` cutoff as a dashed line. Its
horizontal limits differ by panel because forgetting bounds memory at
`n_eff<=1/(1-rho)`. These are nodes of the continuous interpolation grid, not
a claim that every point is an exactly reachable finite history; after `k`
success/failure observations, observation-time forgetting has
`n_eff=(1-rho^k)/(1-rho)`.

Figure 25 and the accompanying certificate address the extinction question at
the base parameters and `rho=.95`. On the invariant boundary, the Bellman
problem reduces exactly to one dimension. A padded cellwise enclosure finds a
strict product-2 interval, and the analytic recurrence argument shows that the
block of 19 failures followed by 6 successes, repeated twice, returns every
sufficiently mature history to that interval. Thus product 2 is used infinitely often almost
surely, conditional only on the explicitly documented floating-point
enclosure. This is a pathwise statement, not merely a finite-horizon frequency.

## Introductory report in Spanish

`report/reporte_modular_dp.tex` gives a plain-language introduction to the
model, learning rules, Bellman equation, numerical experiments, and main
findings. The compiled version is `report/reporte_modular_dp.pdf`. Rebuild it
from the repository root with:

```bash
latexmk -cd -pdf "Modular DP/report/reporte_modular_dp.tex"
```

## Reproduce

From the repository root:

```bash
uv run python "Modular DP/scripts/validate.py"
uv run python "Modular DP/scripts/run_experiments.py" --config "Modular DP/configs/quick.json"
uv run python "Modular DP/scripts/run_experiments.py" --config "Modular DP/configs/full.json"
uv run python "Modular DP/scripts/run_dynamic_panels.py" --config "Modular DP/configs/quick.json" --update-report
uv run python "Modular DP/scripts/run_dynamic_panels.py" --config "Modular DP/configs/full.json" --update-report
uv run python "Modular DP/scripts/study_observation_forgetting_convergence.py"
uv run python "Modular DP/scripts/verify_observation_forgetting_nonextinction.py"
```

Validation is a gate: the experiment runner checks the unit tests, exact
agreement with the legacy baseline, alias equivalences, Bellman equations, and
one example from each continuous solver family before a sweep. It reruns that
gate before every experiment rather than reusing a stale validation result.

`quick.json` exercises the complete pipeline at smaller numerical sizes.
`smoke.json` is an even smaller local sanity check. Their generated results
and figures are intentionally ignored by Git; only the configs are versioned.
`full.json` contains the requested grids, 1,000 simulations, and 5,000 calendar
periods. Its continuous base resolutions are 201 and 401; resolution 801 is
also used for both forgetting clocks at `rho=0.95` and `rho=0.99`. Here
`grid_resolution=N` is the number of coordinates on a triangle edge, whereas
`grid_node_count=N(N+1)/2` is the stored node count.

Both profiles enable the dynamic matrix, so `run_experiments.py` generates it
as part of the complete pipeline. `run_dynamic_panels.py` is the standalone
entry point when only these simulations and figures need to be regenerated.
The dynamic full profile uses 1,000 Monte Carlo paths over 250 calendar
periods; the quick profile uses 200 paths over 100 periods and visibly
subtitles every dynamic figure **PRELIMINARY — reduced simulation**.

The full discrete profile first compares outer diagonals 800, 1200, and 1600
through the common interior `n<=600`. If that interior censors a last-active
boundary, it additionally compares outer diagonals 3000 and 3600 through
`n<=2200`; the extended convergence rows, diagonals, and raw/robust policy
classifications are exported. A simulation can request a still-larger policy solve, recorded
separately in its `simulation_policy_*` fields.

Outputs are written under `results/<profile>/` and `figures/<profile>/`.
The `full` profile is the curated output intended for version control. The
`quick` and `smoke` directories are disposable pipeline-check artifacts.
Generated CSV files are ignored by the repository's existing global rule; the
run manifest records the experiment configuration hash and the independently
computed `validation_configuration_hash`.

## Dynamic method-by-p0 matrix

The dynamic experiment evaluates ten representative specifications:

- standard Thompson sampling;
- scaled updates with `eta=.5` and `eta=2`;
- mean-preserving temperature with `T=.5` and `T=2`;
- epsilon-greedy with `epsilon` in `{.05,.1,.2}`;
- observation-time and calendar-time forgetting with `rho=.95`.

Each specification is solved and simulated separately at all five configured
values of `p0`, producing 50 series. The same choice and outcome uniforms are
reused across the complete matrix as common random numbers. Epsilon-greedy's
exact balance reduction is checked only at `p0=.5`; its other curves use the
general triangular solver. Forgetting curves use the continuous interpolated
solver and must be interpreted with their grid-convergence checks.

Every numbered dynamic figure retains the informative 2x2 structure: realized
A market share, actual product-2 use conditional on A, calendar-period profit,
and the posterior mean at the start of the period. The original complete set
is retained in `dynamic/`: one figure for each `p0`, comparing the original
eight specifications (including temperature and both forgetting clocks):

```text
13_dynamic_p0_0p1_across_methods
14_dynamic_p0_0p3_across_methods
15_dynamic_p0_0p5_across_methods
16_dynamic_p0_0p7_across_methods
17_dynamic_p0_0p9_across_methods
```

A more readable set is written under `dynamic/focused/`. It excludes
mean-preserving temperature and calendar-time forgetting, and compares only
standard TS, both scaled-update values, epsilon-greedy at `epsilon=.1`, and
observation-time forgetting. Colors identify the same specification in every
complete and focused figure:

```text
18_dynamic_p0_0p1_focused_methods
19_dynamic_p0_0p3_focused_methods
20_dynamic_p0_0p5_focused_methods
21_dynamic_p0_0p7_focused_methods
22_dynamic_p0_0p9_focused_methods
```

The product-2 panel also shows the dashed **quality-maintenance mix**

```text
a_maint(p0) = clip((p0-p1)/(p2-p1), 0, 1).
```

It is the smallest stationary fraction of A interactions assigned to product 2
whose expected quality, `p1+a(p2-p1)`, reaches `p0`. It is a sustainability
benchmark, not the solution of the dynamic profit-maximization problem. When
`p0>p2`, no feasible mix can reach the benchmark and the plot says so instead
of drawing a clipped line.

When only new method specifications were added and all other dynamic settings
are unchanged, compatible saved series can be reused safely:

```bash
uv run python "Modular DP/scripts/run_dynamic_panels.py" --config "Modular DP/configs/full.json" --reuse-existing --update-report
```

Both PNG and PDF versions are written to thematic subdirectories under
`figures/<profile>/`: `demand/`, `policy/`, `simulation/`, `diagnostics/`, and
`dynamic/`. The tidy data
and audit records are:

```text
results/<profile>/dynamic_methods_p0_timeseries.csv
results/<profile>/dynamic_panel_summary.csv
results/<profile>/dynamic_solver_checks.csv
results/<profile>/dynamic_panels_manifest.json
results/<profile>/observation_forgetting_convergence.csv
results/<profile>/observation_forgetting_convergence.json
results/full/observation_forgetting_nonextinction_certificate.json
results/full/OBSERVATION_FORGETTING_NONEXTINCTION.md
```

The plotted curves are contemporaneous cross-path Monte Carlo statistics, not
single paths and not rolling or cumulative averages. A period where B is
selected is excluded from the conditional product-2 denominator; it contributes
zero to calendar-period profit and calendar product-2 use. The CSV also keeps
calendar product-2 use and prescribed product-2 share as distinct quantities.
Per-period 95% intervals use Wilson scores for binary proportions and
Student-t intervals for cross-path means. Effective sample size and full Beta
concentration are saved as additional memory diagnostics, especially for the
forgetting specifications.

The legacy figure titled *Exact discounted simulated paths* also used
per-period cross-sectional Monte Carlo statistics (400 paths, 250 simulated
periods, no rolling/cumulative transform), despite its ambiguous title. It
used a finite-horizon policy solved to horizon 700. The new figures use the
stationary modular policies and report their numerical convergence checks.

## Layout

```text
configs/       validation, quick, and full numerical profiles
modular_dp/    reusable models, solvers, diagnostics, simulation, and plotting
scripts/       command-line entry points
tests/         fast unit and numerical-regression tests
results/       tidy generated tables and manifests
figures/       numbered catalog plus curated full-profile figures
REPORT.md      methods, validation, findings, and numerical limitations
```

No smoothing is applied to computed sensitivities or policy regions. A small
action tolerance is used only to distinguish robust actions from numerical
ties. Policy CSVs retain raw product-2 points plus robust/tie flags; figures
show the robust classification. Both raw and robust grid changes are reported.
Continuous convergence requires zero robust `action_changes` on common nested
nodes under the deliberately strict `converged` flag. That flag is retained as
an exact diagnostic, but it is not sufficient by itself for a discontinuous
argmax boundary: a convergent boundary can cross a handful of nodes as the
number of comparison nodes quadruples. The dedicated observation-forgetting
study therefore also reports RMS value convergence, the share of common-node
policy changes, and coarse-versus-fine disagreement on every simulated state.
Across `p0` the value errors exhibit observed orders from 1.85 to 2.02, all
common-node disagreement shares are below 0.005%, and all simulated-state
shares are below 0.04%. For `p0=.5`, refinement through resolution 3201 reduces
the simulated-state share from 0.0396% to 0.0024%. These results support policy
convergence in measure and show that the resolution-801 policies are stable at
the scale of the Monte Carlo panels, even when exact nodewise equality is not
attained. The full table and interpretation are in
`results/full/OBSERVATION_FORGETTING_CONVERGENCE.md`; figure 23 visualizes the
three diagnostics.

Late-use statistics are finite-horizon diagnostics. The late conditional rate
collects actual product-2 uses over realized A interactions from the configured
inclusive one-indexed calendar period through the end of the run. Threshold
probabilities mean at least one observed use at or after that period within the
horizon. An observed latest-use period is horizon-censored and never establishes a finite
last-use time.
