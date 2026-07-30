# Discounted Model

The discounted study is split into two active pipelines:

- `DP/` contains the exact discounted dynamic program and its outputs.
- `Paper/` contains the discounted manuscript.

Most day-to-day work should start in `DP/` or `Paper/`.

The DP uses exact backward induction on a long finite horizon. The default uses
\(\gamma=0.98\) and \(T=700\), for which
\(\gamma^T=7.22\times10^{-7}<10^{-6}\). Rolling value arrays make this feasible
without imposing a boundary at a fixed number of observations.

The previous implementation truncated the state space at `S+F=100`, made that
boundary absorbing, and projected longer simulated histories back to it. Those
operations generated artificial changes near period 100 and are no longer
used.

## Paper

- [Manuscript source](Paper/main.tex)
- [Compiled manuscript](Paper/main.pdf)

The manuscript presents the infinite-horizon discounted Bellman equation, its
exact long-horizon temporal approximation, the discounted continuation-value
threshold, a deterministic fluid model, and the full
\(\gamma=0.999,\ T=2000\) numerical study.

## DP

Code:

- `DP/exact_dp.py`: Bellman primitives, state space, simulation helpers.
- `DP/main.py`: main discounted DP runner and plots.
- `DP/diagnose_horizon.py`: focused horizon-comparison diagnostics.

Full default study:

```bash
uv run python discounted/DP/main.py
```

Quick smoke test:

```bash
uv run python discounted/DP/main.py --T 80 --comparison-horizon 90 --simulation-periods 20 --policy-period 10 --p0-grid 0.5,0.9 --n-rep 20 --density-paths 11 --outputs-dir /tmp/llm_learning_discounted_smoke
```

Focused comparison of \(T=700\), \(T=850\), and the slower
\(\gamma=1,T=2000\) benchmark:

```bash
uv run python discounted/DP/diagnose_horizon.py
```

To test the more patient calibration \(\gamma=0.999\) at \(p_0=0.5\), compare
the \(T=1500\) and \(T=2000\) policies and simulate the latter:

```bash
uv run python discounted/DP/diagnose_horizon.py --gamma 0.999 --T 1500 --comparison-horizon 2000 --simulate-comparison-policy --skip-same-gamma-benchmark --skip-undiscounted-benchmark --outputs-dir discounted/DP/outputs/diagnostics_gamma_0999
```

Here \(T=2000\) is an empirical policy-convergence check, not a
`gamma^T < 1e-6` approximation: that strict criterion would require
`T=13809`.

To run the full five-value \(p_0\) grid with \(\gamma=0.999\), \(T=2000\),
250 simulated periods, and 400 replications per value:

```bash
uv run python discounted/DP/main.py --gamma 0.999 --T 2000 --simulation-periods 250 --policy-period 50 --skip-horizon-check --outputs-dir /tmp/discounted_outputs_gamma_0999
```

The disposable output path is outside the repository so this run does not
recreate the archived `outputs_gamma_0999` directory.

## Continuation value gap

Define the raw continuation gap

\[
D_t(S,F)=V_{t+1}(S+1,F)-V_{t+1}(S,F+1)
\]

and the discounted continuation value gap

\[
M_t(S,F)=\gamma D_t(S,F).
\]

Conditional on Seller A being chosen,

\[
Q_2-Q_1=-(c_2-c_1)+(p_2-p_1)M_t(S,F).
\]

Thus, when `M_t` is reported, the product-2 threshold is

\[
\frac{c_2-c_1}{p_2-p_1}=\frac{4}{3}.
\]

If the raw gap `D_t` is reported instead, its threshold is
`(c2-c1)/(gamma*(p2-p1))`.

## DP outputs

Data in `discounted/DP/outputs/data/`:

- `discounted_summary.csv`
- `discounted_policy_t50.csv`
- `discounted_simulation_replications.csv`
- `discounted_simulation_timeseries.csv`
- `discounted_horizon_convergence.csv`

Plots in `discounted/DP/outputs/plots/`:

- `discounted_initial_summary.png`
- `best_response_policy_heatmaps.png`
- `best_response_policy_posterior_state_space.png`
- `best_response_value_difference_posterior_state_space.png`
- `discounted_simulation_by_period.png`
- `discounted_user_belief_by_period.png`
- `discounted_product2_paths_with_quality_benchmark.png`
- `discounted_reputation_diagnostic_paths.png`
- `discounted_posterior_density_evolution.png`
- `discounted_horizon_convergence_t50.png`

The focused numerical report is
`discounted/DP/outputs/diagnostics/diagnostic_report.md`.

## What Can Be Ignored

The following are generated and can be regenerated from the commands above:

- `DP/outputs*/data/`
- `DP/outputs*/plots/`
