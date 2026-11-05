# Experiment 12: fluid monopoly and competition with average profit

This experiment reoptimizes constant product mixtures for a monopoly and two
symmetric competing sellers with the same outside option. All code, tables,
figures and experiment documentation are in English.

For a constant product-2 fraction `u` in `[0,1]`, long-run fluid reputation is
`q(u)=p1+(p2-p1)*u` and margin is `m(u)=R-c1-(c2-c1)*u`. The objectives are

```text
Monopoly:    maximize D_M(q(u); p0,rho) * m(u).
Competition: u is an equilibrium iff u maximizes
             D_C(q(v),q(u); p0,rho) * m(v) over own mixture v in [0,1].
```

The competitor's action stays fixed in every best response. A mixture is a
pure decision in this game: each seller chooses one definite fraction, which
can be interior. The symmetric diagonal profit is plotted as a reference,
and is never used to select competition equilibria.

This is a long-run average-profit comparison of constant mixtures. Discount
factors, transient profits, initial states and attraction regions do not enter
the objective. No HJB is solved. The discount factor from the source
configuration is recorded only as provenance.

## Configuration and demand

`config.json` points to the active configuration of experiment 10. Each run
reads its `p0s`, `rhos`, `p1`, `p2`, `c1`, `c2` and `R`; it does not hard-code
another parameter grid. The full source configuration, its hash, and all
values actually used are saved to `data/configuration_used.json`.

The current source specifies 24 cases:

- `p0 = 0.1, 0.3, 0.5, 0.7, 0.8, 0.9`.
- `rho = 0.6, 0.7, 0.8, 0.9`.
- `p1=0.35`, `p2=0.8`, `c1=0.05`, `c2=0.65`, `R=1`.

Demand reuses the existing Beta posterior and fluid demand functions:

```text
theta(q) ~ Beta(1+q/(1-rho), 1+(1-q)/(1-rho))
D_M(q)   = Pr(theta(q)>p0)
D_C(q,z) = integral from p0 to 1 of f_q(t)*F_z(t) dt
```

Thus rho continues to affect demand and optimal mixtures. Common-node
positive quadrature with the endpoint-smoothing substitution
`t=1-(1-p0)*s^4` accelerates global scans of the same competition integral.
Peak refinement uses the existing Jacobi fluid-demand function directly,
and independent scaled adaptive integration checks every competitive
candidate and its detected best responses.

## Numerical search and verification

Every monopoly optimization and every fixed-rival best response explores the
whole action interval, refines all sampled local peaks, and compares both
endpoints. It does not assume unimodality. Tied detected global optima are
retained separately.

The equilibrium search evaluates
`g(u)=max_v Pi_C(v,u)-Pi_C(u,u)`, scans its local minima, refines them and
checks the endpoints. Zeros need no sign change. Partial-own-profit FOC
roots provide additional candidates, including stationary tangencies;
every candidate still has to pass the global deviation test. A FOC root
alone does not establish an equilibrium.

The explicit profit tolerance is `1e-12 + 1e-9*profit_scale`, where
`profit_scale` is the maximum absolute profit in the relevant full search.
Actual per-solution scales and tolerances are in the tables. Numerical copies
within `1e-6` in `u` are deduplicated by retaining the best residual, never
averaging. The production run doubles quadrature order from 128 to 256,
refines own-action scans from 501 to 1001 points, and refines equilibrium
scans from 241 to 481 points. All solutions from the finer run are saved.

Coarse/fine solution sets, global deviation gains, independent integration,
accelerated scans against shared Jacobi quadrature on additional probes,
and market accounting are checked. These are numerical verification and
sensitivity checks, not analytic completeness certificates. Cases without a
verified equilibrium retain their minimum residual, location and candidate
diagnostics. No candidate is filtered by initial conditions or dynamics.

## Three different measures

| Measure | Monopoly | Competition |
| --- | --- | --- |
| Product-2 fraction among own sales | `u_M` | `u_C` |
| Product-2 use per seller per time unit | `D_M*u_M` | `D_C*u_C` |
| Product-2 use in the whole market per time unit | `D_M*u_M` | `2*D_C*u_C` |

Plots retain the blue solid monopoly and orange dashed competition style,
with rho on the horizontal axis and panels by p0. Every solution is shown.
At multiple-solution cases lines are interrupted and all set elements are
plotted, avoiding an arbitrary branch continuation. A focal diagnostic at
`p0=0.8,rho=0.9` is produced when that case belongs to the active grid.

`data/comparisons.csv` contains every monopoly/competition solution pair.
It separates the change in per-seller product-2 use exactly into
`mean(D)*(u_C-u_M)` and `mean(u)*(D_C-D_M)`, distinguishing choice from demand.

## Reproduce

From the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/average-profit-mpl \
  .venv/bin/python -B -m oblivious.fluid_average_profit.code.run_experiment

.venv/bin/python -B -m oblivious.fluid_average_profit.code.run_experiment --figures-only

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/average-profit-mpl \
  .venv/bin/python -B -m unittest discover \
  -s oblivious/12_fluid_average_profit/code -p 'test_*.py' -v
```

Use `--config` and `--output-dir` for another campaign. A relative
`source_config` is resolved relative to the supplied experiment configuration.
The `--figures-only` command reads the saved effective configuration.

## Outputs

- [Results and interpretation](RESULTS.md).
- [All solution rows](data/solutions.csv), plus separate monopoly and
  competition solution tables. Each row saves identifiers, `u`, `q(u)`,
  demand, average profit, seller and market product-2 rates, tolerances,
  detected best-response choices and maximum deviation gain.
- [Case summary](data/case_summary.csv), [all solution pairs](data/comparisons.csv),
  and full coarse/fine search results including rejected candidates.
- [Refinement](data/refinement.csv), [independent integration](data/independent_validation.csv),
  [validation](data/validation.json) and [provenance](data/manifest.json).
- [All figures, combined PDF](figures/overview/average_profit_monopoly_vs_competition.pdf),
  with individual PNG and PDF copies under `figures/overview/`.
- `data/diagnostic*.csv` and `data/diagnostic.json`: focal profits,
  fixed-rival deviations, best responses and candidate residuals.

Historical experiments and their saved results are left intact.
