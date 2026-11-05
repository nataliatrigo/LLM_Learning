# Symmetric fluid oblivious equilibrium with an outside option

This study is separate from experiment 08's stochastic distribution and
conditional-distribution equilibria. A seller conjectures a **fixed rival
reputation** `z`, solves its own-state fluid HJB, and checks whether a
stationary optimal response is consistent with `z=x*`. All detected branches
are retained. The source stochastic models and historical results are reused
read-only.

## Model and calibration

The normalized mature reputation is `x=(1-rho)S`, with `x in [0,1]`. The
posterior and the independent Thompson Sampling selection rule are exactly
those of experiment 08:

```text
theta(x) ~ Beta(1+x/(1-rho), 1+(1-x)/(1-rho)),
D_A(x,z;p0) = integral[p0,1] f_x(t) F_z(t) dt,
D_B(x,z;p0) = D_A(z,x;p0),
D_out(x,z;p0) = F_x(p0) F_z(p0).
```

The continuous-time calibration is the previous fluid model's
`lambda=-log(rho)` and `delta=-log(gamma)`:

```text
delta U(x;z) = D_A(x,z) max_i {R-ci + lambda*(pi-x)*U_x(x;z)},
xdot = lambda*D_A(x,z)*(p1+(p2-p1)*alpha(x)-x).
```

The economic parameters remain `R=1`, `c1=.05`, `c2=.65`, `p1=.35`,
`p2=.8`, `gamma=.98`. The six existing outside options are
`p0=.1,.3,.5,.7,.8,.9`; the default seven persistence values are
`rho=.6,.7,.8,.9,.95,.97,.99`, following `../AGENTS.md`.

## Equilibrium search and validation

Pure stationary candidates are checked at `x*=p1` and `x*=p2`, using the
optimized response and its conditional action advantage. Interior candidates
allow the holding mixture `alpha*=(x*-p1)/(p2-p1)`.

A smooth mixed stationary candidate satisfies the necessary condition

```text
D_x(z,z)*[R-c1-(c2-c1)*(z-p1)/(p2-p1)]
  - (c2-c1)/(lambda*(p2-p1)) * [delta+lambda*D_A(z,z)] = 0.
```

Here `D_x` differentiates **only the own state**, holding the rival fixed.
The scalar scan searches sign changes and sampled tangencies. This condition
generates candidates; it does not establish equilibrium. For each candidate,
the code solves the complete optimized HJB, compares its value with the
stationary holding payoff, and checks the conditional holding defect. A
repelling policy threshold is not accepted merely because drifts have
opposite signs.

For mixed candidates the response-root consistency is refined with full HJB
solutions to `5e-8` or better before the representation is aligned for path
integration. Recorded adjustments must therefore be numerical root tolerance,
not a grid-sized artificial equilibrium. The HJB reuses the project's monotone
upwind Howard solver. The study runs 1001, 2001, and 4001 state nodes, and
doubles the scalar scan from 121 to 241 points on the final grid.

The final two grids must agree on equilibrium count and satisfy the configured
sensitivity limits: reputation `.002`, product-2 fraction `.005`, own share
`.002`, and a policy disagreement interval of length `.005` on `[0,1]`.
The first refinement is also reported. These are sensitivity screens, not
rigorous continuous-state error bounds. The finest-grid conditional holding
and value tolerances, HJB residuals, scalar consistency errors, rejected
candidates, and raw search scans are saved.

Positive Beta quadrature follows the original stochastic implementation.
Every final response is checked by doubling quadrature order on **every own
state node**, and by independent scaled adaptive integration at selected exact
states. The same checks cover rejected conjectures when a search finds no
equilibrium. An integration/solver error is recorded separately from
`no_equilibrium_found`. That status means the declared numerical search found
no verified candidate; it is not a mathematical proof of nonexistence.

## Metrics and the stochastic comparison

At a symmetric fluid equilibrium, let `d*=D_A(x*,x*)` and let `alpha*` be the
fraction of own sales using product 2:

| Metric | Fluid definition |
| --- | --- |
| Primary: product 2 among own sales | `alpha*` |
| Own seller selected and product 2, per calendar unit | `d*alpha*` |
| Secondary: customer receives product 2 from either seller | `2*d*alpha*` |
| Own market share | `d*` |
| Outside-option market share | `1-2*d*` |

At symmetry, own share is evaluated with the cancellation-free exact identity
`d* = survival(x*,p0)*(1+CDF(x*,p0))/2`. Conditional and calendar fractions
are never substituted for each other.

The historical common audited n101 stochastic comparison covers the six
`p0` values at `rho=.6,.7,.8`. The figure conventions retain stochastic
monopoly in blue, distribution OE in orange, and conditional OE in green;
fluid point-conjecture equilibria use purple diamonds. Higher-rho stochastic
counterparts remain explicitly unavailable instead of substituting unstable
or interpolated results. Original data, report audits, and checkpoint hashes
are verified before loading.

The stochastic model has an invariant distribution rather than a scalar
equilibrium reputation. Its **actual calendar-time invariant marginal mean
and standard deviation** are compared with deterministic fluid `x*`, with
that distinction labeled on figures and in tables. The quantitative table
also compares policies on the own-state interval, market shares, and all
three product-2 measures. Policy distances are unweighted interval integrals,
not invariant-weighted welfare measures.

## Deterministic trajectories

Both sellers begin at mature reputation `(x_A,x_B)=(.5,.5)`, exactly as in
experiment 08's saved Monte Carlo. The horizon is 3000 calendar units, no
burn-in is removed, and 25-unit blocks reproduce the original usage plots.
The feedback policy is frozen at the response to its equilibrium conjecture,
but realized demand uses **both actual states**:

```text
xdot_A = lambda*D_A(x_A,x_B)*(q_sigma(x_A)-x_A),
xdot_B = lambda*D_A(x_B,x_A)*(q_sigma(x_B)-x_B).
```

The event-driven integrator stops at policy thresholds, crosses them when
appropriate, and uses the optimal holding mixture at attractive thresholds.
Sale and product-2 counters are integrated in the same ODE. In each block,
product-2 use among own sales is `integral(D_A*alpha_A)/integral(D_A)`, while
the calendar rate is `integral(D_A*alpha_A)/block_duration`. Undefined
conditional shares when there are no sales are left undefined.

Each verified equilibrium branch has its own trajectory from the common
initial condition. Stationary consistency does not imply that every initial
condition converges to that equilibrium; terminal distances are reported.
Saved stochastic Monte Carlo flow curves and their nominal 95% path-cluster
intervals are compared directly at `rho=.7`. Those saved files do not contain
reputation histories, so stochastic stationary reputation bands are labeled
as stationary references, not transient paths.

Realized discounted profit, when reported, is integrated along the actual
joint trajectory. It is not replaced by the seller's perceived HJB value
against a frozen rival. The discounted tail bound is saved.

## Run and inspect

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.run_experiment

# Rebuild plots from the completed numerical study.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.run_experiment --figures-only

# Analytical, independent-demand, equilibrium, trajectory, and metric checks.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover \
  -s oblivious/10_fluid_oblivious/code -p 'test_*.py' -v
```

Change `config.json`, or supply `--config` and `--output-dir` for a separate
campaign. Matching checkpoints resume automatically; scientific input/source
changes invalidate their caches. Use `--recompute` to explicitly recalculate.

The outputs include:

- `data/fluid_summary.csv`, `fluid_policies.csv`: all final candidates and
  explicit case statuses, with diagnostics and policy holding points.
- `data/grid_refinement.csv`, `equilibrium_search.json`, `checkpoints/`:
  numerical sensitivity, rejected candidates, scans, response arrays, and
  independent demand audits.
- `data/trajectories.csv`, `trajectory_bins.csv`, `trajectory_diagnostics.json`:
  deterministic paths, exactly integrated usage counters, events, and profits.
- `data/comparison_summary.csv`, `fluid_vs_stochastic_differences.csv`:
  hash-checked stochastic references and paired quantitative differences.
- `figures/`: the primary/secondary layouts against `p0` and `rho`, policies,
  reputation/market-share comparisons, equilibrium status map, deterministic
  paths, transient flow comparisons, and a combined comparison PDF.
- `data/manifest.json`, `figures/report_manifest.json`: configuration,
  numerical source hashes, saved reference provenance, and generated hashes.

See [RESULTS.md](RESULTS.md) for the completed numerical findings.
