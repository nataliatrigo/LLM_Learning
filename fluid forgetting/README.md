# Fluid monopoly trajectories with forgetting

Deterministic calendar-time trajectories on the **mature boundary**, under the
project's optimal fluid HJB policy. Each parameter pair has one own-state policy;
the five curves apply that same policy from different initial reputations.
These simulations describe established sellers. They do not simulate learning
from an empty history or the random outcomes of the discrete model.

The normalized state is `x=(1-rho)S`, with `x in [0,1]`. Its dynamics are

\[
\dot x(t)=-\log(\rho)\,D_\rho(x(t))\,[p_{i(x(t))}-x(t)],
\]

where the own-sales demand is

\[
D_\rho(x)=\Pr\!\left[\operatorname{Beta}
\left(1+\frac{x}{1-\rho},1+\frac{1-x}{1-\rho}\right)>p_0\right].
\]

Demand uses the original `rho`, as in the model notes. The shared parameters are
`p1=.35`, `p2=.8`, `c1=.05`, `c2=.65`, `R=1`, and `gamma=.98`.
The outside-option sweep uses **`p0=.1,.3,.5,.7,.8,.9`**. Every plot has the same
six panels, `rho=.6,.7,.8,.9,.95,.99`, five initial states,
`x0=.1,.3,.5,.7,.9`, and horizon `0 <= t <= 300`, sampled at 601 common times.
The complete sweep contains 180 trajectories. Titles, axes, legends, captions,
and interactive controls are in English.

## Figures and data

The [combined PDF](outputs/p0_sweep/trajectories-all-p0.pdf) contains one page per
outside option. Individual PNG figures are available for
[p0 = 0.1](outputs/p0_sweep/p0_0p1/trajectories-overview.png),
[p0 = 0.3](outputs/p0_sweep/p0_0p3/trajectories-overview.png),
[p0 = 0.5](outputs/p0_sweep/p0_0p5/trajectories-overview.png),
[p0 = 0.7](outputs/p0_sweep/p0_0p7/trajectories-overview.png),
[p0 = 0.8](outputs/p0_sweep/p0_0p8/trajectories-overview.png), and
[p0 = 0.9](outputs/p0_sweep/p0_0p9/trajectories-overview.png).

Each `outputs/p0_sweep/p0_*/` directory contains:

- `trajectories-overview.png` and `.pdf`: 30 trajectories, policy thresholds,
  stationary fixed-product states, and arrows showing the direction of motion.
- `trajectory-data.json`: parameters, times, policy intervals, threshold types,
  holding mixtures, velocity samples, trajectories, and verification diagnostics.
- `trajectories.csv`: one row per `rho`, initial state, and time.
- `verification.json`: HJB residuals, grid and ODE refinement checks, analytical
  validation, and source/configuration hashes.
- `fluid-monopoly-trajectories.html`: an interactive view of that outside option.

The sweep also saves `trajectory-summary.csv`, with terminal states and drifts
for all 180 paths, and a combined `verification.json`. A finite-horizon terminal
state need not be an equilibrium.

The original paths under `outputs/` remain the default `p0=.5` example, with
English figures and an English interactive view. The earlier Spanish outputs
are preserved in `outputs/archive/original_p0_0p5_spanish/`.

## Optimal feedback policies

The [optimal-policy plot folder](outputs/optimal_policies/README.md) contains
one six-panel PNG/PDF figure per outside option, plus a
[combined six-page PDF](outputs/optimal_policies/optimal-policies-all-p0.pdf).
These graphs show the exact piecewise feedback rules used by the trajectories:
the horizontal axis is current mature reputation `x`, and the vertical axis is
the optimal fraction of product 2, `alpha*(x)`. Each panel is one calculated
optimal policy for a fixed `(p0,rho)`; an initial condition selects where that
rule is applied.

Horizontal blue intervals at 0% and 100% represent products 1 and 2. Orange
diamonds mark isolated holding mixtures at attractive thresholds. Open black
circles identify stationary pure-product states when the corresponding product
is optimal at `p1` or `p2`. Vertical dashed lines show policy thresholds; a
threshold need not be a stationary state. The graphs keep discontinuities and
isolated mixtures explicit, without interpolating artificial intermediate
actions between pure intervals.

The saved policy data already include the 8001-versus-16001-node checks, so
plotting them requires no new HJB solves or trajectory simulations. CSV files
in the same folder list pure policy intervals, actions at thresholds, and all
stationary states. Reproduce these figures with:

```bash
MPLCONFIGDIR=/tmp/fluid-monopoly-policies-mpl \
  .venv/bin/python -B 'fluid forgetting/code/plot_policies.py'
```

## Policy and integration

Policies come from
`forgetting.Code.fluid_convergence.numerics.solve_fluid`, with 8001 nodes and
refinement against 16001 nodes. Switching thresholds interpolate the
Hamiltonian difference, following that solver. Products remain fixed between
thresholds.

Smooth segments use DOP853 with terminal threshold events. When the drift
points toward a threshold from both sides, a holding mixture keeps the state
constant after arrival:

\[
\alpha^*=\frac{x^*-p_1}{p_2-p_1}.
\]

This is the fluid model's holding choice, which avoids artificial numerical
oscillation. A threshold is crossed when both drifts point in the same
direction. At an exactly repelling threshold, product 1 resolves the tie.
Fixed-product stationary states `p_i` within their own policy regions are
approached asymptotically. A policy threshold is not automatically a stationary
state.

Nearly horizontal paths can still be moving. High outside options and high
retention can make sales extremely rare; some displacements over 300 time
units are smaller than floating-point state resolution. Direction arrows use
the drift sign without an absolute small-drift cutoff. The time axis is
calendar time, including the demand factor, rather than the seller's sales
clock.

## Reproduce from the project root

```bash
# Default p0=.5 example, with English labels.
MPLCONFIGDIR=/tmp/fluid-forgetting-mpl \
  .venv/bin/python -B 'fluid forgetting/code/simulate_trajectories.py'

# All six outside options and the combined PDF.
MPLCONFIGDIR=/tmp/fluid-forgetting-mpl \
  .venv/bin/python -B 'fluid forgetting/code/simulate_trajectories.py' \
  --p0s 0.1 0.3 0.5 0.7 0.8 0.9 \
  --output-dir 'fluid forgetting/outputs/p0_sweep'

# Default interactive view, or a view for a selected dataset.
.venv/bin/python -B 'fluid forgetting/code/build_visualization.py'
.venv/bin/python -B 'fluid forgetting/code/build_visualization.py' \
  --data 'fluid forgetting/outputs/p0_sweep/p0_0p8/trajectory-data.json' \
  --output 'fluid forgetting/outputs/p0_sweep/p0_0p8/fluid-monopoly-trajectories.html'

.venv/bin/python -B -m unittest discover \
  -s 'fluid forgetting/code' -p 'test_*.py' -v
```

Use `--config /path/to/config.json` to change economic parameters, initial states,
retention values, or the horizon. `--p0s` overrides only the configured outside
option and records that override in each dataset's diagnostics.

## Verification and interpretation

The checks cover the exact exponential solution with constant demand, crossing
events, analytical arrival time at an attractive threshold, holding mixtures,
separation by a repelling threshold, and stationary initial states. Every
generated path is checked for its initial condition, invariant interval,
monotonicity, and threshold events, then recomputed with stricter ODE tolerances
and a smaller maximum step.

HJB residuals concern the discretized problem. Grid-refinement differences
measure numerical sensitivity; they are not rigorous spatial error bounds or
a proof that the discrete stochastic model converges to this fluid.

In the JSON, `alpha` is the product-2 fraction: 0 for product 1, 1 for product 2,
and an intermediate fraction while holding. `product=0` identifies a holding
mixture in the velocity samples. `holding_time=null` means no attractive
threshold was reached during the simulated horizon. Each case has `root_kinds`
(`crossing`, `attractive`, or `repelling`) in the same order as `roots`.

The equations and holding mixtures are documented in
`Progress notes/04_Progress_notes_fluid_results/04_fluid_threshold_theorem.tex`.
