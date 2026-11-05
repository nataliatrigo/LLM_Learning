# Initial reputation under the two saved fluid competition policies

This comparison uses the two verified symmetric fluid oblivious equilibrium
policies already saved for `p0 = 0.8` and `rho = 0.9`. It changes the initial
reputation used to simulate each policy. It does not solve for new equilibria.
The main experiment's configuration, results, and common-start trajectories
remain the reference study.

## Equilibrium conjecture and realized reputation

The low equilibrium has stationary reputation `x* = 0.35` and uses product 1
at that state. Its policy is optimized against a rival fixed at `z = 0.35`.
The high equilibrium has stationary reputation `x* = 0.8` and uses product 2
at that state. Its policy is optimized against a rival fixed at `z = 0.8`.

For each saved policy, both sellers use that same policy and start at the same
reputation:

```text
(x_A(0), x_B(0)) = (x0, x0),
x0 = 0.1, 0.3, 0.5, 0.55, 0.7, 0.9.
```

The policies remain fixed during each simulation. Realized demand is evaluated
at both actual reputations, rather than replacing the rival's current state
by the equilibrium conjecture:

```text
xdot_A = lambda * D_A(x_A, x_B; p0) * (q_sigma(x_A) - x_A),
xdot_B = lambda * D_A(x_B, x_A; p0) * (q_sigma(x_B) - x_B),
q_sigma(x) = 0.35 + 0.45 * alpha_sigma(x),
lambda = -log(0.9).
```

Identical policies and symmetric initial conditions preserve symmetry, so
the two sellers have the same reputation path. The comparison therefore
shows one reputation curve for each policy and initial state. Time is
continuous calendar time, with horizon `T = 3000`.

An equilibrium's stationary consistency does not imply convergence to its
conjectured state from every initial reputation. Away from that state, these
are realized paths under frozen oblivious policies; the exercise does not
establish optimality against the rival's entire changing path.

## Why initial reputation changes the destination

Both policies have the form product 1, then product 2, then product 1 as own
reputation increases. Their first switching thresholds separate the low and
high basins of attraction:

| Saved policy | First switching threshold | Upper switching threshold |
| --- | ---: | ---: |
| Optimized against `z = 0.35` | `0.5301157623` | `0.8580641062` |
| Optimized against `z = 0.8` | `0.5722495433` | `0.8668577712` |

The first threshold lies between `p1 = 0.35` and `p2 = 0.8`. Immediately
below it, product 1 moves reputation toward `0.35`; immediately above it,
product 2 moves reputation toward `0.8`. Both vector fields point away
from this threshold. It is a boundary between basins, not an additional
verified stationary equilibrium.

The upper threshold is above `0.8`. Both products move reputation downward
there, so a path starting above it crosses into the product-2 region and
continues toward `0.8`. This threshold does not create a third destination.

Demand is positive over these states. It sets the speed of movement and can
make convergence very slow, while the signs of `q_sigma(x) - x` determine
the direction. Under these fixed policies, the predicted asymptotic
destinations are:

| Common initial reputation `x0` | Policy optimized against `z = 0.35` | Policy optimized against `z = 0.8` |
| ---: | ---: | ---: |
| `0.1` | `0.35` | `0.35` |
| `0.3` | `0.35` | `0.35` |
| `0.5` | `0.35` | `0.35` |
| `0.55` | `0.8` | `0.35` |
| `0.7` | `0.8` | `0.8` |
| `0.9` | `0.8` | `0.8` |

The `0.55` start lies between the two first thresholds and distinguishes
the policies most clearly. The original `0.5` start lies below both: both
policies choose product 1 throughout the interval visited, so they generate
the same realized trajectory. Their coincident paths do not merge the two
stationary equilibrium conjectures into one.

These are long-run basin predictions. A simulated terminal reputation at
`T = 3000` may still differ appreciably from its asymptotic destination when
sales are rare. The outside option `p0 = 0.8` is fixed throughout; it does
not evolve along the trajectories.

## Mature initial states, not empty histories

Every starting value here lies on the mature normalized boundary
`x_s + x_f = 1`, with `x_s = x0` and `x_f = 1 - x0`. The posterior is

```text
Beta(1 + x0/(1-rho), 1 + (1-x0)/(1-rho)).
```

Starting from an empty history would mean `(x_s, x_f) = (0, 0)` and posterior
`Beta(1, 1)`. That transient requires an additional evidence-mass state and
is outside this comparison. Changing `x0` within the mature model does not
simulate the accumulation of evidence from an empty history.

## Reproduce and inspect

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.compare_initial_conditions
```

The comparison loads the saved finest-grid equilibrium policies. It uses
the experiment's actual-state demand and event-driven trajectory integrator,
which handles threshold crossings without artificial switching oscillations.
The evidence for the original equilibria remains subject to the numerical
accuracy and continuous-state limitations documented in [README.md](README.md).

- [Initial-condition summary](data/initial_conditions_p0_0p8_rho_0p9/summary.csv).
- [Trajectory figure, PNG](figures/overview/15_initial_conditions_p0_0p8_rho_0p9.png).
- [Trajectory figure, PDF](figures/overview/15_initial_conditions_p0_0p8_rho_0p9.pdf).

## All active outside options and retention values

The full sweep extends this exercise to `p0 = .1, .3, .5, .7, .8, .9` and
`rho = .6, .7, .8, .9`, keeping the six initial states above and `T = 3000`.
Every verified finest-grid policy is included: 27 policies across 24 parameter
cases, producing 162 reputation paths. Each policy is used by both sellers
with symmetric mature initial reputations.

The figures are organized in the new
[15_initial_conditions subfolder](figures/15_initial_conditions/README.md),
with one subfolder per outside option and one PNG/PDF pair per retention value.
Each outside-option subfolder also has a four-page PDF. The
[combined PDF](figures/15_initial_conditions/initial_conditions_all_p0.pdf)
contains all 24 case figures. The original two-policy example above is preserved.

Each column uses one saved policy. Top panels show finite-time trajectories;
bottom panels show asymptotic phase-line outcomes. Dashed levels include all
stationary destinations of that policy, including optimal holding mixtures.
Mixed holding states are marked with diamonds. A single equilibrium policy
can have more than one realized attractor: the number of policy columns and
the number of distinct destinations answer different questions.

The underlying data are in `data/initial_conditions_sweep/`, with matching
`p0_*/rho_*/` directories. Each case contains a `summary.csv`, sampled
`trajectories.csv`, and a `manifest.json` recording the saved policies, source
hashes, actual-state selection checks, and ODE refinement. The
[combined summary](data/initial_conditions_sweep/summary.csv) separates terminal
reputation, asymptotic destination, and whether that destination agrees with
the policy's rival conjecture. Limits are determined from the scalar phase
line, not from a small finite-time slope.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/fluid-initial-conditions-sweep-mpl \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.sweep_initial_conditions

# Rebuild figures from matching, verified saved sweep data.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/fluid-initial-conditions-sweep-mpl \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.sweep_initial_conditions \
  --figures-only
```

The runner reuses case data only when inputs and calculation-source hashes
match. Every new path is repeated with tighter ODE tolerances and a smaller
maximum step. The original common-start results at `x0=.5` provide a saved
baseline check. No HJB, equilibrium conjecture, or main-study configuration is
recomputed or changed by this sweep.

