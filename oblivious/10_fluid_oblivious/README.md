# Fluid monopoly versus symmetric fluid oblivious competition

The main experiment compares **two fluid models** with the same outside option,
Thompson Sampling demand, Beta posteriors, economic parameters, continuous-time
calibration, state grids, and initial reputation. It is implemented separately
from the stochastic models whose demand functions it reuses.

## Shared primitives and demand

The normalized mature reputation is `x=(1-rho)S` in `[0,1]`, with

```text
theta(x) ~ Beta(1+x/(1-rho), 1+(1-x)/(1-rho)),
D_M(x;p0) = Pr(theta(x)>p0),
D_A(x,z;p0) = integral[p0,1] f_x(t) F_z(t) dt,
D_B(x,z;p0) = D_A(z,x;p0),
D_out(x,z;p0) = F_x(p0) F_z(p0).
```

The economic parameters are `R=1`, `c1=.05`, `c2=.65`, `p1=.35`, `p2=.8`,
`gamma=.98`. The active grid is `p0=.1,.3,.5,.7,.8,.9` and
**`rho=.6,.7,.8,.9`**, giving 24 parameter cases per fluid model.
Values `rho>=.95` are excluded from the active study. Earlier campaigns are
preserved in `data/archive/`.

Both models use `lambda=-log(rho)` and `delta=-log(gamma)`. A fixed-rival
competition response solves

```text
delta U(x;z) = D_A(x,z) max_i {R-ci + lambda*(pi-x)*U_x(x;z)}.
```

The monopoly solves the same HJB with `D_M(x)` in place of `D_A(x,z)`.
Both reuse the existing monotone upwind Howard solver in
`forgetting/Code/fluid_convergence/numerics.py`.

## Stationary states, equilibrium, and verification

Competition conjectures a fixed rival reputation `z=x*`, rather than a rival
stationary distribution. Its optimal policy must satisfy
`x*=p1+(p2-p1)*alpha(x*)`. Both pure endpoints and mixed stationary candidates
are considered. A mixed holding fraction is `(x*-p1)/(p2-p1)`.

A smooth interior candidate obeys the necessary condition

```text
D_x(z,z)*[R-c1-(c2-c1)*(z-p1)/(p2-p1)]
  - (c2-c1)/(lambda*(p2-p1))*[delta+lambda*D_A(z,z)] = 0.
```

`D_x` holds the rival fixed. This equation only generates candidates. Each
candidate is checked against the complete optimized HJB, the holding payoff,
conditional action/holding defects, and tight response-root consistency.
A policy crossing or a repelling threshold alone does not establish an
optimal holding mixture. All verified competition equilibrium branches are
retained; failure to find an equilibrium is reported separately from a solver
error and is not a proof of nonexistence.

For monopoly, all stationary states of its globally optimal own-state policy
are enumerated: pure endpoints where optimal and valid holding mixtures.
A threshold below `p1` or above `p2` is not automatically a stationary state.
The monopoly can have multiple optimal stationary states under the same
policy; all are reported.

Both models run on 1001, 2001, and 4001 state nodes. Competition additionally
refines its scalar search from 121 to 241 points. Finest-pair sensitivity
limits are `.002` in reputation, `.005` in product-2 fraction, `.002` in own
market share, and `.005` in policy disagreement interval length. Numerical
HJB residuals and conditional holding defects are checked separately so rare
sales do not hide a poor action. Competition demand is checked by full-grid
quadrature doubling and selected independent scaled adaptive integration.
These checks measure numerical accuracy and sensitivity; they are not rigorous
continuous-state error bounds.

## Primary and secondary metrics

At a stationary state, write `d` for own demand, `alpha` for the product-2
fraction, and `N` for seller count (1 in monopoly, 2 in competition).

| Quantity | Definition |
| --- | --- |
| Primary: product 2 among own sales | `alpha` |
| Own seller selected and product 2 per calendar unit | `d*alpha` |
| Secondary: customer receives product 2 from any seller | `N*d*alpha` |
| Own market share | `d` |
| Outside-option market share | `1-N*d` |

The customer metric counts the single monopolist once and sums both sellers
in symmetric competition. Percentages among own sales and calendar rates
have different denominators. Reputation comparisons use deterministic fluid
stationary states in both models. There are no stochastic invariant-law or
Monte Carlo series in the active comparison.

Plots use blue solid lines for **fluid monopoly** and orange dashed lines for
**fluid competition**. All stationary branches are shown. Paired differences
include every competition/monopoly branch pair rather than selecting one
branch arbitrarily. Isolated holding mixtures have zero Lebesgue mass in
policy-distance calculations and are marked separately on policy plots.
Each monopoly policy panel draws its single strategy `sigma(x)` once and
marks every stationary state on that curve. Competition can have different
strategies for different equilibrium conjectures.

## Deterministic reputation paths

The monopoly starts at mature reputation `x(0)=.5`; competition starts at
`(x_A(0),x_B(0))=(.5,.5)`. Both run for 3000 continuous calendar units with
25-unit usage blocks. The realized competition dynamics use both actual
reputations, while the policy remains the response to its fixed conjecture:

```text
xdot_M = lambda*D_M(x_M)*(q_sigma(x_M)-x_M),
xdot_A = lambda*D_A(x_A,x_B)*(q_sigma(x_A)-x_A),
xdot_B = lambda*D_A(x_B,x_A)*(q_sigma(x_B)-x_B).
```

The event-driven integrator crosses policy thresholds or holds using the
optimal mixture, without artificial oscillation. Integrated counters produce
`integral(d*alpha)/integral(d)` among own sales and
`integral(d*alpha)/block_duration` per calendar unit. Conditional fractions
with no sales remain undefined. Realized discounted profits are integrated
along actual paths.

There is one monopoly policy/path per parameter case, even when it admits
several stationary states. Competition has a separate path for each
equilibrium policy. Stationary consistency does not imply that a state will
be reached from the common initial condition; terminal states and distances
are reported separately.

The [initial-condition comparison](INITIAL_CONDITIONS.md) applies the two saved
policies at `p0=.8, rho=.9` to six symmetric mature starts, including `.55`
between their repelling thresholds. Its separate trajectory figure and
phase-line panels distinguish finite-time states from asymptotic destinations.
The [full initial-condition plot folder](figures/15_initial_conditions/README.md)
extends the comparison to all six active outside options and all four active
retention values, with one subfolder per outside option and combined PDFs.

## Run

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.run_experiment

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious.code.run_experiment --figures-only

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover \
  -s oblivious/10_fluid_oblivious/code -p 'test_*.py' -v
```

Competition checkpoints resume when scientific inputs and source hashes
match. The fluid monopoly benchmark is rebuilt with the same configuration.
Use `--config` and `--output-dir` for another campaign, or `--recompute` to
recalculate competition checkpoints explicitly.

## Outputs

- `data/comparison_summary.csv`, `comparison_policies.csv`, and
  `fluid_vs_fluid_differences.csv`: matched fluid results and all branch-pair
  comparisons.
- `data/fluid_summary.csv`, `fluid_policies.csv`, `grid_refinement.csv`,
  `equilibrium_search.json`, and `checkpoints/`: competition results/audits.
- `data/monopoly_summary.csv`, `monopoly_policies.csv`, and
  `monopoly_diagnostics.json`: optimal fluid monopoly stationary states,
  refinement, and checks.
- `data/trajectories.csv`, `trajectory_bins.csv`: competition paths/flows;
  `monopoly_trajectories.csv`, `monopoly_trajectory_bins.csv`: monopoly paths/flows.
- `figures/overview/`: primary/secondary metrics against both `p0` and `rho`,
  stationary states, market shares, and the original initial-condition example.
- `figures/11_policies/`: plot 11, optimal policies by outside option.
- `figures/13_reputation_trajectories/`: plot 13, deterministic reputation paths.
- `figures/14_transient_metrics/`: plot 14, demand and product-2 metrics over time.
  Plot 05 is excluded; plot 06 retains the own-seller product-2 calendar metric
  against `rho`.
- `figures/overview/fluid_monopoly_vs_competition.pdf`: all comparison figures.
- `data/manifest.json`, `figures/report_manifest.json`, and
  `data/independent_validation.json`: configuration, provenance, and checks.

See [RESULTS.md](RESULTS.md) for the completed comparison.

The [common-initial-state comparison](MONOPOLY_ABOVE_COMPETITION.md) identifies
all cases where monopoly reaches a higher long-run product-2 fraction than
competition from the same initial reputation. It includes a marked copy of the
comparison plot and additional threshold refinement without changing the
historical configurations, results, or figures.

The [reverse comparison](COMPETITION_ABOVE_MONOPOLY.md) lists every common-start
case where competition reaches a higher product-2 fraction than monopoly,
distinguishing this from the extreme comparison of 100% versus 0%.
