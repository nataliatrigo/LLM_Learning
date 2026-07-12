# BinaryBelief — 1-D hidden-product-choice reputation model

A small, reproducible simulation suite for the 1-D binary-belief reputation
model. The user's belief `pi = P(product is the high one)` is the entire state:
in log-odds `ell = log(pi/(1-pi))` the Bayes update is additive with constant
increments `dS = log(p2/p1) > 0` (success) and `dF = log((1-p2)/(1-p1)) < 0`
(failure). The seller solves

```
V(ell) = gamma*(1-D(ell))*V(ell)
       + D(ell) * max_x { R - c_x + gamma*[ p_x V(ell+dS) + (1-p_x) V(ell+dF) ] }
```

by value iteration on the fixed-point form `V = D/(1-gamma(1-D)) * M`, on a
uniform `ell`-grid (N = 2000 over `[logit(1e-3), logit(1-1e-3)]`, linear
interpolation of the shifted values, boundary clamping, sup-norm tol 1e-9).
Product 2 is optimal iff the continuation gap
`g(ell) = gamma*[V(ell+dS) - V(ell+dF)]` exceeds the constant threshold `dc/dp`.

Three user decision rules, each treated as a first-class experiment:

- **TS** (Thompson sampling): `D = pi`
- **EG** (epsilon-greedy): `D = eps + (1-eps)*1{pi >= pibar}`,
  `pibar = (p0-p1)/(p2-p1)` — the indifference **cliff**
- **LOGIT** (smooth cliff): `D = sigmoid(beta*(p1 + pi*dp - p0))`; recovers the
  EG cliff (with `eps = 0`) as `beta -> inf`

## Run

```
python run.py               # all configs in config.CONFIGS; sweeps for gamma <= 0.99
python run.py --sweeps-all  # also run the (slower) sweeps for gamma = 0.999
```

Requires numpy + matplotlib (the repo's `.venv` has both:
`../../.venv/bin/python run.py`). Deterministic: seeded RNG, config-driven.
Band edges `[pi_-, pi_+]` for TS/EG/LOGIT and all sanity checks are printed to
stdout; a non-zero exit code means a *structural* check failed (convergence,
threshold consistency). Everything else (single interval, band brackets the
cliff, parking phenomenology) is reported as PASS/WARN flags with explanations.

## Layout

```
config.py        frozen dataclass Config + the CONFIGS tuple actually run
src/model.py     demand rules, Bellman solver, band extraction, Monte Carlo, fluid ODE
run.py           styling, all figures, sanity checks, sweeps
<outdir>/plots/
    TS/          per-rule figures for Thompson sampling
    EG/          per-rule figures for epsilon-greedy
    LOGIT/       per-rule figures for smooth logit + beta_sweep.png
    band_vs_p0.png, band_vs_params.png   cross-rule comparative statics
```

One `<outdir>` per entry in `CONFIGS` (currently `outputs/` at gamma = 0.95 and
`outputs_gamma_{070,080,085,090,0999}/`).

## Figures

Per rule, in `plots/<RULE>/`:

| file | content |
|---|---|
| `value_and_gap.png` | **Key plot.** Top: `V(pi)`. Bottom: `g(pi)` with the horizontal threshold `dc/dp`; the product-2 band `[pi_-, pi_+]` is shaded in both. Staircase kinks come from the outcome lattice; EG has a jump at the cliff. |
| `policy.png` | Step plot of `x*(pi)` with the band shaded. |
| `demand.png` | `D(pi)` with the cliff `pibar(p0)` marked (belief at which expected quality equals `p0`); the LOGIT variant overlaid dashed for reference. |
| `montecarlo.png` | Three panels. Left: one sample path per start (faceted — overlaid paths orbit the same attractor and tangle), first 200 periods, band shaded; flat stretches are periods without engagement. Middle: ensemble over time — fraction of paths alive (`pi > 0.05`), inside the band, and above the cliff; the decay of these curves is the metastability/ruin story. Right: histogram of the stationary `pi` (last 25% of periods) plus an outline of the occupation density conditional on survival, which reveals the parking mass when ruin dominates. |
| `fluid_check.png` | Fluid check: trajectories of `d(ell)/dn = mu1 + a*(ell)*kappa` with the relaxed control `a* = 1{x*=2}` from the DP, overlaid on the band. Interior starts converge to the upper band edge (the parking / singular-arc attractor); starts below `pi_-` collapse to 0. |
| `combined_panel.png` | All of the above in one panel. |

Cross-rule, in `plots/`:

| file | content |
|---|---|
| `band_vs_p0.png` | **Money plot.** Band edges vs `p0` for TS, EG and LOGIT on the same axes, with `pibar(p0)` dashed. EG and LOGIT track the cliff (`p0` enters their demand); the TS band is exactly `p0`-free (`p0` does not enter the TS HJB, so TS is solved once and drawn flat). |
| `band_vs_params.png` | Small multiples: band edges vs `gamma`, vs `eps` (EG only), vs `dc/dp` (sweeping `c2`), and vs distinguishability `Lambda = dS - dF` (sweeping the spread `p2 - p1` around a fixed midpoint, holding `dc/dp` fixed so distinguishability is not confounded with the cost threshold). Dotted hairline = base-config value. Gaps in a line mean the band is empty there. |
| `LOGIT/beta_sweep.png` | **Extra LOGIT experiment.** Band edges vs the demand slope `beta` (log scale): as `beta` grows the LOGIT band sharpens toward the EG cliff band (dashed reference); for small `beta` demand is too flat for reputation to pay and the band disappears. |

## Findings worth knowing before reading the plots

With the current economics (`dc/dp = 1.6`) the band structure is genuinely
interior and regime-dependent:

- **gamma = 0.95** (`outputs/`): TS `[0.03, 0.96]`, EG `[0.09, 0.92]`,
  LOGIT `[0.06, 0.93]`. Beliefs park at the **upper band edge** `pi_+` (the
  point where the seller stops paying `dc`), not at the cliff — the seller
  holds a multi-failure reputation buffer.
- **gamma = 0.90 / 0.85 / 0.80**: the bands tighten around the cliff; at 0.80
  the TS band is **empty** (a Thompson user never makes quality worth 1.6 per
  period) while EG still supports `[0.47, 0.70]` — note it does **not** bracket
  `pibar = 0.4`: below the cliff engagement is too rare (`D = eps`) to be worth
  investing, so the seller only defends reputations already above the cliff.
  The EG upper edge `0.70` is *exactly one failure above the cliff*
  (`logit(0.7) + dF = logit(pibar)`).
- **gamma = 0.70**: all bands empty — quality is never bought.
- **Stochastically, interior bands are metastable.** Paths hug the band for a
  while, but a run of failures pushes the belief below `pi_-`, the seller
  switches to the cheap product forever, and the belief is absorbed near 0
  (reputational ruin — the dominant long-run outcome for gamma <= 0.90 here).
  The stationary histograms therefore pile up at 0; the violet
  occupation-conditional-on-survival outline shows the parking mass.
- The DP solver was cross-checked against Monte-Carlo policy evaluation
  (discounted-return average matches `V` within sampling error).

## Sanity checks (printed each run)

Fatal: value iteration converged under tol; the argmax policy coincides with the
constant-threshold rule `g >= dc/dp`. Flags (PASS/WARN): product-2 region is a
single interval (component count reported otherwise); EG band brackets `pibar`;
surviving MC mass parks at `pi_+`; EG parking near the cliff (reported with the
buffer size in units of failures and the ruin fraction).

## Note on the fluid drift sign

The fluid ODE is implemented as `d(ell)/dn = mu1 + a*kappa` with
`mu1 = p1*dS + (1-p1)*dF < 0` (drift per engagement under product 1) and
`kappa = dp*Lambda`, so that `a = 1` gives `p2*dS + (1-p2)*dF > 0`. The task
note's `mu0 = -(p1*dS + (1-p1)*dF)` has the opposite sign, which would make the
belief drift *up* under the low product — contradicting Bayes learning — so the
`+mu1` convention is used.

## Current default parameters

`p1=0.3, p2=0.8, p0=0.5, R=1.0, c1=0.05, c2=0.85, gamma=0.95, eps=0.10, beta=25`
→ `dS=0.9808, dF=-1.2528, dc/dp=1.6, pibar=0.4, Lambda=2.234`. Edit
`config.py` (or `dataclasses.replace`) to sweep.
