# Regime experiments

## Scope and solver audit
The project imports `ModelParams`, `beta_ccdf`, `required_horizon`, and
`solve_discounted_finite_horizon` from `discounted/DP/exact_dp.py`.  That
routine is exact backward induction for a finite terminal truncation with zero
value on its terminal layer.  It is not a stationary fixed-point routine.
Here `analysis_outer_diagonal` is the largest displayed history length and
`remaining_tail` is the number of discounted periods after that analysis
slice.  Results are called stationary approximations only after comparing two
tails/grids on a common interior.  The old `t=50` label is a calendar-period
policy slice, not “50 periods remaining.”

Demand is the exact Thompson-sampling probability `betaincc(S+1,F+1,p0)`.
No fluid approximation or Monte Carlo calculation is used.

## Design
Experiment A varies `p0` over 37 values from .05 to .95. Experiment B uses six
quality pairs and seven stratified outside-option locations. Experiment C uses
three regime-representative triples with a stratified design in patience and
incremental cost. This is the full configured run.

Regime I gives both products positive observation-time drift relative to `p0`;
Regime II gives positive drift only to product 2; Regime III gives both
products negative drift.

## Drift identities
For `Z=S+1-p0(S+F+2)`, direct enumeration verifies
`E[Z'-Z|A,x=i]=p_i-p0`. Multiplication by demand gives the calendar-time drift
`D(S,F)(p_i-p0)`. Maximum numerical identity error is
`3.109e-15`.

## Convergence
Every row reports a Bellman residual, common-interior policy disagreement, and
maximum continuation-gap change under a longer tail/larger analysis slice.
Only configurations marked `converged` support firm policy conclusions.
Empirical extinction is reported only when the last active diagonal lies
strictly inside the reliable interior.
Any `last_active_diagonal` that reaches the reliable-grid limit is marked
`last_active_censored=True`, and `reported_last_active_diagonal` is left blank.
After extending 65 boundary cases in Regimes I and II to an interior of 180,
54 of those extended cases remain censored. Including nine boundary cases in
Regime III that were not part of this targeted extension,
`63` configurations are censored overall.

All `106` of `106`
configurations have identical policies across the two standard truncations.
The largest Bellman residual is `8.667e-08`. The
stricter continuation-gap flag is satisfied by `106`
configurations; the largest recorded `max_G_change` is
`5.344e-06`. The separate three-grid table confirms no
policy changes for the baseline and five representative cases.
For the enlarged `n<=180` analysis, `extended_tail_stability.csv` compares
outer/tail pairs `(300,1e-7)` and `(340,1e-8)` in representative Regime I and
II cases. All four comparisons have zero policy changes; the largest
continuation-gap change is `3.581e-06`.

## Policy-region diagnostics
Robust interval-property violations: `0`.
Active-diagonal reappearances: `0`.
Tied states do not count as confirmed separators. These findings are empirical
regularities, not theorems.

## Reachability and occupancy
The mean absolute difference between geometric product-2 state share and
embedded-reach-weighted share is
`0.1880`.
Thus heatmap area is materially different from economic exposure. Discounted
calendar-time occupancy uses the self-loop-adjusted recursions, whereas
reachability is indexed by Seller-A observations.
The occupancy terminal-discount flag passes for
`63` of `106` cases.
The remaining cases (the most patient truncations) retain their occupancy
metrics as boundary-limited diagnostics rather than fully confirmed values.
Unlike the undiscounted intervention count, `discounted_product2_uses=H2(0,0)`
is computed for every `p0` and is the usage measure shown in Figure 3.

The reachability-weighted standardized investment centers by regime are
`{'I: p0<p1<p2': 1.0602, 'II: p1<p0<p2': 0.8027, 'III: p1<p2<p0': -0.6097}`.
They do **not** support the initially proposed sign pattern. In this design the
center is positive on average in Regimes I and II and negative in Regime III.
Figure 7 reports the dense-`p0` pattern directly; this contradiction is retained
rather than forcing the anticipated interpretation.

## Main numerical findings
Among the sampled designs, **II: p1<p0<p2** has the largest mean incremental
value of access to product 2, and **II: p1<p0<p2** has the largest mean discounted
number of product-2 uses. In dense `p0` comparative statics, the
discounted product-2 share peaks at `p0=0.800`, in regime
**III: p1<p2<p0**. These are design-conditional comparisons, not causal regime
effects.

Mean incremental values are
`{'I: p0<p1<p2': 4.2448, 'II: p1<p0<p2': 11.3275, 'III: p1<p2<p0': 0.9219}`.
Mean probabilities of ever reaching a product-2 state are
`{'I: p0<p1<p2': 0.8752, 'II: p1<p0<p2': 0.9778, 'III: p1<p2<p0': 0.7625}`.
Mean frontier-crossing probabilities (excluding `p0=.5`) are
`{'I: p0<p1<p2': 0.1352, 'II: p1<p0<p2': 0.7484, 'III: p1<p2<p0': 0.3085}`.

Across the heterogeneous design, the descriptive correlations of incremental
value with `p1-p0`, `p2-p0`, and `Delta_p` are respectively
`{'p1_minus_p0': 0.13, 'p2_minus_p0': 0.201, 'Delta_p': 0.177}`.
Because the designs differ across regimes, these correlations are descriptive
and are not interpreted causally. Investment-region width is more strongly
associated with incremental value in this sample, with correlation
`0.719`.

The undiscounted intervention count is blank unless extinction is confirmed
away from the grid boundary. It is available for `42` of
`106` configurations. Within that restricted subset, regime means
are `{'I: p0<p1<p2': 5.6731, 'II: p1<p0<p2': 53.925, 'III: p1<p2<p0': 6.4373}`; this subset comparison should not be
generalized to boundary-limited configurations.

## Cost and patience robustness
Mean incremental value by `gamma` is `{0.8: 0.0003, 0.9: 0.5505, 0.95: 1.6206, 0.98: 7.6567, 0.99: 17.0168}`;
mean discounted product-2 share is `{0.8: 0.0076, 0.9: 0.3304, 0.95: 0.3536, 0.98: 0.4165, 0.99: 0.4986}`.
Value rises strongly with patience, but share need not rise monotonically
because patience also adds later engagements served with product 1.

Mean value by incremental cost is `{0.2: 6.4636, 0.6: 5.183, 0.9: 2.2086}` and mean
share is `{0.2: 0.6281, 0.6: 0.3157, 0.9: 0.1329}`. Both decline as product 2 becomes
more expensive in this stratified design.

## Interpretation
Regime I allows product 1 itself to build reputation; product 2 is therefore an
accelerator. Regime II makes product 2 the only action with positive expected
reputational drift. Regime III gives both actions negative drift, so product 2
can at most slow deterioration. The value, intensity, and duration columns in
the tables quantify rather than assume these narratives.

All interval and reappearance findings are empirical numerical regularities.
The analytical localization and extinction results remain the only proved
claims. A useful numerical-section recommendation is to lead with the
reachability-weighted regime panels and dense `p0` comparative statics, while
placing raw `(S,F)` heatmaps and standardized-distance plots in robustness.

## Candidate conjectures and paper recommendation
The absence of robust multi-component diagonals suggests an interval-property
conjecture, but does not prove it. The concentration of value and interventions
in Regime II suggests that product 2 matters most when it is the only
positive-drift action. The numerical section should lead with reachability-
weighted regime panels and dense `p0` comparative statics; raw heatmaps and
standardized-distance plots are better robustness exhibits.

## Evidence classification
- **Proved in the paper:** localization and eventual extinction.
- **Exact within each truncation:** Bellman, reachability, and occupancy recursions.
- **Empirical:** interval geometry, no reappearance, comparative statics.
- **Managerial:** accelerator in I, reputation-building action in II, and
  deterioration-slowing action in III.

## Outputs
Tables are in `outputs/tables`, selected state policies in `outputs/policies`,
and figures in `outputs/figures`. `primary_convergence.csv` contains the
required three-grid confirmations, and `extended_tail_stability.csv` records
the additional long-tail checks on the enlarged interior.
