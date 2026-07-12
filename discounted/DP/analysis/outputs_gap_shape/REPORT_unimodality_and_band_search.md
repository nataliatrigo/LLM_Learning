# Continuation-Gap Unimodality and Band Search

## Method
Inspected `exact_dp.py`, `extinction_map.py`, and `local_extinction_certificate.py`. Demand uses the stable identity `P(Bin(S+F+1,p0)<=S)` via `betaincc`; regression tests agree with `scipy.stats.binom.cdf`. The solver is backward recursion of the rearranged discrete Bellman equation, with zero terminal value on diagonal `N_outer+1`. It is not value iteration. Maximum baseline Bellman residual: `1.421e-14`.

## Baseline
Tested diagonals 0--500 with `N_outer=1300`. Empirical extinction diagonal: `435`. Unimodality violations: `0`. Interval-property violations: `0`. Every active product-2 diagonal has one robust integer interval if the latter count is zero.

## Search
The structured design contains `175` cases and the fixed-seed randomized design contains `1000`, stratified across all three regimes. Every configuration was solved at `N_outer=180` and again at `280` on diagonals through `60`; gap and policy changes are recorded in `parameter_summary.csv`. All apparent violations were rerun on three grids with increments `max(500,ceil(10/(1-gamma)))`.

Confirmed robust unimodality violations: `0`. Confirmed robust interval-property violations: `0`. See CSV files for regime-level and candidate details.

Baseline outer-grid comparison (`1300` versus `1450`) gives maximum gap change `9.318e-08` and `0` policy changes on diagonals 0--500.

Across the broad first-pass search, `78` of `1175` configurations have at least one policy change between outer grids 180 and 280; these are concentrated among patient sellers and are not treated as fully converged policy estimates. The largest gap change was `5.31187`. Neither outer-grid solution generated a shape candidate, but absence of a counterexample in these high-gamma cases is weaker evidence than in converged cases.

```csv
regime,configurations,unimodality_violations,interval_violations,ambiguous_cases,negative_gap_cases,reappearance_cases,max_Bellman_residual
p0<p1<p2,383,0,0,0,0,1,1.4210854715202004e-14
p1<p0<p2,410,0,0,0,0,0,1.4210854715202004e-14
p1<p2<p0,382,0,0,0,0,0,1.4210854715202004e-14
```

Auxiliary checks found `0` configurations with a materially negative continuation gap and `1` cases where product 2 disappeared and later reappeared on the tested interior diagonals. Boundary-direction changes are exploratory and are recorded in `parameter_summary.csv`; they are not implied by within-diagonal unimodality.

The reappearance cases were:

```csv
parameter_id,p0,p1,p2,c1,c2,gamma,regime
p0547,0.12992748945459176,0.6811564293288664,0.7906164153154476,0.07432887025493344,0.2414091313529511,0.8669539440473999,p0<p1<p2
```

The single reappearance case was separately checked at `N_outer=180,680,1180`: product 2 is active on diagonals 6--29, absent on 30, active again on 31--33, and absent from 34 onward. This is robust and does not violate either within-diagonal property.

## Conclusion and theorem recommendation
Numerical results are evidence, not proof. No violation of either property was found, so attempting a proof of general unimodality is reasonable. The weakest economically relevant theorem supported by the evidence is the interval property; it should be targeted as a fallback if a unimodality proof fails. Near ties and terminal artifacts are not counted as counterexamples.

Concise answers: (1) no robust unimodality violation was found; (2) no robust interval-property violation was found; (3) this held in all three regimes; (4) the properties were empirically equivalent in this design because both always held, although they are not logically equivalent; (5) the weakest plausible theorem is the within-diagonal interval property. A separate theorem claiming monotone extinction across diagonals is not supported because robust disappearance and reappearance occurred once.
