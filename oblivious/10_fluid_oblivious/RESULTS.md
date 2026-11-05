# Results: fluid monopoly versus fluid competition

The matched study uses 24 parameter combinations per model, with rho = 0.6, 0.7, 0.8, 0.9 and p0 = 0.1, 0.3, 0.5, 0.7, 0.8, 0.9. Both models use identical economic parameters, Beta posteriors, continuous-time calibration, state grids, and mature initial reputation 0.5.

- **Fluid monopoly:** 27 verified numerical stationary states across 24 cases; 3 cases have multiple states; 0 unresolved rows.
- **Fluid competition OE:** 27 verified numerical stationary states across 24 cases; 3 cases have multiple states; 0 unresolved rows.

The monopoly states are stationary states of its optimal policy. Competition states additionally satisfy the symmetric fixed-rival consistency equation. All branches are retained; no arbitrary pure branch is imposed.

## Matched example: p0=0.5, rho=0.7

| Fluid model | Stationary reputation | Product 2 / own sales | Own market share | Own seller & product 2 / period | Customer receives product 2 / period |
| --- | --- | --- | --- | --- | --- |
| fluid_monopoly | 0.554729 | 45.495% | 56.791% | 25.837% | 25.837% |
| fluid_competition | 0.60175 | 55.945% | 42.966% | 24.037% | 48.074% |

The primary measure is product 2 among own sales. The own-seller calendar measure is demand times that fraction. The secondary customer measure sums all sellers: one in monopoly, two in symmetric competition.

## Multiple stationary states

| Model | p0 | rho | Branch | Stationary reputation | Product 2 / own sales |
| --- | --- | --- | --- | --- | --- |
| fluid_monopoly | 0.8 | 0.9 | 0 | 0.35 | 0.000% |
| fluid_monopoly | 0.8 | 0.9 | 1 | 0.8 | 100.000% |
| fluid_monopoly | 0.9 | 0.7 | 0 | 0.35 | 0.000% |
| fluid_monopoly | 0.9 | 0.7 | 1 | 0.8 | 100.000% |
| fluid_monopoly | 0.9 | 0.8 | 0 | 0.35 | 0.000% |
| fluid_monopoly | 0.9 | 0.8 | 1 | 0.8 | 100.000% |
| fluid_competition | 0.8 | 0.9 | 0 | 0.35 | 0.000% |
| fluid_competition | 0.8 | 0.9 | 1 | 0.8 | 100.000% |
| fluid_competition | 0.9 | 0.7 | 0 | 0.35 | 0.000% |
| fluid_competition | 0.9 | 0.7 | 1 | 0.8 | 100.000% |
| fluid_competition | 0.9 | 0.8 | 0 | 0.35 | 0.000% |
| fluid_competition | 0.9 | 0.8 | 1 | 0.8 | 100.000% |

## Deterministic trajectories

Both models start at mature reputation 0.5 and run for 3000 continuous calendar units. Competition starts at (0.5,0.5); realized demand uses both actual reputations while its policy is optimized against the stationary conjecture. Monopoly uses the same outside option with one seller. Threshold events and holding mixtures avoid artificial oscillations.

Each monopoly case has one optimal policy and one common-start path even if that policy has several stationary states. Competition has one path for each equilibrium policy. A stationary state need not be reached from the common initial condition; terminal reputations and distances are saved separately.

Product-2 use in each time block is the ratio of integrated product-2 sales to integrated own sales. Calendar rates divide the integrated sales by block duration. There are no Monte Carlo confidence bands in this deterministic comparison.

## Verification and numerical sensitivity

Both models use state grids [1001, 2001, 4001]. Competition retains its full HJB optimality, holding-value, tight root-consistency, quadrature-doubling and independent demand audits. Monopoly checks the same HJB and stationary holding optimality, and refines its stationary states and policies across the same grids.

Competition maximum final-grid changes: reputation 0.000738, own-sale product-2 fraction 0.00164, own share 0.000346.

These are numerical optimality and sensitivity checks, not rigorous continuous-state error bounds or an exhaustive existence proof. The additional [near-boundary check](data/boundary_refinement.json) documents weaker relative precision for the small competition mixture at p0=.1, rho=.6.

## Files

- [All comparison figures](figures/overview/fluid_monopoly_vs_competition.pdf).
- [Primary product-2 comparison](figures/overview/02_primary_product2_given_sale_by_rho.png).
- [Secondary customer comparison](figures/overview/04_secondary_customer_product2_by_rho.png).
- [Policies, plot 11](figures/11_policies/), [reputation paths, plot 13](figures/13_reputation_trajectories/), and [transient metrics, plot 14](figures/14_transient_metrics/).
- [Matched summaries](data/comparison_summary.csv) and [all branch-pair differences](data/fluid_vs_fluid_differences.csv).
- [Competition diagnostics](data/manifest.json) and [monopoly diagnostics](data/monopoly_diagnostics.json).

Previous campaigns are preserved in data/archive. The active comparison uses only fluid monopoly and fluid competition at the four configured rho values.
