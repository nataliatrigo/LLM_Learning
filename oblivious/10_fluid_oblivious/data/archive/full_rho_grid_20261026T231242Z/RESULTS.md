# Numerical results: symmetric fluid point-conjecture OE

The full study covers **42 parameter cases**. It retains **47 verified numerical equilibria** across 40 cases; **7 cases have multiple equilibria**. The search found no equilibrium in 2 cases. Unresolved solver/refinement rows: 0.

Verification refers to the declared finite-grid optimality, demand, consistency, and spatial-sensitivity checks. It is not a continuous-state existence or uniqueness proof.

## Cases with no equilibrium found

| p0 | rho | Search outcome |
| --- | --- | --- |
| 0.1 | 0.99 | no_equilibrium_found |
| 0.3 | 0.99 | no_equilibrium_found |

Both pure endpoints and the detected interior candidates are tested. Rejected candidates, holding-value defects, and scans are retained in [equilibrium_search.json](data/equilibrium_search.json). This outcome does not prove nonexistence.

## Multiple stationary branches

| p0 | rho | Branch | Type | x* | Product 2 / own sales | Own share |
| --- | --- | --- | --- | --- | --- | --- |
| 0.7 | 0.95 | 0 | pure_product1 | 0.350000 | 0.000% | 0.057% |
| 0.7 | 0.95 | 1 | mixed | 0.792497 | 98.333% | 47.576% |
| 0.7 | 0.97 | 0 | pure_product1 | 0.350000 | 0.000% | 0.001% |
| 0.7 | 0.97 | 1 | mixed | 0.780966 | 95.770% | 48.435% |
| 0.8 | 0.9 | 0 | pure_product1 | 0.350000 | 0.000% | 0.071% |
| 0.8 | 0.9 | 1 | pure_product2 | 0.800000 | 100.000% | 30.941% |
| 0.8 | 0.95 | 0 | pure_product1 | 0.350000 | 0.000% | 0.000% |
| 0.8 | 0.95 | 1 | pure_product2 | 0.800000 | 100.000% | 32.830% |
| 0.8 | 0.97 | 0 | pure_product1 | 0.350000 | 0.000% | 0.000% |
| 0.8 | 0.97 | 1 | pure_product2 | 0.800000 | 100.000% | 33.892% |
| 0.9 | 0.7 | 0 | pure_product1 | 0.350000 | 0.000% | 0.310% |
| 0.9 | 0.7 | 1 | pure_product2 | 0.800000 | 100.000% | 11.348% |
| 0.9 | 0.8 | 0 | pure_product1 | 0.350000 | 0.000% | 0.062% |
| 0.9 | 0.8 | 1 | pure_product2 | 0.800000 | 100.000% | 10.774% |

## Matched example: p0=0.5, rho=0.7

| Model | Fluid x* / stochastic calendar mean | Product 2 / own sales | Own share | Own seller & product 2 / period | Either seller & product 2 / period |
| --- | --- | --- | --- | --- | --- |
| monopoly | 0.518196 | 49.070% | 52.052% | 25.542% | 25.542% |
| competition | 0.540415 | 56.094% | 39.756% | 22.301% | 44.602% |
| conditional_competition | 0.544930 | 57.100% | 39.989% | 22.834% | 45.668% |
| fluid | 0.601750 | 55.945% | 42.966% | 24.037% | 48.074% |

The own-sale product mix and the calendar product-2 rate have different denominators. The stochastic reputation is an invariant marginal mean, not a deterministic stationary state.

## Numerical sensitivity

State grids: [1001, 2001, 4001]; the last grid uses 241 scalar-search points. The finest-pair maximum changes over matched branches are:

- equilibrium reputation: `0.000738312`.
- product-2 fraction among own sales: `0.00164069`.
- own market share: `0.000346282`.
- policy disagreement interval length: `0.00225`.

Every finest-grid response has passed full-grid quadrature doubling and selected independent adaptive demand audits. The analytical/demand/holding/event/accounting tests are run with the command in [README.md](README.md).

An additional [near-boundary refinement](data/boundary_refinement.json) studies p0=.1, rho=.6 at 8001 and 16001 nodes. The small positive product-2 fraction has weaker relative precision than typical interior cases; the primary sweep is not overwritten.

## Deterministic paths from the common initial condition

All 47 verified branches are simulated from mature (0.5, 0.5) for 3000 continuous calendar units. For 23 branches, the terminal reputation remains more than 0.002 from its conjectured stationary point. This can reflect a different basin of attraction or slow movement at low demand; it is not itself a failed stationary consistency check.

| p0 | rho | Branch | Conjectured x* | x(3000) | Terminal distance |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.97 | 0 | 0.741744 | 0.350000 | 0.391744 |
| 0.3 | 0.97 | 0 | 0.741744 | 0.350000 | 0.391744 |
| 0.5 | 0.97 | 0 | 0.741745 | 0.350559 | 0.391187 |
| 0.5 | 0.99 | 0 | 0.350000 | 0.543809 | 0.193809 |
| 0.7 | 0.95 | 0 | 0.350000 | 0.760442 | 0.410442 |
| 0.7 | 0.95 | 1 | 0.792497 | 0.406391 | 0.386105 |
| 0.7 | 0.97 | 0 | 0.350000 | 0.461608 | 0.111608 |
| 0.7 | 0.97 | 1 | 0.780966 | 0.461608 | 0.319358 |
| 0.7 | 0.99 | 0 | 0.350000 | 0.499942 | 0.149942 |
| 0.8 | 0.9 | 0 | 0.350000 | 0.398178 | 0.048178 |
| 0.8 | 0.9 | 1 | 0.800000 | 0.398178 | 0.401822 |
| 0.8 | 0.95 | 0 | 0.350000 | 0.483781 | 0.133781 |
| 0.8 | 0.95 | 1 | 0.800000 | 0.483781 | 0.316219 |
| 0.8 | 0.97 | 0 | 0.350000 | 0.499459 | 0.149459 |
| 0.8 | 0.97 | 1 | 0.800000 | 0.499459 | 0.300541 |
| 0.8 | 0.99 | 0 | 0.350000 | 0.500000 | 0.150000 |
| 0.9 | 0.7 | 0 | 0.350000 | 0.800000 | 0.450000 |
| 0.9 | 0.8 | 0 | 0.350000 | 0.392344 | 0.042344 |
| 0.9 | 0.8 | 1 | 0.800000 | 0.392344 | 0.407656 |
| 0.9 | 0.9 | 0 | 0.350000 | 0.488353 | 0.138353 |
| 0.9 | 0.95 | 0 | 0.350000 | 0.499969 | 0.149969 |
| 0.9 | 0.97 | 0 | 0.350000 | 0.500000 | 0.150000 |
| 0.9 | 0.99 | 0 | 0.350000 | 0.500000 | 0.150000 |

## Plots and comparison coverage

[Complete comparison PDF](figures/fluid_and_stochastic_comparison.pdf) · [Equilibrium map](figures/10_equilibrium_status.png) · [Primary metric](figures/02_primary_product2_given_sale_by_rho.png) · [Secondary metric](figures/04_secondary_customer_product2_by_rho.png)

Audited stochastic common-grid references cover all six outside options at rho=.6,.7,.8. Higher-rho reference gaps are explicit. The fluid study covers the full seven-rho grid. Saved rho=.7 Monte Carlo flow histories use the same mature initial condition and 25-period blocks; the deterministic fluid curves have no simulation confidence bands. Stochastic reputation bands show stationary mean ±1 SD, not unavailable transient histories.

Paired differences, source hashes, and all three product-2 measures are in [fluid_vs_stochastic_differences.csv](data/fluid_vs_stochastic_differences.csv).
