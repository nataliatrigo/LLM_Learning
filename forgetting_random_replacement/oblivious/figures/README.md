# Random-replacement forgetting: integer capacities 1 through 30

This campaign uses every integer memory capacity `C=1,...,30` directly. It compares monopoly and two-seller marginal/conditional OE at 9 outside options (`0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`), and extends marginal OE through five sellers at `p0=.1,.3` over `C=1,...,8`.

The published campaign contains **858 distinct certified cases**: 810 Experiment-8 cases plus 48 additional three-, four-, and five-seller cases. The original rho-matched numerical data remain in the parent folder's `data/` directory; its older figure exports have been retired.

Current comparison figures display Monopoly and Marginal OE. Conditional OE solutions remain in the saved data, tables, and numerical audits.

The five-seller extension is limited to the certified range `C=1,...,8`. Solver stress tests at larger capacities did not pass the current independent stationary-convergence checks. Those exploratory calculations are excluded from the published results; this does not establish that an equilibrium fails to exist. Details are retained in [exploratory solver checks](data/exploratory_solver_checks.json).

## Open the results

- [Results and selected tables](RESULTS.md).
- [Product 2 among own sales across C](figures/01_product2_own_sales_by_capacity.png).
- [Customer product 2 per period across C](figures/02_customer_product2_by_capacity.png).
- [Product 2 among own sales across the outside option](figures/06_product2_given_sale_by_p0.png).
- [Customer product 2 per period across the outside option](figures/07_customer_product2_by_p0.png).
- [All five seller counts across C](figures/09_product2_by_capacity_and_seller_count.png).
- [Monopoly policies at all capacities](figures/10_monopoly_policy_by_capacity.png).
- [Marginal OE policies at all capacities](figures/11_oblivious_policy_by_capacity.png).
- [Side-by-side policies at p0=.5](figures/policies/01_policy_comparison_p0_0p5.pdf).
- [Adjacent C5, C6, and C7 policies](figures/policies/02_adjacent_capacities_C5_C6_C7_p0_0p5.pdf).
- Policy maps using success counts: [Monopoly](figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf) and [Marginal OE](figures/policies/11_oblivious_policy_by_capacity_success_counts.pdf).
- [Policy comparison using success counts at p0=.1](figures/policies/01_policy_comparison_p0_0p1_success_counts.pdf).
- [Complete case summary](data/summary.csv), [figure data](data/figure_data.csv), [numerical validation](data/validation.json), and [figure manifest](figures/manifest.json).

All 31 figures are available in PDF. Policy comparisons are stored directly in `figures/policies/`, without subfolders or PNG copies. Calendar profit and actual discounted profit are also plotted against C.

## What C means

A mature seller remembers C binary observations. Its retained success count is `S=0,...,C`, with posterior `Beta(1+S,1+C-S)`. After a purchase, one old observation is chosen uniformly, independently of the new outcome, and replaced. Only the selected seller updates; outside-option selection freezes every memory.

The expected fraction retained per own purchase is `rho_effective=1-1/C`: it ranges from zero at C1 to 29/30 at C30. C changes the memory model, rather than refining a numerical approximation. Integer capacities can produce changes in optimal policy and nonmonotone outcome curves; no smoothing is applied.

The outside-option and policy comparison panels use these capacities:

| Capacity C | Effective retention rho |
|---:|---:|
| 1 | 0.000000 |
| 2 | 0.500000 |
| 3 | 0.666667 |
| 4 | 0.750000 |
| 5 | 0.800000 |
| 6 | 0.833333 |
| 7 | 0.857143 |
| 8 | 0.875000 |
| 10 | 0.900000 |
| 30 | 0.966667 |

The sharp peaks arise because adding one memory slot changes the available integer success counts, their posteriors, the optimal product choices, and the stationary probability of selling at each count. Product choices often switch between zero and one at an integer boundary. Straight lines connect separately solved capacities and make those discrete changes look pointed. See the adjacent-capacity example in [RESULTS.md](RESULTS.md).

The actual joint law is retained on the exact seller-permutation quotient, including the correlated law of all four rivals with five sellers. Independent rival marginals do not replace the joint distribution.

## Metrics and initialization

Primary product-2 use is `product2_conditional`, the percentage of own sales using product 2. Customer receipt per calendar period sums the actual product-2 flow across every seller: `N * product2_calendar`. Outside-option selection stays in the calendar denominator and contributes zero receipt. Profit flow and discounted payoff are per seller. Actual discounted payoff is evaluated in the real joint chain, separately from perceived OE value.

Discounted payoff uses the same symmetric mature initialization as the parent campaign: exact C/2 when C is even, and an independent equal floor(C/2)/ceil(C/2) mixture for each seller when C is odd. Both have initial mean posterior one half. The economic parameters remain `p1=.35,p2=.8,c1=.05,c2=.65,revenue=1,gamma=.98`.

## Validation and reproduction

Publication requires every configured case to pass the campaign audit. The figure generator verifies saved record/checkpoint hashes and the accounting identities for own-sales and customer-calendar product 2. `data/validation.json` records the numerical checks for each case. These are floating-point checks of finite exact-count models; they do not prove equilibrium uniqueness or fast convergence to the invariant law.

Run the campaign from the repository root:

```bash
MPLCONFIGDIR=/tmp/rr-capacity-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.capacity_sweep --stage all
```

To rebuild only these figures and English notes from the validated cases:

```bash
MPLCONFIGDIR=/tmp/rr-capacity-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.capacity_report
```
