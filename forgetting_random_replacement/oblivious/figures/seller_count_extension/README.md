# More sellers and outside options: extended plot 08

This separate campaign extends the own-sales product-2 comparison while preserving the exact correlated rival distribution and the numerical acceptance criteria.

The available-case audit passed for 923/928 requested cases. The remaining 5 cases are not published as results; see data/validation.json for missing or failed cases. A passing available-case audit is not complete requested coverage.

Requested finite ranges: C=1: N=1–100, C=3: N=1–50, C=5: N=1–10, C=8: N=1–10.

Outside options: `.1, .3, .5, .7, .8, .9`. Monopoly is N=1; N≥2 uses marginal OE.

- [Extended plot 08](figures/08_product2_by_seller_count.png) ([PDF](figures/08_product2_by_seller_count.pdf)).
- [Finite observations](RESULTS.md) and [near-zero summary](data/near_zero_summary.csv).
- [A conservative eventual-zero proof](THEORY.md).
- [Numerical audit and coverage](data/validation.json), [plotted grid including gaps](data/figure_data.csv), and [figure hashes](figures/manifest.json).

The gray band denotes product 2 in at most 1% of own sales. Missing or rejected points remain gaps. N ranges differ between capacity panels because the exact quotient contains binomial(N+C,C) states.

Markers identify the seller counts actually solved; connecting lines guide the eye. For C3 the grid is dense through N30, followed by the additional checks at N35, N40, and N50.

This is not a per-seller calendar-flow graph. The latter is bounded by 1/N mechanically and cannot answer whether the product mix among own sales vanishes.

## A selected positive-to-zero example

An exploratory search near the investment cutoff selected `C=3, p0=0.8805`. All ten cases N1–10 are separately audited. This extra outside option is highlighted in black only in the C3 panel; it was not part of the pre-specified six-value outside-option grid.

- [Own-sales decline and actual calendar demand](figures/08b_illustrative_decline_and_demand.png) ([PDF](figures/08b_illustrative_decline_and_demand.pdf)).
- [Selected-example data](data/illustrative_decline.csv).

It illustrates a possible decline, not a result for every outside option, a controlled causal comparison, or a claim of equilibrium uniqueness.

## Reproduction

From the repository root:

```bash
MPLCONFIGDIR=/tmp/rr-seller-count-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_sweep --stage solve
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_audit
MPLCONFIGDIR=/tmp/rr-seller-count-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_report
```
