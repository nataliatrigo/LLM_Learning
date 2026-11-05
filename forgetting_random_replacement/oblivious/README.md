# Oblivious equilibria with random-replacement forgetting

The current Experiment 8 campaign is in [figures/](figures/README.md). It solves every integer memory capacity `C=1,...,30` at outside options `.1,.2,.3,.4,.5,.6,.7,.8,.9`, with 858 certified cases including the `C=1,...,8` comparison through five sellers. Documentation and figure labels are in English.

## Open the results

- [Campaign description and reproduction](figures/README.md).
- [Results, interpretation, and numerical tables](figures/RESULTS.md).
- [Product 2 among own sales across capacity](figures/figures/01_product2_own_sales_by_capacity.png).
- [Product 2 among own sales across the outside option, plot 06](figures/figures/06_product2_given_sale_by_p0.png).
- [Customer product 2 per calendar period, plot 07](figures/figures/07_customer_product2_by_p0.png).
- Policies across all capacities: [Monopoly](figures/figures/10_monopoly_policy_by_capacity.png) and [Marginal OE](figures/figures/11_oblivious_policy_by_capacity.png).
- [Policy comparisons at p0=.5](figures/figures/policies/01_policy_comparison_p0_0p5.pdf).
- The same policies against integer successes S: [comparison at p0=.1](figures/figures/policies/01_policy_comparison_p0_0p1_success_counts.pdf), [Monopoly map](figures/figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf), and [Marginal OE map](figures/figures/policies/11_oblivious_policy_by_capacity_success_counts.pdf).
- [Current figure manifest](figures/figures/manifest.json).
- [Current numerical validation](figures/data/validation.json), [case summary](figures/data/summary.csv), and [saved policies](figures/data/policies.csv).

The current publication contains 31 figures across 40 files. Every figure has a PDF; 22 policy PDFs sit directly in `figures/figures/policies/`. Policy plots provide both the remembered success fraction `S/C` and the integer success count `S`, using the same saved actions. Selected-capacity comparisons use five rows and two columns for `C=1,2,3,4,5,6,7,8,10,30`. The capacity overviews and policy maps use three rows and three columns for the nine outside options. In the count maps, gray cells with `S>C` are infeasible. Figures show Monopoly and Marginal OE; Conditional OE remains in the numerical data and audits. Plots 05, 08, and 08b are excluded from the current publication.

## Model and timing

A mature seller remembers `C` binary observations, including `S` successes, with posterior `Beta(1+S,1+C-S)`. Consumers draw independently from sellers' posteriors and select the highest draw above the outside option. The selected seller chooses a product and receives a new observation; other sellers' memories remain unchanged.

After outcome `Y`, one old observation is selected uniformly and independently of `Y` and replaced:

```text
R | S ~ Bernoulli(S/C), independently of Y
S_next = S + Y - R
```

The economic parameters are `p1=.35`, `p2=.8`, `c1=.05`, `c2=.65`, `revenue=1`, and calendar-period discount `gamma=.98`. Count states and the correlated joint seller law are retained exactly. Changing `C` changes the memory model. The [campaign notes](figures/README.md) describe belief closures, outcome metrics, initialization, and validation.

## Reproduction

From the repository root, rebuild the current figures and English notes from the audited saved cases:

```bash
MPLCONFIGDIR=/tmp/rr-capacity-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.capacity_report
```

To run the complete integer-capacity campaign, use `python -m forgetting_random_replacement.oblivious.code.capacity_sweep --stage all` with the same environment settings. The [campaign README](figures/README.md) gives the full commands and scope.

## Historical numerical data

The earlier rho-matched campaign remains under [data/](data/summary.csv): 114 certified cases, comprising 108 Experiment 8 cases and six additional three-to-five-seller cases. Its reference retention values `.6,.7,.8,.9,.95,.97,.99` map to capacities `3,3,5,10,20,33,100`. The first two reference values therefore share the same capacity-3 solution. Historical [validation](data/validation.json), [policies](data/policies.csv), and [retention-indexed results](data/experiment8_by_reference_rho.csv) are preserved.

The historical plot exports were removed. Their publication entrypoints are retired; use the current `capacity_report` command above. Historical case records and numerical validation remain available.
