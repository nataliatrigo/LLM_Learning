# Random-replacement oblivious equilibrium results

The current results are in [figures/RESULTS.md](figures/RESULTS.md), with the campaign description and reproduction commands in [figures/README.md](figures/README.md).

The integer-capacity campaign contains **858 certified cases** over `C=1,...,30` and outside options `.1,.2,.3,.4,.5,.6,.7,.8,.9`, including the certified `C=1,...,8` comparison through five sellers at `p0=.1,.3`. Its current publication has 31 figures across 40 files. Plots 06 and 07 and the outside-option policy comparisons use `C=1,2,3,4,5,6,7,8,10,30` in five rows and two columns. The 22 [policy PDFs](figures/RESULTS.md#optimal-product-policies) are stored directly in `figures/figures/policies/`. Capacity overviews and policy maps show all nine outside options in three rows and three columns.

Policy plots now provide both `S/C` and integer `S` axes for the same saved actions. Compare the [fraction version at p0=.1](figures/figures/policies/01_policy_comparison_p0_0p1.pdf) with its [integer-count version](figures/figures/policies/01_policy_comparison_p0_0p1_success_counts.pdf). Integer-count maps are available for [Monopoly](figures/figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf) and [Marginal OE](figures/figures/policies/11_oblivious_policy_by_capacity_success_counts.pdf); gray `S>C` cells are infeasible.

Current figures show Monopoly and Marginal OE. Conditional OE results remain in the saved numerical data and audits. Plots 05, 08, and 08b are excluded; plot 07 is retained.

## Historical numerical results

The earlier rho-matched campaign retains **114 certified cases**: 108 Experiment 8 cases and six additional three-to-five-seller cases. Its numerical [summary](data/summary.csv), [policies](data/policies.csv), [retention-indexed results](data/experiment8_by_reference_rho.csv), and [validation](data/validation.json) remain under `data/`. The reference retention settings `.6,.7,.8,.9,.95,.97,.99` map to capacities `3,3,5,10,20,33,100`.

Historical plot exports were removed. Use the current [results](figures/RESULTS.md) for figures and interpretation, and `python -m forgetting_random_replacement.oblivious.code.capacity_report` to rebuild them.
