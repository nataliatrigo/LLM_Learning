# Monopoly (forgetting) versus competition (OE p0)

[Índice de experimentos oblivious](../../../README.md).

This comparison solves the **original monopoly problem in `forgetting.Code.model.solve_boundary`** at the saved OE p0 benchmark for each rho. The competition policies, equilibrium benchmarks, joint stationary distributions, and Monte Carlo results are reused from `oblivious/02_oe_p0` and `oblivious/06_equilibrium_comparison`; the competition problem is not solved again. The monopoly benchmark stays fixed: there is no new p0 consistency condition.

All cases use the same 51-point mature-state grid as the saved competition results, rho in `{0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999}`, and the original shared primitives. This grid choice makes the comparison consistent with the existing `Full comparison` figures. It is not a grid-convergence claim.

## Reproduce

Run from the repository root:

```bash
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/run_experiment.py
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/plot_comparison.py
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/test_simulation.py
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/audit_finite_variance.py
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/plot_total_profit.py
.venv/bin/python -B oblivious/07_full_comparison/Monopoly_vs_competition/code/plot_equilibrium_p0.py
```

The first command solves and checks the monopoly cases, simulates them, and writes `data/`. The second writes the first three figures in PNG and PDF to `Figures/`. The third and fourth run behavioral checks and audit finite-window sampling variability. Source results are read-only. `simulation.py` implements the monopoly simulation and exact finite-window expectations. `Figures/` contains only figures; tabular results, arrays, diagnostics, and provenance are in `data/`.

The fifth command evaluates and plots **total discounted profit per seller**, the requested comparison from initial reputation 0.5. It creates `Figures/04_total_discounted_profit_across_rho.{png,pdf}`, `data/total_discounted_profit.csv`, full-state values in `data/total_discounted_profit_arrays.npz`, and definitions, numerical checks, and provenance in `data/total_discounted_profit_manifest.json`.

The final command plots the **selected equilibrium p0 for each rho** for both OE p0 and following monopoly (OE quality), using their saved 51-point-grid solutions. It verifies the sixteen values against their original source files and creates `Figures/05_equilibrium_p0_across_rho.{png,pdf}`, `data/equilibrium_p0_across_rho.csv`, and `data/equilibrium_p0_manifest.json`. Values are annotated on the figure. The two benchmarks have different consistency definitions: OE p0 uses the stationary mean rival TS score; following monopoly uses the stationary true rival quality conditional on rival sales. The figure notes the observed OE p0 grid sensitivity at rho=0.97 and 0.99; it makes no uniqueness claim.

## Total discounted profit

Figure 04 plots `E[sum_{t=0}^infinity gamma^t * profit_t]` with `gamma=0.98`, including the initial period and zero-profit periods when the seller is not selected. It starts the monopoly at x=0.5 and both competing sellers at (x_A,x_B)=(0.5,0.5). Values are per seller, not industry totals, and are not normalized by `1-gamma`.

For monopoly, this is the optimal value of the original forgetting problem at the saved equilibrium benchmark p0(rho). For competition, it is the value of **following the saved OE p0 policy under actual demand between the two sellers**. We solve `V_j = r_j + gamma * P * V_j` on the joint state grid; this evaluates the existing policy and does not reoptimize it or solve a full-state equilibrium. Symmetry makes both sellers' initial expected values equal.

Figure 04 also includes **following monopoly (OE quality)** in purple. This third curve evaluates the saved policy from `oblivious/03_oe_following_monopoly/data/quality_sales_n51` under the same actual two-seller demand, discount factor, and initial state. It uses that model's own quality-consistent benchmark and policy. It is not a second one-seller monopoly calculation. The CSV includes its p0, realized-environment discounted value for both sellers, and its perceived fixed-benchmark Bellman value in separate columns; full-state values and the saved policy are in the NPZ. Source hashes, equilibrium consistency, Bellman residuals, forward discounted sums, and symmetry are checked for all eight cases.

The native OE p0 solver instead computes a perceived value against a fixed scalar benchmark. That Bellman problem matches the monopoly problem, so its value coincides with monopoly; it is retained in the CSV as `oe_p0_perceived_bellman_value` for reference, but is not the competition curve. The following-monopoly source has an analogous perceived value at its own benchmark, retained as `following_monopoly_perceived_bellman_value`; its plotted curve also uses actual competition instead.

This figure needs neither simulation nor convergence to stationarity. Each linear solve is checked against a separate 2,000-period forward discounted sum, with a bound on the remaining discounted tail, Bellman residual bounds, and seller symmetry. The monopoly value also matches the saved original forgetting solver. These checks apply to the common 51-point numerical model; they do not remove possible approximation error in the grid or the saved equilibrium p0. Figure 03 retains the earlier average-profit comparison and measures a different quantity.

The principal data files are:

- `policies.csv`: each model's action probability at every rho and grid node.
- `outcomes.csv`: stationary predictions, exact finite-window expectations, and Monte Carlo estimates with intervals.
- `replications.csv`: per-path sale counts and outcomes; the conditional-usage plot pools these counts.
- `verification.csv`: equilibrium-source, Bellman, stationary, and finite-window checks.
- `tie_sensitivity.csv`: pure-monopoly versus matched-mixture stationary outcomes.
- `arrays/`: per-case JSON/NPZ bundles, including raw monopoly simulation paths and the matched-mixture sensitivity.
- `comparison.json` and `manifest.json`: definitions, configuration, caveats, source provenance, and checksums.
- `independent_audit.json`: separate checks of every case using independently assembled transitions, long-double stationary calculations, augmented reward matrix powers, and exact competition-source reuse.
- `finite_window_variance_audit.csv` and `.json`: exact first and second moments of finite-window sale, product-2, and profit totals, compared with empirical Monte Carlo variability.

## Policies and ties

For own reputation x, the monopoly demand is the Beta survival probability

`d(x) = P(Beta(1 + x/(1-rho), 1 + (1-x)/(1-rho)) > p0)`.

An outside-option purchase gives the monopolist zero profit and leaves its state unchanged. Conditional on its own sale, reputation moves to `rho*x + (1-rho)` after success and `rho*x` after failure. The monopoly Bellman equation is

`V(x) = d(x)/(1 - gamma*(1-d(x))) * max_i {R-c_i + gamma*[p_i*V(x_success) + (1-p_i)*V(x_failure)]}`.

At the same p0 this is also the fixed-benchmark decision problem used to construct OE p0. Therefore the **optimal value and optimal action set coincide**. The actual competition outcomes nevertheless use two sellers' stochastic TS scores and their joint calendar-time state law.

The main monopoly policy uses the original forgetting simulation convention: select product 2 only when its conditional action advantage exceeds `1e-6`; otherwise select product 1. OE p0 retains its saved equilibrium mixture. The differences between the main policies occur only at verified action-indifference nodes: x=0.62 at rho=0.7, x=0.46 at rho=0.95, x=0.48 at rho=0.97, and x=0.34 at rho=0.99. They are tie-selection differences, not evidence of different Bellman objectives. No non-tied action is changed by this tolerance in these cases.

A matched-mixture sensitivity calculation also evaluates the saved OE p0 rule in the monopoly environment after independently verifying that it is optimal there. Under that convention the policy curves coincide. The stationary product-2 share **conditional on the seller's sales** also coincides, because both environments induce the same embedded own-sale transition kernel. Tied policies have the same discounted optimal value but can generate different stationary averages.

## Outcome denominators

Let alpha be the product-2 policy, mu the monopoly calendar-time stationary distribution, and eta its own-sale stationary distribution. Then `mu(x)` is proportional to `eta(x)/d(x)`. Define `s = sum_x mu(x)*d(x)`.

| Quantity | Monopoly | Symmetric competition, per seller |
| --- | --- | --- |
| Sale rate per calendar period | `s` | `1/2` |
| Product-2 share conditional on own sales | `sum_x eta(x)*alpha(x)` | Saved market product-2 share, pooling the symmetric sellers |
| Product-2 sales per calendar period | `s * conditional_share` | `conditional_share/2` |
| Profit per calendar period | `s * [R-c1-(c2-c1)*conditional_share]` | `[R-c1-(c2-c1)*conditional_share]/2` |

Thus the usage figure includes two panels with different, explicit denominators. Profit includes zero-profit periods when the outside option or the rival is selected. Competition simulation outcomes average the two sellers; they do not represent one arbitrarily selected seller's realized market share.

## Simulation, stationarity, and verification

Monopoly simulations use 20 independent paths, each with 10,000 burn-in periods and 100,000 retained periods, starting at reputation 0.5. The seed is 20261017. The demand draw is Bernoulli with the exact Beta survival probability, equivalent in distribution to drawing a TS score and comparing it with fixed p0. Off-grid successor states are projected stochastically to adjacent nodes with the Bellman interpolation weights. Competition simulations are reused with their original matching design.

Simulation conditional usage is the pooled product-2 count divided by the pooled own-sale count. Its nominal 95% interval uses independent-path cluster delta-method standard errors and a Student-t multiplier; zero-sale paths remain in the cluster sample. Other intervals use independent path means. These approximate intervals describe Monte Carlo uncertainty; they do not cover initialization bias, discretization error, or tie-selection sensitivity. With only 20 paths, rare traps can also make the realized sample underestimate ensemble variability: the exact finite-window mean can fall outside a nominal Monte Carlo interval. Exact finite-window expected outcomes are saved and plotted, including the ratio of expected product-2 and own-sale counts.

The separate finite-window variance audit checks this sampling issue using exact Markov reward moments. At rho=0.97 the monopoly sale-rate mean has true ensemble standard error approximately 0.04975 for 20 paths, compared with the realized sample estimate 0.01758. At rho=0.99 these are approximately 0.01412 and 0.001608. The differences between Monte Carlo means and exact means are within 2.26 true ensemble standard errors across the audited calendar-time metrics; this audit does not turn nominal Student-t intervals into exact coverage guarantees.

At high rho, very rare low-reputation states have extremely long waiting times before the next monopoly sale. These states can dominate the **finite-grid stationary calendar law** even though paths starting at 0.5 barely visit them during the simulated horizon. For example, using the main pure policy, rho=0.99 gives stationary monopoly profit approximately `1.696e-12` but expected profit approximately `0.6041` over the specified retained window. At rho=0.999 these quantities are approximately `2.293e-75` and `0.7639`. A larger number of replications does not remove this finite-horizon difference. The outcome figures distinguish three quantities: exact finite-window expectations, Monte Carlo estimates with nominal intervals, and stationary predictions.

Stationary verification uses positive GTH elimination of the own-sale chain, log-space normalization of `eta/d`, direct calendar jump-chain solutions, and reversed state orders. A separate audit reproduced the high-rho answers using long-double event-chain elimination and, for rho=0.99 and 0.999 where the event kernel K is tridiagonal, the independent birth-death recurrence `log(eta[i]/eta[i-1]) = log(K[i-1,i]/K[i,i-1])`. This rules out a signed linear-solver artifact on this grid. The high-rho values remain finite-grid results; continuous-state or finer-grid conclusions require a separate convergence study.

The stored OE p0 certificate concerns its scalar-benchmark equilibrium construction. This comparison does not assert that the saved policy is a full-state Markov-perfect equilibrium.
