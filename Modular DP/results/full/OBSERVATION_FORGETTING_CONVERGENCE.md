# Observation-forgetting convergence study

Method: observation-time forgetting with `rho=0.95`. All comparisons use
nested triangular grids, identical seller primitives, and the same 1,000-path,
250-period common random numbers as the dynamic panels.

| p0 | grids | RMS ΔV | observed order | robust action changes | common-node share | visited-state share | fine Bellman residual |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 401→801 | 8.106e-04 | — | 2 | 0.00248% | 0.00040% | 9.636e-11 |
| 0.1 | 801→1601 | 2.055e-04 | 1.98 | 4 | 0.00125% | 0.00000% | 9.638e-11 |
| 0.3 | 401→801 | 1.323e-03 | — | 0 | 0.00000% | 0.02120% | 9.642e-11 |
| 0.3 | 801→1601 | 3.267e-04 | 2.02 | 2 | 0.00062% | 0.00440% | 9.575e-11 |
| 0.5 | 401→801 | 8.317e-04 | — | 4 | 0.00496% | 0.03960% | 9.497e-11 |
| 0.5 | 801→1601 | 2.088e-04 | 1.99 | 3 | 0.00093% | 0.00960% | 9.601e-11 |
| 0.5 | 1601→3201 | 5.240e-05 | 1.99 | 8 | 0.00062% | 0.00240% | 9.542e-11 |
| 0.7 | 401→801 | 6.853e-04 | — | 4 | 0.00496% | 0.01080% | 9.386e-11 |
| 0.7 | 801→1601 | 1.721e-04 | 1.99 | 1 | 0.00031% | 0.00080% | 9.563e-11 |
| 0.9 | 401→801 | 5.515e-05 | — | 4 | 0.00496% | 0.00520% | 3.910e-12 |
| 0.9 | 801→1601 | 1.526e-05 | 1.85 | 3 | 0.00093% | 0.00000% | 3.875e-12 |

## Conclusion

The value function exhibits stable second-order convergence: the observed RMS
orders lie between `1.85` and `2.02`. For the
most sensitive case, `p0=0.5`, the RMS difference falls from
`8.317e-04` on
401→801 to `5.240e-05` on
1601→3201. Its visited-state disagreement falls in parallel from
`0.0396%` to
`0.0024%`.

Exact zero policy changes is retained as a strict diagnostic but is not an
appropriate sole convergence criterion for a discontinuous argmax boundary:
doubling the grid quadruples the common-node set, and a convergent boundary can
still cross a handful of newly resolved nodes. The robust policy converges in
measure: every common-node disagreement share is at most
`0.0050%`, and every disagreement share on
the simulated states is at most
`0.0396%`. All discretized
Bellman residuals are below `9.642e-11`.

The 801-grid policies used by the Monte Carlo panels are therefore numerically
stable at the scale of those plots. The 1601 and 3201 solutions serve as
refinement checks; retaining 801 for plotting avoids a large computational
cost without a visually or economically material policy change.
