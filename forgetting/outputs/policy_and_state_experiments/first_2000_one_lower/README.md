# First 2,000 interactions: the one-threshold 1 -> 2 case

Parameters: rho = 0.95, gamma = 0.98, p0 = 0.50, p1 = 0.35, p2 = 0.55,
c1 = 0.05, c2 = 0.10, R = 1. Thus Delta p = 0.20 and Delta c = 0.05.

The numerical optimal boundary policy has one crossing at x approximately
0.16702052: product 1 below it and product 2 above it. In the units of the
reference individual-state plots, S = x/(1-rho), so the threshold is
S approximately 3.34041043. This is a numerical boundary-policy finding,
not an analytical threshold proof.

## Matching the previous simulation

The primary experiment begins from the empty history (x,y) = (0,0), as in
`forgetting/outputs/policy_and_state_experiments/first_2000`. Each of the 20 paths has 2,000 interactions
with seller A. This is interaction time, not calendar time. The seed is
4298606458669971020, with the same 20-path random-draw ordering as the reference.
Customer 1 was selected before running the simulation.

The policy is evaluated separately on the transient observation layers before
using the mature approximation. It is not replaced by the boundary threshold
at interaction 1. All 20 simulated paths choose product 2 at every interaction
in this particular 2,000-interaction sample. This does not mean the policy is
product 2 at every state, nor that product 1 is never used over an infinite path.

The plot in exactly the reference format is:

`figures/individual_states/p0_0.5/individual_states_p0_0.5_rho_0.95_path_1_start_1_n_2000.png`

## Additional experiment starting on the boundary

The `boundary_start` folder is a separate experiment, initialized at (x,y) =
(0,1), equivalent to S = 0 and F = 20. It uses the mature boundary policy from
the first interaction and the same outcome uniforms as the primary experiment.
For customer 1, product 1 is used at interactions 1-9, and product 2 at
interactions 10-2000. This example illustrates a realized crossing of the state
threshold. The policy pattern 1 -> 2 refers to increasing state, not a guarantee
of a single switch over time; random outcomes can produce repeated crossings.

`figures/one_lower_initial_conditions.png` compares both 2,000-interaction paths
and labels their different starting conditions. Its dashed horizontal line is
the long-run boundary threshold. It does not determine the early choices in
the empty-history experiment.

`figures/one_lower_boundary_first_100.png` enlarges the first 100 interactions
of the boundary-start path so the nine product-1 choices are easy to see.

## Data, verification, and reproduction

Both experiment folders contain `observations.npy` for all 20 paths,
`records_manifest.json`, and `individual_state_trace.csv` for customer 1.
The CSV records x, y, S, product choice, and outcome before/after the interaction
as appropriate. `trajectory_summary.csv` covers all paths from both starts.

The primary policy uses 8,001 state nodes, terminal gap 1e-8, iteration tolerance
1e-12, and action tolerance 1e-8. Repeating with 16,001 nodes and terminal gap
1e-10 gives identical states, choices, and outcomes for every recorded path.
The boundary is solved on 16,001 nodes and has one upward sign crossing, with
negative advantage at zero and positive advantage at one. These checks are
numerical consistency checks, not certified continuous-state error bounds.

Run from the project root:

```bash
.venv/bin/python -B forgetting/Code/threshold_parameter_map/simulate_one_lower_paths.py
```

No production model or reference simulation files were changed.
