CROSS-RHO COMPARISON OF SAVED POLICIES

Rates: 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999.
20 independent paired paths; 100,000 retained calendar periods; 10,000 burn-in periods; seed 20261017; common initial state (0.5, 0.5).
Simulation mode: grid; saved grid: 51 points.
Both actual Beta posterior draws select the seller in every simulation. Only the chosen seller updates.
Same path-level uniforms across models and rates; independent SeedSequence children across replications.
No equilibrium was solved or changed. The original saved selected candidates were used without switching branches.
Verified means passing the existing saved-solution numerical checks; it is not a uniqueness or exact-equilibrium proof.

UNCERTAINTY AND STATIONARITY
95% Student-t Monte Carlo intervals use independent path means, never individual time observations.
CDF bands are pointwise. Sellers are pooled within each path before constructing CDF intervals.
Average seller profit is (profit_A + profit_B)/2 per calendar period, including unselected periods.
Success is the realized success fraction of all market purchases. There is one purchase per period.
Circles/solid lines represent simulation estimates. Hollow diamonds/dashed lines represent saved stationary predictions.
The rho axes show the discrete evaluated values with equal spacing; connecting lines guide the eye.
Monte Carlo intervals describe finite-window simulation uncertainty and do not include initialization bias.
summary.csv separates deterministic finite-window expectations, stationary predictions and initialization bias.
A small stationary residual does not establish rapid mixing. Check the joint-law L1 distances below.

FIGURES
01: eight panels of saved own-reputation product-2 probabilities. Steps only guide the eye between saved nodes.
02: product-2 share of all purchases, simulated with CIs and saved stationary predictions.
03: average seller profit per calendar period and customer success rate, both with CIs and predictions.
04: eight panels of pooled calendar-time reputation CDFs, simulated with pointwise CIs and saved predictions.
05: 8x2 saved stationary joint laws, which are stationary under the verified finite-grid kernels.
S01: 8x2 empirical joint laws from the common-start simulations; these can retain initialization bias.
Both joint figures use identical axes, a shared viridis log probability-mass scale, and colored model headers.
Green identifies OE distribution; orange identifies OE p0. Heatmaps use the same purple-green-yellow palette and sans-serif type as the original single-rho figures.

VERIFICATION AND FINITE-WINDOW DIAGNOSTICS
rho=0.6: both selected candidates verified.
  oe: expected window joint-law L1 distance 1.42e-12; product-2 initialization bias -1.52656e-14.
  p0: expected window joint-law L1 distance 1.44469e-12; product-2 initialization bias -9.62563e-14.
rho=0.7: both selected candidates verified.
  oe: expected window joint-law L1 distance 1.47178e-12; product-2 initialization bias +5.40401e-14.
  p0: expected window joint-law L1 distance 1.42659e-12; product-2 initialization bias -6.99441e-15.
rho=0.8: both selected candidates verified.
  oe: expected window joint-law L1 distance 1.29265e-12; product-2 initialization bias -5.26246e-14.
  p0: expected window joint-law L1 distance 1.42458e-12; product-2 initialization bias -8.82627e-14.
rho=0.9: both selected candidates verified.
  oe: expected window joint-law L1 distance 1.51456e-12; product-2 initialization bias -1.32228e-13.
  p0: expected window joint-law L1 distance 7.14912e-08; product-2 initialization bias +6.85125e-11.
rho=0.95: both selected candidates verified.
  oe: expected window joint-law L1 distance 5.10997e-05; product-2 initialization bias +1.41731e-08.
  p0: expected window joint-law L1 distance 0.0402957; product-2 initialization bias +2.25938e-05.
rho=0.97: both selected candidates verified.
  oe: expected window joint-law L1 distance 0.234882; product-2 initialization bias +0.000135095.
  p0: expected window joint-law L1 distance 0.214794; product-2 initialization bias +1.09982e-05.
rho=0.99: both selected candidates verified.
  oe: expected window joint-law L1 distance 0.105333; product-2 initialization bias +1.08804e-08.
  p0: expected window joint-law L1 distance 0.519146; product-2 initialization bias +1.55691e-06.
rho=0.999: both selected candidates verified.
  oe: expected window joint-law L1 distance 2; product-2 initialization bias +0.
  p0: expected window joint-law L1 distance 2; product-2 initialization bias +0.

Data: /home/natalia/Projects/LLM_Learning/equilibrium_comparison/data/cross_rho_n51
Figures: /home/natalia/Projects/LLM_Learning/equilibrium_comparison/figures/cross_rho_n51
Per-rho checkpoints preserve numeric data, source hashes, input audits and simulation provenance.
Event-level records for new rates are reproducible from the seeds and existing simulate_pair; they are not duplicated here.
Existing single-rho data and figures were left untouched.
