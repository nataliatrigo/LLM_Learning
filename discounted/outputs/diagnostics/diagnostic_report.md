# Discounted-horizon diagnostic

## 1. Horizon convergence

- gamma=0.98, base T=700, gamma^T=7.21528e-07.
- Comparison T=850 at t=50.
- Differing states: 0 / 1275 (0.000000%).
- Initial-value difference: 7.7883764e-06.
- Direct comparison with T=2000: 0 / 1275 differing states (0.000000%).

This verifies that the discounted policy has converged with respect to
the terminal horizon.

## 2. Continuation value gap

Let

D_t(S,F) = V_{t+1}(S+1,F) - V_{t+1}(S,F+1)

be the raw success-versus-failure continuation gap. The quantity
reported in the corrected code is

M_t(S,F) = gamma * D_t(S,F).

Conditional on Seller A being selected,

Q_2 - Q_1 = -(c_2-c_1) + (p_2-p_1) M_t(S,F).

Thus the M_t threshold is Delta c / Delta p = 1.33333333333 = 4/3. If raw D_t is plotted, its threshold is 1.36054421769.

## 3. p0=0.5 simulation

Over periods 201-250, the exact discounted model has:

- Mean A market share: 82.7150%.
- Mean demand probability: 82.6334%.
- Mean product-2 rate when A is chosen: 33.0580%.
- Mean policy product-2 share across simulated states: 34.4000%.

## 4. Comparison with the gamma=1, T=2000 benchmark

- Policy disagreements at t=50: 599 / 1275 (46.9804%).
- Discounted product-2 state share at t=50: 36.0000%.
- gamma=1 product-2 state share at t=50: 82.9804%.
- gamma=1 mean A market share in the late window: 100.0000%.
- gamma=1 mean product-2 rate in the late window: 57.9050%.

The terminal-horizon bug is fixed, but gamma=0.98 is not
policy-equivalent to gamma=1. If a finite-T=2000 run with
gamma=0.98 reports the gamma=1 behavior, the two runs are not
using the same discounting convention or calibration.
