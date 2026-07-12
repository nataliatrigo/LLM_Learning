# Investment Extinction Verification

## 1. Solver Inspection Reported First

Source inspected: `discounted/DP/exact_dp.py` and `discounted/DP/main.py`.

Solver parameter values found and used:

```text
ModelParams(p1=0.35, p2=0.8, c1=0.05, c2=0.65, revenue=1.0, gamma=0.98, tol=1e-10, demand_floor=1e-14)
p0 = 0.5
DEFAULT_P0_GRID = 0.1,0.3,0.5,0.7,0.9
main.py default T / horizon = 700
main.py default policy_period = 50
main.py default diagnostic_p0 = 0.5
main.py default tail_tolerance = 1e-06
```

Policy/value storage found:

- The solver builds `StateSpace(horizon, S, F, total, success_index, failure_index)`.
- It uses rolling arrays `next_value` and `current_value`; the full value table is not returned.
- Requested cuts are returned in `policy_by_period`, `raw_gap_by_period`, and `discounted_gap_by_period`.
- `discounted_gap_by_period[t]` stores `gamma * [V(S+1,F)-V(S,F+1)]` for the requested period.
- For a stored period `t`, `state_count_for_period(t)=t*(t+1)//2`, so the slice covers `n=S+F=0,...,t-1`.
- Therefore the inspected default policy slice has `n_max=49`.

## 2. Extinction Check

| factor | n_max | policy_period | horizon | n_star_max | x*=2 states | interior? |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 49 | 50 | 700 | 49 | 459 | NO |
| 2 | 98 | 99 | 749 | 98 | 1102 | NO |
| 4 | 196 | 197 | 847 | 196 | 2516 | NO |
| 8 | 392 | 393 | 1043 | 392 | 4897 | NO |
| 16 | 784 | 785 | 1435 | 435 | 5113 | yes |

Grid used for figures: `n_max=784`, `policy_period=785`, `horizon=1435`.
`n_star_max = 435`.
Interior check: 435 <= 0.8 * 784 = 627.2 -> PASS.

## 3. Band And Collar Containment

Band check: every diagonal with investment is a single interval.

Diagonals with investment: 436.
Fraction contained in the c=2 collar: 0.954128.
Smallest c containing all observed investment bands: 2.52982.

## 4. Bound Comparison

`dc = c2-c1 = 0.6`.
`dp = p2-p1 = 0.45`.
`dc/dp = 1.33333`.
`kappa = 1/sqrt(2*pi*p0*(1-p0)) = 0.797885`.
`n_bar = 4.849780e+09`.
`n_star_max / n_bar = 8.969479e-08`.

## 5. Premium Decay

Log-log fitted slope for positive `G(n)` values: -0.458073.

## 6. Figures And Tables

- Investment map: `investment_extinction_map.png`
- Premium decay: `premium_decay.png`
- Band endpoints: `investment_band_endpoints.csv`
- Premium series: `premium_decay.csv`
