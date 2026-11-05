# Eventual product-1 dominance: a conservative theoretical bound

The numerical plots display only audited finite seller counts. The following
argument is separate from the numerical experiment. It proves eventual zero
product-2 use for a fixed finite memory capacity, without predicting where
the finite-N equilibrium curve first approaches zero.

## A uniform selection bound

Set `m=C+1`. The most favorable own posterior is `Beta(m,1)` and the least
favorable rival posterior is `Beta(1,m)`. First-order stochastic dominance
therefore gives, for any own count and any joint rival state,

```text
d <= integral[p0,1] m*t^(m-1) * [1-(1-t)^m]^(N-1) dt
  <= B(1/m,N).
```

For the second inequality, substitute `u=1-t`, bound `(1-u)^(m-1)` by one,
extend the integration interval to `[0,1]`, and substitute `z=u^m`. The factor
m cancels the Jacobian factor 1/m. Rival correlations cannot violate this
pointwise bound; averaging over any rival belief preserves it.

Every calendar-period reward is nonnegative and at most `(revenue-c1)*d`.
Hence every continuation value is bounded by

```text
Vmax <= (revenue-c1) * B(1/m,N) / (1-gamma).
```

The seller chooses a product after selection. Its conditional advantage from
product 2 is at most `-(c2-c1) + gamma*(p2-p1)*Vmax`. There is no extra current
selection factor in that conditional comparison. Thus product 1 strictly
dominates at every own state whenever

```text
B(1/m,N) < (c2-c1)*(1-gamma) / [gamma*(p2-p1)*(revenue-c1)]
         = 80/2793
         = approximately 0.02864303616.
```

Since the beta-function bound tends to zero as N increases, eventual
product-1 dominance follows for each fixed finite C. This argument applies
to marginal and conditional beliefs, and does not require independent rival
reputations.

## Safe sufficient seller counts

For `N>1`, the inequality `(1-z)^(N-1) <= exp(-(N-1)*z)` implies

```text
B(1/m,N) <= Gamma(1/m)/(N-1)^(1/m) <= m/(N-1)^(1/m).
```

The last step uses `Gamma(1+1/m)<=1`. Consequently
`N > 1 + (2793*m/80)^m` is a transparent sufficient condition.

| Capacity C | A safe sufficient N from this generic bound |
|---:|---:|
| 1 | 5,000 |
| 3 | 400,000,000 |
| 5 | 90,000,000,000,000 |
| 8 | 30,000,000,000,000,000,000,000 |

These deliberately conservative numbers are **not observed crossing points,
estimates of a minimal threshold, or forecasts of the plotted curves**. They
show why a moderate finite-N experiment need not reach the guaranteed regime.

## A stronger result for C=1 and the plotted marginal closure

For C=1, a purchase entirely replaces the single observation. Let d0 and d1
be selection probabilities at own counts zero and one against the same
unconditional rival law. If T is the maximum of the outside option and the
rival samples, then

```text
d0 = E[(1-T)^2]
d1 = E[1-T^2] <= 2*sqrt(d0)-d0.
```

The inequality is Jensen's inequality. Evaluating the always-product-1 policy
gives the continuation-value difference

```text
V1-V0 = .95*(d1-d0) / [.02 + .98*(.35*d0+.65*d1)].
```

Its product-2 advantage is nonpositive if
`.98*(.0375*d1-.6375*d0) <= .012`. Writing `x=sqrt(d0)` bounds its left side by

```text
.98*(.075*x-.675*x^2) <= .98/480 < .012.
```

The inequality is strict, so product 1 is the unique best response to every
common marginal rival law under this calibration. Therefore every C=1
marginal oblivious equilibrium, including monopoly, has **zero product-2 use
for every N**. This stronger C=1 argument does not automatically apply to
conditional beliefs, whose rival law may change with the own count.

## What the graph can establish

Product 2 among own sales is `product2_calendar/sale_rate`. A symmetric
seller's calendar product-2 flow is at most `1/N`, which declines mechanically
with N. That observation does not prove its conditional product mix declines.
Finite audited points can show a decline, plateau, or entry into the 1% band;
they do not by themselves establish an asymptotic limit or monotonicity.
