# Continuation value gap

Define the raw gap
D_t(S,F) = V_{t+1}(S+1,F) - V_{t+1}(S,F+1).

The reported discounted continuation value gap is
M_t(S,F) = gamma * D_t(S,F).

Conditional on Seller A being chosen,
Q2 - Q1 = -(c2-c1) + (p2-p1) * M_t(S,F).

Therefore product 2 is optimal iff
M_t(S,F) > (c2-c1)/(p2-p1) = 1.33333333333.

If the raw gap D_t is plotted instead, its threshold is
(c2-c1)/(gamma*(p2-p1)) = 1.33466800133.

The unconditional Bellman action gap multiplies this conditional
gap by rho(S,F), which does not change its sign when rho>0.
