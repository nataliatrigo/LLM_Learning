# Observation-time forgetting: non-extinction certificate

## Claim

For the reported primitives, `rho=0.95`, and the stationary optimal
policy, product 2 is used infinitely often almost surely from the uniform prior.
Indeed, the argument gives a strictly positive (though extremely conservative)
lower bound on its asymptotic realized frequency.

## Exact reduction

At Seller-A observation times the normalized state follows

```text
x' = rho*x + (1-rho)*Y,
y' = rho*y + (1-rho)*(1-Y).
```

The total coordinate is deterministic:
`x_k+y_k=1-rho^k`. Hence it approaches the invariant boundary `x+y=1`.
On that boundary the Bellman equation is exactly one-dimensional in `x`; both
success and failure transitions remain on the boundary. The boundary Bellman
operator is a contraction with modulus at most `gamma=0.98`.

## Strict product-2 region

A cellwise lower/upper Bellman enclosure with `50,000` cells
certifies that product 2 is strictly optimal throughout
`x in [0.33, 0.42]` on the invariant boundary. The minimum
certified advantage is `0.828069` and the
largest final value-enclosure width is
`0.0229189`. SciPy beta-tail endpoint
evaluations were padded outward by `5.0e-13`.
Strict positivity and continuity imply the same action on a sufficiently thin
interior neighborhood of this boundary interval.

## Synchronizing word

The outcome word `(0^19 1^6)^2` (the failure/success block repeated
`2` times) maps
every initial boundary coordinate `x in [0,1]` into
`[0.338390857, 0.415335832]`, which lies strictly
inside the certified product-2 interval. If product 1 is used, this particular
word has probability `2.62739e-13` in
each disjoint block of `50` Seller-A observations. Uniformly over
all adaptive product choices, its conditional block probability is at least
`9.28873e-33`.

## Almost-sure recurrence argument

1. Thompson demand is continuous and strictly positive on the compact
   forgetting simplex. Its exact global floor is
   `4.76837e-07`, so Seller A is observed infinitely often
   almost surely.
2. Divide Seller-A interactions into blocks of `51`. In the
   first `50` outcomes of every block, conditional on the entire
   preceding history, the synchronizing word has probability at least the
   positive policy-uniform number above.
3. The block indicators have conditional expectations bounded below by that
   number. The strong law for bounded martingale differences therefore gives a
   lower asymptotic block frequency at least as large as the bound; independence
   of the adaptive blocks is not required.
4. Once `x+y` is sufficiently close to one, the word enters the strict
   product-2 neighborhood. Observation-time idling leaves that state unchanged,
   so the final Seller-A interaction in each successful block uses product 2.
5. Consequently the asymptotic product-2 frequency among Seller-A interactions
   has lower bound `1.82132e-34`.
   Since the asymptotic Seller-A frequency is at least the demand floor, the
   corresponding calendar-time lower bound is
   `8.68473e-41`.
   These tiny bounds certify persistence; they are not estimates of the actual
   long-run frequency.

## Scope

This certificate is parameter-specific. It proves pathwise non-extinction for
observation-time forgetting at the values above; it does not automatically
extend to calendar-time forgetting or arbitrary primitives. The only
computer-assisted step is the strict Bellman action sign. The recurrence step
is analytic. The Bellman enclosure is rigorous in exact arithmetic; this
implementation evaluates beta tails and arithmetic in double precision and
uses the stated outward padding. A publication-grade machine-verified proof
would replace that numerical layer with directed-rounding interval special
functions. The positive advantage margin is reported so this remaining
numerical assumption is explicit rather than hidden.
