# Discounted fluid-HJB policy convergence study

## Verdict

**The product-2 band in the \(p_0\le p_1\) regime is genuine.**

The band persists with the same qualitative shape and location at uniform grid
steps \(h=1,\ 0.5,\ 0.25\). Grid refinement changes only thin layers along the
switching curve: at the finest consecutive comparison, 0.331% of common-lattice
nodes change for \(p_0=0.10\), and 0.840% change for \(p_0=0.30\). The region
does not shrink toward zero or move away from its coarse-grid location.

At the original \(h=1\) grid, reducing the terminal-tail tolerance from
\(10^{-4}\) to \(10^{-6}\) changes 0 of 51,681 comparison nodes for either
\(p_0\). Doubling the stored domain from \(N=80\) to \(N=160\) also changes 0
of 51,681 nodes on the common \(n=s+f\le80\) triangle. Thus the band is neither
a terminal-tolerance artifact nor a domain-boundary artifact.

The switching curve still has visible first-order grid error, especially for
\(p_0=0.30\). The \(h=0.25\) result should therefore be used for the most
accurate plotted boundary, even though the existence of the band is robust.

## 1. Production code and numerical scheme

The solver is `discounted/Fluid/fluid_model.py`, function
`solve_fluid_hjb`. The policy map is produced from the stored `actions` arrays
by `solution_frame` and the plotting functions in the same file.

### (a) State-space discretization

The grid is uniform, not adaptive. At layer \(k\),

\[
n=s+f=kh,\qquad
(s_i,f_i)=(ih,(k-i)h),\quad i=0,\ldots,k.
\]

The default displayed/stored triangle has \(N=80\) and \(h=1\). The refinement
study uses \(h=1,\ 0.5,\ 0.25\), retaining the same physical triangle
\(s+f\le80\). These contain 3,321, 13,041, and 51,681 states, respectively.

The numerical recursion extends far beyond the displayed triangle. Its
terminal count is

\[
N_T=N+
h\left\lceil
\frac{-\log(\varepsilon_{\rm tail})}{rh}
\right\rceil,\qquad r=-\log\gamma.
\]

With \(\gamma=0.999\) and \(\varepsilon_{\rm tail}=10^{-4}\), the terminal
count is approximately 9,286 for all three meshes. Thus \(N=80\) is not the
terminal computational boundary.

### (b) Scheme

The code uses one monotone backward semi-Lagrangian sweep. From a state
\((s,f)\), one count increment \(h\) takes the frozen-demand calendar time
\(h/D(s,f)\). The deterministic next state under quality \(p_j\),
\((s+p_jh,f+(1-p_j)h)\), lies between the two adjacent next-layer nodes
\((s+h,f)\) and \((s,f+h)\). Its continuation value is obtained by linear
interpolation:

\[
C_j=p_jV_{k+1,i+1}+(1-p_j)V_{k+1,i}.
\]

The code integrates the frozen-demand flow payoff over the count step and uses

\[
Q_j=
\frac{D(1-e^{-rh/D})}{r}(R-c_j)
+e^{-rh/D}C_j.
\]

It then stores \(\max(Q_1,Q_2)\) and product 2 iff \(Q_2>Q_1\). Exact numerical
ties are assigned to product 1.

### (c) Derivatives and switching condition

The implementation does not separately compute \(v_s,v_f,w_n,\) or \(w_m\).
The adjacent-node difference

\[
\frac{V_{k+1,i+1}-V_{k+1,i}}{h}
\]

is the scheme's approximation to \(v_s-v_f\). Equivalently, the code compares
the two complete discrete action values:

\[
Q_2-Q_1=
-\Delta c\,\frac{D(1-e^{-rh/D})}{r}
+e^{-rh/D}\Delta p
\left(V_{k+1,i+1}-V_{k+1,i}\right).
\]

After division by \(h\), this converges to the paper's test
\(v_s-v_f\ge\Delta c/\Delta p\) as \(h\downarrow0\).

### (d) Iteration and convergence criterion

There is no value iteration, policy iteration, residual tolerance, or
iteration cap. The acyclic count dynamics permit a single backward sweep from
\(N_T\) to zero.

The solver's only convergence/truncation parameter is `tail_tolerance`. It
chooses \(N_T\) so that
\(e^{-r(N_T-N)}\le\varepsilon_{\rm tail}\). The baseline uses \(10^{-4}\);
the tightened check uses \(10^{-6}\), which moves the terminal boundary from
about 9,286 to 13,889 at \(h=1\).

### (e) Boundary conditions

The only imposed condition is

\[
V(N_T,\cdot)=0.
\]

There is no artificial Dirichlet, Neumann, or extrapolation condition at
\(s=0\) or \(f=0\). The semi-Lagrangian update uses the two valid next-layer
nodes even at those natural edges, so the dynamics are inward or tangent.

Because \(D\le1\), reaching \(N_T\) from the displayed edge takes at least
\(N_T-N\) calendar-time units. A conservative value-error bound at \(N=80\)
is

\[
\frac{R-c_1}{r}\,\varepsilon_{\rm tail},
\]

which is about 0.095 at \(10^{-4}\) and 0.00095 at \(10^{-6}\).

## 2. Grid-refinement results

All policy extents below are evaluated on the same \(h=0.25\) lattice over
\(s+f\le80\), using the production solver's nearest-grid feedback convention.
The area is a node-based estimate (`product-2 nodes` \(\times0.25^2\)).

| \(p_0\) | \(h\) | Product-2 share | Area estimate | \(s_{\max}\) | \(f_{\max}\) |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 1.00 | 12.378% | 399.8125 | 9.25 | 80.00 |
| 0.10 | 0.50 | 12.328% | 398.1875 | 9.25 | 80.00 |
| 0.10 | 0.25 | 12.291% | 397.0000 | 9.25 | 80.00 |
| 0.30 | 1.00 | 35.868% | 1158.5625 | 31.25 | 69.25 |
| 0.30 | 0.50 | 36.658% | 1184.0625 | 32.25 | 69.50 |
| 0.30 | 0.25 | 37.300% | 1204.8125 | 32.25 | 69.75 |

For \(p_0=0.10\), the area decreases by 0.406% and then 0.298%; the total
coarse-to-fine change is \(-0.703\%\). For \(p_0=0.30\), the area increases by
2.201% and then 1.752%; the total coarse-to-fine change is \(+3.992\%\).
Neither region shrinks, disappears, or relocates.

On the native meshes, the \(p_0=0.10\) band has \(s_{\max}=9,\ 9,\ 9.25\)
and its \(s=0\) arm spans \(f=1\) to 80 at \(h=1,0.5\), and \(f=0.75\) to 80
at \(h=0.25\). For \(p_0=0.30\), \(s_{\max}=31,\ 32,\ 32.25\), and the
\(s=0\) arm ends at \(f=23,\ 25,\ 25.75\). These are small switching-boundary
adjustments rather than movement of the region as a whole.

## 3. Consecutive policy-map changes

The policy indicator is binary, so the requested maximum absolute change is
necessarily 1 whenever even one node changes. The mismatch share and symmetric
difference are the informative convergence measures.

| \(p_0\) | Comparison | Max. abs. change | Changed nodes / 51,681 | Changed share | Symmetric-difference area |
|---:|:---|---:|---:|---:|---:|
| 0.10 | \(h=1\to0.5\) | 1 | 168 | 0.325% | 10.5000 |
| 0.10 | \(h=0.5\to0.25\) | 1 | 171 | 0.331% | 10.6875 |
| 0.30 | \(h=1\to0.5\) | 1 | 568 | 1.099% | 35.5000 |
| 0.30 | \(h=0.5\to0.25\) | 1 | 434 | 0.840% | 27.1250 |

Visual inspection confirms that changed nodes form thin layers along the two
switching boundaries, not isolated product-2 islands or wholesale band motion.

## 4. Tightened solver criterion

At \(h=1,N=80\), changing `tail_tolerance` from \(10^{-4}\) to \(10^{-6}\)
changes **0 of 51,681** common-lattice policy nodes for both \(p_0=0.10\) and
\(p_0=0.30\).

The value at the origin does change, as expected from moving a zero terminal
condition farther away:

| \(p_0\) | \(V_{10^{-4}}(0,0)\) | \(V_{10^{-6}}(0,0)\) | Absolute change |
|---:|---:|---:|---:|
| 0.10 | 948.097929 | 948.184605 | 0.086677 |
| 0.30 | 933.606697 | 933.692910 | 0.086213 |

Those value changes do not cross any switching threshold on the original
policy grid.

## 5. Domain-extension and boundary check

At \(h=1\), doubling the stored domain from \(N=80\) to \(N=160\) moves the
zero terminal condition from 9,286 to 9,366. On the common \(s+f\le80\)
triangle, the extended-domain policy changes **0 of 51,681** comparison nodes
for either \(p_0\).

The full \(N=160\) map shows that each product-2 region continues smoothly
across the old display edge \(s+f=80\). For \(p_0=0.10\), the product-2 arm on
the natural edge \(s=0\) ends near \(f=83\), but the band itself continues
toward larger \(s\) and reaches the new \(s+f=160\) edge. Thus the visible
contact with \(s=0\) and \(s+f=80\) is part of the computed switching geometry,
not contamination from an imposed edge condition.

## 6. Saved outputs and reproduction

- `plots/policy_grid_refinement.png`: requested \(2\times3\) summary figure.
- `plots/policy_tail_tolerance_comparison.png`: baseline, tightened tail, and
  difference panels.
- `plots/policy_domain_extension_comparison.png`: statewise comparison over the
  common \(N=80\) triangle.
- `plots/policy_extended_domain_full.png`: full doubled-domain maps.
- `tables/refinement_policy_changes.csv`: consecutive policy-map metrics.
- `tables/policy_extents_common_lattice.csv`: normalized region extents.
- `tables/sensitivity_policy_changes.csv`: tolerance and boundary comparisons.
- `tables/run_metadata.csv`: parameters, terminal counts, runtime, and values.
- `data/*.csv.gz` and `metadata/*.json`: every intermediate solver output.

Reproduce or resume the study from the repository root with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  discounted/Fluid/convergence_study.py --workers 2
```

Existing complete runs are reused. Add `--force` only to recompute all ten
expensive solver runs.
