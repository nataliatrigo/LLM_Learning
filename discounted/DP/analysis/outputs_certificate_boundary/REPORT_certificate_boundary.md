# Certificate Boundary Report

## Implementation

Inspected `local_extinction_certificate.py`, `exact_dp.py`, and the previous figure `plot3_empirical_vs_noncertified_N800.png`. Modified the certificate recursion to use the stable binomial PMF and added this outer-grid study.

The valid constant terminal bound `U=(R-c1)/(1-gamma)=47.5` is imposed on diagonal `n=N_outer+1`; recursion starts at `N_outer`. Only diagonals `n<=N_plot=800` are retained. The operator is `U=A+B max_x[p_x U(S+1,F)+(1-p_x)U(S,F+1)]`, with `A=(R-c1) PMF/(ell(D_A)ell(D_B))`, `B=gamma D_B/ell(D_B)`, `D_A=P(Bin(n+2,p0)<=S+1)`, `D_B=P(Bin(n+2,p0)<=S)`, and `PMF=P(Bin(n+2,p0)=S+1)`.

## Boundary prediction and convergence

`U_crit=Delta_c/(gamma Delta_p)=1.36054421769` and `k_pred=log(U_crit/U_terminal)/log(gamma)=175.860` diagonals.

```csv
N_outer,predicted_wall_diagonal,observed_wall_diagonal,maximum_U_difference,mean_U_difference,fraction_classification_changes,maximum_lower_boundary_change,maximum_upper_boundary_change,largest_changed_diagonal,runtime_seconds
800,624.1401572852405,626,,,,,,,0.4963119649999044
1000,824.1401572852405,826,45.73128108550362,2.6821896877670324,0.18660589475126166,0.00997506234413964,0.479002624671916,800.0,0.7511419249999562
1200,1024.1401572852405,1026,0.8043193299430802,0.04282784637585935,0.00014321250556505118,0.0,0.003759398496240629,800.0,1.0631443989998388
1600,1424.1401572852405,1426,0.014395130245300841,0.0006955463350936432,0.0,0.0,0.0,-1.0,1.8556588520000332
2400,2224.1401572852405,2226,4.45430752879712e-06,1.723068481640395e-07,0.0,0.0,0.0,-1.0,4.287121513000102
```

The comparison of the two largest grids (`1600` versus `2400`) gives `n_stable=800` under exact classification agreement plus the one-state boundary-resolution criterion. Diagonals with multiple non-certified components: `none`.

## Bellman validity

Bellman remaining-horizon comparison: `650` versus `800`; policy disagreements on `n<=800`: `0`. The largest empirical product-2 diagonal is `435` and therefore does not touch the grid boundary. Certificate violations on empirical product-2 states: `0`.

## Asymmetry and conclusion

The binomial PMF source numerator is reflection-symmetric at `p0=0.5`. The denominators in `A`, demand-dependent `B`, and asymmetric product probabilities make the full operator asymmetric. The upper vertical wall moves with `N_outer` and is a terminal-boundary artifact; asymmetry that is unchanged between the two largest outer grids is inherent to the certificate operator, not imposed or symmetrized numerically.

Maximum absolute reflected-state differences on representative diagonals:

```csv
n,max_abs_A_reflection_difference,max_abs_B_reflection_difference,max_abs_U_reflection_difference
20,13.053978215013457,0.9799882101464304,13.320338423121076
50,8.302628081816767,0.9799999999999889,10.394780731080338
100,5.93688578967116,0.98,8.480402983997408
200,4.1943654351129815,0.98,6.745009320565368
400,2.975234547890953,0.98,5.244904701704783
600,2.4284026006075177,0.98,4.475333291926863
800,2.1063256614079844,0.98,3.9821242187829062
```

Use `N_outer=2400` and display no more than the verified stable range (`N_plot<=800`). Non-certified means only that this sufficient product-1 certificate is inconclusive; it is not a prediction of product 2.
