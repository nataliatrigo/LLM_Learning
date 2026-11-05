# Demanda de A al aumentar el número de sellers

[PDF con todos los rho](demand_A_by_seller_count.pdf)

N=1..50 cuenta a A y sus N-1 rivales. Cada nuevo rival tiene reputación fija x_R=0.5; A toma los valores 0.3, 0.5, 0.7.

El eje vertical es la probabilidad de que el comprador elija A en un período, incluyendo los períodos en que elige outside. Cada panel fija p0. Cada panel ajusta su escala vertical para mostrar también las demandas pequeñas. Las líneas discontinuas indican reputaciones iguales para A y todos sus rivales.

```text
D_A(N) = integral_p0^1 f_A(t) F_R(t)^(N-1) dt
Beta(x;rho) = Beta(1+x/(1-rho), 1+(1-x)/(1-rho))
Caso simétrico: D_A(N) = [1-F(p0)^N]/N
```

Es demanda condicionada a reputaciones fijas. No usa políticas, distribuciones estacionarias ni un equilibrio oblivious. rho solo parametriza las Betas. Para demanda promedio de equilibrio habría que promediar sobre la ley conjunta de reputaciones correspondiente a cada N.

Todos los puntos se contrastan con integración adaptativa independiente; la cuadratura positiva se duplica en orden. También se comprueban la fórmula simétrica y que la demanda no aumente al agregar rivales o aumentar p0.

| rho | Figura |
| --- | --- |
| 0.6 | [PNG](demand_A_rho_0p6.png) |
| 0.7 | [PNG](demand_A_rho_0p7.png) |
| 0.8 | [PNG](demand_A_rho_0p8.png) |
| 0.9 | [PNG](demand_A_rho_0p9.png) |

[Datos CSV](../../data/conditional_demand_n50/demand.csv) · [Validaciones](../../data/conditional_demand_n50/validation.json) · [Manifiesto](../../data/conditional_demand_n50/manifest.json)

Reproducir desde la raíz con un nombre nuevo:

```bash
MPLCONFIGDIR=/tmp/conditional-demand-mpl .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.conditional_demand --name conditional_demand_n50_new --max-sellers 50 --p0s 0.1,0.3,0.5,0.7,0.8,0.9 --rhos 0.6,0.7,0.8,0.9 --own-reputations 0.3,0.5,0.7 --rival-reputation 0.5 --order 128
```
