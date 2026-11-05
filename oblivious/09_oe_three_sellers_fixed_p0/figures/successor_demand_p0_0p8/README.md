# Demanda en estados sucesores: p₀ = 0.8

[Vista conjunta](00_successor_demand_overview.png) · [PDF con una página por ρ](successor_demand_by_rho.pdf)

Se reutilizan las demandas percibidas y políticas guardadas de monopolio (N=1) y OE marginal (N=2,3) en `high_p0_refinement_41`, sin resolver equilibrios. Los colores y estilos provienen de la comparación vigente de políticas.

En los mismos nodos originales x de los tres modelos se calculan `x_s = rho*x + 1-rho` y `x_f = rho*x`; se interpola linealmente la demanda guardada de cada modelo en esos estados. La primera columna muestra `(d(x_s)+d(x_f))/2`: el nivel de oportunidades futuras con pesos iguales para éxito y fracaso. No pondera por las probabilidades de éxito del producto. La segunda muestra `d(x_s)-d(x_f)`: la ganancia de demanda por un éxito frente a un fracaso, en unidades de probabilidad y sin dividir por `1-rho`.

El sombreado marca celdas alrededor de nodos donde la diferencia máxima entre probabilidades de producto 2 supera `1e-7`. Incluye políticas mixtas. Los bordes son puntos medios entre nodos vecinos, solo para visualizar: no son umbrales continuos estimados. La tercera columna muestra las políticas originales y cruces en los nodos con diferencias.

Los círculos sobre las dos curvas de demanda y en las franjas inferiores marcan las frecuencias de la reputación propia. Su **área** es proporcional a la probabilidad estacionaria del nodo, con la misma escala para todos los modelos y rho. Las franjas separan N=1,2,3 para que las probabilidades se vean incluso si sus curvas se superponen. Se calcula `mu_N(x) = sum_{reputaciones rivales} joint_N(x, rivales)` a partir de la distribución conjunta guardada. Son porcentajes de todos los períodos de calendario, incluidos los períodos sin venta propia.

| ρ | Grilla | Control | Figura | Nodos con diferencias |
|---|---:|---:|---|---|
| 0.6 | 41 | 31 | [PNG](rho_0p6.png) | 0.85 |
| 0.7 | 41 | 31 | [PNG](rho_0p7.png) | 0.025, 0.05, 0.075, 0.875 |
| 0.8 | 41 | 31 | [PNG](rho_0p8.png) | 0.3, 0.325 |
| 0.9 | 41 | 31 | [PNG](rho_0p9.png) | Ninguno |

Cada diagnóstico tiene su propio contraste de grillas: las demandas de la grilla de control se evalúan en los mismos estados sucesores de la grilla publicada. Si algún modelo cambia más de 0.05 en un diagnóstico, se omiten las tres curvas de ese diagnóstico para ese ρ. Este filtro es independiente de los filtros de mezcla de productos o beneficios.

| Diagnóstico | Máximo cambio absoluto entre grillas |
|---|---:|
| average_successor_demand | 0.002278813 |
| outcome_sensitivity | 0.001870010 |

Las frecuencias tienen un control separado entre las grillas: distancia Wasserstein-1 entre las marginales discretas y cambio máximo de los cuantiles 10/25/50/75/90%, ambos con tolerancia 0.05 en unidades de reputación. Se ocultan los círculos de los tres modelos si alguno falla. No se interpolan masas nodales ni se afirma estabilidad puntual del modo o de cada probabilidad de nodo.

Frecuencias retenidas: 4/4 valores de rho. Máximo Wasserstein-1: 0.022828; máximo cambio de cuantil: 0.033333.


4 de 4 valores de ρ pasan ambos contrastes. La auditoría y los contrastes son verificaciones del modelo finito; no certifican umbrales continuos ni unicidad.

Datos: [todos los nodos](../../data/successor_demand_p0_0p8/successor_demand.csv) · [resumen de frecuencias](../../data/successor_demand_p0_0p8/state_frequency_summary.csv) · [contraste por modelo](../../data/successor_demand_p0_0p8/grid_stability.csv) · [manifiesto y hashes](../../data/successor_demand_p0_0p8/manifest.json).

Reproducción desde la raíz del repositorio:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/successor-demand-mpl .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.successor_demand --name successor_demand_p0_0p8_new
```
