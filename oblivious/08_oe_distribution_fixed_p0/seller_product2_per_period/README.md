# Producto 2 por seller y período

Comparación separada del experimento 08: monopolio, OE marginal y OE condicional.

La medida es la probabilidad de que un seller sea seleccionado y entregue producto 2 en un período calendario.
En competencia se cuenta un seller simétrico, sin sumar al rival. Se incluyen todos los períodos en el denominador.

```text
seller_product2_calendar = product2_calendar
                        = sale_rate * product2_conditional
```

## Relación con el paper

Sí: es el mismo evento que mide el panel de producto 2 de las simulaciones Monte Carlo del paper.
El paper lo define como la fracción de trayectorias en que el seller A es seleccionado y elige producto 2 en esa fecha.
El simulador lo guarda como `product2_unconditional_rate`, calculado con `seller_a_chosen & product2_prescribed`.
La diferencia es temporal: el paper muestra esa probabilidad fecha a fecha desde el estado inicial; aquí se muestra su nivel estacionario.
Se revisaron `OverleafPaper/main.tex`, `forgetting/Code/simulation.py` y `forgetting/Code/experiments/run_experiments.py`.

## Datos y validación

Se usan las mismas soluciones estacionarias auditadas de 101 nodos que en la comparación por customer: seis p0 y tres rho comunes.
Son flujos deterministas, sin Monte Carlo ni intervalos de simulación.
El cambio máximo entre las grillas de validación de esta medida es 1.6478 puntos porcentuales; límite: 5.
La campaña de 201/251 nodos permanece pausada; se preservan los gráficos anteriores y los del customer.

## Figuras

- [Todos los gráficos en un PDF](figures/seller_product2_comparison.pdf).
- [Frente a rho, un panel por p0](figures/01_seller_product2_by_rho.png).
- [Frente a p0, un panel por rho](figures/02_seller_product2_by_p0.png).

- [p0 = 0.1](figures/p0_0p1_seller_product2.png).
- [p0 = 0.3](figures/p0_0p3_seller_product2.png).
- [p0 = 0.5](figures/p0_0p5_seller_product2.png).
- [p0 = 0.7](figures/p0_0p7_seller_product2.png).
- [p0 = 0.8](figures/p0_0p8_seller_product2.png).
- [p0 = 0.9](figures/p0_0p9_seller_product2.png).

- `data/seller_product2_summary.csv`: flujos por seller y período.
- `data/competition_vs_monopoly.csv`: diferencias por seller en puntos porcentuales.
- `data/seller_product2_grid_stability.csv`: sensibilidad de la medida entre grillas.
- `manifest.json`: procedencia y verificaciones.

## Reproducción

```bash
MPLCONFIGDIR=/tmp/seller-product2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.seller_product2_report
```
