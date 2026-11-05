# Producto 2 recibido por el customer por período

Comparación separada del experimento 08: monopolio, OE marginal y OE condicional.

La medida es la probabilidad de que el comprador reciba producto 2 de algún seller en un período calendario.
En competencia se suman los dos sellers simétricos; en monopolio se cuenta el único seller.
La opción externa permanece en el denominador de períodos y no cuenta como producto 2.
No se condiciona en que el customer compre a un seller ni se multiplica por la probabilidad de éxito p2.

```text
Monopolio:   customer_product2_calendar = product2_calendar
Competencia: customer_product2_calendar = 2 * product2_calendar
Ambos:       customer_product2_calendar = (1-outside_rate) * product2_conditional
```

Se usan las soluciones estacionarias auditadas de 101 nodos, con los seis p0 y los tres rho comunes publicados.
Son flujos estacionarios deterministas, sin Monte Carlo ni intervalos de simulación.
La nueva medida agregada también pasa el filtro de sensibilidad: cambio máximo 1.6478 puntos porcentuales entre las grillas de validación, frente al límite de 5.
El contraste de grillas no prueba convergencia al modelo continuo.
La campaña de 201/251 nodos permanece pausada y no se usa en estas figuras.

## Figuras

- [Gráficos de producto 2 en un PDF](figures/customer_product2_comparison.pdf).
- [Comparación frente a rho, un panel por p0](figures/01_customer_product2_by_rho.png).
- [Comparación frente a p0, un panel por rho](figures/02_customer_product2_by_p0.png).

- [p0 = 0.1](figures/p0_0p1_customer_product2.png).
- [p0 = 0.3](figures/p0_0p3_customer_product2.png).
- [p0 = 0.5](figures/p0_0p5_customer_product2.png).
- [p0 = 0.7](figures/p0_0p7_customer_product2.png).
- [p0 = 0.8](figures/p0_0p8_customer_product2.png).
- [p0 = 0.9](figures/p0_0p9_customer_product2.png).

## Datos y reproducción

- `data/customer_product2_summary.csv`: flujos por seller y por customer, con sus identidades contables.
- `data/competition_vs_monopoly.csv`: diferencias de competencia menos monopolio en puntos porcentuales.
- `data/customer_product2_grid_stability.csv`: contraste de la medida agregada entre grillas.
- `manifest.json`: procedencia, huellas y verificaciones.

```bash
MPLCONFIGDIR=/tmp/customer-product2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.customer_product2_report
```

## Compras a la opción externa, A y B

Estos gráficos incluyen compras de cualquier producto, midiendo el porcentaje de períodos en que el customer compra a p0, A o B.
En la comparación general, las filas corresponden a rho=0.6, 0.7 y 0.8; las columnas a monopolio, OE marginal y OE condicional.
Cada barra suma 100%: `outside_rate + sale_rate = 1` en monopolio y `outside_rate + 2*sale_rate = 1` en competencia simétrica.
B no existe en monopolio. En competencia, A y B tienen la misma probabilidad estacionaria de vender.

- [Comparación de los tres modelos, tres rho y seis p0](figures/03_customer_purchase_sources_by_p0.png).
- [Detalle de p0=0.8 y rho=0.6](figures/04_customer_purchase_sources_p0_0p8_rho_0p6.png).
- [Ambos gráficos de compras en PDF](figures/customer_purchase_sources.pdf).
- `data/customer_purchase_sources.csv`: probabilidades por origen de la compra.
- `data/customer_purchase_sources_grid_stability.csv`: contraste entre grillas.
- `purchase_sources_manifest.json`: procedencia y comprobación de los gráficos anteriores sin cambios.

```bash
MPLCONFIGDIR=/tmp/customer-purchase-sources-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.customer_purchase_sources_report
```
