# Compilados: métrica principal y complementaria

Calibración común: c2=0.65, grilla de 101 puntos, seis p0 y rho=0.6,0.7,0.8.
Se reúnen los gráficos existentes y auditados; no se ejecutan simulaciones ni se resuelven nuevos equilibrios.

| Métrica | Frente a p0 | Frente a rho |
| --- | --- | --- |
| **Principal:** producto 2 como porcentaje de las ventas propias | [Tres paneles](01_primary_product2_given_sale_by_p0.png) | [Seis paneles](02_primary_product2_given_sale_by_rho.png) |
| **Complementaria:** producto 2 recibido por el customer por período, sumando A y B | [Tres paneles](03_complementary_customer_product2_by_p0.png) | [Seis paneles](04_complementary_customer_product2_by_rho.png) |

La principal es `P(producto 2 | venta propia)`. La complementaria es `P(customer recibe producto 2 de algún seller)` por período calendario.
En monopolio la complementaria cuenta al único seller; en competencia suma ambos. La opción externa no cuenta como producto 2.

- [PDF con los cuatro compilados](primary_and_complementary_metrics.pdf): páginas 1-2 principal y páginas 3-4 complementaria.
- [Tabla conjunta de las dos métricas](main_metrics_data.csv).
- [Carpetas individuales por p0](individual%20p0s/).

Los seis casos individuales del barrido están en `individual p0s/p0_0p1`, `p0_0p3`, `p0_0p5`, `p0_0p7`, `p0_0p8` y `p0_0p9`.
La comparación histórica ampliada de p0=0.5 está en `individual p0s/p0_0p5_common_grid`.
Los accesos `p0_sweep` y `p0_0p5_common_grid` conservan las rutas históricas necesarias para los certificados y enlaces originales.
La campaña `p0_sweep_201` continúa pausada y conserva su ubicación.

Los gráficos originales del customer, incluidos los de compras a p0, A y B, siguen en `../customer_product2_per_period/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.compile_main_metrics
```
