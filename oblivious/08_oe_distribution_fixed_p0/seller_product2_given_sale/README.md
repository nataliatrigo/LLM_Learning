# Métrica original: producto 2 entre las ventas propias

Ambos formatos solicitados usan `product2_conditional`, la métrica original del experimento 08.
Es la probabilidad de producto 2 condicionada en una venta propia, promediada sobre las reputaciones de las ventas reales.
Se calcula como `product2_calendar / sale_rate`. No es la probabilidad conjunta de vender y entregar producto 2 por período.
Se conserva c2=0.65 y la misma selección auditada de 101 nodos: seis p0 y rho=0.6,0.7,0.8.
Todas las figuras tienen eje vertical común de 0% a 100%.

- [Tres paneles frente a p0](figures/01_product2_given_sale_by_p0.png).
- [Seis paneles frente a rho](figures/02_product2_given_sale_by_rho.png).
- [Ambos gráficos en PDF](figures/original_metric_comparison.pdf).

`data/` conserva los 54 resultados originales y su contraste de grilla; `manifest.json` registra procedencia y huellas.
El contraste de grilla no demuestra convergencia al modelo continuo.

```bash
MPLCONFIGDIR=/tmp/seller-product2-given-sale-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.seller_product2_given_sale_report
```
