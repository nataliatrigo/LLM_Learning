# Figuras vigentes: uno, dos y tres sellers

Solo se presentan las versiones actuales con las grillas más finas validadas
disponibles para cada caso, comunes a monopolio y competencia. Las figuras
anteriores de grillas menores se retiraron de esta carpeta.

| Resultados | Índice | Figura principal |
| --- | --- | --- |
| Políticas, producto 2, beneficios y outside | [Comparaciones actuales](high_p0_refinement_41/README.md) | [Producto 2 según número de sellers](high_p0_refinement_41/05_product2_by_seller_count.png) |
| Elecciones del comprador y evolución del juego | [Frecuencias y simulaciones](choice_dynamics_41/README.md) | [Quién vende y quién elige outside](choice_dynamics_41/01_stationary_buyer_choices.png) |
| Demanda sucesora con p0=.8 y frecuencias de estados | [Diagnósticos por rho](successor_demand_p0_0p8/README.md) | [Promedio, sensibilidad y estados más frecuentes](successor_demand_p0_0p8/00_successor_demand_overview.png) |
| Piloto de cinco sellers, p0=.1,.3 y rho=.6 | [Comparación N=1,2,3,4,5](five_sellers_low_p0/README.md) | [Producto 2 con cinco sellers](five_sellers_low_p0/01_product2_by_seller_count.png) |
| Demanda de A con reputaciones fijas, N=1..50 | [Demanda condicional](conditional_demand_n50/README.md) | [Demanda de A para rho=.8](conditional_demand_n50/demand_A_rho_0p8.png) |

La demanda condicional es un cálculo estático separado de las campañas de
equilibrio: `D_A(N) = integral_{p0}^1 f_A(t) F_R(t)^(N-1) dt`, con N total
incluyendo A, rivales fijos en `.5` y A en `.3,.5,.7`. Usa
`p0=.1,.3,.5,.7,.8,.9` y `rho=.6,.7,.8,.9`; rho solo parametriza las Betas.
No usa OE, políticas ni resultados estacionarios. El
[PDF](conditional_demand_n50/demand_A_by_seller_count.pdf) reúne los cuatro rho;
los [datos CSV](../data/conditional_demand_n50/demand.csv) y las
[validaciones](../data/conditional_demand_n50/validation.json) conservan los 3,600
puntos, contrastados con integración adaptativa independiente (diferencia
absoluta máxima `2.72e-13`). La [fuente](../code/conditional_demand.py) y el
[índice del diagnóstico](conditional_demand_n50/README.md) incluyen su reproducción.

Grillas utilizadas para los cuatro rho `.6,.7,.8,.9`:

| p0 | Grilla publicada | Grilla de contraste |
| --- | ---: | ---: |
| .1, .3, .5 | 31 | 21 |
| .7, .8, .9 | 41 | 31 |

Cada métrica omite los casos que fallan su contraste entre grillas. Producto 2
retiene 22/24 pares p0/rho; outside y frecuencias de elección, 23/24;
beneficio estacionario, 19/24; valor descontado inicial, 23/24. Las políticas
son soluciones certificadas de la grilla, sin prueba de estabilidad puntual
de sus umbrales. Las simulaciones ilustran la dinámica en esas grillas con
políticas fijas; no tienen un contraste propio de estabilidad del transitorio.

Los datos de validación de grillas menores permanecen disponibles en `../data/`.
La [guía del experimento](../README.md) explica el modelo y la validación.

El piloto de cinco sellers usa 31 puntos, contrastados con 21, para los cinco
modelos y los dos p0. Los dos pares pasan los controles de producto 2, outside,
beneficio y valor inicial. N5 cambia 1.08 puntos porcentuales en producto 2
entre grillas. No amplía el barrido de rho: se limita a rho=.6. Esta comparación
reemplaza los plots previos del piloto de cuatro sellers.
