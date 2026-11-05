# Uno, dos y tres sellers con outside fijo

Este experimento agrega un tercer seller estratégico al modelo de
[outside fijo](../08_oe_distribution_fixed_p0/README.md). Compara monopolio,
OE marginal con dos y tres sellers. Por indicación del usuario, la comparación
actual se limita a distribución marginal.
Cada modelo resuelve sus propias políticas y distribución estacionaria con
la misma economía. El barrido 201/251 del experimento 08 sigue pausado.

La ampliación actual usa `rho = {0.6, 0.7, 0.8, 0.9}` y
`p0 = {0.1, 0.3, 0.5, 0.7, 0.8, 0.9}`. El refinamiento completo usa 41 puntos
para `p0=0.7,0.8,0.9`, contrastados con 31; los p0 menores conservan 31,
contrastados con 21. Su manifiesto está en
[data/high_p0_refinement_41](data/high_p0_refinement_41/manifest.json), y sus figuras
en [figures/high_p0_refinement_41](figures/high_p0_refinement_41/README.md).
El [plot 05 con referencia del monopolio](figures/high_p0_refinement_41/05_product2_by_seller_count.png)
añade líneas horizontales discontinuas por p0 y rho.
El [índice de figuras vigentes](figures/README.md) reúne las comparaciones y
las simulaciones actuales. Las publicaciones anteriores de grillas menores se
retiraron de `figures/`; los checkpoints se conservan para validar los
refinamientos. Para cada nuevo reporte se usará la grilla común más fina de la
campaña completa y auditada más reciente, con sus filtros por métrica.

El [diagnóstico de demanda sucesora para p0=.8](figures/successor_demand_p0_0p8/README.md)
añade, para cada rho `.6,.7,.8,.9`, el promedio
`(d(rho*x+1-rho)+d(rho*x))/2` y la diferencia
`d(rho*x+1-rho)-d(rho*x)` de N=1,2,3. Interpola directamente las demandas
percibidas guardadas en la grilla común de 41 puntos, sin resolver modelos.
El sombreado localiza diferencias en las probabilidades de producto 2,
incluidas las políticas mixtas; sus bordes solo delimitan celdas de nodos.
Ambos diagnósticos pasan su propio contraste con las demandas guardadas de
31 puntos para los cuatro rho. La
[vista conjunta](figures/successor_demand_p0_0p8/00_successor_demand_overview.png)
y el [PDF por rho](figures/successor_demand_p0_0p8/successor_demand_by_rho.pdf)
mantienen los colores y estilos de la comparación vigente de políticas.
Los círculos tienen área proporcional a la probabilidad estacionaria de cada
reputación propia en tiempo de calendario, obtenida marginalizando la conjunta
guardada. Las franjas bajo las políticas separan las frecuencias de N=1,2,3;
la misma escala de área se usa en todas las figuras. Los controles de transporte
Wasserstein-1 y cuantiles entre 31 y 41 puntos pasan para los cuatro rho.
La versión previa está respaldada con hashes en
`archive/successor_demand_probability_update/`, sin duplicar sus plots vigentes.

## Demanda de A al aumentar el número de sellers

El [diagnóstico de demanda condicional](figures/conditional_demand_n50/README.md)
calcula la probabilidad de elegir a A con reputaciones fijas, para `N=1..50`
sellers en total: A y `N-1` rivales. Cada rival tiene `x_R=.5`; se comparan
`x_A=.3,.5,.7`, los seis p0 actuales y `rho=.6,.7,.8,.9`. Basta especificar
las distribuciones Beta y p0; rho solo se usa para parametrizar esas Betas.
Con sorteos independientes y rivales idénticos:

```text
D_A(N) = integral_{p0}^1 f_A(t) F_R(t)^(N-1) dt
```

No requiere resolver OE, políticas ni distribuciones estacionarias. Es demanda
condicionada a esas reputaciones; para una demanda promedio de equilibrio
habría que integrar sobre la ley conjunta correspondiente a cada N.
El [PNG para rho=.8](figures/conditional_demand_n50/demand_A_rho_0p8.png) muestra
los seis p0, y el [PDF](figures/conditional_demand_n50/demand_A_by_seller_count.pdf)
reúne los cuatro rho. Los 3,600 puntos pasan el contraste con integración
adaptativa independiente, con diferencia absoluta máxima `2.72e-13`.
El [CSV](data/conditional_demand_n50/demand.csv), las
[validaciones](data/conditional_demand_n50/validation.json) y el
[manifiesto](data/conditional_demand_n50/manifest.json) registran los resultados;
el generador está en [conditional_demand.py](code/conditional_demand.py).
La publicación previa N=1..20 está respaldada con hashes en
[archive/conditional_demand_n20](archive/conditional_demand_n20/manifest.json).

Para reproducirlo desde la raíz con un nombre nuevo:

```bash
MPLCONFIGDIR=/tmp/conditional-demand-mpl .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.conditional_demand --name conditional_demand_n50_new --max-sellers 50 --p0s 0.1,0.3,0.5,0.7,0.8,0.9 --rhos 0.6,0.7,0.8,0.9 --own-reputations 0.3,0.5,0.7 --rival-reputation 0.5 --order 128
```

## Extensión a cinco sellers con p0 bajo

El [piloto N=1,2,3,4,5](figures/five_sellers_low_p0/README.md) agrega un quinto
seller para los mismos p0 (.1 y .3) y `rho=.6`. Importa byte a byte los 16
controles N=1,2,3,4 ya auditados y resuelve cuatro casos N5: dos p0 y grillas 21
y 31. El [plot vigente](figures/five_sellers_low_p0/01_product2_by_seller_count.png)
publica únicamente la grilla de 31 puntos, contrastada con 21 en todos los modelos:

| p0 | Monopolio | 2 sellers | 3 sellers | 4 sellers | 5 sellers |
| --- | ---: | ---: | ---: | ---: | ---: |
| .1 | 0.00% | 15.97% | 41.46% | 51.02% | 58.71% |
| .3 | 18.94% | 30.44% | 44.09% | 51.02% | 58.71% |

Son porcentajes de producto 2 entre ventas propias. Agregar el quinto seller
lo aumenta 7.69 puntos porcentuales frente a cuatro sellers, en ambos casos.
Este resultado se limita a `rho=.6`. Las políticas N5 coinciden entre los dos
p0 en esta grilla; por eso coincide la mezcla entre ventas propias, aunque
pueda diferir la frecuencia de venta. No se infiere igualdad en el continuo.

Los 20 resultados modelo/grilla pasan la
[auditoría independiente](data/five_sellers_low_p0/independent_audit.json).
Los dos pares pasan los controles de producto 2, outside, beneficio y valor
inicial para los cinco modelos. N5 cambia 1.08 puntos porcentuales en producto
2 entre 21 y 31; los [diagnósticos](data/five_sellers_low_p0/grid_stability.csv)
conservan los resultados de grilla menor como validación. Las
[políticas actuales](figures/five_sellers_low_p0/02_policy_comparison.png) son
soluciones certificadas en los nodos de 31 puntos, sin prueba de estabilidad
puntual de los umbrales ni de convergencia al continuo.

La extensión conserva la distribución conjunta de los cuatro rivales. El
cociente exacto por permutaciones representa 28,629,151 estados completos de
N5 con n=31 mediante 324,632 órbitas, sin aproximar rivales independientes.
El valor descontado del cociente es el promedio de cinco sellers; en el estado
inicial simétrico (.5,.5,.5,.5,.5) coincide con el valor individual. No identifica
el valor de una identidad concreta en un estado asimétrico.

Las fuentes nuevas son `five_seller_kernel.py`, `five_seller_model.py`,
`five_seller_audit.py` y `five_seller_experiment.py` en `code/`. Usan GMRES con
normalización exacta desde el inicio. Si el redondeo produce un signo inválido
en una probabilidad muy rara, se rechaza ese vector, se prolonga el arranque
positivo y se vuelve a comprobar con GMRES; no se recortan probabilidades ni
se relajan tolerancias. Se mantienen dos órdenes/referencias independientes,
iteraciones positivas desde dos leyes iniciales, invariancia 1e-11 y estabilidad
1e-7. Las 27 pruebas de la extensión incluyen un juego completo de 243 estados,
conjunta correlacionada, incentivos, valores y recuperación positiva. Las fuentes
y datos anteriores permanecen intactos.

La comparación N=1..5 reemplaza los plots del piloto N=1..4. El respaldo exacto
de estos últimos está en `archive/five_seller_update/previous_four_seller_figures.zip`,
con [registro de hashes](archive/five_seller_update/manifest.json). Los
[datos N4](data/four_sellers_low_p0/manifest.json) siguen disponibles como controles.

## Elecciones del comprador y evolución del juego

La publicación [choice_dynamics_41](figures/choice_dynamics_41/README.md)
usa las políticas ya certificadas del refinamiento, sin resolver nuevos
equilibrios:

- [Frecuencias de elección](figures/choice_dynamics_41/01_stationary_buyer_choices.png):
  barras de A/B/C/outside que suman 100% de los períodos, para uno, dos y tres
  sellers. Se muestran 23 de los 24 pares p0/rho con el filtro de estabilidad
  de outside; se omite `p0=.7,rho=.9`.
- [Evolución promedio](figures/choice_dynamics_41/02_expected_evolution.png):
  probabilidades de elegir a cada seller, reputaciones esperadas y ventas
  acumuladas esperadas, propagando exactamente la ley conjunta del juego finito.
- Trayectorias aleatorias para
  [p0=.5](figures/choice_dynamics_41/03_sample_path_p0_0p5.png) y
  [p0=.8](figures/choice_dynamics_41/03_sample_path_p0_0p8.png): reputaciones,
  seller elegido, producto entregado y ventas acumuladas de cada identidad.

Las trayectorias usan `rho=.6`, 600 períodos y reputaciones iniciales `.5`;
las grillas son 31 para p0=.5 y 41 para p0=.8. Las políticas permanecen fijas.
Solo se actualiza el seller que vende y se usa la misma lotería de interpolación
del kernel; outside congela todas las reputaciones. La simetría iguala las
frecuencias estacionarias de A/B/C, pero una realización puede tener rachas
distintas. Los transitorios son ilustraciones del juego certificado en su
grilla; no tienen un contraste propio de estabilidad entre grillas.
Los [datos y manifiesto](data/choice_dynamics_41/manifest.json) guardan semillas,
cuentas, trayectorias, expectativas y hashes. Siete tests contrastan la simulación
con el kernel y comprueban las actualizaciones y la propagación de probabilidades.

## Modelo

Se mantienen `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`, `p2=0.80`,
`gamma=0.98`. El comprador sortea una valoración Beta independiente por
seller, condicionada en sus reputaciones. Compra al mayor sorteo si supera
al outside conocido `p0`. Para tres sellers:

```text
D_A(x,y,z;p0) = integral_{p0}^1 f_x(t) F_y(t) F_z(t) dt
D_out(x,y,z)  = F_x(p0) F_y(p0) F_z(p0)
D_A + D_B + D_C + D_out = 1
```

El seller seleccionado decide qué producto entregar. Su reputación cambia a
`rho*x + 1-rho` con éxito y a `rho*x` con fracaso. Las reputaciones de los
otros sellers quedan fijas. Si gana el outside, ninguna cambia. La política
es simétrica y depende solo de la reputación propia: `alpha(x)`.

La ley conjunta real en tiempo calendario es `nu(x,y,z)`. En OE marginal, el
seller usa la distribución **conjunta** de los otros dos:

```text
R_BC(y,z) = sum_x nu(x,y,z)
d_marginal(x) = sum_yz D_A(x,y,z;p0) R_BC(y,z)
```

No se sustituye esa conjunta por un producto de marginales independientes.
El Bellman unidimensional incluye los períodos sin venta y el
descuento; el punto fijo vuelve a resolver la ley conjunta y la política.
Esto es un equilibrio oblivious, no un equilibrio de información completa.

## Validación y métricas

La cuadratura de demanda se contrasta con orden doblado y con integrales
adaptativas independientes en los tests. La simetría permite agrupar
exactamente los estados que difieren solo en la identidad de los sellers;
se comprueba esa reducción contra todas las filas del kernel completo.

La ley estacionaria usa tasas de salida calculadas sumando flujos positivos,
evitando restar probabilidades casi iguales a uno. Se comprueban métodos y
órdenes independientes, positividad, normalización e invariancia de flujos.
Las cadenas pequeñas incluyen eliminación positiva GTH en ambos órdenes;
las grandes requieren además iteraciones positivas desde dos distribuciones
iniciales. Agotar un presupuesto de comprobación produce un fallo explícito.
En el cierre condicional se comparan también todas las filas positivas por
separado. No se descartan probabilidades pequeñas mediante un umbral de masa.

Las figuras comparan:

- Política de producto 2 según reputación propia.
- Producto 2 como porcentaje de las **ventas propias**.
- Beneficio por seller y período, y valor descontado real desde reputaciones
  iniciales `0.5` para todos los sellers.
- Outside como porcentaje de los períodos de todo el mercado: `1 - N*sale_rate`.

Se contrasta la ley de reputación por venta propia con una cadena independiente
unidimensional. El valor descontado se evalúa en el juego conjunto real,
separado del valor percibido del Bellman OE. Las comparaciones entre grillas
admiten como máximo 5 puntos porcentuales en probabilidades y 5% de diferencia
relativa simétrica en beneficios y valor. Estos controles no prueban
convergencia al continuo, unicidad ni estabilidad exacta de los umbrales.

## Barrido N=1,2,3: resultados y estabilidad

La campaña `high_p0_refinement_41` conserva los 144 resultados del barrido
21/31 y añade 36 cálculos de 41 puntos para los p0 altos. Los 180 resultados
pasaron la auditoría independiente. Para cada `(p0,rho)`, los tres modelos
comparten la misma grilla publicada y la misma pareja de grillas de contraste:

| p0 | Grilla publicada | Grilla de contraste |
| --- | ---: | ---: |
| .1, .3, .5 | 31 | 21 |
| .7, .8, .9 | 41 | 31 |

Las figuras aplican el filtro por métrica: producto 2 conserva 22/24 pares,
outside y frecuencias de elección 23/24, beneficio estacionario 19/24 y valor
inicial 23/24. Si falla algún modelo, se omite la comparación completa de tres
modelos para esa métrica. Las líneas dejan huecos en los casos omitidos.
Para producto 2 se omiten `(p0,rho)=(.3,.8)` y `(.8,.9)`; para outside y
frecuencias, `(.7,.9)`. Una grilla mayor por sí sola no prueba estabilidad.

El [índice de comparaciones](figures/high_p0_refinement_41/README.md) muestra los
contrastes por caso y métrica. Los [diagnósticos de grilla](data/high_p0_refinement_41/expanded_grid_stability.csv),
la [retención por métrica](data/high_p0_refinement_41/metric_retention.json) y la
[auditoría](data/high_p0_refinement_41/independent_audit.json) respaldan la selección.
El [plot 05 vigente](figures/high_p0_refinement_41/05_product2_by_seller_count.png)
conserva las referencias horizontales del monopolio para los mismos p0 y rho.

La coincidencia de las políticas en una grilla no garantiza que coincidan con
más resolución. Se documentan por separado la
[igualdad entre modelos](data/high_p0_refinement_41/high_p0_policy_equality.csv)
y los [cambios entre grillas](data/high_p0_refinement_41/high_p0_policy_refinement.csv).
La auditoría certifica el juego finito; no demuestra convergencia al continuo,
unicidad ni equilibrio de información completa.

## Consultar y validar los resultados actuales

El primer comando comprueba los datos y contrastes de grilla sin publicar
figuras. El segundo genera una nueva publicación de simulaciones a partir de
las políticas certificadas, sin volver a resolver equilibrios:

```bash
MPLCONFIGDIR=/tmp/highp0-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.high_p0_report --check-only
MPLCONFIGDIR=/tmp/choice-dynamics-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.choice_dynamics --name choice_dynamics_new --periods 600 --seed 20261025
```

El generador de simulaciones exige otro `--name` para preservar una publicación
existente. Tras comprobar una nueva versión, se sustituye la anterior en el
índice y se retiran sus figuras de la carpeta activa.

## Registro de la limpieza

Se retiraron 78 archivos PNG/PDF de `marginal_pilot`, `rho_p0_sweep_31` y
`rho_p0_sweep_31_monopoly_reference`, además de sus índices y el índice del
barrido parcial `rho_p0_sweep`. Sus bytes y rutas originales quedan en
`archive/plot_cleanup/superseded_grid_figures.zip`; el
[registro de la limpieza](archive/plot_cleanup/manifest.json) identifica sus hashes.
Este respaldo también conserva la guía previa con los comandos históricos.
Los recibos de finalización originales describen el estado al terminar cada
campaña; la limpieza posterior queda registrada por separado. No se modificaron
los resultados numéricos, las fuentes científicas ni los plots vigentes.
