# OE distribution with fixed outside option

[Índice de experimentos oblivious](../README.md).

Este es un **modelo nuevo**, implementado en una carpeta y una campaña
separadas. Se reutilizan rutinas numéricas del repositorio, pero se vuelve a
resolver la demanda, las políticas y las distribuciones con la opción externa;
no se toma un equilibrio anterior como resultado del nuevo modelo.

El experimento compara **un monopolista** y **dos vendedores estratégicos
simétricos** frente a la misma opción externa conocida y fija, inicialmente
`p0 = 0.5`. En competencia se conservan los dos vendedores y se agrega la
opción externa. Se comparan los cierres **Distribution OE marginal** y
**Conditional Distribution OE**, ambos con políticas que dependen solo de la
reputación propia. `p0` es un parámetro del experimento: no se resuelve un
punto fijo para determinarlo.

Para comparar los rho retenidos en una misma grilla, usa la
[comparación conjunta de 101 puntos](#comparación-conjunta-con-grilla-de-101-puntos).
Las cuatro comparaciones iniciales de `p0=0.5` están en PNG y PDF.
Se retiraron las figuras anteriores; sus datos y certificados se conservan
para revisar la sensibilidad al tamaño de grilla.

La ampliación a los seis `p0` con `rho=0.9` usa una
[grilla común de 201 puntos validada con 251](#barrido-conjunto-de-201-puntos-validado-con-251).

## Demanda y transiciones

Con reputación `x`, el comprador sortea independientemente una valoración TS

```text
theta(x) ~ Beta(1 + x/(1-rho), 1 + (1-x)/(1-rho)).
```

Sean `f_x` y `F_x` su densidad y distribución acumulada. Con dos vendedores,
el comprador elige A cuando su sorteo supera al de B y a la opción externa:

```text
D_A(x,y; p0) = Pr(theta_A >= max(theta_B,p0) | x,y)
             = integral desde p0 hasta 1 de f_x(t) F_y(t) dt.
D_B(x,y; p0) = D_A(y,x; p0).
D_out(x,y)  = F_x(p0) F_y(p0).
D_A + D_B + D_out = 1.
```

Las igualdades tienen probabilidad cero bajo estas Beta continuas. La opción
externa no tiene reputación, política ni beneficios estratégicos. Si se elige
la opción externa, **ninguna** reputación cambia. Si se elige un vendedor, solo
su reputación se actualiza; el rival espera:

```text
T_success(x) = rho*x + 1-rho,
T_failure(x) = rho*x.
```

El vendedor seleccionado elige producto `i`, recibe `R-c_i` y tiene éxito con
probabilidad `p_i`. Si no es seleccionado, recibe beneficio cero y conserva su
estado. En monopolio se elimina al rival estratégico y se conserva exactamente
la misma opción externa:

```text
d_M(x; p0) = Pr(theta(x) >= p0) = 1-F_x(p0).
```

## Cierre marginal y optimización

La política simétrica es `alpha(x) = Pr(producto 2 | seleccionado,x)`. Para una
política candidata se construye la transición conjunta real de competencia,
incluyendo los períodos en que se elige la opción externa. Su ley invariante
en tiempo calendario es `nu(x,y)`. La conjetura sobre el rival se cierra con la
marginal de esa conjunta:

```text
mu_B(y) = sum_x nu(x,y),
d_OE(x) = sum_y D_A(x,y; p0)*mu_B(y).
```

Durante la optimización se mantiene fija esta conjetura. Para monopolio se usa
`d_M`; para competencia se usa `d_OE`. En ambos casos el Bellman incluye los
períodos de espera y los mismos costos, probabilidades y descuento:

```text
U(x) = gamma*(1-d(x))*U(x)
       + d(x)*max_i {R-c_i
                    + gamma*[p_i*U(T_success(x))
                             + (1-p_i)*U(T_failure(x))]}.
```

Una mezcla interior requiere indiferencia entre ambos productos. En
competencia se buscan conjuntamente la política óptima para esa marginal y
la consistencia de la marginal con el juego real generado por la política.
En monopolio basta resolver el problema individual y calcular su invariante.
Los sucesores fuera de la grilla se distribuyen mediante interpolación lineal.

La demanda que usa el vendedor OE para optimizar es marginal; la distribución
real `nu` puede tener dependencia entre reputaciones. Por eso los beneficios
realizados se evalúan en la cadena conjunta real. **El valor realizado no se
reemplaza por el valor percibido `U` del Bellman OE.** Esto es Distribution OE,
sin cierre condicional ni una política que observe el estado completo. No se
lo identifica como un equilibrio perfecto de Markov del juego de dos estados.

## Extensión con distribución condicional

La campaña `p0_0p5_conditional_fixed` agrega el cierre condicional manteniendo
exactamente la misma demanda con dos vendedores y `p0=0.5`, transiciones,
costos, probabilidades de éxito y descuento. Para cada política candidata se
calcula su propia ley conjunta estacionaria `nu_C` y se usa:

```text
mu_B_C(y | x) = nu_C(x,y) / sum_z nu_C(x,z),
d_C(x) = sum_y D_A(x,y; p0) * mu_B_C(y | x).
```

Así, la conjetura sobre el rival puede variar con la reputación propia. El
condicionamiento se hace por reputación en tiempo calendario, antes de saber
quién vende. Cada cierre resuelve su propio punto fijo; la nueva política no
se obtiene simplemente condicionando la distribución del equilibrio marginal.
La política sigue siendo `alpha_C(x)`: el vendedor no observa el estado
realizado del rival ni resuelve un equilibrio del juego con información
completa sobre ambos estados.

Se conservan todas las filas de masa positiva, por pequeñas que sean. Sólo
para una fila de masa exactamente cero se declara la marginal rival como
convención fuera de la trayectoria. Los certificados contrastan las
condicionales fila por fila con eliminación positiva independiente en orden
inverso, además de Bellman, indiferencia, estacionariedad y demanda. Los
beneficios descontados se evalúan nuevamente en el juego conjunto real.

## Comparaciones

Los parámetros económicos comunes son `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`,
`p2=0.80`, `gamma=0.98`. La primera corrida consideró
`rho = {0.6,0.7,0.8,0.9,0.95,0.97,0.99,0.999}` y grillas de 21 y 51 puntos.
Fijó `p0=0.5` en ambos modelos. Los nuevos barridos por defecto excluyen
`rho=0.999` y conservan las otras siete tasas.

Las métricas de flujos se calculan con la ley estacionaria en tiempo
calendario, incluyendo las decisiones de no comprar al vendedor. Para A en
competencia, con `r(x)=R-c1-(c2-c1)*alpha(x)`:

```text
sale_rate             = sum_xy nu(x,y)*D_A(x,y;p0),
product2_calendar     = sum_xy nu(x,y)*D_A(x,y;p0)*alpha(x),
product2_conditional  = product2_calendar/sale_rate,
profit_calendar       = sum_xy nu(x,y)*D_A(x,y;p0)*r(x),
outside_rate          = sum_xy nu(x,y)*D_out(x,y).
```

En monopolio se usan su invariante y `d_M(x)` en las mismas expresiones de una
dimensión. `product2_conditional` es el uso del producto 2 **por venta propia**;
`product2_calendar` es el número esperado de ventas del producto 2 **por
vendedor y período calendario**. `profit_calendar` también es por vendedor,
con cero cuando no vende. Los beneficios de competencia no se suman entre
los dos vendedores para compararlos con el monopolista.

`outside_rate` es la proporción de períodos o compras potenciales del
**mercado** en que el comprador elige la opción externa y ningún vendedor
vende. En monopolio es `1-sale_rate`; en competencia simétrica es
`1-2*sale_rate`, porque `sale_rate` mide las ventas de un solo vendedor.
Comprar al rival B no cuenta como elegir la opción externa.

`value_initial` es el beneficio total descontado esperado por vendedor en su
entorno real, con descuento `gamma=0.98`. Se evalúa la política guardada desde
`x=0.5` para monopolio y `(x_A,x_B)=(0.5,0.5)` para competencia. Se resuelve
`V = r_calendar + gamma*P*V` con la transición real correspondiente; este valor
transitorio se distingue del flujo estacionario. Los dos vendedores de
competencia empiezan en la misma reputación que el monopolista.

Todos estos cálculos son deterministas en la cadena finita. No requieren
Monte Carlo ni intervalos de confianza de simulación.

## Datos y figuras

Los datos de la primera corrida se conservan en `data/p0_0p5/`.
Las tablas de resumen conservan `rho`, `n`, `p0`, `model`,
`verified`, `sale_rate`, `outside_rate`, los dos usos del producto 2,
`profit_calendar` y `value_initial`, junto con diagnósticos numéricos. La tabla de políticas
conserva la reputación y la probabilidad de producto 2 en cada nodo.

- `summary.csv` y `policies.csv`: comparación en la grilla de 51 puntos.
- `summary_all_grids.csv` y `policies_all_grids.csv`: ambas grillas.
- `grid_refinement.csv`: diferencias entre 21 y 51 puntos; para flujos
  positivos también guarda el cambio de órdenes de magnitud, como
  `log10(flujo_51)-log10(flujo_21)`.
- `arrays/`: soluciones, valores reales y percibidos, distribuciones y auditorías.
- `manifest.json`: parámetros, huellas, casos completos y fallos explícitos.
- `RESULTADOS.txt`: tabla de resultados e interpretación de las métricas.

Las cuatro comparaciones conservadas están en `figures/p0_0p5_common_grid/`,
en PNG y PDF:

1. `01_policy_comparison`: políticas frente a la reputación propia, con un
   panel por `rho` y ejes comunes. Los rombos marcan nodos de mezcla; las
   líneas unen los nodos de la grilla para facilitar la lectura.
2. `02_product2_usage_comparison`: uso estacionario del producto 2 como
   porcentaje de las ventas propias, en un solo panel.
3. `03_profit_comparison`: beneficio estacionario por vendedor/período y
   beneficio total descontado desde la reputación inicial común de 0.5,
   evaluado en el entorno real.
4. `04_outside_option_comparison`: porcentaje de períodos en que el comprador
   elige la opción externa, medido para el mercado completo, en un solo panel.

Azul continuo identifica monopolio; naranja discontinuo identifica competencia
con OE marginal; verde identifica competencia con OE condicional. Los ejes
de `rho` de las comparaciones de flujos y beneficios
espacian los casos uniformemente y muestran su valor numérico. Solo se
representan soluciones de grilla verificadas; un caso que no certifica se
conserva como diagnóstico y no se presenta como equilibrio.

Las figuras muestran `rho=0.6, 0.7, 0.8, 0.9, 0.95, 0.97` con 101 puntos.
Las políticas, el uso del producto 2, los beneficios y la elección de la opción
externa muestran los mismos casos seleccionados por el filtro de sensibilidad.

Un rho se conserva solo si **ambos modelos** cumplen simultáneamente:

- Cambio de hasta 5 puntos porcentuales en uso del producto 2 por venta,
  uso por período, tasa de ventas y elección de la opción externa.
- Cambio de hasta 5% en beneficio por período y valor inicial descontado,
  usando `abs(fino-grueso)/max(abs(fino),abs(grueso))`.
- Resultados verificados en ambas grillas para ambos modelos.

Este es un criterio de presentación para reducir sensibilidad, **no una
prueba de convergencia al modelo continuo**. El detalle del filtro está en
[grid_stability.csv](data/p0_0p5_common_grid/grid_stability.csv).
Los datos completos, las políticas guardadas, las configuraciones y los
certificados históricos se conservan.
La figura de uso tiene un solo panel, con el porcentaje de producto 2 entre
las ventas propias. La de beneficios conserva sus dos paneles. El uso por
vendedor/período calendario se conserva en los datos, pero ya no se grafica.

## Alcance de la precisión

La certificación exige optimalidad, Bellman, estacionariedad y consistencia
marginal dentro de las tolerancias numéricas; una mezcla debe satisfacer
indiferencia. Una marginal calculada a partir de una conjunta cualquiera no
certifica por sí sola el equilibrio. Los resultados guardados permiten revisar
esos residuos y comparar las dos grillas.

Un certificado se refiere a la grilla finita y a la demanda integrada
numéricamente. No prueba unicidad, completitud de la búsqueda ni convergencia
al modelo continuo. En particular, con `rho` cercano a uno, una grilla pequeña
puede distorsionar las leyes y los flujos incluso si el residual del problema
discreto es pequeño. Las diferencias entre 21 y 51 puntos deben interpretarse
como sensibilidad de discretización, sin tratarlas como error Monte Carlo.

## Primera corrida: p0 = 0.5

Los 16 casos de la corrida `p0_0p5` (ocho valores de rho y dos grillas)
obtuvieron certificado numérico. Pasaron 15 pruebas independientes del nuevo
modelo y de la comparación económica.
[summary.csv](data/p0_0p5/summary.csv) conserva los valores en 51 puntos y
[grid_refinement.csv](data/p0_0p5/grid_refinement.csv) compara las grillas.

En la grilla de 51 puntos, la competencia usa más producto 2 por venta propia
para rho entre 0.6 y 0.97. Para rho=0.95, la proporción es 56.34% en monopolio
y 74.52% en competencia. El beneficio estacionario por vendedor es 0.48794 y
0.25005 por período, respectivamente; el beneficio total descontado desde
reputación 0.5 es 21.3100 y 10.3399. En los ocho casos el valor inicial por
vendedor es menor en competencia.

Estos primeros resultados **no demuestran convergencia de grilla**. La
sensibilidad ya es fuerte en rho=0.97: el uso competitivo pasa de 50.35% con
21 puntos a 68.54% con 51. En rho=0.99 las tasas de venta cambian 8.49 órdenes
de magnitud entre ambas grillas, y en rho=0.999 cambian 5.61. Las ventas
estacionarias de la grilla de 51 puntos son aproximadamente 3.87e-18 y
1.06e-292 por vendedor y período en esos dos casos. Esas tasas estacionarias
no describen necesariamente una trayectoria finita desde reputación 0.5;
por eso se muestra también el valor inicial descontado. Los casos de alta
persistencia requieren mayor refinamiento antes de interpretar los niveles
como aproximaciones precisas del modelo continuo.

El filtro inicial entre 21 y 51 puntos retenía solo `rho=0.6, 0.7, 0.8, 0.9`.
Por ejemplo, en `rho=0.95`, el uso competitivo del producto 2 cambiaba 7.25
puntos porcentuales y el valor inicial 8.29%, por encima de los límites.
Entre aquellos casos retenidos, el cambio máximo en probabilidades era de
2.70 puntos porcentuales y el de beneficios/valor de 2.24%. El refinamiento
posterior permitió recuperar 0.95 y 0.97 para la comparación conjunta actual.

## Refinamiento adicional de rho = 0.95 y 0.97

La campaña `p0_0p5_refinement` conserva la calibración y la demanda del nuevo
modelo: se aumenta la grilla uniforme a 101, 151 y 201 reputaciones por
vendedor. Cada política interpolada es solo un inicio; cada nueva grilla
requiere certificados nuevos de optimalidad, estacionariedad y demanda.
Las corridas originales de 21 y 51 puntos se leen sin modificarlas.

Los dos rho se pueden resolver en procesos separados, con un hilo por
proceso y checkpoints independientes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fixed-p0-mpl \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.refine_case --rho .95
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fixed-p0-mpl \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.refine_case --rho .97
```

Se agrega `--resume` para continuar un proceso interrumpido con las mismas
fuentes y configuración. Se guardan las soluciones en
`data/p0_0p5_refinement/rho_0p95/` y `rho_0p97/`, con sus huellas y diagnósticos.
Por defecto se completan las tres grillas solicitadas; no se reduce la
precisión de los certificados para obtener convergencia.

En `rho=0.97`, la continuación directa desde la política interpolada se
atascó. Se completó la búsqueda proponiendo bandas de decisiones y ajustando
la probabilidad del nodo mixto para satisfacer indiferencia. En 201 puntos
se usó una solución dispersa de la ley estacionaria solo durante la búsqueda;
la propuesta final pasó la evaluación completa con eliminación positiva GTH
y las mismas tolerancias del modelo. El certificado comprueba también todas
las decisiones fuera del nodo mixto. Los scripts, registros y comandos de
reproducción de esta búsqueda se conservan en
`data/p0_0p5_refinement/rho_0p97/provenance/`.

El informe compara grillas **consecutivas**, incluyendo 101→151 y 151→201,
con el criterio de 5 puntos porcentuales en probabilidades y 5% relativo en
beneficios/valor. La tabla agregada usa el denominador simétrico
`max(abs(fino),abs(grueso))`; los checkpoints del trabajador conservan además
el diagnóstico más estricto que divide por el valor de la grilla fina.
También se guardan diferencias entre políticas ponderadas por ventas y
distancias entre las distribuciones marginales. Estas son comprobaciones de
sensibilidad observada, sin afirmar una prueba de convergencia continua ni
unicidad del equilibrio.

Los datos y el informe se conservan en `data/p0_0p5_refinement/`.

### Resultados del refinamiento

Las seis soluciones nuevas (dos valores de rho por tres grillas) obtuvieron
certificado. Ambos valores de rho pasan el filtro tanto entre 101 y 151 como
entre 151 y 201 puntos. En el último refinamiento, los mayores cambios entre
los dos modelos son:

| rho | Uso/ventas, cambio máximo (puntos porcentuales) | Beneficio/valor, cambio relativo máximo |
| --- | ---: | ---: |
| 0.95 | 0.273 | 0.2105% |
| 0.97 | 0.493 | 0.5329% |

Los porcentajes relativos de esta tabla usan el denominador simétrico del
informe agregado. Los límites siguen siendo 5 puntos porcentuales y 5%.
Las políticas también cambian cerca de sus umbrales; los datos de todas las
grillas se conservan junto con los indicadores agregados.
En competencia, la diferencia absoluta entre políticas ponderada por ventas
es 1.56% para `rho=0.95` y 4.90% para `rho=0.97` entre 151 y 201 puntos.
La mezcla de `rho=0.97` pasa de reputación 0.47333 a 0.48. La estabilidad de
los agregados no implica convergencia uniforme de las decisiones locales.

En la grilla de 201 puntos:

| rho | Modelo | Producto 2 por venta | Beneficio por vendedor y período | Valor inicial real |
| --- | --- | ---: | ---: | ---: |
| 0.95 | Monopolio | 54.23% | 0.48483 | 21.40770 |
| 0.95 | Competencia | 75.99% | 0.24599 | 10.16441 |
| 0.97 | Monopolio | 51.53% | 0.52104 | 21.86998 |
| 0.97 | Competencia | 72.99% | 0.25515 | 9.91918 |

La estabilidad agregada mejora sustancialmente respecto de 21→51 puntos.
Esto permite recuperar estos dos casos en la comparación conjunta, con el
mismo `p0=0.5`, manteniendo la interpretación de estabilidad observada de
grilla. Las 15 pruebas existentes del modelo y de la comparación económica
también pasan.

- [Comparaciones consecutivas](data/p0_0p5_refinement/consecutive_grid_comparison.csv).
- [Auditoría independiente](data/p0_0p5_refinement/independent_review.json).

## Comparación conjunta con grilla de 101 puntos

La campaña `p0_0p5_common_grid` reúne `rho=0.6, 0.7, 0.8, 0.9, 0.95, 0.97`
en una misma grilla uniforme de **101 reputaciones por vendedor**, con
`p0=0.5`. Políticas, uso del producto 2, beneficios y elección de la opción
externa usan los mismos casos y la misma resolución. Las líneas de las
políticas conectan las decisiones guardadas en esos 101 nodos.

La extensión condicional añade una tercera serie a los gráficos de comparación:
azul continuo para monopolio, naranja discontinuo para competencia con OE
marginal y verde para competencia con OE condicional. Se conserva la grilla
de 101 puntos para los tres modelos. Los nuevos datos de comparación se
guardan en `data/p0_0p5_conditional_fixed/`; los resultados marginales
anteriores se reutilizan con sus certificados y hashes originales.

Se resolvió y certificó la grilla de 101 puntos para los cuatro rho más
bajos. Para 0.95 y 0.97 se reutilizan directamente los checkpoints de 101
puntos de `p0_0p5_refinement`, con sus hashes y certificados. Las soluciones
de 51 puntos de la campaña original sirven como contraste. La comparación
conjunta conserva el filtro de sensibilidad de 5 puntos porcentuales en
uso/ventas y 5% en beneficios/valor. El contraste base es entre 51 y 101.
Para el OE condicional de `rho=0.97` se contrasta además 101 con 151: entre
51 y 101 el uso cambia 6.80 puntos porcentuales y beneficios/valor hasta
6.96%, por encima de los límites. El contraste adicional se guarda en
`data/p0_0p5_conditional_refinement/rho_0p97/`. La curva publicada sigue
usando sus decisiones y resultados de **101 puntos**.
Los resultados marginales más finos de 0.95 y 0.97 siguen disponibles en la
sección anterior.

Los doce casos condicionales de 51/101 puntos y el caso adicional de 151
obtuvieron certificado. Para `rho=0.97`, entre 101 y 151 el cambio máximo
es 0.348 puntos porcentuales en uso/ventas y 0.610% en beneficios/valor,
dentro de los límites. Los otros cinco rho pasan el contraste 51→101.
Las ocho pruebas nuevas del modelo, las siete del informe y las quince
pruebas previas del modelo con opción externa pasan. La
[auditoría independiente](data/p0_0p5_conditional_fixed/independent_review.json)
registra los trece casos, sus residuos, referencias y huellas.

`rho=0.999` sigue descartado por preferencia del experimento. `rho=0.99` se
revisó por separado y sigue mostrando fuerte sensibilidad entre 101 y 151,
incluso en el monopolio; se conserva como diagnóstico y no se reincorpora a
las figuras principales. Los intentos de mayor orden de cuadratura tienen
sus propias campañas y no reemplazan datos históricos.

Las fuentes, configuraciones, certificados y registros de los cuatro rho
nuevos se guardan por separado en `data/p0_0p5_common_grid/rho_*/`.
Los manifiestos registran el paso de 101 puntos certificado y la interrupción
del paso adicional de 151 al elegir 101 para la comparación común.
Para resolver la nueva variante se ejecuta cada rho por separado:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-conditional OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.conditional_run \
  --rho .6
```

Se repite con `.7`, `.8`, `.9`, `.95` y `.97`. Cada proceso guarda sus grillas
de 51 y 101 puntos en una subcarpeta independiente; `--resume` continúa una
corrida con la misma configuración y fuentes. La comprobación adicional de
151 puntos requiere primero el caso condicional de 101 certificado:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-conditional OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.conditional_refine_case \
  --rho .97 --grid 151
```

Para regenerar el informe y los cuatro gráficos con todas las series y esa
comprobación adicional:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-common OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report \
  --grid 101 --coarse-grid 51 \
  --conditional-refinement-name p0_0p5_conditional_refinement
```

El informe nuevo verifica las fuentes y la configuración de cada caso y
registra la procedencia de los resultados reutilizados. La exclusión previa
de 0.99 y su diagnóstico de monopolio en 151 se conservan en los datos de
`p0_0p5_common_grid` y las campañas de mayor orden de cuadratura.

- [Todas las políticas en 101 puntos](figures/p0_0p5_common_grid/01_policy_comparison.png).
- [Uso del producto 2](figures/p0_0p5_common_grid/02_product2_usage_comparison.png).
- [Beneficios y valor inicial](figures/p0_0p5_common_grid/03_profit_comparison.png).
- [Elección de la opción externa](figures/p0_0p5_common_grid/04_outside_option_comparison.png).
- [Resultados comparables de los tres modelos en CSV](data/p0_0p5_conditional_fixed/comparison_summary.csv).
- [Comparaciones de sensibilidad y grilla usada para validar](data/p0_0p5_conditional_fixed/comparison_grid_stability.csv).
- [Diagnóstico previo de los rho excluidos](data/p0_0p5_common_grid/overview.csv).

## Barrido de los p0 de base

El barrido inicial usa `p0 = 0.1, 0.3, 0.5, 0.7, 0.8`. Retoma los valores de la
[configuración base](<../../Fluid base model/config.json>) y, por elección
del usuario, reemplaza `p0=0.9` por `p0=0.8`; la configuración histórica se
conserva. Posteriormente se añadió también `p0=0.9`, cuyos datos y figuras
están en las mismas carpetas `p0_sweep`, como detalla la ampliación de abajo.
Para cada valor se comparan monopolio, OE marginal y OE condicional
con los mismos parámetros económicos y
`rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97`. Cada nuevo `p0` requiere
resolver otra vez las políticas y sus distribuciones. Los resultados de
`p0=0.5` se leen de los certificados existentes, sin modificar sus campañas
ni sus figuras.

El ejecutor guarda los casos nuevos en `data/p0_sweep/p0_*/rho_*/`. Primero
resuelve las grillas de 51 y 101 puntos; cuando sus agregados no pasan el
filtro de sensibilidad, solicita un contraste adicional con 151. Todas las
curvas publicadas conservan la solución de **101 puntos**: los resultados de
151 solo sirven para comprobar su sensibilidad. Se aplican los mismos límites
de 5 puntos porcentuales en probabilidades y 5% relativo simétrico en
beneficios/valor. La comparación inicial entre los cinco `p0` representa únicamente
los rho que tienen resultados certificados y estables en los tres modelos
para **todos** los `p0` solicitados; los casos pendientes o excluidos quedan
registrados en el informe.

Los contrastes adicionales de 151 puntos pueden detenerse si un modelo ya
descarta ese rho del conjunto común. No se necesitan entonces soluciones de
151 para los demás modelos para tomar esa decisión. Se conservan los
certificados obtenidos, las interrupciones y los pasos incompletos; un paso
de 151 sin terminar no se presenta como certificado ni reemplaza los datos
completos de 51 y 101 puntos.

Para ejecutar o continuar el barrido completo y generar el informe:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-sweep OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.multi_p0_sweep \
  --report
```

El ejecutor usa tres trabajadores por defecto. Reanuda automáticamente cada
caso que ya tenga manifiesto y verifica su configuración y fuentes. `--p0s`
y `--rhos` permiten solicitar un subconjunto de los valores seleccionados; por
ejemplo, `--p0s .1 .3 .5 .7 .8 --rhos .6 .7 .8 .9 .95 .97`. Para continuar un
solo caso, se puede usar el trabajador directamente:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-sweep OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery \
  --p0 .3 --rho .8 --resume
```

Para regenerar las comparaciones desde los datos guardados, sin ejecutar
solvers:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-sweep OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.multi_p0_report \
  --name p0_sweep
```

El informe del barrido inicial prepara cuatro comparaciones por `p0` en
`figures/p0_sweep/p0_0p1/`, `p0_0p3/`, `p0_0p5/`, `p0_0p7/` y `p0_0p8/`:
políticas, uso del producto 2 por venta propia, beneficios por vendedor y
elección de la opción externa para el mercado completo. En competencia,
comprar al rival no cuenta como outside; la tasa es `1-2*sale_rate`. El
beneficio descontado se evalúa en el juego real desde reputación inicial 0.5.
Los nombres de los cuatro gráficos, sus colores y el significado de sus
métricas coinciden con los de la comparación original.

| p0 | Políticas | Producto 2 por venta propia | Beneficios por vendedor | Opción externa del mercado |
| ---: | --- | --- | --- | --- |
| 0.1 | [Políticas](figures/p0_sweep/p0_0p1/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p1/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p1/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p1/04_outside_option_comparison.png) |
| 0.3 | [Políticas](figures/p0_sweep/p0_0p3/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p3/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p3/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p3/04_outside_option_comparison.png) |
| 0.5 | [Políticas](figures/p0_sweep/p0_0p5/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p5/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p5/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p5/04_outside_option_comparison.png) |
| 0.7 | [Políticas](figures/p0_sweep/p0_0p7/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p7/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p7/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p7/04_outside_option_comparison.png) |
| 0.8 | [Políticas](figures/p0_sweep/p0_0p8/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p8/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p8/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p8/04_outside_option_comparison.png) |
| 0.9 | [Políticas](figures/p0_sweep/p0_0p9/01_policy_comparison.png) | [Producto 2](figures/p0_sweep/p0_0p9/02_product2_usage_comparison.png) | [Beneficios](figures/p0_sweep/p0_0p9/03_profit_comparison.png) | [Outside](figures/p0_sweep/p0_0p9/04_outside_option_comparison.png) |

`data/p0_sweep/` reúne las tablas `comparison_summary.csv`,
`comparison_policies.csv`, `comparison_grid_stability.csv`, el detalle de
selección y las huellas de fuentes y artefactos. Los resultados de todas las
grillas, incluidos los casos que no se grafican, se conservan por separado.
Las soluciones de 51 y 101 puntos para los tres modelos están certificadas
en los 24 pares nuevos de `p0/rho`; para `p0=0.5` se reutilizan los resultados
anteriores. La comparación inicial de los cinco `p0` conserva
**`rho=0.6, 0.7, 0.8`**, todos con 101 puntos y los mismos tres modelos.
Los rho mayores no pasan el filtro para todos los `p0`: los contrastes de
151 puntos en `p0=0.7` descartan 0.95 y 0.97; en `p0=0.8`, el monopolio de
`rho=0.9` cambia 6.20% en beneficio por período entre 101 y 151, superando
el límite de 5%. Los resultados de 51/101 de los casos excluidos permanecen
en los datos. Esta selección común no modifica las figuras originales de
`p0=0.5`, que conservan sus seis rho y su propia validación.

La [auditoría independiente](data/p0_sweep/independent_review.json) verifica
104 checkpoints de los casos nuevos y un contraste adicional del monopolio.
Comprueba estacionariedad, valores realizados, optimalidad y las identidades
de las tablas exportadas, además de las huellas de fuentes, archivos y
referencias reutilizadas.

## Ampliación con p0 = 0.9

Esta campaña añade `p0=0.9` a las comparaciones anteriores; conserva
`p0=0.8` y los resultados históricos del barrido de cinco valores. Compara
monopolio, OE marginal y OE condicional con los mismos parámetros económicos
y las mismas definiciones de políticas, uso del producto 2 por venta propia,
beneficios por vendedor y opción externa del mercado completo.

Los casos numéricos se guardan en `data/p0_sweep/p0_0p9/rho_*/`, las tablas y
la auditoría de esta ampliación en `data/p0_sweep/p0_0p9/`, y sus figuras en
`figures/p0_sweep/p0_0p9/`. El informe agregado del barrido inicial de cinco
valores se conserva en `data/p0_sweep/`.
Se contrastan las soluciones de 51 y 101 puntos y, cuando hace falta, se
comprueba también la de 151. Las figuras conservan las políticas y agregados
de **101 puntos**; la grilla adicional solo valida su sensibilidad. Se
solicitan `rho=0.6, 0.7, 0.8`, que son los casos del conjunto común anterior,
y se publican únicamente los que pasan el filtro para los tres modelos.
Los rho 0.9, 0.95 y 0.97 ya quedaron fuera del conjunto común del barrido
anterior y no se vuelven a resolver para esta ampliación.

Los tres rho solicitados **`0.6, 0.7, 0.8`** pasaron el contraste de 51 a
101 puntos en los tres modelos; no fue necesario ampliar a 151. Quedan
certificadas las 18 soluciones de modelo, rho y grilla, y las figuras muestran
las nueve combinaciones de modelo y rho en 101 puntos. El mayor cambio entre
51 y 101 puntos para cada rho, tomando el máximo entre los tres modelos, es:

| rho | Cambio relativo del beneficio por período | Cambio en producto 2 por venta propia |
| ---: | ---: | ---: |
| 0.6 | 0.4263% | 1.1683 puntos porcentuales |
| 0.7 | 0.6948% | 0.2840 puntos porcentuales |
| 0.8 | 2.9067% | 0.1669 puntos porcentuales |

Todos quedan por debajo de los límites establecidos de 5% relativo y 5
puntos porcentuales. La certificación comprueba que los solvers resolvieron
sus problemas en cada grilla; este contraste adicional comprueba la
sensibilidad de los agregados al refinarla. Bajo ese criterio no hace falta
ampliar a 151, aunque el contraste de dos grillas no demuestra convergencia
al límite continuo. El porcentaje de
elección de la opción externa está muy cerca de 100%; ese gráfico amplía el
eje vertical cerca de 100% para mostrar las diferencias, sin cambiar la
definición de la probabilidad.

Para resolver un caso nuevo, por ejemplo `rho=0.7`:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-09 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery \
  --p0 .9 --rho .7 --grids 51 101 --max-evaluations 250
```

Se repite para `.6` y `.8`. Si el caso ya tiene manifiesto, se añade
`--resume` para continuar con la misma configuración. Una vez cerrados los
tres casos de 51 y 101 puntos, se ejecuta la auditoría independiente:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-09 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.p0_0p9_extension_audit
```

Para regenerar solo el informe y las figuras a partir de los resultados
guardados y esa auditoría:

```bash
MPLCONFIGDIR=/tmp/fixed-p0-09 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.p0_0p9_extension_report
```

| Comparación con p0 = 0.9 | PNG | PDF |
| --- | --- | --- |
| Políticas | [Ver](figures/p0_sweep/p0_0p9/01_policy_comparison.png) | [Descargar](figures/p0_sweep/p0_0p9/01_policy_comparison.pdf) |
| Producto 2 por venta propia | [Ver](figures/p0_sweep/p0_0p9/02_product2_usage_comparison.png) | [Descargar](figures/p0_sweep/p0_0p9/02_product2_usage_comparison.pdf) |
| Beneficios por vendedor y valor inicial | [Ver](figures/p0_sweep/p0_0p9/03_profit_comparison.png) | [Descargar](figures/p0_sweep/p0_0p9/03_profit_comparison.pdf) |
| Opción externa del mercado | [Ver](figures/p0_sweep/p0_0p9/04_outside_option_comparison.png) | [Descargar](figures/p0_sweep/p0_0p9/04_outside_option_comparison.pdf) |

Los [agregados comparables](data/p0_sweep/p0_0p9/comparison_summary.csv),
las [políticas](data/p0_sweep/p0_0p9/comparison_policies.csv) y el
[contraste entre grillas](data/p0_sweep/p0_0p9/comparison_grid_stability.csv)
permiten revisar los valores detrás de las figuras. La
[auditoría independiente](data/p0_sweep/p0_0p9/independent_review.json)
comprueba los checkpoints, las identidades del mercado y las huellas de sus
fuentes y referencias.

## Barrido conjunto de 201 puntos validado con 251

Esta ampliación solicitada por el usuario compara los seis valores
**`p0=0.1, 0.3, 0.5, 0.7, 0.8, 0.9`** con
**`rho=0.6, 0.7, 0.8, 0.9`**. Los tres modelos se vuelven a resolver en
**201 y 251 puntos para cada par de parámetros**: son 24 pares y 144
soluciones de modelo, parámetros y grilla. Las políticas anteriores sirven
como propuestas iniciales; la demanda, las distribuciones estacionarias,
la optimalidad y los valores realizados se comprueban de nuevo en la grilla
y el `p0` correspondientes. Los resultados previos de 101 puntos conservan
sus propios datos, figuras y alcance.

Las curvas de la nueva comparación usan siempre **201 puntos**; los
resultados de 251 permiten medir su sensibilidad. Se conserva el criterio
de 5 puntos porcentuales de cambio absoluto en probabilidades y 5% de
cambio relativo simétrico en beneficios por período y valor inicial. El
denominador de ese cambio relativo es el mayor valor absoluto de las dos
grillas. Un `rho` se incluye en las comparaciones conjuntas cuando los
tres modelos pasan estos contrastes para los seis `p0`.

La certificación del solver comprueba el equilibrio de cada problema
discreto. El contraste entre 201 y 251 comprueba la estabilidad de los
agregados al refinar esa grilla; ambos controles se registran por separado.
La auditoría de esta campaña requiere los 24 pares completos en ambas
grillas antes de publicar la comparación. Reconstruye los eventos, las
ecuaciones de estacionariedad y valor, los incentivos, las creencias
condicionales y las tablas exportadas a partir de los checkpoints guardados.

Los datos están en `data/p0_sweep_201/p0_*/rho_*/` y las nuevas figuras en
`figures/p0_sweep_201/p0_*/`. Cada `p0` conserva los cuatro gráficos:
políticas, producto 2 por venta propia, beneficios por vendedor y opción
externa del mercado completo. La tasa de opción externa cuenta únicamente
los períodos en que no compra a ningún vendedor.

El [índice de seguimiento y figuras](figures/p0_sweep_201/README.md) se
actualiza automáticamente después del cálculo, la auditoría y la verificación
de los archivos.

La fase de ejecución actual distribuye seis pares de parámetros independientes
en seis trabajadores y usa dos hilos de GTH por trabajador. El almacenamiento
de GTH pasa a una matriz densa, conservando el orden de eliminación, las
operaciones dentro de cada fila y la sustitución final. Las pruebas verifican
igualdad bit a bit con el cálculo original en ambos órdenes, incluidas las
masas positivas pequeñas; los certificados positivos y de orden inverso
siguen siendo los originales. Las reservas de memoria para esas matrices
comparten un presupuesto de **24 GiB**, repartido en seis unidades de 4 GiB.
Cada GTH de 201 puntos reserva una unidad y cada GTH de 251 reserva dos:
pueden ejecutarse hasta seis del primero o tres del segundo simultáneamente.
La búsqueda prueba mejoras de acciones puras y después resuelve las
condiciones de optimalidad de los nodos que aún las incumplen. Permite
acciones puras estrictas sin imponerles indiferencia. Esa búsqueda y
el cálculo exploratorio rápido solo proponen políticas: cada política
aceptada pasa de nuevo por la certificación completa original. Las fuentes
adicionales de Python y C++ se registran mediante sus huellas en las
referencias de los casos y en el historial de ejecución del manifiesto.

Para continuar la campaña completa con esa fase:

```bash
MPLCONFIGDIR=/tmp/sweep201-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.sweep201_complementarity_entry \
  --batch --workers 6 --max-evaluations 250 --resume
```

Para iniciar una campaña nueva se omite `--resume`. Para un
solo par de parámetros se puede usar `sweep201_complementarity_entry`, por ejemplo
`--p0 .8 --rho .9 --grids 201 251 --resume`. Al terminar los 24 pares,
se ejecutan la auditoría y el informe, en ese orden:

```bash
MPLCONFIGDIR=/tmp/sweep201-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.sweep201_audit \
  --workers 6

MPLCONFIGDIR=/tmp/sweep201-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.sweep201_report
```

El informe exige una auditoría vigente y comprueba las huellas de todos los
datos, fuentes y referencias utilizados. Los agregados de ambas grillas,
las políticas, los contrastes de sensibilidad y la selección común quedan
guardados por separado en la carpeta de datos de esta campaña.

Para ejecutar el cálculo restante, la auditoría paralela y la generación de
plots como una sola cadena automática, con la campaña detenida:

```bash
MPLCONFIGDIR=/tmp/sweep201-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.sweep201_finish
```

El estado se registra en `data/p0_sweep_201/finalization_manifest.json` y cada
etapa tiene su registro en `data/p0_sweep_201/logs/finalization_*.log`. El índice
de figuras se actualiza después de verificar la auditoría, las tablas, los PNG
y los PDF.
La auditoría distribuye los pares de parámetros entre hasta seis procesos,
con un hilo de álgebra lineal por proceso y resultados en orden fijo; se puede
usar `--workers 1` para ejecutarla secuencialmente. Revisa los archivos
guardados sin volver a resolver equilibrios ni distribuciones.

## Marginal estacionaria para p0=0.8 y rho=0.9

El [gráfico de la marginal OE](figures/marginal_distributions/p0_0p8/rho_0p9/stationary_marginal_n201.png)
usa los 201 nodos guardados de `p0_sweep_201`. Representa la masa por nodo en
tiempo calendario, obtenida de la distribución conjunta real; las líneas
unen los nodos sin suavizar. La marginal está normalizada y su reputación
media es aproximadamente `0.127584`. Se conserva también la versión SVG.

Para reproducir el gráfico desde los checkpoints verificados:

```bash
MPLCONFIGDIR=/tmp/exp08-marginal-mpl .venv/bin/python -B \
  -m oblivious.oe_distribution_fixed_p0.code.plot_marginal_distribution \
  --p0 .8 --rho .9 --n 201
```

El CSV con las masas y el JSON de procedencia se guardan en
`data/marginal_distributions/p0_0p8/rho_0p9/`.

### Evolución desde reputación inicial 0.9

El [gráfico transitorio](figures/marginal_distributions/p0_0p8/rho_0p9/transient_marginal_n201_x0_0p9_t_100_1000_10000.png)
inicia ambos vendedores en `x_A=x_B=0.9` y mantiene la misma política OE marginal
guardada. Propaga de forma determinista la ley conjunta con la transición real
en tiempo calendario y muestra la marginal después de 100, 1.000 y 10.000
períodos, junto a la estacionaria. No calcula un nuevo equilibrio ni usa Monte Carlo.

```bash
MPLCONFIGDIR=/tmp/exp08-marginal-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  -m oblivious.oe_distribution_fixed_p0.code.plot_transient_marginal \
  --p0 .8 --rho .9 --initial .9 --periods 100 1000 10000
```

En esta cadena finita hay una sola componente fuertemente conexa y todas las
entradas diagonales son positivas: la estacionaria es única y es el límite
desde cualquier estado inicial. La convergencia puede ser lenta: a los 10.000
períodos la media es `0.34709`, frente a `0.12758` en la estacionaria. El CSV
de las marginales y el JSON de diagnósticos quedan en la misma carpeta de
datos indicada arriba, con el estado inicial y los horizontes en el nombre.

### Dos atractores de la deriva y dos relojes de observación

Para `p0=0.8`, `rho=0.9`, la política marginal guardada usa producto 1 hasta
`x=0.540`, producto 2 desde `0.545` hasta `0.895`, y producto 1 desde `0.900`.
La deriva esperada por venta propia es `(1-rho)*(q(x)-x)`, con
`q(x)=p1+(p2-p1)*alpha(x)`: tiene atracción local hacia `p1=0.35` y `p2=0.8`.
Los resultados aleatorios permiten cruzar entre ambas zonas.

El [contraste entre ventas y calendario](figures/marginal_distributions/p0_0p8/rho_0p9/sale_vs_calendar_marginal_n201.png)
muestra que la marginal observada en ventas tiene dos concentraciones visibles.
En calendario, la segunda loma queda casi oculta en escala lineal. La selección
efectiva real es `d_eff(x)=sum_y nu(x,y)*D_A(x,y)/mu_calendar(x)`, y se cumple
`mu_sale(x)=mu_calendar(x)*d_eff(x)/sale_rate`. Por eso la ocupación calendario
da mucho más peso a reputaciones bajas, donde las ventas son muy raras.

La masa en `x>=0.545` es `83.1919%` condicionando a venta propia, frente a
`0.060301%` en calendario. Las medias respectivas son `0.708186` y `0.127584`.
El diagnóstico reconstruye la demanda conjunta y contrasta la ley ponderada
por ventas con la invariante independiente de la cadena de ventas propias.

```bash
MPLCONFIGDIR=/tmp/exp08-marginal-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  -m oblivious.oe_distribution_fixed_p0.code.diagnose_stationary_bimodality
```

El CSV y el JSON de comprobaciones se guardan junto a las marginales previas,
con el prefijo `sale_vs_calendar_marginal_n201`.
