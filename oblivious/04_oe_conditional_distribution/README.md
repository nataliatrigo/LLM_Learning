# OE conditional distribution (Conditional OE)

[Índice de experimentos oblivious](../README.md).

Variante separada de **Distribution OE** (`oblivious/01_oe_distribution`), con una
política simétrica que depende solo de la reputación propia y una conjetura
sobre el rival condicionada a esa reputación. No modifica el código ni los
resultados de los experimentos existentes.

## Modelo y cálculo

Para una política común `alpha(x) = P(producto 2 | seleccionado, x)`, se
construye la transición conjunta dispersa del juego TS real. En cada período
calendario exactamente un vendedor es seleccionado según
`D(x_A,x_B) = P(theta_A > theta_B)`. Solo ese vendedor actualiza su estado:
`rho*x + 1-rho` si hay éxito y `rho*x` si hay fracaso. Los sucesores fuera de
la grilla usan la misma interpolación lineal de Distribution OE.

De la conjunta estacionaria `nu[a,b]` se obtiene:

```text
mu_A[a]       = sum_b nu[a,b]
mu[b | a]     = nu[a,b] / mu_A[a]
d[a]          = sum_b mu[b | a] * D[a,b]
```

Se resuelve el mismo Bellman individual, manteniendo fija esa demanda durante
la optimización:

```text
U(x) = gamma*(1-d(x))*U(x)
       + d(x)*max_i {R-c_i + gamma*[p_i*U(T_success(x))
                                  + (1-p_i)*U(T_failure(x))]}.
```

La búsqueda alterna política, estacionaria conjunta y demanda condicional.
Comienza con iteraciones amortiguadas y después usa continuación y condiciones
de complementariedad para permitir mezclas. Una mezcla interior solo se acepta
cuando ambos productos son óptimos. Las derivadas de la estacionaria aceleran
la búsqueda del punto fijo; no cambian el problema individual de elección con
demanda fija. Se prueban los mismos tres inicios que Distribution OE: producto
1, mezcla 0.5 y solución de la grilla anterior (producto 2 en la primera).

**Filas sin masa estacionaria:** se divide toda fila con masa estrictamente
positiva, aunque sea muy pequeña. No se introduce un piso que elimine estados
raros. En una fila con masa exactamente cero la condicional no está definida
por Bayes: el código adopta explícitamente la marginal incondicional del rival
como conjetura fuera de la trayectoria. Guarda esas filas y comprueba la
optimalidad de la política bajo esa convención. La consistencia con Bayes se
exige en las filas con masa positiva.

La certificación final recalcula la estacionaria mediante métodos independientes
y exige errores pequeños de Bellman, optimalidad, indiferencia, estacionariedad
y consistencia de las conjeturas y demandas condicionales. También compara las
condicionales fila por fila bajo eliminación positiva en dos órdenes de estados,
porque un error global pequeño en la conjunta puede ocultar errores relativos
en filas raras. Cerrar la condicional
por construcción no basta para declarar equilibrio. Se conservan los intentos
que no convergen, identificados como **no verificados**; no se simulan como
equilibrios. Encontrar una solución no demuestra unicidad ni convergencia al
modelo continuo.

## Ejecutar

Desde la raíz `LLM_Learning`, con un solo hilo de álgebra lineal:

```bash
# Pruebas numéricas y de comportamiento.
MPLCONFIGDIR=/tmp/conditional-oe-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest oblivious.oe_conditional_distribution.code.test_model

# Experimento completo con la configuración de Distribution OE.
MPLCONFIGDIR=/tmp/conditional-oe-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -B -m oblivious.oe_conditional_distribution.code.run_experiment

# Una corrida nueva para un rho o grillas particulares.
MPLCONFIGDIR=/tmp/conditional-oe-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -B -m oblivious.oe_conditional_distribution.code.run_experiment \
  --name rho095_n101 --rhos 0.95 --grids 51 101

# Reanudar una corrida interrumpida con su configuración y código originales.
MPLCONFIGDIR=/tmp/conditional-oe-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -B -m oblivious.oe_conditional_distribution.code.run_experiment --resume

# Regenerar únicamente las figuras desde resultados guardados.
MPLCONFIGDIR=/tmp/conditional-oe-mpl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.oe_conditional_distribution.code.run_experiment --plots-only
```

Por defecto se usan `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`, `p2=0.80`,
`gamma=0.98`; rho en `{0.6,0.7,0.8,0.9,0.95,0.97,0.99}`; grillas
de 21, 31 y 51 puntos; cuadratura TS de orden 128; tolerancia de consistencia
`1e-7` y del Bellman `1e-10`. La amortiguación inicial es `0.35`, con 20
iteraciones; se pueden cambiar con `--damping` y `--picard-iterations`.
`--max-evaluations` es un presupuesto orientativo de evaluaciones exploratorias
por inicio; las llamadas internas finales del método de raíces y la certificación
o pulido independiente pueden excederlo. Los conteos se guardan en los datos.
`--help` lista las opciones.

Los resultados existentes están protegidos contra sobrescritura. Para cambiar
parámetros, usar otro `--name`. La reanudación comprueba código y huellas de
los resultados guardados.

## Simulación y figuras

Se reutiliza el simulador real de Distribution OE: se sortean ambas Beta,
ambos vendedores usan la misma política, el no seleccionado obtiene profit
cero y solo cambia el estado del seleccionado. Se simulan todas las soluciones
verificadas de la grilla más fina que verificó para cada rho.

Se conservan cuatro trayectorias de 100,000 períodos después de 10,000 períodos
de calentamiento, semilla 20261017 con el mismo desplazamiento por rho. Cada
trayectoria empieza en un par sorteado de la conjunta estacionaria de la
grilla. Después los estados evolucionan de forma continua y la política se
interpola entre nodos, igual que en Distribution OE. Esa simulación facilita
la comparación, pero no prueba mezcla desde cualquier estado inicial ni
certifica optimalidad fuera de la grilla.

```text
oblivious/04_oe_conditional_distribution/
├── code/                  Modelo, diagnósticos, runner y pruebas
├── data/rho_forgetting/   Tablas, JSON/NPZ, trayectorias y verificaciones
└── figures/rho_forgetting/ Solo figuras PNG y PDF
```

Las seis figuras conservan la disposición de paneles, ejes, métricas,
configuración de simulación y formatos de Distribution OE. Sus títulos usan
**Conditional OE**:

1. `01_policy`: probabilidad de producto 2 frente al estado propio.
2. `02_marginal_consistency`: CDF de la marginal del rival implícita al agregar
   `mu(y|x)` con `mu_A(x)`, frente a la marginal real. Es la misma métrica CDF
   del gráfico original; la consistencia condicional completa se verifica
   aparte, fila por fila, y no se deduce solo de estas CDF.
3. `03_joint_stationary`: conjunta estacionaria en tiempo calendario.
4. `04_demand`: demanda usada para optimizar (azul), demanda con marginal
   incondicional real (naranja) y demanda condicional real (verde). En
   Conditional OE deben coincidir azul y verde donde la condicional está
   definida; la naranja puede diferir.
5. `05_joint_dependence`: diferencia entre la conjunta y el producto de sus
   marginales.
6. `06_advantage`: ventaja de producto 2 condicionada a ser seleccionado,
   con los estados de mezcla marcados.

`summary.csv`, `initializations.csv`, `solver_history.csv` y `RESULTADOS.txt`
reportan errores y convergencia. `arrays/` incluye las políticas, valores,
conjuntas, condicionales completas y simulaciones. `demand_summary.json`,
`simulation_summary.json` y `deviations_summary.json` conservan los diagnósticos
comparables. Los manifiestos registran parámetros, procedencia y huellas.

La política sigue siendo función de un solo estado. Las conjeturas provienen
de la condicional estacionaria, no de un filtro de toda la historia de ventas.
Este es el cierre **Conditional OE** solicitado; no se lo identifica como un
equilibrio de Markov perfecto del juego de dos estados. El diagnóstico de
desviaciones contra el rival permite cuantificar esa diferencia.

## Resultados de la corrida incluida

La corrida `rho_forgetting` del 20 de octubre de 2026 terminó con **23 de 24
combinaciones verificadas**. Hay una solución verificada en 51 puntos para
cada uno de los ocho rho; esas son las que aparecen en las seis figuras PNG
y PDF. Se guardaron cuatro trayectorias de 100,000 períodos por rho, además
del calentamiento: 3.2 millones de períodos retenidos en total. Las 20 pruebas
numéricas y de comportamiento pasaron.

El único caso sin convergencia es **rho=0.97, n=21**: ninguno de los tres
inicios pasó optimalidad (error de Bellman del candidato seleccionado
`0.0807761`). Esto es un fallo de búsqueda, no una prueba de inexistencia.
Para ese mismo rho, las grillas de 31 y 51 puntos sí verificaron. Se encontró
mezcla interior en rho=0.97 con 31 puntos y en rho=0.99 con 31 y 51 puntos.

Entre los candidatos seleccionados y verificados, los errores máximos fueron:

| Comprobación | Error máximo |
|---|---:|
| Bellman | `1.01e-12` (cota redondeada hacia arriba) |
| Estacionariedad de la conjunta | `3.27e-16` |
| Consistencia condicional y de demanda | `0` por construcción |
| Diferencia entre condicionales de dos cálculos estacionarios independientes, máximo L1 por fila | `1.70e-15` |

La precisión respecto al modelo continuo sigue requiriendo refinamiento.
Por ejemplo, para rho=0.97 la reputación media pasa de `0.574354` en 31 puntos
a `0.631033` en 51; para rho=0.99 pasa de `0.276006` a `0.419616`. En este
último caso, la fracción de compras del producto 2 es `0.353455` en la
estacionaria de 51 puntos y `0.339975` en la simulación continua, cuyo IC95 es
`[0.332686, 0.347264]`. Por tanto, verificar el equilibrio discreto no implica
que simulación y grilla ya coincidan. `grid_refinement.csv` y los reportes de
simulación conservan estas diferencias.

Para rho=0.999, el uso estacionario del producto 2 es aproximadamente
`6.87e-8` por compra y no se observaron compras de ese producto en los 400,000
períodos retenidos. El intervalo calculado entre las cuatro trayectorias es
entonces `[0, 0]`; esa ausencia de eventos no demuestra una tasa verdadera
exactamente cero.

`data/rho_forgetting/independent_audit.json` documenta la reconstrucción
independiente de las condicionales, transiciones, residuos, simulaciones y
métricas, junto con las versiones del entorno y las verificaciones de huellas.
