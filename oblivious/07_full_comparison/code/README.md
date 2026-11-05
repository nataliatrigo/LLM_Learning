# Full comparison

[Índice de experimentos oblivious](../../README.md).

Comparación guardada de políticas, uso del producto 2 y beneficio de **OE distribution**, **OE p0**, **Following monopoly**
(nombre público del modelo: **OE quality**) y **Conditional OE** para
`rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999`.
Los cuatro modelos utilizan la misma grilla de **51 reputaciones** por vendedor.
Se conservan las políticas seleccionadas en los experimentos originales.

## Regenerar desde la raíz del proyecto

Con los datos ya preparados, regenerar las tres figuras:

```bash
MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.plot_comparison
```

Para calcular o reanudar las simulaciones de Conditional OE y actualizar solo
las figuras de uso y beneficio:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.prepare_conditional_outcomes
MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.plot_comparison --outcomes-only
```

El primer comando reutiliza los checkpoints compatibles ya completos y guarda
el avance por rho. No vuelve a resolver el equilibrio. Si cambian las fuentes,
la configuración o el código de simulación, rechaza los checkpoints anteriores;
para una nueva corrida se deben archivar primero los archivos
`data/conditional_outcomes*` y `data/conditional_replications.csv`.

Para una primera preparación sin snapshots previos, ejecutar en este orden:

```bash
MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.prepare_data
MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.prepare_conditional_policy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.prepare_conditional_outcomes
MPLCONFIGDIR=/tmp/full-comparison-mpl .venv/bin/python -B -m oblivious.full_comparison.code.plot_comparison
```

`prepare_data.py` verifica y copia los resultados de los tres modelos originales.
`prepare_conditional_policy.py` guarda las ocho políticas seleccionadas y
verificadas de Conditional OE, comprobando parámetros, grilla y certificados.
Ambos leen resultados existentes. Sus manifiestos identifican los snapshots;
no es necesario regenerarlos para actualizar las figuras.
`plot_comparison.py` verifica las huellas de los datos antes de dibujar.
La opción `--policies-only` permite regenerar únicamente la figura de políticas.

## Refinamiento paralelo de los cuatro modelos: 101 y 201

Desde la raíz del proyecto:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_refinement
```

El lanzador procesa, en este orden, **OE p0 (endogenous)**, **OE distribution**,
**Conditional OE** y **Following monopoly / OE quality**. Termina las grillas de
un modelo antes de empezar el siguiente. Usa los siete valores de rho entre
0.6 y 0.99 del experimento original; **excluye 0.999**. Ejecuta hasta **tres rho
en paralelo en 101** y **uno en 201**, con un hilo BLAS/OpenMP por proceso.
Dentro de cada modelo termina la etapa de 101 antes de pasar a 201.

Los resultados originales se leen sin modificarlos. Los candidatos de 101
ya guardados se certifican de nuevo y se reutilizan cuando verifican. En
Distribution y Conditional se interpolan las ramas verificadas de la grilla
anterior y se intenta primero una corrección local; si falla, se usa la
búsqueda original. `--multistart` añade los inicios genéricos para explorar
otras soluciones; continuar ramas no demuestra unicidad. En OE p0 y OE quality
se mantiene la búsqueda escalar original en cada grilla nueva: una política
mixta interpolada no sustituye la indiferencia ni la consistencia del escalar.

La eliminación estacionaria positiva ahora utiliza almacenamiento disperso y
un núcleo C++17. Conserva las comprobaciones independientes de estacionariedad,
optimalidad y condicionales en filas raras, sin recortar masas ni relajar
tolerancias. La primera llamada compila la biblioteca en `/tmp` con `g++` o
`c++`; no requiere instalar paquetes de Python. Si falta el compilador, las
grillas grandes dan un error explícito en lugar de intentar la matriz densa.

Cada caso verificado se simula con el mismo diseño del Full comparison:
20 trayectorias, 10.000 períodos de burn-in, 100.000 períodos retenidos, inicio
`(0.5, 0.5)` y semilla `20261017`. Se conservan resúmenes y distribuciones, no
eventos individuales. El modo predeterminado `grid` valida el proceso discreto;
no prueba por sí solo convergencia al continuo. Para añadir una comprobación
continua de las políticas interpoladas, iniciar otra corrida con
`--simulation-modes grid linear` o `grid nearest`. Estas extensiones tampoco
certifican optimalidad fuera de los nodos.

El avance queda en `oblivious/07_full_comparison/data/refinement_101_201/`, separado por
modelo, rho y grilla. `summary.csv` reúne resultados y simulaciones;
`grid_refinement.csv` compara 101 con 201. Las diferencias del uso del producto
2 están expresadas como fracciones: multiplicar por 100 para puntos porcentuales.
Cada carpeta contiene `worker.log`, `status.json`, la solución y sus huellas.
Si no se obtiene equilibrio verificado, el caso figura como `unverified` y no
se simula; no se declara convergencia por haber terminado la búsqueda.

Para ver el avance o reanudar después de `Ctrl+C` o de apagar el equipo:

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.run_refinement --status
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_refinement --resume
```

Las búsquedas escalares guardan cada evaluación terminada y las reutilizan al
reanudar. En Distribution y Conditional se conservan casos completos, pero
una búsqueda de equilibrio interrumpida se reinicia. Las simulaciones
completas se reutilizan y una simulación interrumpida se repite. `Ctrl+C` envía
la interrupción a los workers y espera su salida; si siguen bloqueados en un
cálculo numérico, termina esos procesos sin aceptar archivos parciales.

### Completar primero 101 en los cuatro modelos

Si la campaña ya entró en 201 y se quiere priorizar la comparación en 101,
interrumpir el controlador con `Ctrl+C`, esperar a que vuelva el prompt y ejecutar:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.finish_101
```

Este lanzador continúa la misma campaña: reutiliza los casos y simulaciones
completos de 101 y procesa únicamente los pendientes de esa grilla, un modelo
a la vez y hasta tres rho en paralelo. No inicia casos de 201 ni borra sus
soluciones o evaluaciones parciales. Si se utilizó otro nombre de campaña,
añadir `--name NOMBRE`; `--workers 2` reduce la concurrencia.

Se conservan la configuración, las huellas del código numérico y la identidad
de los resultados. El seguimiento de esta etapa se guarda por separado en
`finish_101.json`. El lanzador exige que el controlador anterior esté detenido.
Para continuar con las grillas de 201 después del reintento certificado de
OE distribution, usar el lanzador `run_201.py` descrito a continuación.

Si el controlador anterior ya se detuvo pero aparece el error
`The campaign is running`, puede haber quedado desactualizada la marca del
manifiesto. En ese caso, ejecutar localmente:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.finish_101 --recover-interrupted
```

La recuperación comprueba procesos locales de la misma campaña y se bloquea
si encuentra otro controlador o worker activo. Conserva resultados, evaluaciones
parciales, configuración y huellas científicas; registra la recuperación en
los metadatos operativos. Este flag se usa después de detener el controlador
anterior, no para ejecutarlo simultáneamente.

La reanudación conserva la configuración científica y exige el mismo código;
solo permite cambiar el nombre de campaña y el número de workers. Un caso
`unverified` conserva su diagnóstico y no vuelve a buscarse automáticamente:
para probar otro presupuesto o más inicios, usar otro `--name`. Evitar ejecutar
dos controladores al mismo tiempo sobre el mismo nombre de campaña.

### Correr únicamente 201 desde los resultados certificados de 101

Desde la raíz del proyecto:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_201 --workers 2
```

Resuelve y simula **los cuatro modelos**, un modelo a la vez y hasta dos rho
en paralelo. Incluye `.6, .7, .8, .9, .95, .97, .99`; excluye `.999` y no
repite la grilla de 101. Si no se indica `--workers`, ejecuta un solo caso
a la vez. Cada proceso usa un hilo BLAS/OpenMP. Para priorizar los casos más
sensibles, ejecutar `.95`, `.97` y `.99` en una campaña separada:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_201 --name refinement_201_high_rho --rhos .95 .97 .99 --workers 2
```

El nuevo lanzador lee los resultados de `refinement_101_201` y usa el
equilibrio reparado de `oe099_retry_n101` como referencia para OE distribution
en `.99`. Conserva las tolerancias y la simulación del Full comparison:
20 trayectorias, 10.000 períodos de burn-in y 100.000 retenidos. Los resultados
se guardan por separado en `oblivious/07_full_comparison/data/refinement_201/`, con un
`worker.log` y un `status.json` por modelo y rho. La campaña anterior y el
reintento de 101 son entradas de solo lectura.

La terminal informa cada 30 segundos del tiempo transcurrido y de la última
línea del log de cada caso activo. `summary.csv` reúne los certificados y los
resultados simulados; `comparison_101_201.csv` compara las predicciones
estacionarias de ambas grillas. Su columna `product2_change_pp` expresa el
cambio del uso de producto 2 en puntos porcentuales.

Las evaluaciones escalares de 201 terminadas antes de la interrupción se
importan cuando coinciden su configuración y sus huellas numéricas. Las
políticas de 101 sirven como referencia y como inicio de búsqueda para
Distribution y Conditional; en los modelos escalares se conserva la búsqueda
completa de 201. Una política interpolada no cuenta como equilibrio certificado.

Para consultar el estado, reanudar o inspeccionar el plan sin calcular:

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.run_201 --status
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_201 --resume --workers 2
.venv/bin/python -B -m oblivious.full_comparison.code.run_201 --dry-run
```

Si se eligió otro nombre, añadir `--name NOMBRE` también al consultar o
reanudar. `Ctrl+C` detiene los workers del lanzador y conserva los checkpoints
completos. La reanudación restaura la configuración guardada; permite cambiar
el número de workers. `--solve-only` crea una campaña que resuelve los
equilibrios sin las simulaciones posteriores. Los certificados en 201 y la
comparación con 101 se reportan por separado: terminar la corrida no implica
por sí solo convergencia entre grillas.

Cuando termine la campaña, generar las comparaciones con:

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_refinement_comparison --name refinement_201 --grid 201
```

Las figuras quedan en `oblivious/07_full_comparison/Figures/refinement_201/n201/`.

### Reintento de OE distribution en rho=0.99, n=101

El nuevo buscador permite revisar tanto las decisiones puras como las mixtas
cuando una cara de indiferencia no satisface las otras condiciones de
optimalidad. Conserva el evaluador independiente y las tolerancias originales.
No impone un umbral de reputación ni restringe la búsqueda a estrategias puras.

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.retry_oe099
```

Lee el caso fallido y guarda el reintento por separado en
`oblivious/07_full_comparison/data/oe099_retry_n101/`. Conserva intentos completos y sus
huellas; no modifica la campaña original. Para continuar y generar su
simulación con los mismos 20 caminos y horizontes de la comparación:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.retry_oe099 --resume --simulate
.venv/bin/python -B -m oblivious.full_comparison.code.plot_refinement_comparison --oe-retry oe099_retry_n101
```

El segundo comando actualiza las figuras de comparación usando el resultado
aislado y su simulación, verificando ambas procedencias y la política simulada.
Los archivos originales del caso fallido se conservan. La búsqueda no prueba
unicidad ni convergencia al estado continuo.

### Comparar las grillas de 51 y 101

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.audit_grid_refinement --oe-retry 'oblivious/07_full_comparison/data/oe099_retry_n101' --output 'oblivious/07_full_comparison/data/grid_audit_51_101_with_oe_retry'
```

Guarda `comparison.csv` y `audit.json` con el uso estacionario del producto 2,
cambios en puntos porcentuales, distribuciones y políticas. No ejecuta solves
ni simulaciones. Sin `--oe-retry` compara únicamente la campaña original y
marca su caso OE 0.99/101 como no certificado.

La certificación de un equilibrio en cada grilla y la convergencia entre
grillas son comprobaciones distintas. En rho=0.99 los resultados seleccionados
todavía cambian mucho entre 51 y 101; se necesita contrastar con 201 y seguir
las ramas cuando hay varios equilibrios. Las diferencias de Monte Carlo
frente a la estacionaria se reportan por separado porque también pueden
reflejar un inicio o un horizonte insuficientes.

Ejemplos para ejecutar solo un modelo, inspeccionar el plan sin calcular y
ajustar el paralelismo:

```bash
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_refinement --models p0 --name p0_refinement_101_201
.venv/bin/python -B -m oblivious.full_comparison.code.run_refinement --dry-run
.venv/bin/python -u -B -m oblivious.full_comparison.code.run_refinement --resume --workers-101 2 --workers-201 1
```

Pruebas pequeñas de integración del lanzador:

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.test_run_refinement
```

### Figuras de OE p0 desde los resultados guardados

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_p0_refinement
```

Genera ocho familias de figuras en PNG y PDF dentro de
`oblivious/02_oe_p0/figures/refinement_101_201/n101/`: mapa de consistencia, política,
ventaja del producto 2, conjunta estacionaria, CDF de reputaciones, resultados
simulados, demandas y comparación de políticas con OE distribution. Utiliza
únicamente soluciones y simulaciones guardadas; no vuelve a resolver ni simular.
La comparación usa referencias verificadas de la misma grilla y señala si
falta alguna. `--grid 201` permite dibujar esa grilla cuando todos sus casos y
simulaciones estén completos. `figure_manifest.json` registra entradas y figuras.

### Comparaciones de los cuatro modelos desde el refinamiento

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_refinement_comparison
```

Guarda las comparaciones de políticas óptimas, uso del producto 2 y beneficio
en `oblivious/07_full_comparison/Figures/refinement_101_201/n101/`, en PNG y PDF. Conserva
los colores originales y utiliza líneas continuas. Los círculos con intervalos
MC representan simulaciones; los rombos huecos representan la predicción
estacionaria. Los casos sin equilibrio certificado no se dibujan y quedan
señalados explícitamente; no se sustituyen por otra grilla. También guarda
los datos dibujados y sus huellas en esa subcarpeta. No ejecuta solvers ni
simulaciones nuevas.

### Monopolio para distintas outside options frente a OE distribution

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_monopoly_p0_refinement
```

Genera dos comparaciones en PNG y PDF dentro de
`oblivious/07_full_comparison/Figures/refinement_101_201/n101/monopoly_p0_comparison/`:

- `01_product2_usage_vs_p0`: un panel por rho con la curva del monopolio
  `u^M(rho,p0)` y la línea horizontal del equilibrio `u^OE(rho)`.
- `02_product2_usage_vs_rho`: las curvas del monopolio para los p0 previamente
  utilizados (`0.1, 0.3, 0.5, 0.7, 0.9`) y la curva de OE distribution.

En ambos modelos, `u` es la **fracción de las ventas propias del proveedor que
utiliza producto 2**. El monopolio se calcula con el solver original de
`forgetting`, los parámetros comunes y la misma grilla uniforme de 101 nodos.
Se conserva su selector puro: producto 2 cuando la ventaja condicional supera
`1e-6`, producto 1 en caso contrario. El barrido usa los rho del refinamiento
que cumplen `rho > 0.5` y recorre `p0` desde `0.05` hasta `0.95`, con paso
`0.005`; las líneas unen las evaluaciones y no localizan
exactamente los cambios de política o los cruces con OE.

La referencia competitiva es **OE distribution**, usando los siete rho del
refinamiento y el reintento certificado de `.99/101`. Se leen sus equilibrios
guardados, sin volver a resolverlos. La ley del monopolio por venta se calcula
con la cadena de reputaciones en cada venta propia; se verifican de forma
independiente el Bellman, la estacionariedad y la ponderación de la ley
calendario por la probabilidad de venta. Las figuras muestran predicciones
estacionarias de la grilla, sin intervalos Monte Carlo.

Los datos dibujados, las políticas, las leyes estacionarias y las verificaciones
quedan en `oblivious/07_full_comparison/data/refinement_101_201/monopoly_p0_comparison/n101/`.
`comparison.csv` incluye la diferencia monopolio menos OE en puntos porcentuales;
`fixed_p0_comparison.csv` conserva los cinco p0 anteriores y `oe_reference.csv`
identifica las fuentes competitivas. Los manifiestos registran configuración,
huellas de entradas, código y figuras. Las siguientes ejecuciones reutilizan
el barrido compatible; `--plots-only` exige esos datos guardados y `--recompute`
los recalcula. Para otra campaña/grilla, usar `--name NOMBRE --grid N` cuando
estén disponibles sus equilibrios certificados.

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.test_plot_monopoly_p0_refinement
```

### Outside option igual a la calidad media del rival OE

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_quality_matched_refinement
```

Para cada `rho > 0.5` disponible en el refinamiento, calcula
`p0_ref(rho) = p1 + (p2-p1)*u_OE(rho)`, donde `u_OE` es la fracción de las
**ventas propias del rival** que utiliza producto 2. Se comprueba esta referencia
directamente con la conjunta OE y su demanda: pondera la calidad verdadera del
rival por sus ventas. No utiliza la media de su score TS ni el p0 del modelo
endógeno. No hay filtro sobre `p0_ref`: se incluyen los siete rho, aunque su
referencia sea menor que `0.5`.

Resuelve el monopolio original en esa referencia exacta, sin redondearla al
barrido anterior de p0. Conserva los parámetros, la grilla de 101 nodos y el
selector puro original. Los equilibrios OE distribution, incluido el reintento
certificado de `.99`, se leen sin modificarlos. Esta es una calibración desde
OE; no se impone un nuevo cierre de consistencia.

Guarda en `oblivious/07_full_comparison/Figures/refinement_101_201/n101/quality_matched_comparison/`:

- `01_product2_usage_quality_matched`: uso de producto 2 entre ventas propias
  en ambos modelos frente a rho, con un panel que identifica cada `p0_ref`.
- `02_policies_quality_matched`: un panel por rho con las decisiones de producto
  según reputación y la referencia utilizada; los rombos indican mezcla en OE.

Ambas figuras se generan en PNG y PDF. Los valores de referencia, diferencias
en puntos porcentuales y verificaciones quedan en `comparison.csv`; las
políticas dibujadas, en `policies.csv`. Estos archivos, el bundle numérico y el
manifiesto con huellas quedan en
`oblivious/07_full_comparison/data/refinement_101_201/quality_matched_comparison/n101/`.
`--name NOMBRE --grid N` selecciona otra campaña y grilla con resultados OE
certificados. Igualar la calidad media de la alternativa no iguala toda su
distribución de scores ni obliga a que el monopolista entregue la misma calidad.

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.test_plot_quality_matched_refinement
```

### Calidad media del rival en endogenous p0

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_quality_matched_p0_refinement
```

Repite la comparación anterior tomando como referencia los siete equilibrios
certificados de **endogenous p0 (OE p0)**, con `rho > 0.5` y grilla de 101 nodos.
Calcula `p0_ref = p1 + (p2-p1)*u_p0`, donde `u_p0` utiliza como denominador las
ventas propias del rival, y resuelve el monopolio original en cada referencia
exacta. No interpola ni redondea al barrido anterior de outside options.

`p0_ref` es la calidad verdadera del rival entre sus ventas. Es distinto del
escalar `equilibrium_p0`, que cierra el modelo endógeno con la media del score
TS del rival en tiempo calendario. El CSV conserva ambos. Usar directamente
ese escalar en el monopolio daría el mismo Bellman percibido de OE p0; esta
comparación cambia el benchmark a la calidad media del rival.

Guarda los dos gráficos (uso de producto 2 y políticas) en PNG y PDF dentro de
`oblivious/07_full_comparison/Figures/refinement_101_201/n101/quality_matched_endogenous_p0/`.
Los datos, referencias certificadas, políticas, bundle y manifiesto con huellas
quedan en
`oblivious/07_full_comparison/data/refinement_101_201/quality_matched_endogenous_p0/n101/`.
Conserva los parámetros y el selector puro original del monopolio, sin filtro
sobre p0. `--name NOMBRE --grid N` permite seleccionar otra campaña y grilla
con equilibrios p0 certificados.

### Distribuciones conjuntas: Conditional y Distribution

```bash
.venv/bin/python -B -m oblivious.full_comparison.code.plot_joint_comparison
```

Compara las conjuntas estacionarias certificadas de ambos modelos en 101,
incluyendo el resultado reparado de OE distribution en `.99`. Guarda una vista
ampliada de `.95`, `.97` y `.99` y otra de los siete rho, en PNG y PDF, dentro de
`oblivious/07_full_comparison/Figures/refinement_101_201/n101/joint_comparison/`.
Las columnas muestran Distribution, Conditional y Conditional menos Distribution.
Los dos modelos comparten la escala de masa por fila; el panel de diferencias
usa verde para más masa en Distribution y azul para más masa en Conditional.
La vista ampliada conserva al menos el 99.8% de cada conjunta. Guarda también
las matrices dibujadas, métricas y huellas. No resuelve ni simula experimentos.
`--rhos .99` selecciona un caso; `--grid 201 --name refinement_201` permite
comparar esa grilla cuando ambos modelos tengan resultados certificados.

## Figuras

Las figuras originales de la grilla de 51 se guardan en
`oblivious/07_full_comparison/Figures/n051/`, cada una en **PNG a 300 dpi y PDF**:

- `01_optimal_policies_across_rho`: ocho paneles, uno por rho, con las cuatro
  políticas `alpha(x) = P(producto 2 | vendedor seleccionado, reputación x)`.
  Los rombos identifican mezcla en un nodo; las líneas unen los nodos resueltos.
- `02_product2_usage_across_rho`: proporción de todas las compras que utilizan
  producto 2.
- `03_profit_across_rho`: beneficio medio por vendedor y período calendario,
  incluidos los períodos en que no es seleccionado.

Verde identifica OE distribution, naranja OE p0, morado Following monopoly
y azul Conditional OE. Los cuatro modelos aparecen en las tres figuras.
En las dos figuras de resultados, los círculos y líneas continuas muestran
simulaciones con intervalos MC del 95%; los rombos huecos y líneas discontinuas
muestran predicciones estacionarias de la grilla. Los valores de rho están
equiespaciados y las líneas orientan la lectura.

## Fuentes y alcance

Las políticas y leyes estacionarias seleccionadas provienen de:

- `oblivious/01_oe_distribution/data/rho_forgetting/`
- `oblivious/02_oe_p0/data/rho_forgetting_mixed/`
- `oblivious/03_oe_following_monopoly/data/quality_sales_n51/`
- `oblivious/04_oe_conditional_distribution/data/rho_forgetting/` (políticas y leyes conjuntas de 51 puntos)

Las simulaciones de los dos primeros modelos provienen de
`oblivious/06_equilibrium_comparison/data/cross_rho_n51/`; sus políticas y leyes conjuntas
se comprueban contra los experimentos originales. Following monopoly usa sus
propios diagnósticos guardados. Para Conditional OE se generan nuevas
simulaciones mediante el mismo simulador de `oblivious/06_equilibrium_comparison`, usando
sus políticas guardadas. Los cuatro usan TS real entre dos vendedores,
grilla de 51 puntos, inicio `(0.5, 0.5)`, semilla `20261017`,
**20 trayectorias independientes**, **10.000 períodos de burn-in** y
**100.000 períodos retenidos** por trayectoria.
Las transiciones se proyectan aleatoriamente a la grilla con los pesos de la
interpolación del solver. Las simulaciones originales de `oblivious/04_oe_conditional_distribution`
(cuatro trayectorias continuas iniciadas en la ley estacionaria) no se usan aquí.

Los parámetros comunes son `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`,
`p2=0.80`, `gamma=0.98`. La optimalidad está verificada numéricamente para
el problema de decisión propio de cada modelo en esta grilla; no certifica
un equilibrio de Markov perfecto del juego completo, unicidad ni convergencia
al modelo continuo. Una mezcla interpolada entre nodos no demuestra
indiferencia fuera de la grilla.

Si `nu[i,j]` es la ley conjunta y `D[i,j]` la probabilidad de seleccionar A,
el uso estacionario de producto 2 es
`sum(nu[i,j] * (D[i,j]*alpha[i] + D[j,i]*alpha[j]))`.
Hay una compra por período. El beneficio medio de los dos vendedores es
`0.5 * (R-c1-(c2-c1)*usage) = 0.475-0.3*usage`; no es un valor descontado.
Los intervalos Student-t se calculan entre medias de trayectorias y excluyen
el sesgo de inicialización. Con rho alto, simulación finita y estacionaria
pueden diferir.

## Datos y trazabilidad

Todo lo siguiente queda en `oblivious/07_full_comparison/data/`:

- `policies.csv`: una fila por modelo, rho y nodo; incluye reputación,
  probabilidad de producto 2 e indicador de mezcla para los tres modelos originales.
- `conditional_policies.csv`: las 408 filas correspondientes a Conditional OE
  (ocho rho por 51 nodos), con el mismo esquema de política.
- `conditional_policy_manifest.json`: procedencia y huellas de las políticas
  Conditional OE, parámetros comunes, certificados por rho y estilo de su curva.
- `outcomes.csv`: una fila por modelo, rho y métrica; incluye predicción
  estacionaria, media simulada, error estándar, extremos del intervalo
  y número de trayectorias válidas, para los tres modelos originales.
- `replications.csv`: uso de producto 2 y beneficio de cada trayectoria
  de los tres modelos originales.
- `conditional_outcomes.csv`: las mismas dos métricas para Conditional OE
  en los ocho rho (16 filas).
- `conditional_replications.csv`: las 160 trayectorias de Conditional OE.
- `conditional_outcomes_manifest.json`: configuración homogénea, semillas,
  procedencia, verificaciones y huellas de código, fuentes y datos nuevos.
- `conditional_outcomes/`: checkpoints JSON/NPZ por rho, con políticas,
  leyes estacionarias, resúmenes de trayectorias, diagnósticos y semillas
  reproducibles. No guarda los eventos individuales de cada trayectoria.
- `verification.csv`: verificación por modelo/rho, residuo estacionario,
  número de nodos mixtos, p0 cuando corresponde y carpeta fuente.
- `stationary_inputs.npz`: grilla, demanda de A, políticas y leyes conjuntas;
  las claves siguen `rho_<valor>_n051/grid`, `.../demand_A`,
  `.../<modelo>/policy` y `.../<modelo>/stationary_joint`.
- `comparison.json`: configuración, etiquetas, colores y definiciones.
- `manifest.json`: huellas SHA-256 de las fuentes y datos del snapshot original
  de tres modelos, y del código usado al crearlo.
- `figure_manifest.json`: huellas SHA-256 de las figuras y procedencia por
  figura, incluyendo los manifiestos adicionales de políticas y resultados
  Conditional OE.

Los identificadores internos son `oe`, `p0`, `monopoly` y `conditional`.
Al preparar los datos se verifican parámetros, políticas, estacionariedad,
predicciones, identidad contable del beneficio y resúmenes Monte Carlo.
