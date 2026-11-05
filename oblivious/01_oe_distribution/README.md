# OE distribution

[Índice de experimentos oblivious](../README.md).

En este experimento de **OE distribution con estrategias mixtas**, la distribución del rival
usada para optimizar debe coincidir con la marginal de la distribución conjunta
del juego real de dos vendedores. En las figuras se identifica como
**OE distribution**.

Este experimento reutiliza las funciones de demanda, evaluación de políticas y
transiciones de `oblivious/02_oe_p0/code/policy.py`, la verificación estacionaria de
`oblivious/02_oe_p0` y las funciones del juego conjunto en `oblivious/02_oe_p0/code/game.py`
y `oblivious/02_oe_p0/code/primitives.py`. Sus gráficos muestran los resultados de
este experimento. Sus políticas verificadas también sirven de referencia para
la comparación con **OE p0** en `oblivious/02_oe_p0`.

## Organización

```text
oblivious/01_oe_distribution/
├── README.md
├── code/
│   ├── model.py             Solver y comprobaciones de equilibrio
│   ├── step_by_step.py      Explicación ejecutable, con un caso pequeño
│   ├── diagnostics.py       Simulación, demanda y desviaciones
│   ├── run_experiment.py    Experimento completo para varios rho
│   └── test_model.py        Pruebas independientes
├── data/rho_forgetting/     Tablas, distribuciones y trayectorias
└── figures/rho_forgetting/  Figuras PNG y PDF
```

## Las condiciones que resolvemos

1. Proponemos una distribución **mu** para la reputación del rival.
2. El vendedor optimiza usando `d_mu(x) = sum_y D(x,y)*mu(y)`, donde
   `D(x,y) = P(theta(x) > theta(y))` es la competencia TS real entre esas dos
   reputaciones. Su estrategia `alpha(x)` indica la probabilidad de elegir
   el producto 2, condicionado a ser seleccionado y a su reputación propia.
3. Ambos vendedores usan esa misma estrategia en el juego real: A utiliza
   `alpha(x_A)` y B utiliza `alpha(x_B)`. El cliente compara los dos sorteos TS;
   exactamente un vendedor es seleccionado y solo él actualiza su reputación.
4. Calculamos la distribución conjunta estacionaria **nu(x_A,x_B)** del juego.
5. Extraemos la marginal de B y exigimos `mu(y) = sum_x nu(x,y)` en **todos**
   los puntos de la grilla. Por simetría, la marginal de A coincide con la de B.

La estrategia debe ser óptima frente a esa misma mu. Si `A(x)=Q2(x)-Q1(x)`:

- Si `A(x)>0`, se elige el producto 2: `alpha(x)=1`.
- Si `A(x)<0`, se elige el producto 1: `alpha(x)=0`.
- Una probabilidad interior `0<alpha(x)<1` exige `A(x)=0`.

La implementación aplica tolerancias numéricas a estas condiciones. Permitir
mezcla también permite los extremos 0 y 1. No se impone una política de umbral
ni un número fijo de estados con mezcla.

El vendedor sigue usando la marginal **incondicional**, aunque en la conjunta
las reputaciones puedan estar relacionadas. La demanda condicionada en su
propia reputación se guarda como diagnóstico; no sustituye a `D@mu` en la
optimización. Por ello este cierre no convierte el experimento en un MPE.

## Cómo lo encuentra el programa

En vez de mantener mu y alpha como dos incógnitas separadas, el solver prueba
una política alpha y obtiene inmediatamente su distribución conjunta y su
marginal real. Con esa marginal evalúa la política en el problema del OE distribution.
Después ajusta alpha hasta satisfacer la optimalidad y la indiferencia en los
estados con mezcla. Así, **mu es siempre la marginal generada por la política
que estamos evaluando**; todavía falta comprobar que esa política sea óptima.

Esta reducción explica por qué el error de consistencia marginal puede ser
cero por construcción incluso en un intento que no es equilibrio. La etiqueta
`verified` exige además optimalidad, indiferencia y una comprobación independiente
de la distribución estacionaria. No se verifica solamente una media.

Para facilitar la búsqueda se usan problemas suavizados intermedios y después
las condiciones originales sin suavizar. Se verifica de nuevo el candidato
con una evaluación de política, una mejor respuesta independiente y el solver
estacionario reforzado del juego conjunto. En rho altos se comprueban flujos
muy pequeños con eliminación positiva. Un mensaje de éxito del algoritmo de
búsqueda no basta para aceptar el resultado.

El programa principal usa tres inicios: elegir siempre el producto 1, una
probabilidad 0.5 en todos los estados y la solución de la grilla anterior. En
la primera grilla, el tercer inicio elige siempre el producto 2. Las nuevas
ejecuciones conservan todos sus intentos y las soluciones distintas que se
verifican. Esto no demuestra unicidad ni que se hayan encontrado todas.

## Ejecutar

Desde la carpeta `LLM_Learning`:

```bash
# Leer y ejecutar un caso pequeño, paso a paso.
.venv/bin/python -B -m oblivious.oe_distribution.code.step_by_step

# Barrido completo, primero rho=0.95.
.venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment

# Otro experimento, conservando los resultados existentes.
.venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment --name otro_caso --rhos 0.95 --grids 21 31 51

# Reanudar una ejecución interrumpida con los parámetros y código originales.
.venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment --resume

# Pruebas y ayuda.
.venv/bin/python -B -m unittest oblivious.oe_distribution.code.test_model
.venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment --help
```

La limpieza del 17 de octubre de 2026 conserva exactamente los resultados
seleccionados, todos los candidatos y las simulaciones guardados. Los datos y metadatos
previos se copiaron fuera del repositorio; la procedencia se registra mediante
su hash de archivo y una nota explícita en los metadatos. El historial previo
de inicios e intentos se retiró de los resultados activos: `initializations.csv`
y `solver_history.csv` están vacíos en esta corrida guardada. No se volvió a
ejecutar el solver. Las nuevas corridas registran sus propios inicios e intentos.
Para ejecutar el código actualizado usa otro `--name`; la reanudación mantiene
la comprobación de integridad del código y de los resultados.

Parámetros por defecto: `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`, `p2=0.80`,
`gamma=0.98`; rho = **0.6, 0.7, 0.8, 0.9, 0.95, 0.97 y 0.99**.
Las grillas tienen **21, 31 y 51 puntos por vendedor**: la de 51 representa
2,601 pares de reputaciones. Se audita la integración TS con el doble de puntos
y con una integración independiente. Los parámetros son configurables.

Los experimentos existentes están protegidos contra sobrescritura. Para nuevos
parámetros se utiliza otro `--name`. La reanudación comprueba que el código
coincida con el registrado, para no combinar casos de programas diferentes.
Si todos los casos ya terminaron y se corrige únicamente el código de informes,
`--resume --refresh-diagnostics` permite regenerar todos los diagnósticos y
figuras. Conserva los archivos de equilibrio y sus huellas originales, registra
la revisión del código y los informes reemplazados, y rechaza cambios en las
fuentes del solver. Esta opción se utilizó para corregir la validación de una
probabilidad que excedía 1 por un único redondeo de máquina; no se cambiaron
las tolerancias de equilibrio.

## Leer los resultados

- `RESULTADOS.txt`: resumen breve del barrido.
- `summary.csv`: un resultado seleccionado por rho y grilla. Se elige el
  primer inicio que verifica; si ninguno verifica, se identifica un diagnóstico.
- `candidates.csv`: todas las soluciones distintas verificadas; incluye las
  reputaciones con mezcla y sus probabilidades.
- `initializations.csv` y `solver_history.csv`: intentos y progreso numérico de
  nuevas corridas; vacíos en la corrida guardada tras la limpieza documentada.
- `grid_refinement.csv`: cambios entre grillas. Las soluciones comparadas no
  tienen por qué pertenecer a una misma rama cuando existe multiplicidad.
- `demand_summary.json`: demanda usada para optimizar, demanda con la marginal
  real incondicional y demanda real condicionada en la reputación propia.
- `simulation_summary.json`: promedios TS, intervalos entre trayectorias y
  comparación de las distribuciones empíricas con la marginal de la grilla.
- `deviations_summary.json`: ganancias de desviarse pudiendo usar ambas
  reputaciones, manteniendo fija la estrategia del rival.
- `arrays/`: políticas, valores, conjunta, marginales y trayectorias en NPZ,
  con metadatos JSON. Se pueden abrir sin permitir pickle.
- `manifest.json`: parámetros, archivos completados y huellas del código.
- `independent_audit.json`: comprobaciones independientes de los resultados
  guardados y de las transiciones realizadas en las simulaciones.

Las seis figuras, guardadas en PNG y PDF, muestran las probabilidades de
producto 2, la consistencia de toda la marginal, la distribución conjunta,
las demandas, la dependencia entre reputaciones y la ventaja del producto 2.
La figura [06_advantage.png](figures/rho_forgetting/06_advantage.png)
([PDF](figures/rho_forgetting/06_advantage.pdf)) muestra `A(x)=Q2(x)-Q1(x)`,
condicionada a que el vendedor haya sido seleccionado, con la misma definición
que en **OE p0**. Valores positivos favorecen el producto 2 y negativos
el producto 1; cero indica indiferencia. Los puntos marcan los nodos resueltos
y las líneas interpolan entre ellos. Los diamantes naranjas identifican los
nodos donde la política mezcla ambos productos; allí la ventaja debe ser cero
dentro de la tolerancia numérica.

En **Demand comparison**, las etiquetas y colores significan:

- Azul, **Used for product choice: rival marginal**: demanda calculada con la
  marginal del rival que usa el vendedor para elegir producto.
- Naranja, **Real game: unconditional rival distribution**: demanda promediada
  sobre toda la distribución estacionaria real del rival, sin condicionarla
  en la reputación propia.
- Verde, **Real game: conditional on own reputation**: demanda promediada sobre
  la distribución del rival condicionada en la reputación propia, calculada
  a partir de la distribución conjunta nu.

En una solución verificada, las curvas azul y naranja coinciden dentro de la
tolerancia numérica. La verde puede diferir porque conocer la reputación propia
cambia la distribución relevante del rival cuando las reputaciones están
relacionadas. Solo la azul se usa para optimizar. Las etiquetas y colores
naranja y verde son los mismos que en el experimento **OE p0**.

Se simulan todas las soluciones verificadas en la grilla más fina que haya
verificado para cada rho. Cada una utiliza cuatro trayectorias independientes
de 100,000 períodos, después de 10,000 períodos de calentamiento. Se guardan
los dos sorteos TS y las dos medias posteriores en cada período, el producto
realizado y las monedas independientes que determinan producto y éxito.
Cada trayectoria comienza sorteando un par de reputaciones de la conjunta
estacionaria calculada en la grilla. Después evoluciona con estados continuos.
Por tanto, estas simulaciones no prueban convergencia desde cualquier condición
inicial.

## Resultados del barrido guardado

Se completaron las **24 combinaciones de rho y grilla**, con al menos una
solución verificada en cada una. Se encontraron 25 soluciones distintas dentro
de sus respectivas grillas; seis tienen algún nodo con mezcla interior.
Permitir estrategias mixtas también admite soluciones que eligen un producto
con probabilidad 0 o 1 en todos los nodos.

Para **rho=0.97 y 51 puntos** se encontraron dos soluciones diferentes. Ambas
se guardan y se simulan: el candidato 0 mezcla en x=0.46 con probabilidad de
producto 2 de aproximadamente 0.3353; el candidato 1 mezcla en x=0.46 y x=0.62,
con probabilidades aproximadas 0.3419 y 0.6634. Las figuras generales muestran
el candidato 0; `candidates.csv` y los archivos de diagnóstico contienen ambos.

La auditoría independiente comprobó las 25 soluciones y las nueve simulaciones
(3.6 millones de períodos guardados). Actualmente se incluyen cinco figuras
en PNG y PDF. El máximo
error de indiferencia en nodos mixtos fue 4.61e-11 y el error de consistencia
de la marginal completa fue cero. El programa incluye comprobaciones numéricas
automatizadas.

**La verificación en la grilla no resuelve todavía la precisión en estados
continuos.** Hay sensibilidad entre grillas, especialmente para rho=0.97 y
0.99. En la simulación del candidato seleccionado de rho=0.97, la media TS de B
fue aproximadamente 0.6149, frente a 0.5746 bajo la estacionaria de la grilla.
Para rho=0.999 las medias agregadas fueron cercanas, pero las trayectorias
mantuvieron vendedores dominantes diferentes y los intervalos fueron muy
amplios: esa coincidencia no demuestra convergencia temporal.

## Alcance de las comprobaciones

La certificación corresponde a la grilla y a la demanda numérica utilizadas.
Refinar la grilla permite observar sensibilidad, pero no prueba por sí solo
convergencia al modelo continuo. La simulación continua interpola alpha entre
nodos: una mezcla interpolada no implica indiferencia fuera de la grilla.

Una mezcla en un punto con masa positiva en la grilla puede tener otro efecto
en un modelo continuo. No se demuestra aquí la existencia de una solución
continua equivalente. Tampoco se deduce rapidez de convergencia temporal de
un residuo estacionario pequeño. Los intervalos del 95% usan medias de
trayectorias independientes; cuatro trayectorias pueden omitir regímenes poco
frecuentes cuando hay mucha persistencia.

El valor `U` es el del problema de decisión con demanda promediada. Los beneficios
realizados del juego conjunto se calculan con nu y las probabilidades reales
de selección. No se presume una identidad estacionaria entre esos beneficios
y el promedio de U bajo la marginal real.
