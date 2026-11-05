# OE p0 con estrategias mixtas

[Índice de experimentos oblivious](../README.md).

Este experimento permite **estrategias mixtas**. La política
`alpha(x)` es la probabilidad de elegir el producto 2 cuando el vendedor con
reputación `x` es seleccionado. El producto 1 se elige con probabilidad
`1-alpha(x)`.

Las funciones numéricas compartidas están en `code/policy.py`; las del juego
de dos vendedores están en `code/game.py` y `code/primitives.py`. La comparación
de políticas usa únicamente resultados guardados y verificados de
[OE distribution](../01_oe_distribution/README.md), con los mismos
parámetros económicos. Las leyendas identifican los modelos como
**OE p0** y **OE distribution**.

## Organización

```text
oblivious/02_oe_p0/
├── README.md
├── experiment_specification.txt
├── code/
│   ├── core.py              Optimización, mezcla y consistencia de p₀
│   ├── policy.py            Demanda, evaluación de políticas y ley estacionaria compartidas
│   ├── game.py              Transiciones, evaluación y desviaciones del juego conjunto
│   ├── primitives.py        Parámetros, demanda Beta e interpolación
│   ├── diagnostics.py       Simulaciones, demanda y comparación con OE distribution
│   ├── run_experiment.py    Programa principal
│   └── test_experiment.py   Comprobaciones numéricas
├── data/rho_forgetting_mixed/
│   ├── RESULTADOS.txt
│   ├── summary.csv
│   ├── fixed_points.csv
│   ├── manifest.json
│   └── arrays/              Datos numéricos y metadatos
└── figures/rho_forgetting_mixed/
```

Abre el [resumen de resultados](data/rho_forgetting_mixed/RESULTADOS.txt), la
[tabla de candidatos](data/rho_forgetting_mixed/fixed_points.csv) o la
[comparación de demanda](figures/rho_forgetting_mixed/07_demand_comparison.png).

El barrido guardado usó **rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99 y 0.999**, con
`R=1, c1=0.05, c2=0.65, p1=0.35, p2=0.80, gamma=0.98` y grillas de 21, 31 y
51 puntos de reputación. Los parámetros son configurables.
Los nuevos barridos por defecto excluyen `rho=0.999`.
Una grilla de 51 puntos representa 51 reputaciones posibles por vendedor y
51 × 51 = 2,601 pares de reputaciones en el juego de dos vendedores.

## Resultados guardados

La ejecución terminó para las **24 combinaciones** de ocho rho y tres grillas.
Se detectaron **34 candidatos verificados**, de los cuales **11 tienen mezcla
interior**. Todos satisfacen las condiciones numéricas de optimalidad mixta,
estacionariedad conjunta y consistencia en su propia grilla; el mayor error
`abs(F-p₀)` entre esos candidatos es aproximadamente `2.92e-9`, menor que la
tolerancia `1e-7`. El programa incluye comprobaciones numéricas automatizadas.

Esta tabla muestra el p₀ del candidato seleccionado para cada caso. Un asterisco
indica que ese candidato mezcla en al menos un punto de la grilla. Si existen
varios candidatos, todos se conservan en `fixed_points.csv`.

| rho | p₀: 21 puntos | p₀: 31 puntos | p₀: 51 puntos |
| --- | ---: | ---: | ---: |
| 0.6 | 0.498541 | 0.502760* | 0.499987 |
| 0.7 | 0.533610 | 0.540564* | 0.529007* |
| 0.8 | 0.576941* | 0.590278 | 0.588874 |
| 0.9 | 0.656640 | 0.665967 | 0.669681 |
| 0.95 | 0.619018 | 0.662668* | 0.680475* |
| 0.97 | 0.506782* | 0.581377* | 0.639605* |
| 0.99 | 0.210241 | 0.279225 | 0.434271* |
| 0.999 | 0.175649 | 0.175649 | 0.175649 |

Por ejemplo, con **rho=0.95 y 51 puntos**, p₀=0.680474794 y la política mezcla
en reputación 0.46: elige el producto 2 con probabilidad 0.783872824. En ese
punto los dos productos tienen el mismo valor dentro de la tolerancia numérica.

**La verificación en la grilla no demuestra convergencia al modelo continuo.**
Los cambios de p₀ entre grillas son considerables para algunos rho. Además,
con rho=0.97 la simulación continua produce un promedio TS de B de 0.677597,
frente al valor de grilla 0.639605; su intervalo del 95% es [0.676436, 0.678759].
Para el candidato seleccionado de rho=0.99, los valores son 0.476831 y
0.434271, respectivamente, y también quedan separados por el intervalo
reportado. Estas discrepancias se conservan en las tablas; no se ha
determinado cuánto corresponde a discretización,
interpolación de la política o dinámica de largo plazo. Los intervalos usan
solo cuatro trayectorias independientes y pueden omitir regímenes poco frecuentes.
En rho=0.999 el intervalo TS es muy amplio: que contenga F aporta poca precisión.
Se simularon los 12 candidatos encontrados en las grillas de 51 puntos;
9 intervalos TS contienen F y 10 intervalos de TS menos media posterior
contienen cero. Son diagnósticos de estas trayectorias, no pruebas de igualdad.

La consistencia de p₀ tampoco elimina los incentivos a desviarse en el juego
real. Por ejemplo, para rho=0.95 la ganancia descontada por desviación,
promediada con la distribución estacionaria, es aproximadamente 0.712075 por
vendedor. La diferencia absoluta ponderada entre demanda fija y demanda
condicional es de 7.12 puntos porcentuales. Los diagnósticos completos están
en `deviation_summary.csv` y `demand_errors.csv`.

## Qué se resuelve

Para un p₀ dado, la demanda usada por el vendedor para optimizar es
`d_fixed(x) = P(theta(x) > p₀)`, calculada con la cola superior de la
correspondiente distribución Beta. No se sustituye por una comparación de medias.

El programa busca p₀ y la política alpha conjuntamente, exigiendo:

1. **Optimalidad:** si el producto 2 tiene mayor valor, alpha=1; si tiene menor
   valor, alpha=0; una probabilidad interior exige que ambos productos tengan
   el mismo valor, dentro de la tolerancia numérica.
2. **Estacionariedad:** ambos vendedores siguen alpha en el juego real. El
   cliente compara sus dos sorteos TS, y solo el vendedor elegido actualiza su
   reputación. De ahí se calcula la distribución conjunta estacionaria nu.
3. **Consistencia:** p₀ debe coincidir con la media del sorteo TS del vendedor B
   bajo esa distribución: `p₀ = sum_x mu_B(x)*(1-rho+x)/(3-2*rho)`.

El costo esperado al ser seleccionado es `c1+alpha*(c2-c1)` y la probabilidad
esperada de éxito es `p1+alpha*(p2-p1)`. En la simulación se sortea un producto
concreto y después su éxito, usando números aleatorios independientes.

**Permitir mezcla no obliga a mezclar en todos los estados.** Las probabilidades
0 y 1 son extremos admisibles dentro del mismo modelo. No se fuerza un umbral
ni un número de intervalos de uso del producto 2.

Cuando hay indiferencia, varias políticas óptimas pueden corresponder al mismo
p₀ y producir distintos promedios TS. Por eso la consistencia depende de
**F(p₀, alpha)**: no basta con buscar raíces de una única función definida por
una regla arbitraria para resolver empates. El programa localiza cambios de
política, comprueba los empates de valores y busca probabilidades de mezcla
compatibles con el promedio TS. Las decisiones iniciales puras son pasos de
búsqueda; todos los candidatos finales se verifican con las condiciones mixtas.

## Ejecutar

Desde `LLM_Learning`:

```bash
# Los ocho rho; primero valida 0.95.
.venv/bin/python -B -m oblivious.oe_p0.code.run_experiment

# Un experimento propio, con un nombre nuevo.
.venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --name mi_experimento --rhos 0.95 --grids 21 31 51

# Reanudar una ejecución interrumpida con sus parámetros guardados.
.venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --resume

# Ayuda y comprobaciones.
.venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --help
.venv/bin/python -B -m unittest oblivious.oe_p0.code.test_experiment
```

Los resultados existentes están protegidos. Usa un nombre nuevo para otros
parámetros. `--resume` conserva los casos terminados y usa la configuración
guardada. También se puede abrir y ejecutar directamente
[code/run_experiment.py](code/run_experiment.py).

## Cómo leer las tablas y figuras

- `02_policy.pdf` / `.png`: políticas seleccionadas de OE p0, en paneles de dos
  columnas (cuatro filas para los ocho rho). Las líneas unen los nodos guardados
  y los rombos indican elecciones mixtas en esos nodos.
- `03_advantage.pdf` / `.png`: ventaja del producto 2 respecto al producto 1,
  con el mismo formato y una escala vertical común. La línea en cero indica
  indiferencia. El gráfico combinado anterior `03_policy_and_advantage` se
  conserva en la carpeta de figuras de la corrida existente.
- `summary.csv`: un candidato seleccionado por rho y grilla, número total de
  candidatos y cuántos tienen mezcla interior.
- `fixed_points.csv`: todos los candidatos detectados y verificados; incluye
  las reputaciones con mezcla y las probabilidades de elegir producto 2.
- `map_evaluations.csv`: valores utilizados para localizar candidatos. La rama
  con una regla pura para desempatar es una referencia de búsqueda, no toda
  la correspondencia de estrategias óptimas.
- `grid_refinement.csv`: cambios entre grillas; no demuestra continuidad de
  una misma rama cuando hay varios candidatos.
- `simulation_summary.csv`: medias TS y posteriores, intervalos de confianza,
  productos, beneficios y cuotas de mercado.
- `demand_errors.csv`: diferencias entre las tres demandas, tanto máximas como
  ponderadas por la distribución estacionaria de reputación propia.
- `model_comparisons.csv`: comparación con políticas verificadas de
  `oblivious/01_oe_distribution` que tienen los mismos parámetros económicos.
- `deviation_summary.csv`: ganancias de desviarse usando ambas reputaciones.
- `arrays/`: valores, alpha, ventajas, demandas, nu y trayectorias. Los archivos
  NPZ tienen metadatos JSON y se abren sin pickle.

Las figuras principales usan la grilla más fina con candidatos verificados.
Si hay candidatos con mezcla interior se muestra el de menor p₀; si no,
se muestra el candidato de menor p₀ entre los extremos puros admisibles.
Se guardan todos los demás y se simulan todos los candidatos de esa grilla.
Si ninguna grilla verifica un candidato se muestra un diagnóstico identificado
como tal. La búsqueda numérica no garantiza haber encontrado todos los candidatos.
Si varios estados son indiferentes simultáneamente, puede haber muchas
combinaciones de probabilidades, incluso un continuo de soluciones. El programa
explora una colección finita de combinaciones y guarda las que verifica; el
número de filas guardadas no constituye un recuento de todas las soluciones.

La figura de políticas muestra puntos resueltos y líneas que interpolan alpha.
Una probabilidad interior entre puntos de la grilla **no demuestra indiferencia
fuera de ella**. El simulador utiliza esa interpolación como extensión explícita,
igual que el experimento de OE distribution.

En una grilla, un punto de mezcla puede tener masa estacionaria positiva. Si
en el modelo continuo un punto aislado tuviera probabilidad cero, mezclar solo
en ese punto podría no cambiar los promedios de largo plazo. Aquí no se demuestra
si la distribución continua tiene puntos con masa positiva. Por eso, verificar
una solución mixta en la grilla no demuestra que exista la misma solución en
el modelo continuo; la interpolación del simulador también debe tenerse en cuenta.

## Qué se comprueba y qué no está demostrado

Se verifica de nuevo el valor de cada política, la optimalidad de las dos
acciones, la indiferencia en todos los estados mixtos, la distribución conjunta
real y el error `abs(F-p₀)`. Un resultado no se acepta solo porque un algoritmo
anuncie éxito o una curva cambie de signo. Los logaritmos o suavizados que puedan
usarse durante una búsqueda no sustituyen la indiferencia del modelo original.

La demanda conjunta proviene de `code/game.py` y `code/primitives.py`. Se comprueba
con integración independiente y el doble de puntos de cuadratura. La
estacionariedad se verifica en la matriz completa y con órdenes numéricos
independientes; para rho altos se utiliza además eliminación positiva para
preservar flujos muy pequeños. Las evaluaciones exploratorias no sustituyen
estas comprobaciones de los candidatos finales.

En **Demand comparison**, las etiquetas y colores significan:

- Azul, **Used for product choice: fixed p₀**: demanda contra el p₀ fijo usado
  para elegir producto. p₀ es constante dentro del problema de decisión y se
  determina mediante la condición de consistencia.
- Naranja, **Real game: unconditional rival distribution**: demanda promediada
  sobre toda la distribución estacionaria del rival, sin condicionarla en la
  reputación propia.
- Verde, **Real game: conditional on own reputation**: demanda promediada sobre
  la distribución del rival condicionada en la reputación propia, calculada
  a partir de la distribución conjunta nu.

**Solo la curva azul se usa para optimizar en este experimento.** Las otras dos
son diagnósticos del juego real y tienen las mismas etiquetas y colores en el
experimento de OE distribution. Las diferencias entre ellas reflejan que
conocer la reputación propia puede cambiar la distribución relevante del rival.

Cada simulación guarda ambos sorteos TS y ambas medias posteriores antes de
la decisión, incluso para el vendedor no seleccionado. Las trayectorias se
inicializan con la distribución conjunta. Se reportan intervalos del 95%
calculados entre trayectorias independientes, sin tratar los períodos como
observaciones independientes. Un calentamiento o un intervalo amplio no
prueban convergencia a largo plazo.

El sorteo TS menos su media condicional es una diferencia de martingala
acotada. Esto justifica comparar sus promedios. Identificar además ese promedio
con la media estacionaria requiere condiciones de largo plazo apropiadas.
La coincidencia TS–media posterior no garantiza que la simulación continua
coincida con el promedio calculado en una grilla.

**Consistencia escalar con mezcla no implica MPE.** Las ganancias por desviación
se calculan en el juego real permitiendo observar ambas reputaciones. Las cotas
numéricas corresponden al modelo finito, no al error del modelo continuo.
Las referencias de **OE distribution** conservan su certificado en la
grilla y cuadratura en que se verificaron; transferir una política a otra grilla
permite compararla, pero no transfiere ese certificado.
