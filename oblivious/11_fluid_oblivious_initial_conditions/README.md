# Equilibrios fluid oblivious desde una reputación inicial común

Este experimento aplica la condición de consistencia de la imagen al modelo
de competencia fluida del [experimento 10](../10_fluid_oblivious/README.md).
El resultado de cada equilibrio es la terna **política óptima, reputación
consistente y conjunto de condiciones iniciales**:

\[
 (\sigma_{\bar x},\bar x,B_{\bar x}),\qquad
 B_{\bar x}=\{x_0\in[0,1]:\lim_{t\to\infty}x(t;x_0,\sigma_{\bar x})=\bar x\}.
\]

La política se calcula suponiendo que el rival tiene reputación fija
\(\bar x\), exactamente como en el 10. La dinámica que determina el conjunto de condiciones iniciales
usa las dos reputaciones efectivas, que coinciden cuando ambos vendedores
siguen la misma política y parten de un estado común.

## Condición implementada

Se comprueban simultáneamente:

\[
 p_1+(p_2-p_1)\sigma_{\bar x}(\bar x)=\bar x,
 \qquad
 \lim_{t\to\infty}x(t;x_0,\sigma_{\bar x})=\bar x,
\]

con

\[
 \dot x(t)=\lambda D(x(t),x(t);p_0)
 [p_1+(p_2-p_1)\sigma_{\bar x}(x(t))-x(t)],
 \quad x(0)=x_0,\quad\lambda=-\log\rho.
\]

El HJB usado para optimizar contra la conjetura fija es

\[
 \delta U(x;\bar x)=D(x,\bar x;p_0)
 \max_{a\in[0,1]}\{R-c_1-(c_2-c_1)a+
 \lambda[p_1+(p_2-p_1)a-x]U_x(x;\bar x)\},
 \quad\delta=-\log\gamma.
\]

Por tanto, para una condición inicial dada, la correspondencia de equilibrio es
\(E(p_0,\rho;x_0)=\{(\sigma_{\bar x},\bar x):x_0\in B_{\bar x}\}\).
Dos reputaciones estacionarias bajo políticas distintas pueden tener conjuntos de condiciones iniciales
separadas: que existan dos ramas estacionarias no implica dos equilibrios desde
el mismo \(x_0\).

## Parámetros y alcance

Se conserva la grilla **activa** del experimento 10:

- `p0 = 0.1, 0.3, 0.5, 0.7, 0.8, 0.9`.
- `rho = 0.6, 0.7, 0.8, 0.9`: 24 combinaciones.
- `R=1`, `c1=0.05`, `c2=0.65`, `p1=0.35`, `p2=0.8`, `gamma=0.98`.
- Mallas de 1001, 2001 y 4001 nodos; cuadratura de orden 128;
  búsqueda escalar de 121 puntos, refinada a 241 en la malla final.

Los estados iniciales son reputaciones normalizadas maduras, \(x_0\in[0,1]\).
La condición de consistencia se aplica a la competencia simétrica. El benchmark
de monopolio del 10 puede consultarse allí.

## Cálculo de los conjuntos de condiciones iniciales y extremos

Para los parámetros activos, la demanda diagonal se puede escribir como
\(D(x,x;p_0)=\tfrac12[1-F_x(p_0)^2]>0\). Se evalúa de forma estable como
`0.5 * survival * (1 + cdf)`. El factor de demanda cambia la velocidad,
mientras que el signo de la dinámica viene de
\(p_1+(p_2-p_1)\sigma(x)-x\).

Se divide la línea de fase en todos los umbrales de política y en `p1,p2`.
El destino asintótico de cada intervalo abierto es constante; cada extremo se
clasifica por separado. Se unen las piezas con el mismo destino. Así se obtiene
el conjunto completo de inicios de cada política calculada, incluidos posibles
singletons, sin muestrear una malla de `x0` ni inferir el límite a partir de
`x(T)`.

Las políticas y el flujo usan esta convención explícita:

- En los puntos de holding verificados, se utiliza la mezcla que mantiene la
  reputación. En los umbrales atractivos de los resultados actuales, esa mezcla
  coincide con el equilibrio mixto verificado.
- En un umbral repulsivo se elige producto 1 y se toma la continuación
  inmediata hacia la izquierda. Los cruces siguen la acción del tramo de salida.
- Un umbral que coincide con una reputación pura estacionaria usa la acción
  pura que la mantiene.

La continuación inmediata es parte de la convención de flujo por eventos.
Para una ODE discontinua interpretada sólo casi en todas partes, fijar la acción
en el punto inicial no garantiza por sí solo una trayectoria única. Los
extremos abiertos/cerrados declarados aquí corresponden al flujo seleccionado;
esta salvedad afecta los puntos frontera, no los interiores ni las medidas de
los conjuntos de condiciones iniciales actuales. No se afirma haber enumerado todos los selectores óptimos
en puntos de indiferencia.

## Fuentes y verificación

La condición nueva añade el conjunto de condiciones iniciales a los candidatos estacionarios: no modifica
el HJB contra el rival fijo. Por ello la ejecución predeterminada reutiliza los
checkpoints del 10 **tras comparar todos los parámetros y las huellas de las
seis fuentes numéricas**, y guarda copias idénticas en `data/checkpoints/`.
`--recompute` vuelve a resolverlos y escribe sólo en este experimento.

Se recalculan los controles de refinamiento estacionario desde los checkpoints.
También se comparan la topología y las fronteras de los conjuntos de condiciones iniciales entre las mallas
2001 y 4001; el cambio máximo permitido es `0.002`. Las fronteras guardadas son
aproximaciones numéricas, no cotas certificadas. Cerca de un umbral, la
pertenencia es sensible a ese error espacial.

La validación contrasta seis condiciones iniciales por candidato con las
predicciones guardadas del 10. Además integra nuevamente la dinámica con
demanda efectiva desde `x0=0.5`, desde cada `x_bar` y a ambos lados de cada
umbral repulsivo. Comprueba simetría, dirección, ausencia de sobrepaso del límite,
estacionariedad en `x_bar` y estabilidad al refinar el integrador. Los límites
asintóticos proceden de la línea de fase.

La búsqueda conserva la limitación del 10: detecta numéricamente candidatos
puros y mixtos y verifica el HJB, el holding y la demanda. No constituye una
prueba matemática de exhaustividad o de inexistencia cuando no encuentra uno.

## Ejecución

Desde la raíz del repositorio:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious_initial_conditions.code.run_experiment

# Resolver otra vez los HJB y búsquedas estacionarias en el experimento 11.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious_initial_conditions.code.run_experiment --recompute

# Volver a dibujar las figuras a partir de las ternas calculadas.
.venv/bin/python -B -m oblivious.fluid_oblivious_initial_conditions.code.run_experiment --figures-only

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover \
  -s oblivious/11_fluid_oblivious_initial_conditions/code -p 'test_*.py' -v
```

`--config`, `--output-dir` y `--workers` permiten otra configuración, destino o
número de procesos. Para parámetros sin checkpoints compatibles se utiliza
`--recompute`.

## Resultados y archivos

Las figuras mantienen los paneles y la tipografía del experimento 10, con líneas
sólidas. Las métricas y trayectorias comparan desde el inicio común `x0=0.5`.
Las figuras de `11_policies` y `policies_and_initial_conditions` muestran las
27 políticas de equilibrio con leyendas al pie que indican `rho` y `x_bar`.
Se usan sólo tres colores: azul para monopolio, naranja para el equilibrio único
o inferior, y verde para el superior cuando hay dos. La misma regla se aplica
en ambas carpetas. También se muestran los
conjuntos completos de condiciones iniciales.

Ver [RESULTS.md](RESULTS.md) y el [PDF de figuras](figures/overview/equilibria_and_initial_conditions.pdf).

- [equilibria.json](data/equilibria.json): ternas completas, políticas por
  intervalos, acciones exactas en umbrales, conjuntos de condiciones iniciales, línea de fase y diagnósticos.
- [equilibria.csv](data/equilibria.csv): resumen de cada terna.
- [initial_conditions_intervals.csv](data/initial_conditions_intervals.csv): intervalos y extremos
  abiertos/cerrados con toda la precisión guardada.
- [policies.csv](data/policies.csv): proporción de producto 2 en cada intervalo
  y en cada punto de cambio de la política.
- [initial_state_correspondence.csv](data/initial_state_correspondence.csv):
  qué equilibrios admite cada conjunto de condiciones iniciales, incluidos los
  conjuntos sin ninguno detectado.
- [case_summary.csv](data/case_summary.csv): número de ramas y cobertura por caso.
- `stationary_refinement.csv`, `initial_state_checks.csv`, `ode_checks.csv`,
  `validation_trajectories.csv`, `validation.json`, `manifest.json` y
  `checkpoints/`: controles, trayectorias de validación y procedencia.
- [figures/](figures/README.md): conjuntos de condiciones iniciales, multiplicidad y políticas; todos los
  plots están en subcarpetas.

## Comprobación independiente de los intervalos sin cubrir

Se ejecutó una búsqueda nueva para los cuatro casos con intervalos sin cubrir.
El [informe independiente](EXISTENCE_CHECK.md) registra 1698 resoluciones nuevas del HJB,
una enumeración independiente de candidatos y desviaciones rentables que
descartan seis candidatos. La conclusión es numérica; se detalla su alcance
y la sensibilidad a la malla.

Para reproducirla:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_oblivious_initial_conditions.code.verify_existence
```
