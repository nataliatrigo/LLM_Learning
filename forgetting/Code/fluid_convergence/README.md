# Política fluida en el monopolio estocástico con forgetting

Este experimento mide la pérdida de usar una política calculada del fluido en
el monopolio estocástico original. Trabaja sobre la **frontera madura**, con
estado normalizado `x∈[0,1]` y horizonte infinito. Se mantienen fijos `rho`,
`p0`, `gamma`, las probabilidades de éxito y los costes al refinar las grillas.

## Cantidad estudiada

Para cada combinación de parámetros se calcula

\[
L_\rho(x)=V_\rho^*(x)-V_\rho^{\widehat\pi_f}(x).
\]

`V*` es el óptimo del monopolio estocástico y `V^pi_f` es el valor de seguir
la candidata fluida en **ese mismo problema estocástico**. La política se toma
del fluido; su valor se calcula con las transiciones y el descuento originales.

Cuando vende, la reputación pasa a `rho*x+(1-rho)` tras éxito y a `rho*x` tras
fracaso. Sin venta conserva el estado y obtiene cero. La demanda es

\[
D_\rho(x)=\Pr\!\left[\operatorname{Beta}
\left(1+\frac{x}{1-\rho},1+\frac{1-x}{1-\rho}\right)>p_0\right].
\]

Cada venta del producto `i` produce `R-c_i`; el descuento por período es
`gamma`. Estos elementos del modelo permanecen iguales en todas las grillas.

## Política calculada del fluido

Se resuelve el fluido existente del estudio de forgetting:

\[
\delta v(x)=D_\rho(x)\max_i\{R-c_i+\lambda(p_i-x)v'(x)\},
\qquad \lambda=-\log\rho,\quad\delta=-\log\gamma.
\]

El solver utiliza Howard y diferencias upwind monótonas. Se interpola la
**diferencia de Hamiltonianos** a los nodos estocásticos y se elige producto 2
cuando supera `policy_action_tolerance=1e-8`; en caso contrario, producto 1.
La política binaria no se interpola como una probabilidad, para evitar introducir
una franja de mezclas distinta de la candidata definida aquí.

Esta convención define una política numérica, sin presentarla como una política
óptima exacta del fluido continuo. No se añade una mezcla de sostén en el punto
exacto de un umbral atractivo. Se comprueba la sensibilidad a la grilla que
produce la candidata.

## Evaluación en el modelo estocástico original

Para una política fija `alpha(x)∈[0,1]`, se forma el kernel original con la
probabilidad de éxito `p_alpha=p1+(p2-p1)*alpha`:

\[
P^\alpha=(1-D_\rho)I+D_\rho[p_\alpha S_\rho+(1-p_\alpha)F_\rho].
\]

`S_rho` y `F_rho` proyectan los estados sucesores a nodos adyacentes con los
pesos de interpolación del solver estocástico. El valor satisface

\[
(I-\gamma P^\alpha)V^\alpha
=D_\rho[R-c_1-(c_2-c_1)\alpha].
\]

Se resuelve mediante LU dispersa, sin formar la inversa ni matrices densas.
El óptimo estocástico se calcula por Howard y se comprueban las ventajas
condicionales incluso donde la demanda es muy pequeña. No se interpolan
soluciones entre valores de `rho` o `p0`.

## Refinamiento numérico separado

El experimento tiene dos controles:

1. Se congela la candidata de la grilla fluida más fina y se refina la grilla
   estocástica. Se comparan `V*`, `V^pi` y `L` en estados comunes.
2. Se congela el kernel estocástico más fino y se refina la grilla que calcula
   la política fluida. Se comparan los valores fluidos, los umbrales y los
   valores estocásticos de las distintas candidatas.

Refinar la grilla estudia la estabilidad de la **estimación de la pérdida**.
Una pérdida positiva que se estabiliza representa el coste de usar la política
fluida en el modelo estocástico; no se exige que desaparezca.

Se informa la sensibilidad de la función de pérdida en norma supremo, en el
estado inicial `x0=0.5`, en L1 respecto de `x` y del escalar de pérdida máxima.
L1 integra diferencias absolutas sobre `[0,1]`: no es un promedio según una
distribución estacionaria.

La candidata binaria es discontinua en sus umbrales, donde los productos no
necesariamente son indiferentes en el problema estocástico. Desplazar un umbral
puede mantener grande una diferencia uniforme aunque la región afectada se
estreche. Por eso se conservan los cuatro diagnósticos, sin sustituir uno por
otro para declarar estabilidad.

El criterio configurado requiere que los dos últimos cambios de la pérdida
entre grillas estocásticas y el último cambio de candidata sean pequeños
respecto de la pérdida observada: 1%, con piso absoluto `1e-4`. Los casos que
no cumplen el criterio quedan sin declarar estables. Las diferencias entre
grillas son indicadores de sensibilidad y no cotas rigurosas del error espacial.

Las normas se calculan sobre los nodos de evaluación. Los CSV de curvas usan
una grilla común para dibujar; las normas se calculan antes de reducir la
resolución gráfica. El experimento no cubre las capas transitorias anteriores
a la frontera madura.

## Comprobaciones

Los residuos producen indicadores `residual/(1-gamma)` para el valor
estocástico discretizado y `residual/delta` para el fluido discretizado. Se
exige que no superen `1e-8`; para los óptimos también se exige regret condicional
menor que `1e-9`. Las pérdidas negativas mayores que la tolerancia del solver
provocan un error. Estas comprobaciones usan doble precisión y no incluyen el
error espacial ni constituyen certificados de aritmética de intervalos.

Las pruebas independientes incluyen demanda constante con valor analítico,
coincidencia con el solver original, evaluación mixta con un sistema construido
independientemente, períodos sin venta, enumeración de políticas en una grilla
pequeña, cachés y separación de los dos refinamientos.

## Ejecutar

Desde la raíz del proyecto:

```bash
.venv/bin/python -u -B forgetting/Code/fluid_convergence/run_experiment.py
```

[`config.json`](config.json) utiliza `p0=[.1,.3,.5,.7,.9]`,
`rho=[.6,.7,.8,.9,.95,.97,.99]` y grillas `1001,2001,4001`, en 35 combinaciones.
No se filtra `p0`. El número de intervalos de cada grilla, las tolerancias,
el estado inicial y el número de casos simultáneos son configurables.

Las soluciones completas se guardan por caso y se reutilizan si coinciden sus
entradas y huellas numéricas. Interrumpir y ejecutar de nuevo reanuda desde
esos checkpoints. Dos casos se procesan simultáneamente por defecto;
`--workers 1` ejecuta uno a la vez, con BLAS/OpenMP limitado a un hilo.

```bash
# Prueba pequeña, en una subcarpeta independiente
.venv/bin/python -u -B forgetting/Code/fluid_convergence/run_experiment.py --quick

# Regenerar figuras desde los datos verificados
.venv/bin/python -B forgetting/Code/fluid_convergence/run_experiment.py --figures-only

# Otra configuración y destino
.venv/bin/python -u -B forgetting/Code/fluid_convergence/run_experiment.py \
  --config /ruta/config.json --output-dir /ruta/resultados

# Pruebas analíticas e independientes
.venv/bin/python -B -m unittest discover -s forgetting/Code/fluid_convergence -p 'test_*.py' -v
```

Si cambia la configuración científica o el núcleo numérico, se requiere otro
destino o `--recompute` para reemplazar los resultados derivados de esa corrida.

## Resultados

Los resultados están en
[`forgetting/outputs/fluid_convergence/`](../../outputs/fluid_convergence/).
`summary.md` describe los hallazgos reales; `manifest.json` registra
configuración, criterios, huellas y diagnósticos por caso.

- `original_summary.csv`, `original_curves.csv`: pérdidas y valores estocásticos.
- `grid_refinement.csv`: refinamiento estocástico con candidata fluida fija.
- `fluid_refinement.csv`: refinamiento fluido con kernel estocástico fijo.
- `solver_diagnostics.csv`: residuos, indicadores de error e iteraciones.
- `checkpoints/`: soluciones y evaluaciones reutilizables.
- `figures/`: cinco figuras en PNG y PDF.
