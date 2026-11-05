# Competencia fluida oblivious: respuestas cruzadas desde el mismo x0

Este experimento ejecuta la propuesta del profesor: calcular una política para
A suponiendo que B está en `p2`, y otra para B suponiendo que A está en `p1`.
Después comprueba si, partiendo de la misma reputación inicial, las dos
políticas llevan efectivamente a A hacia `p1` y a B hacia `p2`.

Los [resultados](RESULTS.md) incluyen intervalos iniciales completos,
trayectorias conjuntas, un control negativo y verificaciones numéricas.
Los experimentos anteriores conservan sus archivos originales.

## Modelo y condición comprobada

Se mantienen las primitivas de los experimentos 10 y 11:
`R=1`, `c1=.05`, `c2=.65`, `p1=.35`, `p2=.8`, `gamma=.98`,
`lambda=-log(rho)` y `delta=-log(gamma)`.

Se resuelven de nuevo los dos HJB globales de reputación propia:

```text
delta U_A(x;p2) = D(x,p2) max_a {
  R-c1-(c2-c1)*a + lambda*[p1+(p2-p1)*a-x]*U_A'(x;p2)
},

delta U_B(x;p1) = D(x,p1) max_a {
  R-c1-(c2-c1)*a + lambda*[p1+(p2-p1)*a-x]*U_B'(x;p1)
}.
```

La demanda es la misma demanda Thompson Sampling con posteriores Beta y
opción externa fija `p0`. Se comprueba que la política de A mantiene `p1` y la
de B mantiene `p2`, comparando la acción y el valor de holding con la solución
global del HJB.

Luego se calculan los conjuntos completos de inicios de cada política:

```text
B_A = {x0 : A converge a p1 usando sigma_A},
B_B = {x0 : B converge a p2 usando sigma_B},
K = B_A intersección B_B.
```

Para `x0` en `K` se integra la demanda efectiva conjunta:

```text
xdot_A = lambda*D(x_A,x_B)*[p1+(p2-p1)*sigma_A(x_A)-x_A],
xdot_B = lambda*D(x_B,x_A)*[p1+(p2-p1)*sigma_B(x_B)-x_B],
x_A(0) = x_B(0) = x0.
```

La demanda es continua y estrictamente positiva en el dominio compacto
`[0,1]^2` para estos parámetros. Para cada vendedor el cambio de tiempo
`s_i(t)=lambda*integral_0^t D_i(u) du` tiende a infinito. Por ello la demanda
conjunta cambia la velocidad de su línea de fase, pero conserva su destino.
Así se obtiene todo `K` sin discretizar los estados iniciales ni confundir
`x(T)` con un límite asintótico.

## Casos y verificaciones

Es una prueba dirigida a los cuatro casos con inicios sin cubrir en el
experimento 11, con `(p0,rho)=(.8,.9),(.9,.7),(.9,.8),(.7,.9)`.
No es un nuevo barrido general de `rho`. El último sirve de control: se prueba
el mismo par puro aunque su rama simétrica existente sea mixta.

- **24 HJB nuevos:** dos respuestas por caso en mallas de 4001, 8001 y 16001
  nodos. Se guardan las políticas y los diagnósticos en cada malla.
- Refinamiento de todas las raíces y fronteras, topología de los conjuntos de
  inicios, acción óptima, valor de holding y residuos del HJB.
- Cuadratura original de orden 128 frente a 256 en toda la malla fina,
  contrastada además con integración Legendre directa independiente mediante
  el cambio `t=1-(1-p0)*u^4`, con órdenes 256 y 512. Este cambio suaviza el
  extremo `t=1`, donde la integral Legendre sin transformar pierde precisión
  para reputaciones próximas a 1.
- Simulaciones conjuntas desde tres inicios interiores por caso positivo,
  intercambio de identidades y refinamiento de tolerancias, pasos y cuadratura.
- Identidad independiente de reputación y ventas acumuladas durante los
  tramos de producto puro.
- Simulación larga hasta `T=1e6` para cada caso positivo. Se usan coordenadas
  exactas de logaritmo de la distancia al destino para resolver la lentitud de
  A sin inestabilidad numérica cuando B ya está cerca de `p2`. Se contrasta
  con el integrador por eventos durante las primeras 3000 unidades de tiempo.

Las raíces y las fronteras son aproximaciones numéricas. En un umbral
repulsivo se usa producto 1 y su continuación inmediata, como en el
experimento 11. Los extremos abiertos y cerrados corresponden a ese selector.
El refinamiento no proporciona cotas matemáticas certificadas del error.

La optimalidad corresponde al **criterio oblivious con rival fijo** de los
experimentos anteriores. Esta prueba no comprueba mejores respuestas frente
a un rival que se mueve durante toda la transición, ni enumera todos los
posibles pares asimétricos con reputaciones mixtas.

## Reproducción

Desde la raíz del proyecto:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m oblivious.fluid_asymmetric.code.run_experiment

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover \
  -s oblivious/13_fluid_asymmetric/code -p 'test_*.py' -v

# Regenerar el informe y los gráficos a partir de los datos calculados.
.venv/bin/python -B -m oblivious.fluid_asymmetric.code.run_experiment --figures-only
```

`--config` y `--output-dir` permiten guardar otra campaña por separado.
La ejecución completa recalcula las respuestas; no reutiliza checkpoints
históricos.

`--recheck-demand` permite repetir únicamente la auditoría de demanda sobre
resultados con la misma configuración y las mismas fuentes numéricas
originales. La primera auditoría sin suavizar el extremo se conserva en
`data/demand_audit_before_endpoint_transform.json`, cuando existe. Su
discrepancia motivó la transformación; no se relajaron las tolerancias.

## Archivos

- [RESULTS.md](RESULTS.md): informe en español con las figuras y su alcance.
- [data/summary.csv](data/summary.csv): tabla principal.
- [data/results.json](data/results.json): políticas, conjuntos de inicios y
  controles completos con toda la precisión calculada.
- `data/grids/`: resultados en cada malla, antes de las simulaciones.
- `data/trajectories.csv`, `data/trajectory_bins.csv`: reputaciones, acciones,
  demanda, ventas y beneficios realizados hasta `T=3000`.
- `data/long_horizon_trajectories.csv`: trayectorias largas.
- `data/manifest.json`: configuración, versiones y huellas SHA-256 de las
  fuentes y resultados usados.
- `figures/`: intervalos iniciales, políticas y trayectorias en PNG.
