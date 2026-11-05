# OE following monopoly (OE quality)

[Índice de experimentos oblivious](../README.md).

Nuevo experimento: el vendedor usa la estrategia de **monopolio de forgetting**
para un benchmark fijo `p0`, pero ese benchmark debe coincidir con la **calidad
media entre las ventas del rival en el juego real de dos vendedores**.

Se conservan los experimentos OE distribution y OE p0. El nuevo nombre público
es **OE quality** y sus curvas son moradas, para distinguirlas del verde y el
naranja de los otros modelos.

## Definición

Para una política propia `alpha(x) = P(producto 2 | seleccionado, x)`, su calidad
esperada es `q(x) = p1 + (p2-p1) alpha(x)`. Dados `p0` y `rho`, el vendedor resuelve
el problema de monopolio en la frontera madura de forgetting, con demanda
`P(theta(x)>p0)`. La espera cuando
no vende y el descuento son los mismos que en `forgetting/Code/model.py`.

Ambos vendedores emplean esa política en el juego TS real: se sortean las dos
Beta, gana el sorteo mayor y solo actualiza su reputación el vendedor elegido.
Si `nu(i,j)` es su distribución conjunta estacionaria y `D_B(i,j)` la
probabilidad real de elegir a B, el cierre solicitado es

```text
F(p0, alpha) = sum_ij nu(i,j) D_B(i,j) q(x_j)
              / sum_ij nu(i,j) D_B(i,j)

p0 = F(p0, alpha).
```

El denominador convierte el promedio en una tasa **entre ventas de B**. No es
el promedio de sus samples TS ni un promedio de calidad entre períodos sin venta.
El promedio por períodos también se conserva como diagnóstico con otro nombre.
Los valores factibles del punto fijo están entre `p1` y `p2`.

La política debe ser óptima para ese mismo `p0`. Las probabilidades interiores
de producto 2 solo son admisibles donde ambos productos tienen el mismo valor.
Se conservan todos los candidatos detectados; la búsqueda no demuestra unicidad
ni exhaustividad. Un candidato de este cierre no es un certificado de MPE.

El espacio de estrategias es **mixto**, igual que en los otros experimentos:
`0 <= alpha(x) <= 1`. Los gráficos siguen el mismo criterio de selección que
OE p0: se prefiere un candidato interior verificado cuando se encuentra uno y,
entre alternativas de ese grupo, el de menor `p0`. Los extremos puros siguen
siendo admisibles. Todos los candidatos permanecen guardados.

## Cómo se calcula

La búsqueda de equilibrio examina políticas puras y caras de indiferencia,
resolviendo conjuntamente la mezcla y la consistencia de `p0`.

También se conserva un **diagnóstico auxiliar con selector puro** desde
`p0=0.5`, y una versión amortiguada de ese diagnóstico. Sus ciclos se deben a
esa selección particular de políticas y no describen el comportamiento de un
algoritmo de equilibrio mixto. La convergencia de este diagnóstico tampoco
tiene prioridad al seleccionar el equilibrio de los gráficos.

Para acelerar la búsqueda se utiliza una identidad exacta: la reputación de un
vendedor, observada solo en sus propias ventas, sigue una cadena unidimensional
que depende de su política. La competencia afecta cuánto espera entre ventas,
pero no esa transición en tiempos de venta. Por eso el promedio entre sus ventas
se puede calcular con esa cadena durante la búsqueda.

**La certificación final vuelve al juego conjunto TS completo.** Comprueba la
distribución estacionaria con métodos independientes, el promedio condicionado
a ventas, optimalidad e indiferencia, y la ecuación de Bellman original de
forgetting en la misma grilla. La verificación es numérica en la grilla; no
demuestra convergencia al modelo continuo ni rapidez de mezcla.

## Ejecutar

Desde la raíz del proyecto:

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -u -B -m oblivious.oe_following_monopoly.code.run_experiment \
  --name quality_sales_n51
```

Por defecto: `rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99`, grilla de
51 puntos, `R=1, c1=0.05, c2=0.65, p1=0.35, p2=0.80, gamma=0.98`. Los parámetros
son los mismos de las comparaciones anteriores.

```bash
# Reanudar; reutiliza casos y diagnósticos terminados.
.venv/bin/python -u -B -m oblivious.oe_following_monopoly.code.run_experiment \
  --name quality_sales_n51 --resume

# Regenerar figuras desde los resultados guardados.
.venv/bin/python -B -m oblivious.oe_following_monopoly.code.run_experiment \
  --name quality_sales_n51 --plots-only

# Una corrida nueva con grillas mayores; no sobrescribe resultados previos.
.venv/bin/python -u -B -m oblivious.oe_following_monopoly.code.run_experiment \
  --name quality_sales_n101 --grids 51 101

# Resolver y guardar candidatos sin las simulaciones; continuar con --resume.
.venv/bin/python -u -B -m oblivious.oe_following_monopoly.code.run_experiment \
  --name solo_equilibrios --solve-only

.venv/bin/python -B -m unittest oblivious.oe_following_monopoly.code.test_model
```

## Datos y figuras

Los resultados se guardan en `data/<name>/` y las figuras en `figures/<name>/`.
Cada caso y diagnóstico tiene un checkpoint JSON/NPZ; `manifest.json` registra
parámetros, hashes y casos terminados. La corrida pausada de OE p0 es independiente.

Las simulaciones usan 20 trayectorias independientes, inicio común `(0.5,0.5)`,
10,000 períodos de burn-in y 100,000 retenidos. Reutilizan el simulador de la
comparación, con la demanda TS real y las transiciones de la grilla guardada.
Los intervalos del 95% se calculan entre trayectorias, sin tratar los períodos
como independientes. Esos intervalos no incluyen sesgo de inicialización;
con `rho` alto puede persistir una separación respecto a la ley estacionaria.

Se generan PDF y PNG para:

1. `01_policy`: políticas en 4×2 paneles; rombos en nodos mixtos.
2. `02_reputation_cdfs`: CDF estacionaria y simulada, con bandas MC del 95%.
3. `03_joint_stationary`: distribuciones conjuntas con escala log común y viridis.
4. `04_demand`: demanda usada al optimizar, marginal real y condicional real.
5. `06_advantage`: ventaja del producto 2, con línea de indiferencia.
6. `07_quality_fixed_point_map`: calidad generada frente a `p0`, con candidatos verificados.
7. `08_usage_profit_success`: uso de producto 2, beneficio y éxito frente a `rho`.
8. `S01_simulated_joint`: conjunta simulada, en la misma escala que la estacionaria.

`summary.csv`, `candidates.csv`, `iteration_history.csv`, `simulation_summary.csv`
y `RESULTADOS.txt` permiten revisar los valores, las alternativas y los errores.
Los diagnósticos numéricos permanecen en los datos guardados. El archivo
`iteration_history.csv` conserva el diagnóstico auxiliar con selector puro.
