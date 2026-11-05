# Experimento nuevo: forgetting por reemplazo aleatorio uniforme

Cada resultado nuevo de A reemplaza **una de las n observaciones anteriores elegida uniformemente al azar**, independientemente de si el resultado nuevo es éxito o fracaso. Los conteos son enteros y el estado no necesita conservar el orden.

Este experimento está separado en `forgetting_random_replacement/`. El experimento de reemplazo opuesto se conserva en [`../forgettin_fixed/`](../forgettin_fixed/README.md).

## Archivos principales

- [Resultados y cifras](RESULTADOS.md).
- [Trayectoria del consumidor 1, n=20 y p₀=0.5](figures/individual_states/individual_states_p0_0.5_n_20.png).
- [Políticas óptimas por capacidad y p₀](figures/optimal_policies_by_count.png).
- [Ventaja y política detallada de n=20](figures/exact_policy_detail_p0_0.5_n_20.png).
- [Monte Carlo de todas las capacidades](figures/monte_carlo_by_window.png).
- [Comparación descriptiva: reemplazo opuesto frente a aleatorio](figures/opposite_vs_random_replacement_p0_0.5_n_20.png).
- [Comparación con forgetting geométrico](figures/geometric_vs_random_replacement_p0_0.5_rho_0.95_n_20.png).
- [Bandas detectadas](data/policy_intervals.csv), [transiciones](data/conditional_transitions.csv), [diagnósticos](data/solver_diagnostics.csv) y [auditoría](data/validation.json).

## Regla exacta

Sea S el número de éxitos y F=n−S el de fracasos en una memoria llena. El vendedor elige producto i, el resultado nuevo es Y∼Bernoulli(pᵢ) y se elimina una observación vieja. Si R indica que la eliminada era un éxito:

\[
R\mid S\sim\operatorname{Bernoulli}(S/n),\qquad R\perp Y\mid(S,i),\qquad S'=S+Y-R.
\]

Por ejemplo, con S=18 y n=20, si llega un fracaso:

| Dato eliminado | Probabilidad | S siguiente |
|---|---:|---:|
| Éxito | 18/20 | 17 |
| Fracaso | 2/20 | 18 |

Un resultado nuevo puede reemplazar otro del mismo tipo, dejando S igual. Los bordes quedan definidos por esta misma regla: con S=0 se elimina necesariamente un fracaso, y con S=n se elimina necesariamente un éxito.

Las probabilidades incondicionales de transición bajo producto i son:

\[
P_i(S,S+1)=p_i(1-S/n),\quad
P_i(S,S-1)=(1-p_i)S/n,
\]

\[
P_i(S,S)=p_iS/n+(1-p_i)(1-S/n).
\]

Ésta es la razón por la que los conteos son un estado Markov exacto: la probabilidad del dato que sale depende únicamente de S/n. La memoria no conserva necesariamente los últimos n resultados cronológicos.

## Inicialización y relación con ρ

Para comparar con los experimentos anteriores, la configuración por defecto empieza **con memoria vacía y prior Beta(1,1)**. Se agregan los primeros n resultados de A sin borrar nada. A partir de entonces se recuerdan exactamente n observaciones y cada compra realiza un reemplazo aleatorio. Elegir la opción externa deja el estado intacto.

El código también admite comenzar con memoria llena. En `code/config.json`, `initialization: "balanced_full"` inicia en S=⌊n/2⌋ y F=n−S. En la API, `initial_successes=k` permite cualquier estado lleno con k éxitos. La inicialización efectiva queda registrada en cada manifiesto.

Se conserva la correspondencia n≈1/(1−ρ), redondeada al entero más cercano y con empates hacia arriba. Para esta regla:

\[
E[S'\mid S,i]=(1-1/n)S+p_i.
\]

La retención esperada es ρₙ=1−1/n; para n=20 equivale exactamente a ρ=0.95. Cada recuerdo sobrevive k reemplazos con probabilidad (1−1/n)ᵏ. La igualdad en medias con el forgetting geométrico no iguala el riesgo de transición ni las políticas, porque el valor y la demanda son no lineales. Para los ρ que requieren redondeo, se publica también la capacidad resultante en `data/rho_to_n.csv`.

| ρ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.97 | 0.99 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 5 | 10 | 20 | 33 | 100 | 1000 |

## Demanda y política óptima

Durante el inicio, el estado es (m,S), con m≤n observaciones y F=m−S. La demanda de A mantiene el modelo original:

\[
D(m,S)=\Pr\{\operatorname{Beta}(1+S,1+m-S)>p_0\}.
\]

Después de integrar los períodos de espera por la opción externa:

\[
V(h)=\frac{D(h)}{1-\gamma[1-D(h)]}
\max_i\{R-c_i+\gamma[p_iE(V(h')\mid Y=1)+(1-p_i)E(V(h')\mid Y=0)]\}.
\]

Con la memoria llena y r=S/n, las continuaciones condicionadas son

\[
E(V'\mid Y=1)=(1-r)V(S+1)+rV(S),
\]

\[
E(V'\mid Y=0)=rV(S-1)+(1-r)V(S).
\]

Se promedia el reemplazo **antes de maximizar sobre los productos**: el vendedor no conoce de antemano el sorteo de eliminación. Antes de alcanzar n se usa la actualización de acumulación normal, sin eliminación.

Se resuelve el MDP completo, con n+1 estados maduros y (n+1)(n+2)/2 estados incluyendo el inicio. No hay interpolación ni aproximación por historias. Se escoge producto 2 cuando Q₂−Q₁>10⁻⁸; las bandas se detectan después de resolver, sin imponer su forma. Para n=20,p₀=0.5 se obtiene **producto 2 en S=5–12**.

## Calibración y simulaciones

| Parámetro | Valor |
|---|---:|
| p₁, p₂ | 0.35, 0.8 |
| c₁, c₂ | 0.05, **0.65** |
| Ingreso | 1 |
| γ por período calendario | 0.98 |
| p₀ | 0.1, 0.3, 0.5, 0.7, 0.9 |

- **Calendario:** 35 escenarios, siete capacidades distintas y cinco p₀; 20.000 consumidores × 3.000 períodos. Se guardan datos por período, bloques de 25 períodos y por consumidor, incluyendo beneficios descontados.
- **Interacciones:** 11 escenarios, 20 consumidores × 20.000 compras de A. Los gráficos e histogramas muestran al consumidor 1, incluyendo el transiente.
- **Estabilidad:** n=20,p₀=0.5; 100 consumidores × 200.000 interacciones, burn-in de 20.000 y bloques de 10.000.

Cada período calendario sortea tres vectores independientes: demanda, resultado y eliminación. Los uniformes se comparten entre capacidades a igual p₀. Los modelos anteriores usaban otro número de sorteos, por lo que las comparaciones con sus trayectorias son descriptivas y no están emparejadas.

Los intervalos de confianza al 95% usan consumidores como clústeres independientes. Las proporciones condicionadas a comprar A usan el método delta. Los intervalos miden incertidumbre Monte Carlo, no incertidumbre de parámetros ni tiempos de mezcla.

## Datos y validación

Los registros de interacción contienen `S_before`, `m_before`, `product2`, `success`, `demand` y **`removed_success`**: 1 si se elimina éxito, 0 si se elimina fracaso y −1 si todavía se acumula memoria. El estado se registra antes de elegir y antes del resultado. Los resúmenes por consumidor permiten comprobar:

\[
S_{final}=S_{inicial}+\text{éxitos observados}-\text{éxitos eliminados}.
\]

Las 12 pruebas verifican la agregación por conteos frente al modelo completo de slots binarios, iteración independiente de políticas en tiempo calendario, el ejemplo 18/20, independencia entre resultado y eliminación, inicialización, bordes, opción externa y reproducción escalar de trayectorias y beneficios. La auditoría reconstruye los estados y las decisiones guardadas y compara los beneficios Monte Carlo con el valor de Bellman.

Los certificados numéricos usan residuos de Bellman en punto flotante y excluyen error de redondeo no acotado por intervalos. Todas las capacidades se calculan con el mismo método exacto por estados, incluida n=1000.

## Reproducción

```text
forgetting_random_replacement/
├── code/       modelo, simulación, gráficos, configuración, auditoría y pruebas
├── data/       políticas NPZ, historias NPY, tablas CSV y manifiestos
├── figures/    gráficos del experimento nuevo
├── README.md
└── RESULTADOS.md
```

Desde la raíz del proyecto:

```bash
MPLCONFIGDIR=/tmp/random-replacement-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgetting_random_replacement.code.run_experiment --stage all --workers 3
MPLCONFIGDIR=/tmp/random-replacement-mpl .venv/bin/python -B -m unittest discover -s forgetting_random_replacement/code -p 'test_*.py' -v
```

Se puede ejecutar por etapas con `--stage solve|calendar|interactions|figures|audit`. Se reutilizan los escenarios completados cuando coincide su configuración y política; `--force` los recalcula. Usar `--force` si se modifica código de simulación. Dependencias del entorno existente: NumPy, SciPy, pandas y Matplotlib. Se reutiliza la clase de primitivas de `forgetting/Code/model.py`.
