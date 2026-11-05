# Monopolio con forgetting de ventana fija (FIFO)

Réplica del experimento de `forgetting/`, manteniendo los últimos **n resultados observados de A**, en orden. El consumidor empieza con una memoria vacía y prior Beta(1,1). Un resultado sale exactamente después de n nuevas compras de A; elegir la opción externa no modifica la memoria.

**Alcance de la optimalidad:** para n=1,2,3,5,10,20 se resuelve el MDP completo de historias ordenadas, incluidos los transientes. Todas las decisiones calculadas para esos casos quedaron certificadas por sus márgenes numéricos y el residuo de Bellman. Para n=33,100,1000 se entregan **políticas aproximadas**, con ejecución FIFO real, cotas explícitas y comparación de refinamientos. Sus cotas globales en madurez son amplias: estos casos no deben interpretarse como una solución globalmente óptima certificada.

## Archivos para empezar

- [Resultados y cifras principales](RESULTADOS.md).
- [Trayectoria equivalente al gráfico adjunto: p₀=0.5, n=20](figures/individual_states/individual_states_p0_0.5_n_20.png).
- [Políticas por ventana y p₀](figures/optimal_policies_by_count.png). Naranja significa que el orden cambia la decisión a igual número de éxitos; no es una política aleatoria.
- [Política completa para las 1.048.576 historias con n=20](figures/exact_policy_all_histories_p0_0.5_n_20.png).
- [Monte Carlo en tiempo calendario](figures/monte_carlo_by_window.png).
- [Referencia geométrica frente a FIFO](figures/geometric_vs_fixed_p0_0.5_rho_0.95_n_20.png).
- [Diagnósticos numéricos](data/solver_diagnostics.csv), [refinamiento](data/refinement/state_comparison.csv) y [equivalencia ρ→n](data/rho_to_n.csv).

## Calibración y equivalencia

Se usa la calibración del gráfico adjunto, confirmada para esta réplica:

| Parámetro | Valor |
|---|---:|
| p₁, p₂ | 0.35, 0.8 |
| c₁, c₂ | 0.05, **0.65** |
| Ingreso R | 1 |
| Descuento γ por período calendario | 0.98 |
| p₀ | 0.1, 0.3, 0.5, 0.7, 0.9 |

El Monte Carlo posterior del proyecto usaba c₂=0.6665; aquí se conserva **0.65 en todos los experimentos**, conforme a la elección del usuario. No se mezclan las dos calibraciones.

Igualamos la masa total de evidencia geométrica en madurez, Σρʲ=1/(1−ρ), con la masa de la ventana fija n. Se redondea al entero más cercano, con los empates hacia arriba. Esta equivalencia conserva evidencia total; no iguala la distribución de edades ni la varianza de los pesos.

| ρ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.97 | 0.99 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 5 | 10 | 20 | 33 | 100 | 1000 |

Los ρ que corresponden al mismo n generan el mismo modelo: se calculan y simulan una sola vez. Los casos 33 y 1000 reproducen las referencias adicionales de historias individuales; el barrido calendario usa n=1,2,3,5,10,20,100.

## Modelo y política

Sea h la lista ordenada de los m=min(n,t) resultados retenidos, S su número de éxitos y F=m−S. La demanda de A es

\[
D(h)=\Pr\{\operatorname{Beta}(1+S,1+F)>p_0\}.
\]

Si A vende el producto i, se observa Y∼Bernoulli(pᵢ). Se agrega Y y, si la ventana está llena, se elimina su primer resultado. Denotamos por h⁰ y h¹ esas dos historias siguientes. Eliminar los períodos de espera por la opción externa da

\[
V(h)=\frac{D(h)}{1-\gamma(1-D(h))}
\max_i\left\{R-c_i+\gamma[p_iV(h^1)+(1-p_i)V(h^0)]\right\}.
\]

Se elige producto 2 cuando

\[
-(c_2-c_1)+\gamma(p_2-p_1)[V(h^1)-V(h^0)]>10^{-8}.
\]

Las tablas NPZ conservan la política completa, no sólo su proyección por S. Una historia binaria se codifica con el dato más antiguo como bit más significativo. `model.py` enumera 2ⁿ historias maduras y obtiene los niveles m<n por inducción hacia atrás. n=20 tiene 1.048.576 estados maduros y 2.097.151 estados totales. No se usa interpolación en esos casos.

Para las ventanas grandes, `large_window.py` conserva los k resultados más antiguos en orden y el conteo de éxitos en el resto. El planificador aproxima ese resto como intercambiable. **Al ejecutar la política se consulta el siguiente bit real y se mantiene la cola FIFO completa**; no se elimina un resultado al azar. Se usan k=10 para n=33, k=12 para n=100 y k=6 para n=1000. El residuo del valor aproximado sobre el MDP FIFO completo se calcula sobre todos los contextos relevantes, no sobre una muestra. Se reporta su cota uniforme; los cuantiles de residuo describen contextos y no frecuencias de consumidores. La comparación de k no prueba optimalidad.

Las cotas basadas en residuos se calculan en punto flotante y no mediante aritmética de intervalos. Los intervalos de confianza Monte Carlo sólo miden error de simulación, no error de la política aproximada.

## Experimentos ejecutados

1. **Calendario:** 20.000 consumidores independientes × 3.000 períodos, siete n distintos y cinco p₀: 35 escenarios. Se guardan demanda de A, frecuencia incondicional del producto 2, frecuencia condicionada a comprar A, datos por período, bloques de 25 períodos e información por consumidor. Los intervalos al 95% de los bloques tratan a cada consumidor como un clúster independiente; para las proporciones condicionadas se usa el método delta. Se comparte semilla entre ventanas a igual p₀, de modo que las comparaciones no son muestras independientes.
2. **Historias del gráfico adjunto:** 20 consumidores × 20.000 interacciones de A en cada uno de 11 escenarios. El gráfico y el histograma muestran sólo al consumidor 1, incluyendo el transiente, y registran S antes de elegir. Hay una vista adicional de las primeras 2.000 interacciones. El estado es discreto: los histogramas cuentan cada S entero.
3. **Estabilidad por interacción:** n=20, p₀=0.5, 100 consumidores × 200.000 interacciones, burn-in 20.000 y bloques de 10.000. Se guardan frecuencias acumuladas, por bloque, estabilidad entre mitades y comparación de longitudes 50.000/100.000/200.000.
4. **Refinamiento de políticas grandes:** comparación de k=8/10 para n=33, k=10/12 para n=100 y k=4/6 para n=1000. Los desacuerdos sobre historias de validación se distinguen de las cotas uniformes.

Las interacciones de A condicionan en la compra y no sortean la opción externa. La política sigue optimizando beneficios descontados en tiempo calendario. Las frecuencias por interacción y por período calendario tienen denominadores diferentes.

## Estructura y reproducción

```text
forgettin_fixed/
├── code/       modelo exacto, aproximación, simulación, figuras, configuración y pruebas
├── data/       políticas NPZ, historias NPY, tablas CSV, diagnósticos y manifiestos
├── figures/    gráficos PNG
├── README.md
└── RESULTADOS.md
```

Desde la raíz del repositorio, con el entorno existente:

```bash
MPLCONFIGDIR=/tmp/fixed-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgettin_fixed.code.run_experiment --stage all --workers 3
MPLCONFIGDIR=/tmp/fixed-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgettin_fixed.code.refinement
MPLCONFIGDIR=/tmp/fixed-mpl .venv/bin/python -B -m forgettin_fixed.code.run_experiment --stage figures
MPLCONFIGDIR=/tmp/fixed-mpl .venv/bin/python -B -m unittest discover -s forgettin_fixed/code -p 'test_*.py' -v
MPLCONFIGDIR=/tmp/fixed-mpl .venv/bin/python -B -m forgettin_fixed.code.audit_outputs
```

`--stage solve|calendar|interactions|figures` permite ejecutar por partes. Los cálculos completados se reutilizan; `--force` los vuelve a generar. Si se cambia código, configuración, tolerancia o profundidad de aproximación, usar `--force` en los experimentos afectados. Las semillas, tamaños efectivos y hashes del código quedan en los manifiestos. Dependencias: NumPy, SciPy, pandas y Matplotlib del entorno del proyecto. Se reutiliza únicamente la clase de primitivas de `forgetting/Code/model.py`; no se altera el experimento original.

Para consultar la acción exacta de una historia concreta:

```python
from forgettin_fixed.code.model import WindowSolution
p = WindowSolution.load('forgettin_fixed/data/policies/p0_0.5_n_20.npz')
word = int('10110101010101101010', 2)  # antiguo → reciente
product2, demand = p.evaluate(20, word)
```

## Validación

Las pruebas comparan el solver con iteración independiente de políticas en tiempo calendario, reproducen las trayectorias con colas Python, verifican que la opción externa no actualiza evidencia y que cada estado se guarda antes del resultado. Para la aproximación se compara el residuo publicado con la enumeración del MDP FIFO verdadero y se verifica que k=n−1 recupera el solver exacto en problemas pequeños. La auditoría final de archivos se guarda en `data/validation.json`.
