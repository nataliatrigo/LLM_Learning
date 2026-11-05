# Monopolio con reemplazo del resultado opuesto

Esta es la versión activa del experimento solicitado: el estado contiene **cuántos éxitos y fracasos se recuerdan**, sin conservar su orden. Cuando la memoria está llena, un éxito nuevo reemplaza un fracaso y un fracaso nuevo reemplaza un éxito. Los gráficos, datos y políticas se recalcularon con esta regla.

La versión anterior, que conservaba los últimos n resultados mediante FIFO, está en [`archive/fifo_previous/`](archive/fifo_previous/README.md). Los gráficos antiguos de historias ordenadas y los diagnósticos de aproximación sólo están en ese archivo.

## Resultados y gráficos

- [Resumen de resultados](RESULTADOS.md).
- [Trayectoria equivalente al gráfico adjunto, n=20 y p₀=0.5](figures/individual_states/individual_states_p0_0.5_n_20.png).
- [Políticas óptimas por conteo, capacidad y p₀](figures/optimal_policies_by_count.png).
- [Ventaja del producto 2 y política exacta para n=20](figures/exact_policy_detail_p0_0.5_n_20.png).
- [Comparación Monte Carlo de las capacidades](figures/monte_carlo_by_window.png).
- [Referencia geométrica frente al reemplazo opuesto](figures/geometric_vs_replacement_p0_0.5_rho_0.95_n_20.png).
- [Bandas detectadas](data/policy_intervals.csv), [diagnósticos numéricos](data/solver_diagnostics.csv) y [auditoría](data/validation.json).

**Todas las capacidades se resuelven directamente como MDP finitos de conteos**, incluida n=1000. Las bandas se detectan después de resolver; su forma no se impone al algoritmo. Para p₀=0.5 y n=20, la política madura elige producto 2 si **4≤S≤14**, y producto 1 fuera de ese intervalo.

## Estado y actualización

El estado es (m,S), con m≤n resultados retenidos, S éxitos y F=m−S fracasos. La memoria empieza vacía, con prior Beta(1,1).

Antes de llenarse, cada compra de A agrega una observación:

\[
(m',S')=(m+1,S+Y),\qquad m<n.
\]

Con la memoria llena:

\[
S'=\begin{cases}
\min(n,S+1),&Y=1,\\
\max(0,S-1),&Y=0,
\end{cases}\qquad m'=n,\quad F'=n-S'.
\]

Si todos los resultados recordados son éxitos y llega otro éxito, S permanece en n; el caso simétrico permanece en cero. Elegir la opción externa no modifica el estado. La actualización ocurre sólo al observar un resultado de A.

n representa la **capacidad de memoria**. El borrado selectivo del resultado opuesto no conserva necesariamente los últimos n resultados cronológicos. Dos consumidores con el mismo (m,S) reciben la misma acción; durante el inicio, dos consumidores con igual S pero distinto m pueden recibir acciones distintas.

La demanda y el problema descontado mantienen las ecuaciones del experimento original:

\[
D(m,S)=\Pr\{\operatorname{Beta}(1+S,1+m-S)>p_0\},
\]

\[
V(m,S)=\frac{D(m,S)}{1-\gamma[1-D(m,S)]}
\max_i\{R-c_i+\gamma[p_iV(h^1)+(1-p_i)V(h^0)]\}.
\]

h¹ y h⁰ son los estados siguientes definidos arriba. Se elige producto 2 cuando su ventaja Q₂−Q₁ supera 10⁻⁸. El descuento γ corresponde a períodos calendario; la espera hasta una compra de A se elimina analíticamente mediante el prefactor.

## Calibración

| Parámetro | Valor |
|---|---:|
| p₁, p₂ | 0.35, 0.8 |
| c₁, c₂ | 0.05, **0.65** |
| Ingreso R | 1 |
| Descuento γ | 0.98 |
| p₀ | 0.1, 0.3, 0.5, 0.7, 0.9 |

Se conserva c₂=0.65, como en el gráfico adjunto y según la elección del usuario. La versión posterior del Monte Carlo original que usaba c₂=0.6665 no se mezcla con estos resultados.

La correspondencia confirmada es n≈1/(1−ρ), redondeando al entero más cercano con empates hacia arriba. Iguala la capacidad de evidencia; no iguala edades, pesos ni mecanismos de reemplazo.

| ρ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.97 | 0.99 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 1 | 1 | 1 | 2 | 2 | 3 | 3 | 5 | 10 | 20 | 33 | 100 | 1000 |

Se calcula una sola vez cada modelo correspondiente a varios ρ. El barrido calendario usa siete capacidades distintas: 1,2,3,5,10,20,100. Los casos n=33 y n=1000 conservan las referencias adicionales de historias individuales.

## Experimentos

1. **Calendario:** 35 escenarios, cinco p₀ y siete capacidades; 20.000 consumidores × 3.000 períodos por escenario. Se guardan cuota de A, frecuencia incondicional del producto 2 y frecuencia condicionada a comprar A, por período, por bloques de 25 períodos y por consumidor. Los intervalos al 95% usan consumidores como clústeres independientes; las proporciones condicionadas usan el método delta. Las semillas se comparten entre capacidades a igual p₀.
2. **Historias individuales:** 11 escenarios de 20 consumidores × 20.000 interacciones de A. El gráfico y el histograma muestran sólo al consumidor 1, incluyendo el transiente. Cada estado se registra antes de elegir producto y observar su resultado. Se guarda también una vista de las primeras 2.000 interacciones.
3. **Estabilidad por interacción:** n=20, p₀=0.5; 100 consumidores × 200.000 interacciones, burn-in de 20.000 y bloques de 10.000. Se guardan frecuencias acumuladas, por bloque, entre mitades y por longitud de simulación.

Las simulaciones por interacción condicionan en comprar A. Siguen usando la política óptima del problema descontado en tiempo calendario. Sus frecuencias y las frecuencias por período tienen denominadores diferentes.

## Cálculo y validación

La política madura tiene **n+1 estados**. Los estados iniciales m<n se resuelven por inducción hacia atrás. El total de estados es (n+1)(n+2)/2: n=1000 tiene 1.001 estados maduros y 501.501 estados totales, sin explosión por historias ordenadas.

El solver publica el residuo de Bellman y cotas de error de valor y de ventaja. Los márgenes de las acciones de los 37 escenarios superan sus tolerancias numéricas. Estas verificaciones son en punto flotante; no se usa aritmética de intervalos.

Las 11 pruebas comparan contra iteración independiente de políticas en tiempo calendario, reproducen simulaciones con una implementación escalar y verifican crecimiento inicial, saturación en los bordes, congelación ante la opción externa, registros anteriores al resultado e intervalos por consumidor. La auditoría reconstruye todos los estados y acciones de las historias guardadas con la nueva dinámica y comprueba los totales del Monte Carlo.

## Archivos y reproducción

```text
forgettin_fixed/
├── code/       modelo exacto, simulación, gráficos, auditoría, configuración y pruebas
├── data/       políticas NPZ, historias NPY, tablas CSV y manifiestos actuales
├── figures/    gráficos actuales del reemplazo opuesto
├── archive/fifo_previous/  versión FIFO anterior
├── README.md
└── RESULTADOS.md
```

Desde la raíz del proyecto:

```bash
MPLCONFIGDIR=/tmp/opposite-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgettin_fixed.code.run_experiment --stage all --workers 3
MPLCONFIGDIR=/tmp/opposite-mpl .venv/bin/python -B -m unittest discover -s forgettin_fixed/code -p 'test_*.py' -v
```

`--stage solve|calendar|interactions|figures|audit` permite ejecutar por partes. Los archivos terminados se reutilizan cuando coinciden el modelo, la política y la configuración. `--force` vuelve a calcularlos; usarlo si se modifica el código de simulación. Las semillas y hashes están en los manifiestos. Se usan NumPy, SciPy, pandas y Matplotlib del entorno existente y la clase `Primitives` de `forgetting/Code/model.py`.

Para consultar una decisión concreta:

```python
from forgettin_fixed.code.model import CountSolution
p = CountSolution.load('forgettin_fixed/data/policies/p0_0.5_n_20.npz')
product2, demand = p.evaluate(m=20, successes=12)
print(p.product2_intervals())  # [(4, 14)]
```
