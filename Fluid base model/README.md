# Fluid base model

Modelo fluido **sin olvido** de las notas `Notes_261020_165951.pdf`. Calcula la política óptima estacionaria en dos dimensiones, simula sus trayectorias deterministas y compara por separado el corte espacial, el paso de malla y el paso de integración temporal.

Todo el código y los resultados de este experimento están dentro de esta carpeta. No modifica los experimentos existentes.

## Ver los resultados

La ejecución actual extiende las trayectorias hasta **T=200**. La ejecución base está guardada en [outputs/summary.md](outputs/summary.md).

- [Producto óptimo para cada estado (S,F)](outputs/figures/policy_sf.png).
- [Producto óptimo por media posterior e información acumulada](outputs/figures/policy_nx.png).
- [Valor óptimo y demanda](outputs/figures/value_and_demand.png).
- [Trayectorias óptimas y comparación con productos fijos](outputs/figures/trajectories.png).
- [Ventaja local de utilizar el producto 2](outputs/figures/switching_advantage.png).

Los mapas muestran el **interior** `S+F <= report_n`, no la frontera artificial del cálculo. Gris significa empate o acción no resuelta por el intervalo de truncamiento; la ausencia de gris no certifica la precisión de la malla continua.

## Comparar los cinco valores de p₀

El ejercicio también se ejecuta para los valores anteriores `p0 = 0.1, 0.3, 0.5, 0.7, 0.9`. Los resultados están separados de la ejecución base:

- [Resumen comparativo y precisión de cada caso](outputs/p0_comparison/summary.md).
- [Comparación de políticas en coordenadas (x,n)](outputs/p0_comparison/figures/policy_nx_comparison.png).
- [Comparación de políticas en coordenadas (S,F)](outputs/p0_comparison/figures/policy_sf_comparison.png).
- [Valor inicial y regiones de producto 2](outputs/p0_comparison/figures/initial_value_and_policy.png).
- [Trayectorias desde (0,0) para cada p₀](outputs/p0_comparison/figures/trajectories_p0_comparison.png).

Para reproducir todo el barrido desde la raíz del repositorio:

```bash
.venv/bin/python "Fluid base model/code/run_p0_comparison.py" --workers 5 --reuse-hjb
```

`--workers` admite entre uno y cinco escenarios simultáneos. `--reuse-hjb` aprovecha únicamente mallas cuyos parámetros y configuración coinciden; cada nuevo p₀ requiere optimizar otra HJB. Para una prueba más pequeña, añadir `--quick`: sus resultados se guardan en una subcarpeta distinta. `p0_comparison.values` en `config.json` controla la lista, con hasta cinco valores distintos para los paneles comparativos.

Cada carpeta `outputs/p0_comparison/p0_.../` contiene la configuración efectiva, las tres mallas, la política, las trayectorias, cinco figuras individuales, el registro de ejecución y sus diagnósticos. Las figuras agregadas usan las mismas escalas. Los círculos y las letras A–D identifican los cuatro estados iniciales; los cuadrados identifican sus puntos finales en el tiempo de simulación.

Las trayectorias actuales llegan a `T=200` desde los mismos cuatro estados iniciales. El interior se amplía hasta `S+F <= 240`: dado que `dot n = D <= 1` y el máximo n inicial es 20, todas las trayectorias satisfacen `n(200) <= 220`, con un margen de 20 hasta el borde. Se mantienen las tres resoluciones, el paso temporal y la cota de truncamiento. Una trayectoria casi inmóvil a p₀ alto puede deberse a una demanda muy pequeña: un punto final de simulación no se interpreta como estado absorbente. Las trayectorias comparativas parten del origen, que debe figurar entre los estados iniciales de la configuración.

Los resultados anteriores se conservan en [outputs/archive_T80](outputs/archive_T80/summary.md), incluyendo su [comparación de p₀](outputs/archive_T80/p0_comparison/summary.md) y su configuración. Extender el tiempo mostrado no cambia el objetivo de horizonte infinito de la HJB; se recalcula un dominio mayor para consultar la política durante las trayectorias más largas.

Como control estructural adicional se compara el valor en **todos los nodos** entre p₀ consecutivos: con los márgenes no negativos de este modelo, elevar p₀ no debe aumentar el valor óptimo. El resultado se informa en `manifest.json`. No se impone una relación monótona a las regiones de producto 2. La fracción de nodos con un producto es una medida geométrica de la malla, no una probabilidad de visitar esos estados ni una proporción de tiempo de uso.

## Búsqueda de regiones que no sean un intervalo

Se realizó una búsqueda específica de tres estados ordenados de la misma diagonal `n=S+F` con preferencias estrictas `producto 2 → producto 1 → producto 2`. Se variaron el umbral, las probabilidades, el coste normalizado y el descuento. Los resultados están en el [informe de búsqueda](outputs/interval_search/summary.md), con [parámetros y recuentos](outputs/interval_search/search_summary.json).

La búsqueda incluye 24.108 calibraciones normalizadas distintas y 30.498 resoluciones entre dos esquemas numéricos, además de las 15 soluciones de referencia ya guardadas. No se encontró un contraejemplo. Esta evidencia finita no demuestra que la propiedad de intervalo valga para todos los parámetros del modelo continuo. Los casos de cola insuficientemente controlada se identifican como exploratorios.

El [auditor](code/audit_saved_intervals.py) revisa las secciones sin imponer una forma de umbral. Los buscadores [por lotes](code/search_intervals.py) e [independiente](code/search_intervals_independent.py) guardan las configuraciones examinadas. Los controles del detector distinguen una separación estricta de los empates y comprueban que el código pueda detectar dos regiones cuando se cambia artificialmente la demanda; ese control no se presenta como ejemplo del modelo beta.

## Ejecutar

Desde la raíz del repositorio:

```bash
.venv/bin/python "Fluid base model/code/run_experiment.py"
```

El archivo [config.json](config.json) contiene todos los parámetros. La ejecución base utiliza `h = 0.5, 0.25, 0.125`, guarda el interior `S+F <= 240` y elige automáticamente una frontera exterior para que su cota teórica de truncamiento sea como máximo `1e-6`.

Para regenerar trayectorias, diagnósticos y figuras aprovechando las mallas ya calculadas, añadir `--reuse-hjb`. Sólo se reutilizan archivos cuyos parámetros, paso, frontera, interior y tolerancia de acción coinciden; los demás se recalculan. Se conserva una solución NPZ por malla.

Para una ejecución más pequeña, con resultados separados en `outputs/quick`:

```bash
.venv/bin/python "Fluid base model/code/run_experiment.py" --quick
```

Para otra configuración y otro destino:

```bash
.venv/bin/python "Fluid base model/code/run_experiment.py" \
  --config "/ruta/a/config.json" \
  --output-dir "/ruta/a/resultados"
```

Pruebas matemáticas y de simulación:

```bash
.venv/bin/python -m unittest discover -s "Fluid base model/code" -p "test_*.py" -v
```

Dependencias: Python 3.12+, NumPy, SciPy, pandas y Matplotlib, ya incluidas en el entorno del repositorio. Las figuras se generan sin interfaz gráfica y sin exportar PDF.

## Modelo

Los estados `S,F >= 0` son cantidades continuas de éxitos y fracasos. Se conserva la demanda beta del modelo original:

\[
\theta\mid S,F\sim\operatorname{Beta}(S+1,F+1),\qquad
D(S,F)=\Pr(\theta\ge p_0\mid S,F)
=\operatorname{betaincc}(S+1,F+1,p_0).
\]

Para el producto `i`:

\[
\dot S=D(S,F)p_i,\qquad
\dot F=D(S,F)(1-p_i),\qquad
r_i(S,F)=D(S,F)(R-c_i).
\]

El objetivo es maximizar la integral de `exp(-delta*t)*r_i` hasta tiempo infinito. La tasa continua se calibra como `delta = -log(gamma)`; esto iguala el descuento por unidad de tiempo, sin afirmar equivalencia exacta con el modelo estocástico de períodos discretos.

La HJB es

\[
\delta V=D(S,F)\max_i\{R-c_i+p_iV_S+(1-p_i)V_F\}.
\]

La calibración inicial se toma de `extinction/config.json`: `p0=.5`, `p1=.35`, `p2=.8`, `c1=.05`, `c2=.65`, `R=1`, `gamma=.98`.

Se exige `0 <= p1 < p2 <= 1`, `0 < p0,gamma < 1` y `0 <= c1 <= c2 <= R`. Los beneficios no negativos permiten las continuaciones inferior cero y superior constante utilizadas en este experimento.

## Algoritmo y precisión

En una malla de paso `h`, la discretización monótona con vecinos futuros da

\[
V_h(S,F)=\frac{D}{\delta h+D}\max_i
\left\{h(R-c_i)+p_iV_h(S+h,F)+(1-p_i)V_h(S,F+h)\right\}.
\]

Ambos vecinos pertenecen a la siguiente diagonal `S+F+h`. Se resuelve hacia atrás desde `S+F=N`, sin iteración global de políticas. El tiempo crece como `O((N/h)^2)`; sólo se conservan dos diagonales de trabajo y las matrices triangulares del interior solicitado.

La ventaja normalizada del producto 2 es

\[
A_h=-\Delta c+\Delta p\frac{V_h(S+h,F)-V_h(S,F+h)}h.
\]

Su signo determina la acción nominal. El programa también construye un intervalo para esta ventaja a partir de los valores inferiores y superiores de los dos vecinos. Los mapas no imponen que las regiones sean conexas ni que exista un solo umbral.

### Corte espacial del problema infinito

El horizonte temporal de la HJB sigue siendo infinito. En el borde espacial se resuelven simultáneamente dos problemas: continuación cero y continuación constante `B=(R-c1)/delta`. Sus valores encierran el valor del problema infinito **discretizado**, en aritmética exacta. Los cálculos guardados usan doble precisión, no aritmética de intervalos.

Como `dot(S+F)=D<=1`, la cota continua para continuación cero es

\[
|V-V_N|(S,F)\le B e^{-\delta[N-(S+F)]}.
\]

El esquema tiene la cota geométrica correspondiente

\[
|V^h_\infty-V^h_N|(S,F)
\le B(1+\delta h)^{-[N-(S+F)]/h}.
\]

`choose_cutoff` utiliza esta última para elegir `N` a partir de `report_n`, `h` y `truncation_tolerance`. El ancho observado entre soluciones puede redondear a cero; por eso se informa también la cota analítica no nula. La cota no cubre el error de malla.

### Malla y control

Se comparan los valores y las acciones en todos los nodos comunes de las mallas anidadas. Los resultados son evidencia de sensibilidad numérica; no constituyen una prueba ni una cota del error de la HJB continua. Cerca de `A=0`, las acciones pueden variar aunque el valor cambie poco. Los certificados de truncamiento tampoco certifican las acciones frente al error de malla.

El residuo HJB guardado es el de la ecuación **discretizada**; comprueba la aritmética de la solución, no su error frente al continuo. El esquema monótono es consistente con la HJB; la teoría general requiere regularidad y un principio de comparación: [Barles y Souganidis (1991)](https://journals.sagepub.com/doi/10.3233/ASY-1991-4305).

Para limitar la memoria de las figuras en este dominio mayor, los mapas gráficos seleccionan nodos de la solución completa. Los colores categóricos se toman del nodo más próximo, sin promediar productos. Los archivos NPZ y los diagnósticos conservan la resolución numérica completa; el muestreo del dibujo no cambia el solucionador ni las trayectorias.

### Trayectorias

Después de optimizar se integra el campo controlado con Heun y se interpola linealmente la ventaja en los triángulos de la malla. Los empates nominales utilizan producto 1. Se comparan la política óptima y las dos políticas fijas desde los mismos estados iniciales.

Si las dos etapas de Heun eligen productos distintos, `product2_fraction` registra su mezcla numérica aproximada durante el intervalo anterior. No se presenta como una decisión aleatoria del modelo. Se repite la integración con `dt/2`, se informa su sensibilidad y se guardan las trayectorias más finas.

Todas las trayectorias paran al alcanzar `report_n`: nunca se extrapola la política. Para comparar beneficios al mismo horizonte, elegir `horizon <= report_n - max(S0+F0)`, que es suficiente porque `D<=1`. La configuración base lo cumple.

Los beneficios simulados son integrales **hasta el horizonte mostrado**. Se informa una cota del beneficio posterior omitido y se comprueba la identidad de continuación con `V_h`, sin presentar ese beneficio parcial como valor infinito exacto.

La transformación `n=S+F`, `x=(S+1)/(n+2)` se utiliza para visualizar. No reduce el problema a una dimensión: `dot x=D(p_i-x)/(n+2)` sigue dependiendo de `n`.

## Código y datos

| Archivo | Función |
|---|---|
| `code/model.py` | Demanda, HJB, cotas de frontera, interpolación y política |
| `code/simulation.py` | Trayectorias fluidas y productos fijos |
| `code/diagnostics.py` | Comparaciones, exportaciones y regiones de cambio |
| `code/figures.py` | Cinco figuras PNG en español |
| `code/run_experiment.py` | Ejecución reproducible completa |
| `code/run_p0_comparison.py` | Barrido de p₀ con escenarios separados, tablas y controles comunes |
| `code/comparison_figures.py` | Paneles de políticas, valor y trayectorias entre escenarios |
| `code/test_model.py`, `code/test_simulation.py`, `code/test_p0_comparison.py` | Benchmarks analíticos, trayectorias y validación del barrido de p₀ |
| `outputs/diagnostics.json` | Configuración efectiva y controles de precisión |
| `outputs/data/solution.npz` | Solución completa del interior, sin objetos pickle |

`solution.npz` contiene vectores triangulares concatenados. `row_offsets[k]:row_offsets[k+1]` selecciona el nivel `n=k*h`; su entrada `j` representa `S=j*h`, `F=(k-j)*h`. Los códigos de acción son `0` para empate/incertidumbre, `1` y `2` para cada producto. `load_solution` en `diagnostics.py` reconstruye el objeto y sus métodos de consulta.

Este trabajo no modifica ni reemplaza el solucionador estocástico en `extinction`. Reducir `h` aproxima el fluido; evaluar cuánto aproxima al modelo estocástico requiere una comparación posterior independiente.
