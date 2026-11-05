# Comparación económica de OE distribution y OE p0

[Índice de experimentos oblivious](../../README.md).

La carpeta `oblivious/06_equilibrium_comparison` se organiza en `code` (scripts, pruebas y
esta documentación), `data` (datos y resultados) y `figures` (figuras).
Los comandos siguientes se ejecutan desde la raíz del repositorio.

Este experimento aplica las políticas guardadas de los dos modelos al mismo
juego estocástico de dos vendedores. No calcula nuevos equilibrios ni modifica
los resultados de los experimentos de origen.

Las entradas por defecto son `oblivious/01_oe_distribution/data/rho_forgetting` y
`oblivious/02_oe_p0/data/rho_forgetting_mixed`, con ρ=0.95 y 51 puntos. Se usan los
parámetros de esas soluciones: R=1, c₁=0.05, c₂=0.65, p₁=0.35, p₂=0.80 y γ=0.98.
`input_validation.json` documenta los candidatos seleccionados, los parámetros,
las huellas de los archivos y las comprobaciones de consistencia y estacionariedad.
Un resultado ausente, incompatible o no verificado detiene la comparación.

## Comparación conjunta entre tasas de olvido

La extensión `cross_rho` reutiliza el mismo simulador y las políticas seleccionadas
guardadas, sin ejecutar ningún solver ni alterar los modelos. Compara
ρ ∈ {0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99} por defecto en figuras conjuntas.
La comparación histórica guardada incluye también ρ=0.999 y se conserva.
Los resultados nuevos quedan en `data/cross_rho_n51` y las figuras en
`figures/cross_rho_n51`; los archivos de una sola ρ se conservan.

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 .venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.cross_rho

# Reanudar únicamente las tasas pendientes.
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -u -B \
  -m oblivious.equilibrium_comparison.code.cross_rho --resume

# Redibujar desde los checkpoints, sin volver a simular ni leer políticas nuevas.
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B \
  -m oblivious.equilibrium_comparison.code.cross_rho --plots-only
```

La configuración común usa la grilla original de 51 puntos, 20 réplicas pareadas,
100.000 períodos retenidos, 10.000 de descarte y el estado inicial `(0.5, 0.5)`.
Se comparten los mismos uniformes entre modelos y tasas, con réplicas
independientes. Los registros existentes de ρ=0.95 se reutilizan solamente si
coinciden la configuración, las auditorías y las fuentes pertinentes.
`--mode nearest` o `--mode linear`, junto con un nombre distinto, permiten
comparar una extensión continua; las predicciones de la grilla se identifican
entonces como referencias y no como leyes estacionarias del proceso continuo.

Las cinco figuras principales, en PDF y PNG a 300 dpi, son:

1. Ocho paneles de probabilidades óptimas de producto 2 frente a reputación propia.
2. Frecuencia de compras de producto 2 frente a ρ, con intervalos Monte Carlo
   del 95% y predicciones estacionarias identificadas por separado.
3. Dos paneles: beneficio medio por vendedor y período calendario, y éxito del cliente.
4. Ocho paneles de CDFs de reputación, con bandas puntuales y predicciones guardadas.
5. Ocho filas por dos columnas de las leyes conjuntas estacionarias verificadas.

El suplemento `S01` muestra las leyes conjuntas empíricas de las simulaciones,
con los mismos ejes y la misma escala de probabilidad que la figura 5. Una
escala común `viridis` (morado–verde–amarillo), con la misma tipografía de los
heatmaps originales, representa masa en ambos modelos; los encabezados identifican
OE distribution en verde y OE p0 en naranja. La escala de masa sigue siendo logarítmica
y común a todos los paneles. Las tasas del eje horizontal se presentan
como valores discretos igualmente espaciados, y las líneas solo guían la vista.
Los nodos extremos de los mapas se dibujan con ancho de celda completo para
que el marco no oculte los átomos en reputación cero; sus masas se anotan
cuando concentran más de la mitad de la probabilidad.

El suplemento adicional `S02` muestra las simulaciones de todas las tasas con
la escala **lineal original** de `rho095_n51`: `viridis`, mínimo 0 y máximo
0.0528115. Los paneles de ρ=0.95 usan exactamente los mismos histogramas de
aquella corrida. La escala es común para las ocho tasas; las masas que superan
el máximo original se pintan con el color superior, indicado por el triángulo
de la barra de color. Se genera sin volver a simular ni sobrescribir las demás
figuras:

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B \
  -m oblivious.equilibrium_comparison.code.plot_joint_linear
```

El JSON junto al PDF y PNG guarda los límites de color y la procedencia exacta
de la comparación de referencia. `--reference` permite seleccionar otra corrida
anterior de ρ=0.95, como `oblivious/06_equilibrium_comparison/data/rho095_n51_long_grid`.

El suplemento `S03` reúne las distribuciones guardadas y simuladas en una sola
figura de ocho filas y cuatro columnas, en este orden: **OE distribution saved stationary,
OE distribution simulated, OE p0 saved stationary, OE p0 simulated**.
Los 32 paneles comparten los ejes y la misma escala logarítmica `viridis`
que las figuras `05` y `S01`. Se reutilizan directamente sus datos guardados:

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B \
  -m oblivious.equilibrium_comparison.code.plot_joint_combined
```

Se guarda como `S03_joint_stationary_and_simulated_log_scale.pdf` y `.png`;
el JSON adjunto registra la procedencia y el contenido de cada panel.

Los intervalos t de Student se calculan entre medias de trayectorias
independientes. El beneficio medio incluye ambos vendedores y los períodos
en que no son seleccionados; las CDFs agrupan ambos vendedores dentro de cada
réplica antes de calcular bandas. Los intervalos no cubren el sesgo del inicio.
La propagación determinista del kernel guardado separa ese sesgo de la
incertidumbre Monte Carlo en `summary.csv` y los checkpoints por tasa.
La figura 5 usa las leyes guardadas precisamente para representar estacionariedad
sin suponer que un descarte fijo elimina episodios persistentes.

`verification_status.csv` identifica las tasas verificadas. Si una pareja de
políticas no supera el cargador existente, su panel se marca **UNVERIFIED /
UNAVAILABLE** y sus curvas quedan interrumpidas; el programa no sustituye
candidatos ni presenta esa pareja como equilibrio. La verificación es numérica
y conserva las tolerancias y definiciones del experimento original.

El archivo `data/cross_rho_n51/README.txt` detalla los resultados y diagnósticos;
`summary.csv`, `replications.csv`, `paired_differences.csv` y
`reputation_cdfs.csv` contienen los datos de las figuras. Los checkpoints
preservan las leyes conjuntas, políticas, auditorías, semillas y procedencia;
los registros de cada evento de las tasas nuevas se pueden reproducir con el
simulador existente, sin duplicarlos en esta carpeta.

## Ejecutar

Desde la raíz del repositorio:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl

.venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.run_experiment \
  --name rho095_n51
```

La configuración inicial utiliza 20 réplicas independientes, 100.000 períodos
retenidos, 10.000 de descarte inicial y tres trayectorias ilustrativas de 1.000
períodos. Ambas políticas parten de `(0.5, 0.5)`, estados maduros de reputación.
Los períodos se cuentan en tiempo calendario: ocurre una compra por período.

El programa también ejecuta comprobaciones de estabilidad: cada trayectoria
principal llega hasta 220.000 períodos, permitiendo separar el cambio de longitud
(200.000 observaciones) del cambio de descarte inicial (20.000 períodos). Las
ventanas se solapan y no son réplicas independientes adicionales. Para estudiar
la sensibilidad a la grilla, repite la validación discreta y la extensión al
nodo más cercano con las soluciones ya guardadas de 21 y 31 puntos. Esto permite
comparar resoluciones disponibles; no demuestra convergencia fuera de ellas.

También se propaga directamente la distribución inicial mediante el kernel
conjunto guardado, sin sorteos aleatorios. Esta validación distingue el sesgo
del estado inicial de la incertidumbre Monte Carlo. En particular, los estados
de reputación muy baja de OE p0 pueden producir episodios persistentes
que no quedan bien representados en pocas trayectorias.

```bash
# Otra configuración: siempre un nombre nuevo.
.venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.run_experiment \
  --name mas_replicas --replications 40 --periods 200000 --burn-in 20000

# Prueba corta, sin sensibilidad de duración ni otras grillas.
.venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.run_experiment \
  --name prueba_corta --periods 2000 --burn-in 200 --replications 3 \
  --short-periods 1000 --no-stability --refinement-grids

# Retomar configuraciones pendientes tras una interrupción.
.venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.run_experiment \
  --name rho095_n51 --resume

# Validación discreta adicional de mezcla lenta: 500.000 períodos de descarte
# y un millón de observaciones por réplica, conservando la ejecución inicial.
.venv/bin/python -u -B -m oblivious.equilibrium_comparison.code.run_experiment \
  --name rho095_n51_long_grid --modes grid --periods 1000000 \
  --burn-in 500000 --replications 20 --no-stability --refinement-grids

# Pruebas numéricas independientes.
.venv/bin/python -B -m unittest discover -s oblivious/06_equilibrium_comparison/code -p 'test_*.py'
```

Los resultados se guardan en `data/<nombre>` y las figuras en `figures/<nombre>`.
Las figuras conservadas son 01–07, 09, 11 y 14, en PNG y PDF; mantienen su
numeración original.
No se sobrescribe una ejecución existente. La reanudación usa su configuración
guardada y exige las mismas fuentes e insumos; una combinación de grilla y modo
interrumpida vuelve a calcularse, mientras que las completadas se recuperan.
`--oe-candidate` y `--p0-candidate` permiten escoger explícitamente otro candidato
verificado, con índices desde cero. Por defecto se conserva la selección guardada.

La figura 5 tiene dos paneles: el primero compara a A y B juntos, usando el
porcentaje de ventas de producto 2 entre las ventas de cada vendedor (círculos
para A y cuadrados vacíos para B). El segundo muestra el porcentaje de producto
2 entre todas las ventas del mercado. El color identifica la política guardada.

La figura 6 compara a A y B juntos en la columna izquierda, con los mismos
marcadores que la figura 5. La columna derecha muestra la suma de beneficios
de ambos vendedores. La fila superior presenta el beneficio promedio por
período después del ajuste inicial; la inferior, el beneficio descontado desde
el estado inicial común.

La figura 7 contiene un único panel de barras horizontales con el porcentaje
de períodos en que se selecciona al vendedor A. Cada barra muestra el porcentaje
estimado y su intervalo de confianza del 95%. La línea discontinua del 50%
indica la frecuencia esperada por simetría. El eje conserva la escala de 0% a
100%, y los colores distinguen las dos políticas guardadas.

### Uso del producto 2 a lo largo del tiempo

La figura `14_product2_usage_over_time.png` (también en PDF) compara las políticas
guardadas desde el estado común `(0.5, 0.5)`. Su eje horizontal es el período
calendario desde `t=0`. En cada período calcula
`q2(t) = (1/M) sum_r 1{producto[r,t] = 2}`, usando el producto del vendedor
seleccionado y todas las réplicas disponibles. No descarta los períodos iniciales.
El color identifica el experimento de origen de la política; cada pareja de
paneles identifica la regla de evolución de reputaciones usada en la simulación.

El panel superior muestra la frecuencia sin suavizar durante los primeros 500
períodos. Debajo se muestran los primeros 10.000 períodos con un promedio móvil
retrospectivo de 100 períodos. Al inicio usa solamente los períodos disponibles;
no incorpora observaciones futuras. Las bandas del 95% se calculan con los
promedios móviles de cada trayectoria independiente, sin tratar los períodos
correlacionados dentro de una trayectoria como réplicas adicionales. Son
intervalos puntuales de Monte Carlo, no bandas simultáneas ni una prueba de
convergencia a la distribución estacionaria.

Las líneas horizontales discontinuas muestran la frecuencia estacionaria de
producto 2 calculada con cada política guardada y su distribución conjunta:
`q2_stationary = sum_ij nu[i,j] * (D[i,j] * policy[i] + (1-D[i,j]) * policy[j])`.
Aquí `D[i,j]` es la probabilidad real de seleccionar a A en ese par de
reputaciones. Se incluyen tanto la ocupación de cada estado como la selección
del vendedor; no se sustituye la ley conjunta por marginales independientes.
Los valores de la grilla original de 51 reputaciones son 74.5200% para OE distribution y
84.3331% para OE p0. Proceden de `analysis` y coinciden con
`stationary_predictions.csv`, sin estimarse con Monte Carlo.

Esas distribuciones son estacionarias para el proceso discreto original. En
los paneles de reputaciones continuas, las mismas líneas se identifican como
referencias del cálculo original. La cercanía de una curva temporal a la línea
no demuestra por sí sola la convergencia de toda la distribución conjunta.
El suplemento guarda los valores, la definición y la procedencia de las
referencias; el CSV las incluye en la columna `original_stationary_q2`.

Para agregarla a una ejecución existente sin volver a simular ni calcular políticas:

```bash
MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B \
  -m oblivious.equilibrium_comparison.code.plot_product_usage --name rho095_n51

MPLCONFIGDIR=/tmp/equilibrium-comparison-mpl .venv/bin/python -B \
  -m oblivious.equilibrium_comparison.code.plot_product_usage --name rho095_n51_long_grid
```

El comando lee los archivos `raw/n051_<modo>/{oe,p0}.npz`, incluyendo sus 20
réplicas, y añade `product2_usage_over_time.csv` y el suplemento `.json`/`.npz`
con los datos y su procedencia. Conserva los resultados y manifiestos originales.
`--periods`, `--smoothing-window` y `--early-periods` permiten cambiar el horizonte
y el suavizado. Las nuevas ejecuciones también generan la figura automáticamente;
`--usage-periods` y `--usage-smoothing-window` controlan sus valores por defecto.

## Selección real, transiciones y extensión de políticas

En todos los modos se sortean ambas Beta usando las reputaciones actuales,
se selecciona A si su sorteo supera el de B (empates para B), se sortea el
producto según la política y después se sortea el éxito. Solo se actualiza
la reputación del vendedor seleccionado. Nunca se usa la demanda del problema
aproximado para seleccionar vendedores.

- **Reputations restricted to saved values:** las reputaciones evolucionan
  entre los valores usados en los experimentos originales. Después de éxito
  o fracaso se sortea entre nodos vecinos con los pesos de la misma función
  `interpolation_matrix` usada por el solver. La selección con sorteos Beta
  corresponde al kernel del solver salvo su error numérico de integración de
  la demanda, auditado mediante duplicación del orden y cuadratura independiente.
  Esas comprobaciones miden errores observados, no una cota rigurosa global.
- **Continuous reputations: closest saved rule:** las reputaciones evolucionan sin redondeo;
  se utiliza la probabilidad del nodo más cercano (empates al nodo inferior).
  Conserva elecciones puras y mezclas guardadas, sin imponer políticas de umbral.
- **Continuous reputations: blended saved rules:** las reputaciones también son continuas, pero
  se interpola linealmente la probabilidad de producto 2. Esta aproximación
  adicional puede crear mezclas entre nodos con acciones puras diferentes;
  se presenta separadamente como sensibilidad.

Se reutilizan los posteriores, transiciones, demanda y kernel conjunto de
`oblivious/02_oe_p0`, así como la lectura de sus archivos y los intervalos entre
réplicas. La nueva simulación permite el inicio común y las reglas de extensión
que los diagnósticos anteriores no ofrecían.

## Aleatoriedad e incertidumbre

Cada réplica tiene una semilla independiente generada con `SeedSequence`.
Para comparar políticas, reglas de extensión y grillas se comparten los mismos
uniformes de cada réplica y período: dos para los sorteos Beta mediante inversa
de la CDF, uno para producto, uno para éxito y uno para proyección a la grilla.
La extensión del horizonte conserva exactamente el prefijo de la trayectoria.

Los errores estándar e intervalos del 95% usan medias entre réplicas y la
distribución t de Student. Las diferencias entre modelos utilizan parejas con
la misma semilla y se reportan como **OE distribution menos OE p0**. Las observaciones
dentro de una trayectoria no se tratan como independientes. Los intervalos de
CDF y promedios acumulados son puntuales; no son bandas simultáneas.

El beneficio descontado comienza en el período cero, antes del descarte inicial,
y se trunca al horizonte corto configurado. Su cota de cola por vendedor es
`R * gamma**short_periods / (1-gamma)` y se registra en el manifiesto. Es distinto
del beneficio medio estacionario por período calendario.

## Archivos y medidas

- `manifest.json`: configuración, progreso, código y verificación de insumos.
- `input_validation.json`: certificados, residuos recalculados y procedencia.
- `transient_validation.json`, `.csv` y `.txt`: evolución determinista de las
  leyes conjuntas desde `(0.5, 0.5)` y separación entre sesgo del horizonte finito,
  desviación Monte Carlo y predicciones estacionarias.
- `stationary_predictions.csv`: resultados económicos calculados con cada ν
  guardada y las probabilidades reales de selección de vendedores.
- `summary.csv`: estimaciones, errores estándar e intervalos por modelo, modo
  y ventana de análisis. Incluye medias, varianzas, correlación, participación,
  producto 2, beneficios y resultados del cliente.
- `differences.csv`: diferencias pareadas OE distribution menos OE p0.
- `replications.csv`: estadísticas individuales, base de los intervalos.
- `sensitivity.csv` y `sensitivity_differences.csv`: resultados por grilla.
- `analysis.json` y `analysis.npz`: objetos estructurados para reproducir figuras.
- `raw/nNNN_<modo>/`: registros de todos los períodos, incluido el descarte,
  en NPZ sin pickle. Hay estados antes/después, ambos sorteos TS, selección,
  producto, éxito, beneficios y probabilidad de producto 2. Los uniformes
  compartidos y las semillas se conservan por separado. Para reducir el espacio
  ocupado se puede usar `--compress-raw`, a cambio de tiempo de compresión.
  Estos registros extensos se excluyen de Git; las tablas, auditorías y figuras
  quedan disponibles para versionar.
- `checkpoints/`: resultados de configuraciones completas para reanudar.
- Las figuras PNG/PDF y el informe de resultados se generan al terminar.

El uso condicional de producto 2 divide sus compras por las compras atendidas
por ese vendedor. El uso por período calendario divide por todos los períodos.
El uso de mercado cuenta todas las compras; hay exactamente una por período.

Wasserstein y la diferencia máxima entre CDF usan las reputaciones observadas
sin redondearlas. Los histogramas de estados continuos emplean celdas alrededor
de los nodos únicamente para visualización. Los residuos empíricos del kernel
solo se calculan para simulaciones discretas.

El desacuerdo entre políticas usa la diferencia absoluta entre probabilidades
de producto 2. Se calcula uniformemente en todo `[0,1]`, ponderado por estados
visitados y condicionado a la selección de cada vendedor. Una diferencia pequeña
entre probabilidades mixtas no se cuenta como una discrepancia de acción completa.

La comparación principal utiliza perfiles simétricos. La función de simulación
también admite asignaciones cruzadas mediante `assignments`; no forman parte del
barrido inicial ni se interpretan como equilibrios o mejores respuestas.

## Consultar la ejecución guardada

- [Resumen e interpretación en español, incluida la validación larga](../data/rho095_n51/RESUMEN.txt).
- [Informe de las diez preguntas de investigación](../data/rho095_n51/research_summary.txt).
- [Tabla comparativa con intervalos y diferencias pareadas](../data/rho095_n51/comparison_table.csv).
- [Uso de producto 2 desde el inicio: frecuencia por período y promedio móvil](../figures/rho095_n51/14_product2_usage_over_time.png).
- [La misma evolución con reputaciones en la grilla original](../figures/rho095_n51_long_grid/14_product2_usage_over_time.png).
- [Diagnóstico determinista del horizonte finito](../data/rho095_n51/transient_validation.txt).
- [Validación discreta adicional de un millón de períodos](../data/rho095_n51_long_grid/research_summary.txt).

La ejecución principal conserva los tres modos y las tres grillas. La validación
adicional estudia la mezcla lenta en la grilla de 51 puntos; no demuestra mezcla
rápida en estados continuos ni convergencia a grillas más finas.
