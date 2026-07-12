# Modelo fluido reparametrizado para todos los \(p_0\)

## Alcance

Este estudio compara los cinco valores de referencia utilizados en el paper,

\[
p_0\in\{0.10,0.30,0.50,0.70,0.90\},
\]

usando la solución principal con \(h=1\), \(\gamma=0.999\),
\(p_1=0.35\) y \(p_2=0.80\).

No es un diagnóstico de convergencia. No usa los archivos de
`convergence_study/` y no compara distintas discretizaciones. Los inputs son
directamente los grids y las trayectorias de producción guardados en
`outputs_gamma_0999/data/`.

Todos los nuevos resultados están separados en
`outputs_gamma_0999/reparameterized_all_p/`.

## 1. Reparametrización

Se aplica el cambio exacto

\[
n=s+f,\qquad
m=\frac{s+1}{n+2},\qquad
w(n,m)=v(s,f).
\]

La política de producto 2 puede escribirse mediante

\[
\varphi(n,m)
=
\frac{w_m(n,m)}{n+2}
=v_s-v_f
\]

y el umbral

\[
\frac{\Delta c}{\Delta p}
=
\frac{0.65-0.05}{0.80-0.35}
=\frac43.
\]

La reconstrucción de la diferencia entre los valores de acción coincide
exactamente con la política original para los cinco \(p_0\). En cada capa de
\(n\), la región de producto 2 es un único intervalo conectado en \(m\).

## 2. Comparación transversal de las bandas

| \(p_0\) | Régimen | Fracción de estados con producto 2 | Banda en \(n=80\) | Anchura en \(n=80\) |
|---:|:---|---:|:---|---:|
| 0.10 | \(p_0<p_1\) | 13.04% | \([0.0122,\,0.1220]\) | 0.1098 |
| 0.30 | \(p_0<p_1\) | 35.47% | \([0.1463,\,0.3902]\) | 0.2439 |
| 0.50 | \(p_1\le p_0\le p_2\) | 45.80% | \([0.3293,\,0.6463]\) | 0.3171 |
| 0.70 | \(p_1\le p_0\le p_2\) | 44.05% | \([0.5488,\,0.8415]\) | 0.2927 |
| 0.90 | \(p_0>p_2\) | 22.40% | \([0.8171,\,0.9878]\) | 0.1707 |

La extensión de la región de producto 2 es no monótona en \(p_0\). Crece
desde los benchmarks bajos, alcanza su máximo alrededor del régimen
intermedio y vuelve a reducirse cuando \(p_0>p_2\).

Una regularidad notable es que, en \(n=80\), \(m=p_0\) pertenece a la banda de
producto 2 en los cinco casos. La banda se desplaza hacia reputaciones más
altas conforme aumenta el benchmark, pero sigue centrada económicamente cerca
de la calidad contra la que compite Seller A.

## 3. Los tres regímenes

### \(p_0<p_1\): \(p_0=0.10,0.30\)

Las bandas se encuentran en la parte baja del espacio de reputación.

- Para \(p_0=0.10\), la frontera inferior coincide con la frontera natural
  \(s=0\), es decir \(m=1/(n+2)\), en todas las capas con producto 2.
- Para \(p_0=0.30\), la banda es mucho más ancha. Inicialmente toca las
  fronteras naturales y luego se vuelve interior por abajo.
- En ambos casos la frontera superior desciende hacia una reputación cercana
  a \(p_0\).

### \(p_1\le p_0\le p_2\): \(p_0=0.50,0.70\)

Éste es el régimen con las bandas más amplias.

- Para \(p_0=0.50\), la anchura en \(n=80\) es 0.317, la mayor de los cinco
  casos.
- Para \(p_0=0.70\), la banda permanece amplia y se desplaza hacia
  reputaciones altas.
- Las dos fronteras se mueven gradualmente hacia \(p_0\) conforme aumenta
  \(n\), reflejando la creciente rigidez de la reputación.

### \(p_0>p_2\): \(p_0=0.90\)

La banda queda pegada a reputaciones muy altas. En \(n=80\), producto 2 sólo
es óptimo aproximadamente para

\[
0.817\le m\le0.988.
\]

Como incluso la calidad alta \(p_2=0.80\) está por debajo del benchmark,
mantener demanda positiva se vuelve cada vez más difícil. La reparametrización
separa bien este efecto: la dinámica en \(n\) continúa, pero el reloj
calendario \(d\tau/dn=1/D\) se vuelve extremadamente lento.

## 4. Trayectorias óptimas

| \(p_0\) | Fracción de observaciones usando producto 2 | \(m(60)\) | Producto en \(n=60\) | Tiempo calendario en \(n=60\) |
|---:|---:|---:|:---|---:|
| 0.10 | 0.67% | 0.3577 | 1 | 60.30 |
| 0.30 | 17.64% | 0.4318 | 1 | 61.03 |
| 0.50 | 74.04% | 0.6771 | 2 | 62.32 |
| 0.70 | 100.00% | 0.7903 | 2 | 76.85 |
| 0.90 | 79.28% | 0.7007 | 1 | 179,305.88 |

La intensidad de producto 2 sobre la trayectoria también es no monótona.

- Con \(p_0=0.10\), sólo se usa brevemente al comienzo y la reputación converge
  hacia una zona cercana a \(p_1\).
- Con \(p_0=0.50\), la trayectoria alterna cerca de la frontera de switching y
  usa producto 2 durante la mayor parte de las observaciones.
- Con \(p_0=0.70\), producto 2 se utiliza durante toda la trayectoria y
  \(m(n)\) se aproxima a \(p_2\).
- Con \(p_0=0.90\), la trayectoria usa producto 2 inicialmente, pero termina
  fuera de la banda. Al quedar \(m<p_0\), la demanda se hace extremadamente
  pequeña: alcanzar \(n=60\) requiere aproximadamente 179,306 unidades de
  tiempo calendario.

Los múltiples cambios de producto en algunos paths provienen del feedback
bang--bang evaluado cerca de una frontera sobre la malla \(h=1\); deben leerse
como seguimiento discreto de la banda, no como ciclos económicos separados.

## 5. Lectura general

La reparametrización muestra que no hay una anomalía exclusiva de
\(p_0=0.10\) o \(0.30\). Existe una familia continua de bandas de inversión:

1. la banda empieza en reputaciones bajas cuando \(p_0<p_1\);
2. se ensancha al entrar en el intervalo \([p_1,p_2]\);
3. se desplaza hacia reputaciones altas y vuelve a estrecharse cuando
   \(p_0>p_2\).

El ancho máximo aparece en el régimen donde Seller A puede cruzar el benchmark
alternando entre las dos calidades. Fuera de ese régimen, una de las dos
fronteras naturales restringe progresivamente la región económicamente útil.

## 6. Archivos

- `plots/policy_nm_all_p.png`: mapas de política para los cinco \(p_0\).
- `plots/product2_boundaries_nm_all_p.png`: bandas y regímenes.
- `plots/product2_intervals_selected_n_all_p.png`: comparación transversal en
  \(n=10,40,80\).
- `plots/optimal_paths_nm_all_p.png`: trayectorias \(m(n)\) y uso de producto 2.
- `plots/phi_switching_margin_nm_all_p.png`: margen
  \(w_m/(n+2)-\Delta c/\Delta p\).
- `tables/all_p_policy_summary.csv`: métricas principales de las bandas.
- `tables/all_p_optimal_path_summary.csv`: métricas de las trayectorias.
- `tables/all_p_product2_boundaries_nm.csv`: fronteras capa por capa.
- `data/reparameterized_grid_p0_*.csv.gz`: grids transformados.
- `data/reparameterized_path_p0_*.csv.gz`: trayectorias transformadas.

Reproducción:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  discounted/Fluid/reparameterized_all_p_study.py
```
