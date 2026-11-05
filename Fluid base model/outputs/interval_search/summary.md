# Búsqueda de una región del producto 2 que no sea un intervalo

**Resultado: no se encontró un contraejemplo para el modelo fluido beta sin olvido.** Es un resultado de una búsqueda finita, no una demostración de inexistencia.

## Qué significaría un contraejemplo

Se mantiene fijo n=S+F. Se ordenan los estados por S, o equivalentemente por x=(S+1)/(n+2). Buscamos tres estados x₁<x₂<x₃ con preferencia estricta por los productos 2, 1 y 2, respectivamente. No se usan empates para fabricar una desconexión.

\[A(S,n)=-(c_2-c_1)+(p_2-p_1)(V_S-V_F),\qquad A(S_1,n)>0,\ A(S_2,n)<0,\ A(S_3,n)>0.\]

La condición se refiere a una sección del mapa bidimensional. No equivale a preguntar si la región completa en (S,F) es convexa o conexa.

## Evidencia numérica

Se resolvieron 30,498 problemas numéricos nuevos, correspondientes a 24,108 calibraciones normalizadas distintas, además de auditar las 15 soluciones ya guardadas. Los refinamientos no se cuentan como parámetros nuevos.

| Comprobación | Resoluciones | Secciones revisadas | Resultado |
|---|---:|---:|---|
| Diferencias monótonas por vecinos futuros | 28,438 | 3,502,922 | Ningún patrón +,−,+ |
| Método semi-Lagrangiano independiente | 2,060 | 179,200 | Ningún patrón +,−,+ |
| Calibración base guardada, cinco p₀ y tres mallas | 15 | 16,815 niveles, incluidos los degenerados | Ningún patrón +,−,+ |

El primer buscador utiliza 22.228 calibraciones distintas y mallas h=1, 0.5 y 0.25. Incluye p₀ entre .001 y .999, varios pares de calidad, costes normalizados, y δ∈{.01,.03,.1,.3,1,3}. Los dominios de interés llegan hasta n=200 en la exploración de δ=.01, n=80 en varias otras rondas y n=40 en la de h=.25. La auditoría base llega hasta n=240 y h=.125. Estos rangos son conjuntos muestreados, no cajas continuas cubiertas.

La búsqueda independiente usa 1.880 calibraciones distintas, con brechas de calidad tan pequeñas como 1.30×10⁻⁵ y costes tan pequeños como 3.71×10⁻⁹. Abarca δ aproximadamente entre .00321 y 31.31; también refina 180 casos de Δn=1, ΔS=.25 a Δn=.5, ΔS=.125.

Las preferencias nominales se comprobaron respecto de cero, sin tolerancia ni anchura mínima: tampoco aparecieron pequeñas islas nominales. Por separado se aplicaron márgenes de signo y cotas inferiores/superiores de continuación para descartar candidatos debidos al corte.

## Precisión y límites

- El barrido principal elige el corte para que su cota geométrica de error de valor sea menor que 10⁻⁵; su detector con cotas exige márgenes de ventaja de 10⁻⁴.
- En la búsqueda independiente, 216 calibraciones de descuento pequeño tienen una cola insuficientemente controlada y se consideran exploratorias. Las otras 1.664 calibraciones únicas tienen anchura de continuación menor que 10⁻⁶. La ronda de costes escalados tiene anchura máxima menor que 5.3×10⁻¹².
- Las cotas de continuación controlan el corte del problema discretizado. No son cotas del error de discretización frente a la HJB continua; se usa doble precisión, no redondeo dirigido.
- Parámetros y mallas finitos pueden omitir una región muy estrecha, estados fuera de los dominios examinados o regímenes no muestreados. La ausencia de ejemplos numéricos no permite afirmar que siempre exista un solo intervalo.

Los márgenes se normalizan a R=1, c₁=0 y c₂=d=(c₂ original−c₁ original)/(R original−c₁ original). Esta normalización conserva la política cuando R−c₁>0 y evita barrer escalas equivalentes de beneficios. Se conserva la demanda beta y el objetivo de horizonte infinito.

## Controles del buscador

El solucionador por lotes se contrastó con model.solve en diez calibraciones. El detector pasó controles sintéticos de un intervalo, de dos intervalos, de empates, de signos de magnitud 10⁻¹⁰ y de incertidumbre en el punto separador. La suite completa pasa 39 pruebas.

Como control positivo se sustituyó temporalmente la demanda por dos escalones suaves: el programa sí detectó zonas separadas. Ese control cambia la demanda y se excluye de todos los recuentos y conclusiones sobre el modelo beta.

## Por qué aún falta una prueba

El teorema de intervalo que existe en las notas corresponde al fluido con olvido y concentración fija. El modelo actual conserva n como variable dinámica, por lo que ese resultado no se aplica directamente.

Una reducción exacta es $k=\Delta c/\Delta p$, $a=R-c_1+kp_1$ y $V(n,S)=kS+W(n,S)$. Entonces
\[\delta W=D\,[a+W_n+\max_i p_iW_S]-\delta kS.\]

La propiedad buscada sería que {S:W_S(n,S)>0} sea un intervalo para cada n. La forma de la demanda beta por sí sola no constituye aquí una demostración de esa propiedad para el valor óptimo futuro. Tampoco basta mostrar que un gap es no unimodal: debe cruzar el nivel real de elección con signos +,−,+.

## Archivos

- [Resumen de búsqueda en JSON](search_summary.json).
- [Parámetros y resultados del barrido principal](engine_a/combined_summary.json).
- [Cobertura del método independiente](engine_b/coverage.json) y [reproducción](engine_b/README.md).
- [Auditoría de las soluciones anteriores](baseline_audit.json).
- [Curvas de ventaja en n=20 para las calibraciones base](figures/advantage_sections.png): ilustran el criterio, no toda la búsqueda.
- [Buscador por lotes](../../code/search_intervals.py), [buscador independiente](../../code/search_intervals_independent.py) y [auditor de soluciones guardadas](../../code/audit_saved_intervals.py).

## Reproducir desde la raíz del repositorio

```bash
.venv/bin/python "Fluid base model/code/audit_saved_intervals.py"
.venv/bin/python "Fluid base model/code/search_intervals.py" --output "Fluid base model/outputs/interval_search/engine_a/grid_coarse"
.venv/bin/python "Fluid base model/code/search_intervals.py" --p0 .01,.05,.1,.3,.5,.7,.9,.95,.99 --delta .01 --report-n 200 --output "Fluid base model/outputs/interval_search/engine_a/grid_low_discount"
.venv/bin/python "Fluid base model/code/search_intervals.py" --p0 .1,.3,.5,.7,.9 --delta .03,.1,.3 --h .5 --output "Fluid base model/outputs/interval_search/engine_a/grid_half"
.venv/bin/python "Fluid base model/code/search_intervals.py" --p0 .1,.3,.5,.7,.9 --delta .03,.1,.3 --p 0,.2,.5,.8,1 --h .25 --report-n 40 --output "Fluid base model/outputs/interval_search/engine_a/grid_quarter"
.venv/bin/python "Fluid base model/code/search_intervals.py" --p0 .001,.01,.99,.999 --delta .03,.1,.3,1,3 --random-configurations 512 --report-n 120 --output "Fluid base model/outputs/interval_search/engine_a/random_extreme_p0"
.venv/bin/python "Fluid base model/code/plot_interval_sections.py"
```

El método independiente y sus tres rondas se reproducen con los comandos guardados en su README. Los archivos de cada grupo incluyen todos los parámetros usados y la semilla 261020.
