# Búsqueda independiente: modelo fluido sin olvido

Se examinaron 1880 calibraciones diferentes y 2060 soluciones contando el refinamiento. No se encontró ninguna sección a n fijo con el patrón estricto 2–1–2. El detector nominal usa signo >0, sin tolerancia ni filtro de anchura; incluye todas las diagonales con al menos tres nodos, también n<2.

| Fase | Calibraciones | dn | dS | Máximo n | Componentes nominales máximas |
|---|---:|---:|---:|---:|---:|
| extremes | 800 | 1 | 0.25 | 80.0 | 1 |
| scaled_costs | 1080 | 1 | 0.25 | 80.0 | 1 |
| refined_dn_0.5 | 180 | 0.5 | 0.125 | 80.0 | 1 |

El método es semi-Lagrangiano en n=S+F: mantiene constante la demanda durante cada paso, integra exactamente el descuento bajo esa aproximación e interpola linealmente el sucesor S+p_i dn. Es monótono y consistente con la HJB; difiere de la transición Bernoulli de model.py.

Se resuelve dos veces simultáneamente con continuaciones 0 y 1/delta para diagnosticar la influencia del corte. El barrido de costes escalados tiene anchura máxima de ese bracket inferior a 5.3e-12. Algunos extremos con descuento muy pequeño tienen brackets amplios y son exploratorios. Estas cifras usan punto flotante, no redondeo certificado.

La ausencia de ejemplos en esta búsqueda no demuestra que todas las secciones del modelo continuo sean intervalos. Los brackets de cola tampoco incluyen el error de discretización.

Detalles de parámetros y recuentos: [coverage.json](coverage.json). Cada directorio contiene todos los parámetros para reproducirlo.

Reproducir desde la raíz del repositorio:

```bash
.venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design extremes --count 90 --cutoff-n 400 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/extremes_nominal'
.venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design scaled --count 120 --cutoff-n 360 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/scaled_nominal'
.venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design scaled --count 20 --dn 0.5 --cutoff-n 360 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/refined_h05_nominal'
.venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --coverage-only
```
