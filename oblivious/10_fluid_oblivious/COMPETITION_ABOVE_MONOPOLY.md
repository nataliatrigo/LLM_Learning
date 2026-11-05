# Competencia por encima de monopolio desde el mismo inicio

Se revisaron 24 combinaciones y 27 políticas competitivas del experimento 10.

## Competencia 100% y monopolio 0%

No aparece ningún caso en la grilla activa, ni usando otras políticas guardadas cuya trayectoria no confirme su conjetura.

## Competencia con una proporción mayor de producto 2

Sí aparece en 18 combinaciones. En todas las filas siguientes la competencia alcanza su propia conjetura; varios resultados son mixtos, no 100% frente a 0%.

| p0 | rho | condiciones iniciales x0 | producto 2 monopolio | producto 2 competencia |
| --- | --- | --- | --- | --- |
| 0.1 | 0.6 | [0, 1] | 0.0000% | 0.5002% |
| 0.1 | 0.7 | [0, 1] | 0.0000% | 19.8432% |
| 0.1 | 0.8 | [0, 1] | 0.0000% | 43.4225% |
| 0.1 | 0.9 | [0, 1] | 0.0000% | 72.0331% |
| 0.3 | 0.6 | [0, 1] | 0.0000% | 20.1255% |
| 0.3 | 0.7 | [0, 1] | 6.5626% | 32.1260% |
| 0.3 | 0.8 | [0, 1] | 13.0688% | 47.2153% |
| 0.3 | 0.9 | [0, 1] | 15.7495% | 72.0487% |
| 0.5 | 0.6 | [0, 1] | 37.6873% | 46.7391% |
| 0.5 | 0.7 | [0, 1] | 45.4954% | 55.9445% |
| 0.5 | 0.8 | [0, 1] | 52.2481% | 65.4939% |
| 0.5 | 0.9 | [0, 1] | 55.9294% | 76.9431% |
| 0.7 | 0.6 | [0, 1] | 71.5916% | 74.3232% |
| 0.7 | 0.7 | [0, 1] | 79.1431% | 82.2317% |
| 0.7 | 0.8 | [0, 1] | 86.2571% | 90.1095% |
| 0.7 | 0.9 | (0.420713457523, 1] | 91.4439% | 97.1615% |
| 0.8 | 0.6 | [0, 1] | 87.3084% | 88.2178% |
| 0.8 | 0.7 | [0, 1] | 94.5547% | 95.5551% |

Se comparan destinos asintóticos de las políticas guardadas de 4001 nodos, con idéntico inicio en monopolio y en ambos vendedores competitivos. Las fronteras siguen el selector guardado y son aproximaciones numéricas. Las reputaciones finales determinan la proporción de producto 2 mediante `(x-p1)/(p2-p1)`.

Datos y revisión de todos los casos: [summary.json](data/competition_above_monopoly/summary.json).
