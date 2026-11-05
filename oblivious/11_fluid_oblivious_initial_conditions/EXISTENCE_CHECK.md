# Comprobación independiente de los intervalos sin equilibrio

Se volvió a buscar en los cuatro casos con condiciones iniciales sin cubrir, sin utilizar los candidatos guardados del experimento 10.

## Qué se comprobó

- HJB nuevo para 401 conjeturas por caso en todo `[p1,p2]`, con 1001 nodos de estado.
- Enumeración independiente de candidatos interiores mediante cuadratura Legendre, derivada analítica, refinamiento de la malla y búsqueda de extremos. También se cubren todos los intervalos de conjeturas con cotas analíticas de derivadas, condicionadas a los presupuestos numéricos de error de cuadratura.
- Verificación global de holding y consistencia en mallas nuevas [4001, 8001, 16001].
- Cada candidato rechazado tiene una desviación factible, calculada sin el HJB, que da un beneficio mayor que mantener la reputación propuesta.

## Resultado por caso

| p0 | rho | reputaciones de equilibrio, malla más fina | condiciones iniciales sin cubrir, malla más fina |
| --- | --- | --- | --- |
| 0.7 | 0.9 | 0.786937002 | [0, 0.420499495] |
| 0.8 | 0.9 | 0.350000000, 0.800000000 | (0.530152617, 0.57228024] |
| 0.9 | 0.7 | 0.350000000, 0.800000000 | (0.464294441, 0.485554151] |
| 0.9 | 0.8 | 0.350000000, 0.800000000 | (0.645802163, 0.664441836] |

## Candidatos adicionales descartados

La última columna es la mejora de beneficio de una estrategia explícita: usar un producto puro hasta otra reputación y después mantenerla.

| p0 | rho | candidato x_bar | reputación de destino de la desviación | mejora de beneficio |
| --- | --- | --- | --- | --- |
| 0.7 | 0.9 | 0.350000000 | 0.750000000 | 0.121586701 |
| 0.7 | 0.9 | 0.800000000 | 0.790000000 | 0.014263041 |
| 0.7 | 0.9 | 0.404435378 | 0.504435378 | 0.287636324 |
| 0.8 | 0.9 | 0.567964443 | 0.667964443 | 0.190459605 |
| 0.9 | 0.7 | 0.528646970 | 0.628646970 | 0.045164519 |
| 0.9 | 0.8 | 0.649937955 | 0.549937955 | 0.051306508 |

## Alcance de la conclusión

La nueva búsqueda no encontró equilibrios adicionales que cubran los huecos. Los candidatos descartados fallan optimalidad incluso al compararlos con estrategias factibles sencillas.

La exhaustividad de los candidatos tiene una comprobación numérica refinada. Las cotas de cuadratura y del HJB no son cotas certificadas por aritmética intervalar, por lo que este informe no se presenta como una prueba matemática formal de inexistencia en el modelo continuo.

Las fronteras de condiciones iniciales son aproximadas: se informa su cambio al refinar la malla. La inclusión exacta de los umbrales repulsivos conserva la convención de continuación del experimento.

El argumento de necesidad permite incluir candidatos aunque el valor óptimo no sea suave: ver [derivación](code/EXISTENCE_ARGUMENT.md). Los datos de la comprobación están en [data/existence_check/](data/existence_check/summary.json).

## Sensibilidad al refinamiento

El barrido directo resolvió 1633 HJB nuevos; la verificación de candidatos añadió 65, para un total de **1698 resoluciones nuevas**. Los 22 tests del experimento, incluidos los métodos independientes, pasaron.

| p0 | rho | mayor cambio de x_bar, 8001 a 16001 | mayor cambio de frontera inicial, 8001 a 16001 |
| --- | --- | --- | --- |
| 0.7 | 0.9 | 6.14e-05 | 5.67e-05 |
| 0.8 | 0.9 | 0 | 2.27e-05 |
| 0.9 | 0.7 | 0 | 1.08e-05 |
| 0.9 | 0.8 | 0 | 1e-05 |

Estos cambios miden sensibilidad espacial; no son cotas matemáticas certificadas. Los archivos originales y sus figuras conservan la malla inicial de 4001 nodos; este informe registra la comprobación nueva y las aproximaciones de 16001 nodos por separado.
