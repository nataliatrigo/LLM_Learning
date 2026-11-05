# Resultados: equilibrio desde una reputación inicial común

**Comprobación posterior:** la [búsqueda independiente](EXISTENCE_CHECK.md), con mallas de hasta 16001 nodos y 1698 HJB nuevos, no encontró equilibrios adicionales que cubran los intervalos indicados. El informe conserva por separado las fronteras refinadas y el alcance numérico de la conclusión.

Se detectaron **27 ternas verificadas** en **24 combinaciones** de parámetros.

Cada fila identifica una política óptima calculada contra el rival fijo 
`x_bar`, su reputación estacionaria y el conjunto completo de inicios comunes `x0`.

| p0 | rho | política | x_bar | sigma(x_bar) | condiciones iniciales x0 |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.6 | 0 | 0.352250968 | 0.005002 | [0, 1] |
| 0.1 | 0.7 | 0 | 0.439294596 | 0.198432 | [0, 1] |
| 0.1 | 0.8 | 0 | 0.545401392 | 0.434225 | [0, 1] |
| 0.1 | 0.9 | 0 | 0.674148954 | 0.720331 | [0, 1] |
| 0.3 | 0.6 | 0 | 0.440564561 | 0.201255 | [0, 1] |
| 0.3 | 0.7 | 0 | 0.494566947 | 0.321260 | [0, 1] |
| 0.3 | 0.8 | 0 | 0.562468880 | 0.472153 | [0, 1] |
| 0.3 | 0.9 | 0 | 0.674218981 | 0.720487 | [0, 1] |
| 0.5 | 0.6 | 0 | 0.560325868 | 0.467391 | [0, 1] |
| 0.5 | 0.7 | 0 | 0.601750464 | 0.559445 | [0, 1] |
| 0.5 | 0.8 | 0 | 0.644722452 | 0.654939 | [0, 1] |
| 0.5 | 0.9 | 0 | 0.696243811 | 0.769431 | [0, 1] |
| 0.7 | 0.6 | 0 | 0.684454272 | 0.743232 | [0, 1] |
| 0.7 | 0.7 | 0 | 0.720042513 | 0.822317 | [0, 1] |
| 0.7 | 0.8 | 0 | 0.755492683 | 0.901095 | [0, 1] |
| 0.7 | 0.9 | 0 | 0.787226768 | 0.971615 | (0.420713458, 1] |
| 0.8 | 0.6 | 0 | 0.746980298 | 0.882178 | [0, 1] |
| 0.8 | 0.7 | 0 | 0.779997746 | 0.955551 | [0, 1] |
| 0.8 | 0.8 | 0 | 0.800000000 | 1.000000 | [0, 1] |
| 0.8 | 0.9 | 0 | 0.350000000 | 0.000000 | [0, 0.530115762] |
| 0.8 | 0.9 | 1 | 0.800000000 | 1.000000 | (0.572249543, 1] |
| 0.9 | 0.6 | 0 | 0.800000000 | 1.000000 | [0, 1] |
| 0.9 | 0.7 | 0 | 0.350000000 | 0.000000 | [0, 0.464244581] |
| 0.9 | 0.7 | 1 | 0.800000000 | 1.000000 | (0.485527169, 1] |
| 0.9 | 0.8 | 0 | 0.350000000 | 0.000000 | [0, 0.64579659] |
| 0.9 | 0.8 | 1 | 0.800000000 | 1.000000 | (0.664455417, 1] |
| 0.9 | 0.9 | 0 | 0.350000000 | 0.000000 | [0, 1] |

## Condiciones iniciales sin equilibrio detectado

Estos conjuntos se refieren a las políticas numéricamente detectadas y al selector declarado.

| p0 | rho | x0 sin equilibrio detectado |
| --- | --- | --- |
| 0.7 | 0.9 | [0, 0.420713458] |
| 0.8 | 0.9 | (0.530115762, 0.572249543] |
| 0.9 | 0.7 | (0.464244581, 0.485527169] |
| 0.9 | 0.8 | (0.64579659, 0.664455417] |

Máximo de equilibrios para un mismo x0: **1**.

## Verificación e interpretación

- 162 condiciones iniciales contrastadas; 162 coinciden con las predicciones guardadas del experimento 10.
- 68 integraciones nuevas con demanda efectiva D(x,x), con cambio máximo 4.28e-11 al refinar el integrador.
- Todas las políticas mantienen x_bar; se comprobaron las mallas 1001, 2001 y 4001 y la sensibilidad de las fronteras de los conjuntos de condiciones iniciales en la pareja más fina.
- En umbrales repulsivos se selecciona producto 1 y se toma su continuación inmediata hacia la izquierda (convención de flujo por eventos). La ecuación diferencial discontinua interpretada sólo casi en todas partes no fija por sí sola esa continuación. Los puntos de holding verificados usan la mezcla que mantiene su reputación. Los extremos abiertos/cerrados corresponden a esta convención; los interiores de los conjuntos de condiciones iniciales no dependen de ella.
- Los conjuntos de condiciones iniciales son intervalos de la política numérica: sus umbrales tienen error de discretización. Las búsquedas no son una prueba de exhaustividad ni se enumeran todos los selectores óptimos en indiferencias.
- La convergencia usa la línea de fase, no la cercanía de x(T) a x_bar en un horizonte finito.

Ver [políticas y conjuntos de condiciones iniciales completas](data/equilibria.json), [tabla de equilibrios](data/equilibria.csv), [correspondencia por condición inicial](data/initial_state_correspondence.csv) y [figuras](figures/README.md).
