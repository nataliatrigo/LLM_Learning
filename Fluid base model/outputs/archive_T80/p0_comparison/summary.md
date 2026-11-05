# Comparación del modelo fluido para distintos p₀

Se resuelve de nuevo la HJB para p₀ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. Cada escenario conserva los demás parámetros económicos, las mallas, el dominio de interés y los estados iniciales.

La HJB tiene horizonte infinito y descuento δ = −log(γ). Las trayectorias se integran hasta el tiempo calendario T = 80; su beneficio acumulado hasta T no incluye la continuación posterior. El eje n = S+F mide información acumulada, no tiempo.

Los 4 estados iniciales comunes (S₀,F₀) son (0,0), (2,8), (8,2), (10,10).
Mallas h = [0.5, 0.25, 0.125]; interior S+F ≤ 100. La frontera de cálculo queda más lejos y se elige mediante una cota de truncamiento.

## Política y valor en el origen

| p₀ | Vₕ(0,0) | Producto en (0,0) | Ventaja producto 2 | Fracción de malla con producto 2 |
|---:|---:|---|---:|---:|
| 0.1 | 46.568980 | producto 1 | -0.142012 | 10.49% |
| 0.3 | 41.357801 | producto 2 | 0.423545 | 21.79% |
| 0.5 | 27.884106 | producto 2 | 1.0048 | 22.26% |
| 0.7 | 14.100959 | producto 2 | 2.11655 | 15.16% |
| 0.9 | 1.956014 | producto 2 | 0.827536 | 4.21% |

La fracción anterior cuenta nodos en la malla uniforme de (S,F). No es una probabilidad de visitar estados, ni una fracción de tiempo con producto 2; depende del dominio y de las coordenadas de la malla.

Comprobación de estructura: al aumentar p₀, el valor no debe aumentar (con márgenes no negativos). Se compararon todos los nodos entre escenarios consecutivos: 0 aumentos superiores a 10⁻¹⁰; máximo aumento positivo observado = 0. El detalle se guarda en `manifest.json`.

## Trayectoria óptima desde (0,0)

| p₀ | Tiempo final | n final | Media x final | Demanda final | Beneficio hasta T=80 | Tiempo con producto 2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 80 | 79.5874 | 0.353677 | 1 | 37.319283 | 0.00% |
| 0.3 | 80 | 75.1993 | 0.383687 | 0.938833 | 32.613928 | 7.02% |
| 0.5 | 80 | 73.6109 | 0.575926 | 0.908301 | 21.628470 | 51.45% |
| 0.7 | 80 | 62.148 | 0.763464 | 0.881265 | 10.435520 | 94.59% |
| 0.9 | 80 | 8.85994 | 0.728516 | 0.0732002 | 1.626049 | 94.17% |

Todas las trayectorias óptimas y de productos fijos alcanzan T = 80 sin salir del interior calculado.

## Precisión numérica

| p₀ | Cota truncamiento | Máximo cambio de V, últimas mallas | Acciones distintas, últimas mallas | Cambio de beneficio al dividir dt por 2 |
|---:|---:|---:|---:|---:|
| 0.1 | 9.98e-07 | 0.900403 | 0.107% | 9.67675e-06 |
| 0.3 | 9.98e-07 | 0.797033 | 0.387% | 0.000272307 |
| 0.5 | 9.98e-07 | 0.481869 | 0.071% | 0.0024207 |
| 0.7 | 9.98e-07 | 0.295685 | 0.252% | 0.00178284 |
| 0.9 | 9.98e-07 | 0.058055 | 0.056% | 0.000103526 |

La cota controla el corte espacial de la HJB discretizada. Las diferencias entre mallas y pasos de tiempo son comprobaciones de sensibilidad, no cotas del error continuo. El residuo algebraico de la HJB se guarda en `case_summary.csv`. La clasificación de indiferencia/incertidumbre no incorpora todo el error de malla. No se presupone que toda trayectoria llegue a una frontera entre productos ni se ha probado la aproximación al modelo estocástico.

## Figuras comparativas

- [policy_nx_comparison](figures/policy_nx_comparison.png)
- [policy_sf_comparison](figures/policy_sf_comparison.png)
- [initial_value_and_policy](figures/initial_value_and_policy.png)
- [trajectories_p0_comparison](figures/trajectories_p0_comparison.png)

## Detalle y datos

- [p0_0.1: resumen y figuras](p0_0.1/summary.md); [configuración](p0_0.1/effective_config.json); [registro](p0_0.1/run.log).
- [p0_0.3: resumen y figuras](p0_0.3/summary.md); [configuración](p0_0.3/effective_config.json); [registro](p0_0.3/run.log).
- [p0_0.5: resumen y figuras](p0_0.5/summary.md); [configuración](p0_0.5/effective_config.json); [registro](p0_0.5/run.log).
- [p0_0.7: resumen y figuras](p0_0.7/summary.md); [configuración](p0_0.7/effective_config.json); [registro](p0_0.7/run.log).
- [p0_0.9: resumen y figuras](p0_0.9/summary.md); [configuración](p0_0.9/effective_config.json); [registro](p0_0.9/run.log).
- [Tabla de escenarios](case_summary.csv).
- [Resumen de todas las trayectorias](trajectory_summary.csv).
- [Extremos de las trayectorias óptimas desde el origen](origin_path_endpoints.csv).
- [Configuración y archivos del barrido](manifest.json).

Los resultados base en `outputs/` se conservan; este barrido guarda sus escenarios y comparaciones en una carpeta separada.
