# Resultados del modelo fluido base

Experimento determinista sin olvido; horizonte infinito en la HJB, dominio espacial truncado.

## Parámetros y precisión

- Parámetros: `{"p0": 0.5, "p1": 0.35, "p2": 0.8, "c1": 0.05, "c2": 0.65, "revenue": 1.0, "gamma": 0.98}`.
- Tasa de descuento continuo: δ = 0.02020270732 = −log(γ).
- Malla final: h = 0.125; interior mostrado S+F ≤ 240; frontera de cálculo N = 1115.62.
- Cota teórica del efecto del truncamiento en el valor de la malla: 9.98e-07.
- Máxima anchura observada del intervalo superior/inferior: 4.23e-08 (puede redondear a cero en doble precisión).
- Residuo algebraico de la HJB discretizada: 1.31e-13.
- V_h(0,0) = 27.88410556; ventaja del producto 2 en el origen = 1.00479762.

La cota anterior controla solamente el corte espacial del problema de malla. El residuo es una comprobación algebraica; ninguno certifica el error de discretización de la HJB continua. Las cotas se calculan en aritmética de doble precisión, sin aritmética de intervalos.

## Comparación de mallas

| h anterior → h nuevo | Máximo cambio de V | Fracción de acciones distintas |
|---|---:|---:|
| 0.5 → 0.25 | 0.904048 | 0.055% |
| 0.25 → 0.125 | 0.481869 | 0.046% |

Estas diferencias son evidencia numérica y no una demostración de convergencia. Cerca de la indiferencia, cambios pequeños del valor pueden cambiar el producto elegido.

## Trayectorias

Los beneficios de esta tabla son integrales hasta T; no incluyen el beneficio posterior. Todas las políticas se comparan con el mismo horizonte si ninguna alcanza antes la frontera del interior.

| Estado inicial | Política | Tiempo final | Beneficio hasta T | Tiempo con producto 2 |
|---|---|---:|---:|---:|
| 1: S0=0,F0=0 | optimal | 200 | 27.455626 | 41.55% |
| 1: S0=0,F0=0 | product1 | 200 | 9.849392 | 0.00% |
| 1: S0=0,F0=0 | product2 | 200 | 16.300279 | 100.00% |
| 2: S0=2,F0=8 | optimal | 200 | 6.753446 | 66.34% |
| 2: S0=2,F0=8 | product1 | 200 | 1.466794 | 0.00% |
| 2: S0=2,F0=8 | product2 | 200 | 4.675567 | 100.00% |
| 3: S0=8,F0=2 | optimal | 200 | 31.279600 | 37.51% |
| 3: S0=8,F0=2 | product1 | 200 | 22.566944 | 0.00% |
| 3: S0=8,F0=2 | product2 | 200 | 16.971815 | 100.00% |
| 4: S0=10,F0=10 | optimal | 200 | 25.129702 | 42.44% |
| 4: S0=10,F0=10 | product1 | 200 | 12.759929 | 0.00% |
| 4: S0=10,F0=10 | product2 | 200 | 15.396742 | 100.00% |

Al dividir dt por dos, el máximo cambio del beneficio acumulado es 0.00266727. Se conserva la trayectoria con el paso más fino.

Comprobación adicional: máximo |beneficio hasta T + e^(−δT)V_h(S_T,F_T) − V_h(S_0,F_0)| para la política óptima interpolada = 0.171702. Incluye errores de malla, interpolación y tiempo; no se presenta como error de truncamiento.

## Archivos

- `figures/policy_sf.png`: producto por estado (S,F) y trayectorias óptimas.
- `figures/policy_nx.png`: producto por media posterior e información acumulada.
- `figures/value_and_demand.png`: valor óptimo y demanda.
- `figures/trajectories.png`: estados, creencias, demanda y beneficios frente a productos fijos.
- `figures/switching_advantage.png`: ventaja local y frontera de cambio de producto.
- `data/solution.npz`: valores, intervalos, demanda y acciones del interior completo.
- `data/switching_regions.csv`: todas las componentes de producto 2 por diagonal, sin imponer un solo intervalo.
- `data/trajectories.csv`, `data/trajectory_summary.csv`, `data/trajectory_diagnostics.csv`: trayectorias y comprobaciones.
- `diagnostics.json`: configuración efectiva y diagnósticos de malla, frontera y tiempo.

La clasificación gris indica empate o incertidumbre por truncamiento. La ausencia de gris no elimina el error de malla. No se ha demostrado aquí la aproximación del modelo estocástico original por el fluido.
