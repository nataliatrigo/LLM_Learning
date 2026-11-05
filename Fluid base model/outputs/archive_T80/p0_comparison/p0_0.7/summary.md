# Resultados del modelo fluido base

Experimento determinista sin olvido; horizonte infinito en la HJB, dominio espacial truncado.

## Parámetros y precisión

- Parámetros: `{"p0": 0.7, "p1": 0.35, "p2": 0.8, "c1": 0.05, "c2": 0.65, "revenue": 1.0, "gamma": 0.98}`.
- Tasa de descuento continuo: δ = 0.02020270732 = −log(γ).
- Malla final: h = 0.125; interior mostrado S+F ≤ 100; frontera de cálculo N = 975.625.
- Cota teórica del efecto del truncamiento en el valor de la malla: 9.98e-07.
- Máxima anchura observada del intervalo superior/inferior: 0 (puede redondear a cero en doble precisión).
- Residuo algebraico de la HJB discretizada: 1.17e-13.
- V_h(0,0) = 14.10095873; ventaja del producto 2 en el origen = 2.11654984.

La cota anterior controla solamente el corte espacial del problema de malla. El residuo es una comprobación algebraica; ninguno certifica el error de discretización de la HJB continua. Las cotas se calculan en aritmética de doble precisión, sin aritmética de intervalos.

## Comparación de mallas

| h anterior → h nuevo | Máximo cambio de V | Fracción de acciones distintas |
|---|---:|---:|
| 0.5 → 0.25 | 0.537534 | 0.414% |
| 0.25 → 0.125 | 0.295685 | 0.252% |

Estas diferencias son evidencia numérica y no una demostración de convergencia. Cerca de la indiferencia, cambios pequeños del valor pueden cambiar el producto elegido.

## Trayectorias

Los beneficios de esta tabla son integrales hasta T; no incluyen el beneficio posterior. Todas las políticas se comparan con el mismo horizonte si ninguna alcanza antes la frontera del interior.

| Estado inicial | Política | Tiempo final | Beneficio hasta T | Tiempo con producto 2 |
|---|---|---:|---:|---:|
| 1: S0=0,F0=0 | optimal | 80 | 10.435520 | 94.59% |
| 1: S0=0,F0=0 | product1 | 80 | 3.712876 | 0.00% |
| 1: S0=0,F0=0 | product2 | 80 | 9.926684 | 100.00% |
| 2: S0=2,F0=8 | optimal | 80 | 0.021679 | 0.00% |
| 2: S0=2,F0=8 | product1 | 80 | 0.021679 | 0.00% |
| 2: S0=2,F0=8 | product2 | 80 | 0.008127 | 100.00% |
| 3: S0=8,F0=2 | optimal | 80 | 12.947564 | 89.67% |
| 3: S0=8,F0=2 | product1 | 80 | 8.156283 | 0.00% |
| 3: S0=8,F0=2 | product2 | 80 | 11.860516 | 100.00% |
| 4: S0=10,F0=10 | optimal | 80 | 0.839324 | 0.00% |
| 4: S0=10,F0=10 | product1 | 80 | 0.839324 | 0.00% |
| 4: S0=10,F0=10 | product2 | 80 | 0.438052 | 100.00% |

Al dividir dt por dos, el máximo cambio del beneficio acumulado es 0.00178284. Se conserva la trayectoria con el paso más fino.

Comprobación adicional: máximo |beneficio hasta T + e^(−δT)V_h(S_T,F_T) − V_h(S_0,F_0)| para la política óptima interpolada = 0.26853. Incluye errores de malla, interpolación y tiempo; no se presenta como error de truncamiento.

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
