# Scripts de visualización y comparación

Los scripts están reunidos en esta carpeta. Los gráficos, tablas y datos se guardan en las
carpetas temáticas de `forgetting/outputs/`.

| Script | Uso |
| --- | --- |
| `build_threshold_data.py` | Datos para visualizar umbrales y conjuntos alcanzables |
| `plot_thresholds_static.py` | Figuras estáticas a partir de esos datos |
| `plot_advantage_by_rho.py` | Ventaja de producto 2 al variar rho |
| `plot_advantage_by_p0_rho.py` | Comparación conjunta de p0 y rho |
| `plot_fixed_policy_advantage_by_rho.py` | Ventaja de una política fija |
| `compare_optimal_and_fixed_advantage.py` | Comparación de la política óptima y la fija |
| `compare_beta_logistic_demand.py` | Comparación de demanda Beta y logística |
| `compare_policy_geometry.py` | Barrido de la geometría de políticas |
| `plot_policy_geometry_comparison.py` | Gráficos del barrido de geometría |
| `test_beta_logistic_demand.py` | Pruebas automáticas de la comparación de demandas |

Los dos HTML, sus PNG y `threshold-data.json` están en
[outputs/threshold_visualizations](../../outputs/threshold_visualizations).
Los comandos se ejecutan desde la raíz del proyecto. Por ejemplo:

```bash
uv run python -B forgetting/Code/visualizations/compare_beta_logistic_demand.py --help
```

Los scripts de dibujo que leen resultados guardados requieren que esos
resultados existan; la reorganización no vuelve a ejecutar los barridos.
