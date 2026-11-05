# Resultados de forgetting

Esta es la carpeta única de resultados del estudio. Contiene los resultados
anteriores de `forgetting/outputs/` y los estudios trasladados desde `output/`.
Todos los scripts están agrupados en `forgetting/Code/`, en subcarpetas
con el nombre del estudio correspondiente.

| Carpeta | Contenido |
| --- | --- |
| [policy_and_state_experiments](policy_and_state_experiments) | Comparaciones de políticas, ventanas de inversión, trayectorias y Monte Carlo |
| [high_cost_policy_experiments](high_cost_policy_experiments) | Carpeta reservada para experimentos de costes altos; actualmente vacía |
| [fluid_thresholds](fluid_thresholds) | Análisis y resultados de umbrales en el modelo fluido |
| [fluid_convergence](fluid_convergence/summary.md) | Pérdida de la política fluida en el monopolio original y refinamiento separado de grillas |
| [threshold_parameter_map](threshold_parameter_map) | Mapas de políticas al variar diferencias de probabilidades y costes |
| [full_delta_map](full_delta_map) | Barridos ampliados de probabilidades y costes |
| [investment_connectedness](investment_connectedness) | Búsquedas numéricas sobre la conexión de la región de inversión |
| [threshold_visualizations](threshold_visualizations) | Visualizaciones de umbrales y conjuntos alcanzables en HTML y PNG |
| [numerical_validations](numerical_validations) | Certificados y verificaciones numéricas para revisión posterior |
| [advantage_by_p0_and_rho](advantage_by_p0_and_rho) | Ventaja de producto 2 para distintos p0 y rho |
| [fixed_policy_advantage_by_rho_p0_0.5](fixed_policy_advantage_by_rho_p0_0.5) | Ventaja bajo política fija al variar rho |
| [optimal_vs_fixed_advantage_p0_0.5](optimal_vs_fixed_advantage_p0_0.5) | Comparación de ventajas bajo políticas óptima y fija |
| [logistic_demand_comparison_p0_0.5](logistic_demand_comparison_p0_0.5) | Comparación de demanda Beta y logística |
| [beta_logistic_policy_geometry_p0_0.5](beta_logistic_policy_geometry_p0_0.5) | Geometría de las políticas con ambas demandas |

Los scripts generales están en [Code/experiments](../Code/experiments) y los de
comparaciones gráficas en [Code/visualizations](../Code/visualizations).
La [guía de organización](../README.md) muestra las ubicaciones anteriores.
