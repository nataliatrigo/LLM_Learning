# Validaciones numéricas de umbrales

Resultados conservados para revisarlos más adelante. Las pruebas automáticas
que comprueban el funcionamiento del código están en
[Code/model_and_simulation_validation](../../Code/model_and_simulation_validation).

| Carpeta | Qué contiene |
| --- | --- |
| [threshold_certificates](threshold_certificates) | Certificados de umbrales para intervalos de rho; incluye rho de 0.5 a 0.8 |
| [local_threshold_intervals](local_threshold_intervals) | Certificación de umbrales en vecindades de los parámetros de referencia, con sus scripts en `forgetting/Code/numerical_validations/` |
| [discrete_gap_counterexample](discrete_gap_counterexample) | Contraejemplo certificado a la unimodalidad de la diferencia de continuación discreta, con sus scripts en `forgetting/Code/numerical_validations/` |

Los scripts de certificación por intervalos requieren la dependencia opcional
`python-flint`; sus comandos de reproducción figuran en los documentos de cada estudio.
