# Experimentos de forgetting

Los scripts para ejecutar experimentos, generar gráficos y resumir resultados
y sus archivos `config_*.json` están reunidos en esta carpeta.

Los comandos se ejecutan desde la raíz del proyecto:

```bash
uv run python -B -m forgetting --help
uv run python -B -m forgetting paths --help
uv run python -B -m forgetting paper --config forgetting/Code/experiments/config_policy_comparisons.json --skip-simulation
```

| Archivo | Uso |
| --- | --- |
| `run_experiments.py` | Experimentos del artículo y barridos de parámetros |
| `run_monte_carlo.py` | Comparaciones de Monte Carlo |
| `interaction_paths.py` | Trayectorias individuales por interacción |
| `interaction_grid.py` | Experimentos sobre una cuadrícula de parámetros |
| `occupation_diagnostic.py` | Frecuencias de visita de los estados |
| `states.py` | Gráficos de estados, historias iniciales y conjuntos alcanzables |
| `policy_summary.py` | Tablas de políticas, umbrales y estados observados |
| `config_paper.json` | Configuración de los experimentos del artículo |
| `config_policy_comparisons.json` | Comparaciones de políticas y ventanas de inversión |
| `config_interaction*.json`, `config_occupation.json` | Configuraciones de trayectorias y ocupación |

Los resultados se guardan en `forgetting/outputs/`. El código compartido del
modelo está en `forgetting/Code/model.py`, `forgetting/Code/layers.py` y
`forgetting/Code/simulation.py`. Las notas matemáticas están en `Progress notes/`.

Consulta el [README del proyecto](../../../README.md) para los comandos y las
opciones de cada experimento.
