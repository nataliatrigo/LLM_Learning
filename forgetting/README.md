# Organización del estudio de forgetting

El estudio tiene dos carpetas principales: `Code` y `outputs`. Las notas matemáticas están en `../Progress notes/`.

```text
forgetting/
├── Code/
│   ├── model.py, layers.py, simulation.py  Modelo y simulación compartidos
│   ├── cli.py                            Entrada de comandos
│   ├── experiments/                      Experimentos y configuraciones JSON
│   ├── visualizations/                   Comparaciones y gráficos
│   ├── model_and_simulation_validation/   Pruebas automáticas
│   ├── fluid_thresholds/                 Scripts del modelo fluido
│   ├── fluid_convergence/                Política fluida en el estocástico y refinamiento numérico
│   ├── threshold_parameter_map/          Scripts de mapas de umbrales
│   ├── full_delta_map/                   Barridos de probabilidades y costes
│   ├── investment_connectedness/         Búsquedas de regiones desconectadas
│   └── numerical_validations/            Scripts de certificación numérica
└── outputs/                              Datos, figuras y resultados por estudio
```

Todo el código de cálculo está dentro de `Code/`, organizado por función.
Los únicos archivos Python fuera de esa carpeta son `__init__.py` y el pequeño
adaptador `__main__.py`, que conservan el comando `python -m forgetting`.

## Ejecutar experimentos

Desde la raíz del proyecto:

```bash
uv run python -B -m forgetting --help
uv run python -B -m forgetting paths --help
uv run python -B -m forgetting paper --config forgetting/Code/experiments/config_policy_comparisons.json --skip-simulation
```

Las configuraciones están en `Code/experiments/`: `config_paper.json` para los
experimentos del artículo y `config_policy_comparisons.json` para las
comparaciones de políticas y ventanas de inversión. Las demás configuraciones
indican su experimento en el nombre.

## Comprobar el código

```bash
uv run python -B -m unittest discover -s forgetting/Code/model_and_simulation_validation -v
uv run python -B -m unittest discover -s forgetting/Code/visualizations -p 'test_*.py' -v
```

## Resultados y procedencia

Los resultados se guardan en `outputs/`, separados por estudio. La antigua
carpeta `test` se llama `policy_and_state_experiments`; `test_costes_altos` se
llama `high_cost_policy_experiments`. Los certificados de umbrales están en
`outputs/numerical_validations/` y sus scripts en `Code/numerical_validations/`.
Los estudios antes guardados en `output/research/` están bajo `outputs/`;
`discrete_unimodality` y `local_interval` se encuentran en
`outputs/numerical_validations/discrete_gap_counterexample/` y
`outputs/numerical_validations/local_threshold_intervals/`, respectivamente.
Los HTML, PNG y datos de la antigua `output/visualizations/` están en
`outputs/threshold_visualizations/`; sus scripts están en `Code/visualizations/`.

Los cuatro PDFs de la antigua carpeta `output/pdf/` se enviaron a la papelera.
Las notas y PDFs de `../Progress notes/` se conservan en sus carpetas temáticas.
Los manifiestos de trayectorias conservan las rutas históricas de procedencia:
son registros del experimento original y forman parte de las claves de caché.

Consulta [la guía de experimentos](Code/experiments/README.md),
[el índice de resultados](outputs/README.md) y [la documentación del proyecto](../README.md).

El [experimento de comparación fluido–estocástico](Code/fluid_convergence/README.md)
evalúa la política fluida en el monopolio con forgetting original y refina por
separado las grillas fluida y estocástica. Sus
[resultados](outputs/fluid_convergence/summary.md) estudian la estabilidad
numérica de la pérdida de usar esa política.
