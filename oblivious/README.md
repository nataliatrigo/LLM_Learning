# Experimentos oblivious

Los experimentos y sus comparaciones se agrupan en estas carpetas,
ordenadas con prefijos numéricos. Cada una conserva sus datos y figuras.

| Orden | Experimento | Carpeta y guía | Alias Python |
| --- | --- | --- | --- |
| 1 | OE distribution | [01_oe_distribution](01_oe_distribution/README.md) | `oblivious.oe_distribution` |
| 2 | OE p0 | [02_oe_p0](02_oe_p0/README.md) | `oblivious.oe_p0` |
| 3 | OE following monopoly | [03_oe_following_monopoly](03_oe_following_monopoly/README.md) | `oblivious.oe_following_monopoly` |
| 4 | OE conditional distribution | [04_oe_conditional_distribution](04_oe_conditional_distribution/README.md) | `oblivious.oe_conditional_distribution` |
| 5 | OE p0 temporal | [05_oe_p0_temporal](05_oe_p0_temporal/README.md) | `oblivious.oe_p0_temporal` |
| 6 | Equilibrium comparison | [06_equilibrium_comparison](06_equilibrium_comparison/code/README.md) | `oblivious.equilibrium_comparison` |
| 7 | Full comparison | [07_full_comparison](07_full_comparison/code/README.md) | `oblivious.full_comparison` |
| 8 | Nuevo OE distribution con opción externa fija | [08_oe_distribution_fixed_p0](08_oe_distribution_fixed_p0/README.md) | `oblivious.oe_distribution_fixed_p0` |
| 9 | Comparación de uno, dos y tres sellers con opción externa fija | [09_oe_three_sellers_fixed_p0](09_oe_three_sellers_fixed_p0/README.md) | `oblivious.oe_three_sellers_fixed_p0` |
| 10 | Fluid monopoly versus symmetric fluid oblivious competition | [10_fluid_oblivious](10_fluid_oblivious/README.md) | `oblivious.fluid_oblivious` |
| 11 | Equilibrios fluid oblivious y conjuntos de condiciones iniciales de condiciones iniciales | [11_fluid_oblivious_initial_conditions](11_fluid_oblivious_initial_conditions/README.md) | `oblivious.fluid_oblivious_initial_conditions` |
| 12 | Fluid monopoly and competition with long-run average profit | [12_fluid_average_profit](12_fluid_average_profit/README.md) | `oblivious.fluid_average_profit` |
| 13 | Competencia fluida asimétrica desde el mismo x0 | [13_fluid_asymmetric](13_fluid_asymmetric/README.md) | `oblivious.fluid_asymmetric` |

La comparación [Monopoly vs competition](07_full_comparison/Monopoly_vs_competition/code/README.md)
está dentro de `07_full_comparison/`. Las funciones numéricas compartidas están
en `02_oe_p0/code/`.

Para nuevos experimentos, el barrido por defecto es
`rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99`: se excluye `rho=0.999`.
Las corridas y figuras históricas que incluyen `0.999` se conservan intactas.

Desde la raíz del repositorio, consulta las opciones de cada experimento:

```bash
.venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment --help
.venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --help
.venv/bin/python -B -m oblivious.oe_following_monopoly.code.run_experiment --help
.venv/bin/python -B -m oblivious.oe_conditional_distribution.code.run_experiment --help
.venv/bin/python -B -m oblivious.oe_p0_temporal.code.run_experiment --help
.venv/bin/python -B -m oblivious.equilibrium_comparison.code.cross_rho --help
.venv/bin/python -B -m oblivious.full_comparison.code.plot_comparison --help
.venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.run_experiment --help
```

Las rutas físicas llevan números; los alias Python permiten usar comandos
`python -m` con nombres de módulo válidos. Las guías usan estos alias.
Los nombres de campañas, archivos e identificadores internos de datos se
mantienen. Los certificados históricos conservan las rutas y las huellas de
las fuentes originales para permitir su auditoría.

Tras el traslado, usa otro `--name` para iniciar una ejecución con el código
actualizado. `--resume` en campañas parciales sigue exigiendo las huellas
originales del código y de los resultados; el cambio de carpeta no elimina
esa comprobación. Los resultados guardados y sus certificados se conservan.
