# Validación del modelo y de los experimentos de forgetting

Las pruebas de esta carpeta comprueban el solver de Bellman, la simulación, las
trayectorias individuales, la reproducción de estados iniciales, la geometría
del conjunto alcanzable, las ventanas de inversión, las configuraciones y los
comandos de experimentos.

Desde la raíz del proyecto:

```bash
uv run python -B -m unittest discover -s forgetting/Code/model_and_simulation_validation -v
```

Los certificados numéricos usados en la investigación de umbrales están en
[outputs/numerical_validations](../../outputs/numerical_validations).
