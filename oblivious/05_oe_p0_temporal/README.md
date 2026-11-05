# OE p0 temporal

[Índice de experimentos oblivious](../README.md).

Este experimento mantiene **p₀ fijo durante la optimización** y exige que sea
consistente con el promedio temporal **esperado** del sorteo TS del rival:
100.000 períodos de calendario después de 10.000 de descarte, partiendo de
reputaciones `(0.5, 0.5)`. Es una ejecución separada del
[OE p0 original](../02_oe_p0/README.md), cuyo cierre usa una distribución
estacionaria.

La corrida guardada usa los ocho valores `rho = 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999`,
51 nodos por vendedor y los mismos parámetros:
`R=1, c1=0.05, c2=0.65, p1=0.35, p2=0.80, gamma=0.98`.
Los nuevos barridos por defecto excluyen `rho=0.999`.
Una grilla de 51 nodos implica 2.601 pares de reputaciones.

## Qué cambia

La política `alpha(x)` sigue siendo la probabilidad de elegir el producto 2
cuando se selecciona al vendedor. El producto 1 tiene probabilidad
`1-alpha(x)`. Para optimizar, el vendedor utiliza la demanda fija
`P(theta(x) > p₀)`, calculada con la cola Beta. Las mezclas interiores requieren
indiferencia entre los valores de ambas acciones.

Ambos vendedores usan esa política en el juego real: en cada período el cliente
compara sus dos sorteos TS y solo el vendedor elegido actualiza su reputación.
Si `P_alpha` es la matriz de transición conjunta y `nu_0` concentra la masa en
`(0.5,0.5)`, la nueva condición es:

```text
nu_bar = (1/T) sum_{t=B}^{B+T-1} nu_0 P_alpha^t
m_rho(x) = (1-rho+x)/(3-2*rho)
F_T(p₀, alpha) = sum_{x_A,x_B} nu_bar(x_A,x_B) m_rho(x_B)
p₀ = F_T(p₀, alpha)

B = 10.000, T = 100.000
```

Los estados se observan **antes de la selección**, desde `t=10.000` hasta
`t=109.999`, con `X_0=(0.5,0.5)`. Se promedia al rival en todos los períodos,
también cuando no vende. La media condicional del sorteo TS es `m_rho(x)`;
promediarla bajo la ley temporal calcula la esperanza del promedio de los
sorteos. No es el promedio de una única trayectoria ni una media de calidad.

La ley se propaga de forma determinista en la cadena discretizada. Una eventual
aceleración de la cola incluye una cota de truncamiento; esa cota no certifica
todo el error de redondeo. La ley estacionaria de la nueva política se conserva
como diagnóstico, sin sustituir la ventana temporal en la condición de cierre.

## Ejecutar

Desde la raíz de `LLM_Learning`:

```bash
.venv/bin/python -B -m oblivious.oe_p0_temporal.code.run_experiment --name temporal_n51
.venv/bin/python -B -m oblivious.oe_p0_temporal.code.run_experiment --help

# Reanudar usando la configuración guardada.
.venv/bin/python -B -m oblivious.oe_p0_temporal.code.run_experiment --name temporal_n51 --resume

# Regenerar tablas y figuras de una corrida completa.
.venv/bin/python -B -m oblivious.oe_p0_temporal.code.run_experiment --name temporal_n51 --plots-only
```

Los resultados y las figuras se guardan en `oblivious/05_oe_p0_temporal/data/<nombre>/` y
`oblivious/05_oe_p0_temporal/figures/<nombre>/`. La referencia se lee de
`oblivious/02_oe_p0/data/rho_forgetting_mixed`, usando su candidato seleccionado en
la misma grilla. El experimento original se conserva. El comparador es ese
OE p0 original, no el modelo de calidad de `oblivious/03_oe_following_monopoly`.

## Comparaciones guardadas

- `summary.csv`: p₀ nuevo y original, su diferencia, residuos de cierre,
  diferencias entre políticas y diagnósticos de la ley temporal.
- `candidates.csv`: todos los candidatos encontrados, probabilidades de mezcla
  y comprobaciones numéricas.
- `horizon_sensitivity.csv`: la política nueva evaluada con el doble de períodos
  o el doble de descarte; estas filas no son equilibrios resueltos de nuevo.
- `economic_outcomes.csv`: uso de producto 2, beneficios y éxito esperado del
  cliente bajo ambas políticas, evaluadas en la misma ventana temporal.
- `RESULTADOS.txt`: resultados e interpretación en español.
- `01_p0_comparison`: p₀ temporal y estacionario original, para todos los rho.
- `02_policy_comparison`: ambas políticas; la original aparece discontinua y
  los rombos marcan nodos mixtos. Las líneas solo unen los valores guardados.
- `03_reputation_cdf_comparison`: distribuciones del rival bajo ambas políticas,
  evaluadas en **la misma ventana temporal**; las curvas estacionarias aparecen
  punteadas como diagnóstico adicional.
- `04_consistency_map_comparison`: cierre temporal nuevo y cierre estacionario
  original; incluye los candidatos seleccionados y los puntos de búsqueda.
- `05_joint_law_comparison`: leyes conjuntas de ambos modelos, tanto temporales
  como estacionarias, con la misma escala logarítmica.
- `06_advantage_comparison`: ventaja de valor del producto 2 respecto al 1 para
  los dos benchmarks, con una escala vertical común.
- `07_economic_outcome_comparison`: proporción de compras del producto 2,
  beneficio por vendedor y período de calendario, y tasa de éxito del cliente.

Las figuras se exportan en PNG y PDF. Azul identifica el nuevo cierre temporal;
naranja identifica el OE p0 original. Evaluar la política original en la nueva
ventana permite distinguir el cambio de definición del cambio de política.
Los resultados económicos son flujos esperados por período, sin descuento;
la demanda se calcula en el juego real de dos sorteos TS. El beneficio por
vendedor incluye los períodos en los que no es seleccionado.

## Cómo interpretar el resultado

El promedio esperado de una ventana finita depende del horizonte, del descarte
y de la condición inicial. El descarte no garantiza estacionariedad. Que dos
medias escalares se parezcan tampoco demuestra que sus distribuciones coincidan.
Bajo condiciones ergódicas apropiadas, el promedio temporal de largo plazo
puede coincidir con la media estacionaria; esa afirmación no hace idénticas las
dos definiciones para esta ventana finita.

El cálculo no requiere simulación Monte Carlo y no reporta intervalos de
confianza de trayectorias. Su precisión depende de la grilla, la cuadratura y las
tolerancias de optimización y propagación. La cota de truncamiento de la cola
controla solo esa aproximación. La validación de una solución en la grilla no
demuestra convergencia al modelo continuo.

Cada candidato seleccionado se revalida propagando la cadena conjunta completa
durante todos los períodos, sin reducción por simetría ni truncamiento de cola.
Se guardan la diferencia entre leyes y el residual de consistencia recalculado.

Se permiten estrategias mixtas en nodos de indiferencia y se guardan todos los
candidatos que la búsqueda encuentra y verifica. La búsqueda no demuestra que
sean los únicos. Los candidatos guardados del OE p0 original se usan también
como propuestas y se vuelven a verificar bajo el cierre temporal; conservar
uno exige que satisfaga esa nueva condición. Cuando existen varios, comparar los seleccionados no demuestra
que pertenezcan a una misma rama. La consistencia de un escalar tampoco implica
equilibrio perfecto de Markov en el juego de información completa.
