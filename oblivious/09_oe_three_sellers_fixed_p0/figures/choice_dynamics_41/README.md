# Elecciones del comprador y evolución del juego

Fuente: `high_p0_refinement_41`. No se vuelve a resolver ningún equilibrio.

- [01_stationary_buyer_choices](01_stationary_buyer_choices.png)
- [02_expected_evolution](02_expected_evolution.png)
- [03_sample_path_p0_0p5](03_sample_path_p0_0p5.png)
- [03_sample_path_p0_0p8](03_sample_path_p0_0p8.png)

## Cómo leer las figuras

- **01:** porcentajes estacionarios de todos los períodos, incluyendo outside.
  Cada barra corresponde a un mercado N=1,2,3 y suma 100%. A/B/C son identidades
  simétricas: en promedio estacionario cada una tiene la misma frecuencia.
  Se conservan 23/24 pares p0/rho con el filtro de outside (máximo 5 puntos
  porcentuales entre las dos grillas comunes más finas). Se omite p0=.7,rho=.9.
- **02:** expectativas exactas desde todas las reputaciones en .5, propagando
  la ley conjunta completa; no son promedios de una sola trayectoria. El primer
  panel mide la probabilidad de elegir a **cada seller**, por período; las
  referencias discontinuas son estacionarias y no presuponen que se alcancen
  dentro del horizonte. El tercer panel cuenta ventas esperadas por seller.
- **03:** una realización aleatoria de 600 períodos por modelo.
  Izquierda: reputaciones. Centro: a quién eligen y producto entregado
  (círculo=1, triángulo=2). Derecha: ventas acumuladas por identidad.
  Las diferencias entre A/B/C dentro de una realización son aleatorias.

Las políticas OE se mantienen fijas. El comprador elige según las demandas
del estado conjunto real, después el seller elegido sortea el producto según
su política y un éxito/fracaso según su producto. Solo cambia su reputación;
outside congela todo. La actualización sortea los dos vecinos de la grilla
con los mismos pesos de interpolación del kernel certificado, sin redondeo
determinista ni aproximación por reputaciones independientes.

Los p0=.5 y .8, rho=.6 pasan los cuatro filtros estacionarios existentes.
Sus transitorios se muestran en las grillas certificadas n=31 y n=41,
respectivamente; no se ha certificado estabilidad entre grillas del transitorio
ni convergencia al continuo. No hay adaptación o aprendizaje de las políticas.

Datos reproducibles, semillas, cuentas y controles: `../../data/choice_dynamics_41`.
Las figuras y checkpoints anteriores permanecen intactos.

```bash
MPLCONFIGDIR=/tmp/choice-dynamics-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.choice_dynamics --name choice_dynamics_41 --periods 600 --seed 20261025
```

Usar otro `--name` para una ejecución adicional, preservando esta publicación.
