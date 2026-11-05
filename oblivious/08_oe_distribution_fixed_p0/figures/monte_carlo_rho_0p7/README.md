# Monte Carlo: rho=0.7

Comparación de monopolio, OE marginal y OE condicional con c2=0.65 y los seis p0 del experimento 08.
20,000 trayectorias por caso y 3,000 períodos; cada seller comienza en la misma reputación madura x=0.5.
Se simulan las políticas auditadas de 101 nodos sin resolver nuevos equilibrios. La elección del customer usa las reputaciones reales de ambos sellers y la opción externa.
Solo se actualiza el seller que vende; comprar a la opción externa deja ambos estados iguales.
La lotería entre nodos vecinos reproduce exactamente la transición del modelo en grilla.

**Inicio:** no son las políticas de aprendizaje desde cero observaciones del antiguo Monte Carlo del monopolio. Se usa el mismo estado maduro inicial en los tres modelos para comparar los equilibrios guardados.

- [Formato del monopolio: demanda propia y producto 2 propio por período](01_monopoly_style_demand_and_product2.png).
- [Métrica principal: producto 2 entre ventas propias](02_primary_product2_given_sale.png).
- [Complementaria: producto 2 recibido por el customer por período](03_complementary_customer_product2.png).
- [Los tres gráficos en PDF](monte_carlo_rho_0p7_comparison.pdf).

Los gráficos usan bloques de 25 períodos. Las bandas de las dos métricas muestran intervalos nominales del 95% por trayectoria; las líneas punteadas son las referencias estacionarias guardadas.
El cociente condicional de cada bloque divide el total de ventas de producto 2 por el total de ventas propias.
El flujo del customer cuenta efectivamente las entregas de A más las de B; no duplica por aproximación la realización de A.
Se contrastan fechas seleccionadas de los primeros 100 períodos con propagación determinista exacta del mismo kernel.
Los datos por período, por bloque y por trayectoria, las referencias exactas y los chequeos están en ../../data/monte_carlo_rho_0p7/.
Las curvas son transitorias: no se declara estacionariedad por observar una meseta o por simular 3,000 períodos.
