# Preferencias de publicación para el experimento de tres sellers

- Mantener en `figures/` una sola versión vigente de cada comparación: la
  campaña completa y auditada más reciente, con la grilla común certificada
  más fina disponible para cada par `(p0,rho)` y los tres tamaños de mercado.
- Aplicar los filtros de estabilidad por métrica. Omitir los casos que fallan;
  una grilla más grande por sí sola no demuestra que el resultado sea estable.
- Retirar de `figures/` las publicaciones de grillas anteriores cuando sean
  reemplazadas. Conservar su trazabilidad en un respaldo comprimido y registrar
  las rutas y hashes en `archive/`, sin duplicar sus plots en los índices actuales.
- Conservar los checkpoints de grillas menores como controles numéricos y
  validaciones. No publicarlos nuevamente como comparaciones principales.
- Las figuras actuales están en `high_p0_refinement_41` y `choice_dynamics_41`.
  Usan 41 puntos para `p0=.7,.8,.9` y 31 para `p0=.1,.3,.5`; esa selección se
  actualizará al disponer de un nuevo refinamiento completo y auditado.
- La extensión vigente N=1..5 está en `five_sellers_low_p0`, con `p0=.1,.3`,
  `rho=.6` y 31 puntos contrastados con 21. Reemplaza los plots N=1..4 del piloto
  `four_sellers_low_p0`, cuyos datos y fuentes permanecen congelados. Es un piloto
  separado del barrido completo N=1,2,3. Conservar la conjunta de los cuatro
  rivales y la reducción exacta por permutaciones; no sustituirla por marginales
  independientes. GMRES normalizado y el arranque positivo para estados raros
  mantienen los controles de invariancia e independencia y sus tolerancias.
