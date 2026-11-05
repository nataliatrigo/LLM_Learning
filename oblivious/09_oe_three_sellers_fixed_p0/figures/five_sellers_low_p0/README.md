# Cinco sellers: p0 bajos, OE marginal

Grilla publicada: 31, contrastada con 21. Rho=.6; p0=.1,.3.

[Producto 2 por número de sellers](01_product2_by_seller_count.png) · [Políticas](02_policy_comparison.png)

Producto 2 se mide entre ventas propias, incluyendo solo los pares que pasan el límite de 5 puntos porcentuales en los cinco modelos. Los datos de grilla menor sirven como validación y no se publican como versiones alternativas.

| p0 | Monopolio | 2 sellers | 3 sellers | 4 sellers | 5 sellers | Pasa contraste de producto 2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | 0.0000% | 15.9664% | 41.4601% | 51.0154% | 58.7076% | True |
| 0.3 | 18.9390% | 30.4423% | 44.0895% | 51.0154% | 58.7076% | True |

La distribución conjunta de los cuatro rivales se conserva. La reducción por permutaciones es exacta; no se sustituye por marginales independientes.
Las decisiones se toman tras conseguir la venta. Outside congela las reputaciones; solo se actualiza el seller seleccionado.

El valor calculado en el cociente es el promedio entre cinco sellers; en el estado inicial simétrico (.5,.5,.5,.5,.5) coincide exactamente con el valor individual. No representa el valor de una identidad concreta en estados asimétricos.

[Datos y certificado](../../data/five_sellers_low_p0/manifest.json) · [Auditoría](../../data/five_sellers_low_p0/independent_audit.json) · [Sensibilidad a la grilla](../../data/five_sellers_low_p0/grid_stability.csv)

Se certifica el juego numérico finito; el contraste no prueba convergencia al continuo ni unicidad.
