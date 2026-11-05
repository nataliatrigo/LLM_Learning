# Condiciones necesarias y comprobación independiente de candidatos

Este argumento corresponde a las políticas de feedback óptimas en todos los
estados que calcula el HJB del experimento. La reputación del rival permanece
fija en `z` durante la optimización. La condición posterior de convergencia
utiliza la dinámica efectiva simétrica de ambos vendedores.

## Una condición necesaria que no requiere que el valor óptimo sea suave

Escribamos

\[
 b=\frac{c_2-c_1}{p_2-p_1},\qquad
 r(x)=R-c_1-b(x-p_1),\qquad
 k=\frac b\lambda,\qquad
 H_z(x)=\frac{D(x,z;p_0)r(x)}\delta.
\]

Para `x` entre `p1` y `p2`, `H_z(x)` es el valor descontado de mantener
constantemente la reputación `x`, usando la mezcla
`alpha(x)=(x-p1)/(p2-p1)`. Si una política óptima mantiene `z`, necesariamente
su valor desde `z` es `H_z(z)`.

Considérese la estrategia factible de usar producto 2 desde `z` hasta un
objetivo `y>z`, y mantener `y` después. Para `y<z`, utilícese producto 1 hasta
`y`, seguido de holding. Para el producto `i` elegido, definamos

\[
 T_i(y;z)=\int_z^y\frac{du}{\lambda D(u,z;p_0)(p_i-u)},
\]

\[
 J_i(y;z)=\int_z^y e^{-\delta T_i(u;z)}
 \frac{R-c_i}{\lambda(p_i-u)}\,du
 + e^{-\delta T_i(y;z)}H_z(y).
\]

Los signos de la orientación de la integral y de `p_i-u` coinciden, de modo
que el tiempo de viaje y el beneficio acumulado son positivos en ambos casos.
Los objetivos utilizados son interiores; por tanto el viaje tiene tiempo
finito porque la demanda es estrictamente positiva.

Como `R-ci-r(y)=-b(pi-y)`, la derivada del valor de esta estrategia es

\[
 \frac{\partial J_i(y;z)}{\partial y}
 =e^{-\delta T_i(y;z)}\big[H_z'(y)-k\big].
\]

En un equilibrio interior `z`, la derivada de mejora para una excursión de
tamaño positivo hacia arriba es `H_z'(z)-k`, y para una excursión hacia abajo
es `k-H_z'(z)`. La optimalidad exige que ambas sean no positivas. En consecuencia,

\[
 H_z'(z)=k,
\]

equivalentemente,

\[
 F(z):=D_x(z,z;p_0)r(z)
 -\frac{c_2-c_1}{\lambda(p_2-p_1)}
   \big[\delta+\lambda D(z,z;p_0)\big]=0.
\]

La derivada `D_x` mantiene fija la reputación del rival. Este argumento compara
estrategias explícitas con el holding: **no necesita derivar el valor óptimo
`V_z`**, ni supone que éste sea suave en un punto de cambio de política.
Por ello la falta de suavidad del HJB no genera por sí sola una clase adicional
de reputaciones estacionarias interiores que eluda esta condición necesaria.

En los extremos sólo hay una dirección factible de excursión seguida de holding:

\[
 F(p_1)\leq0,\qquad F(p_2)\geq0.
\]

Son condiciones necesarias, no suficientes. Fuera de `[p1,p2]` no puede
cumplirse la primera condición de consistencia porque
`p1+(p2-p1)*sigma(z)` pertenece a ese intervalo.

## Una segunda condición necesaria y testigos de rechazo

En un cero interior de `F`, se tiene

\[
 \frac{\partial^2J_i(y;z)}{\partial y^2}\bigg|_{y=z}=H_z''(z).
\]

Por tanto `H_z''(z)>0` produce una mejora local mediante una excursión y
descarta el holding óptimo. Su expresión es

\[
 H_z''(z)=\frac{r(z)D_{xx}(z,z;p_0)-2bD_x(z,z;p_0)}\delta,
\]

con el rival fijo en ambas derivadas. Ésta no es la derivada total de `F`
cuando ambos argumentos de la demanda varían con `z`.

Para el posterior Beta de concentración fija, poniendo `s=1/(1-rho)` y
`a=1+s*x`, `b_beta=1+s*(1-x)`, el score de la densidad es

\[
 L_x(t)=s\,[\log t-\log(1-t)-\psi(a)+\psi(b_{\rm beta})],
\]

\[
 \partial_xL_x(t)=-s^2[\psi_1(a)+\psi_1(b_{\rm beta})].
\]

Así, `D_x` y `D_xx` se calculan integrando respectivamente
`f_x(t)*L_x(t)*CDF_z(t)` y
`f_x(t)*(L_x(t)^2 + partial_x L_x(t))*CDF_z(t)` sobre `[p0,1]`.

Una evaluación independiente mediante Gauss–Legendre fijo encontró los
siguientes ceros y curvaturas (valores numéricos redondeados):

| p0 | rho | cero interior z | H_z''(z) | consecuencia del test local |
| --- | --- | --- | --- | --- |
| .7 | .9 | .404435377893 | +127.238269 | holding no óptimo |
| .7 | .9 | .786887316024 | -242.166836 | el test local no lo rechaza |
| .8 | .9 | .567964443095 | +116.885597 | holding no óptimo |
| .9 | .7 | .528646969917 | +18.169850 | holding no óptimo |
| .9 | .8 | .649937955386 | +34.485929 | holding no óptimo |

Además se evaluaron estrategias factibles de excursión de tamaño finito. Estas
proporcionan testigos de rechazo independientes de la discretización del HJB:

| p0 | rho | z | producto durante el viaje | destino y | H_z(z) | J_i(y;z) | mejora |
| --- | --- | --- | --- | --- | --- | --- | --- |
| .7 | .9 | .35 | 2 | .75 | .470987539590 | .592574240934 | .121586701344 |
| .7 | .9 | .8 | 1 | .79 | 7.814984594233 | 7.829247635608 | .014263041375 |
| .7 | .9 | .404435377893 | 2 | .504435377893 | .989210552916 | 1.276846876630 | .287636323714 |
| .8 | .9 | .567964443095 | 2 | .667964443095 | 1.046701479066 | 1.237161083888 | .190459604822 |
| .9 | .7 | .528646969917 | 2 | .628646969917 | .564827258408 | .609991777206 | .045164518799 |
| .9 | .8 | .649937955386 | 1 | .549937955386 | .733416232691 | .784722741150 | .051306508459 |

Los valores se obtuvieron con cuadratura Gauss–Legendre fija de orden 384 para
los interiores y 512 para los extremos y curvaturas; la identidad diagonal
de demanda y el método alternativo del experimento permiten contrastarlos.
La integración del viaje utilizó el estado como variable independiente:

\[
 \frac{d\ell}{dx}=-\frac\delta{\lambda D(x,z)(p_i-x)},\qquad
 \frac{dA}{dx}=e^\ell\frac{R-c_i}{\lambda(p_i-x)},
 \qquad (\ell,A)(z)=(0,0).
\]

Al terminar, `J=A(y)+exp(ell(y))*H_z(y)`. Las tolerancias fueron `rtol=2e-10`,
`atol=1e-12` en los interiores y `rtol=2e-11`, `atol=1e-13` en los extremos.

## Qué falta para una prueba exhaustiva de inexistencia

La condición necesaria reduce cualquier reputación estacionaria a los
extremos o a los ceros de `F`. La demanda y `F` son analíticas en un entorno del
intervalo activo: las formas Beta permanecen positivas, y las derivadas de
densidad están dominadas por potencias integrables con factores logarítmicos.
Por tanto un `F` que no sea idénticamente cero tiene un número finito de ceros
en este intervalo compacto.

**Un barrido numérico, aunque se refine y coincida entre cuadraturas distintas,
no certifica por sí solo haber encontrado todos esos ceros.** Para una prueba
formal de exhaustividad hace falta, por ejemplo, una partición completa con
evaluaciones intervalares de `F` que excluyan cero fuera de las cajas candidatas,
y aislamiento validado de cada raíz dentro de ellas. Los errores de cuadratura,
redondeo y derivación también deben estar encerrados por cotas demostradas.

Las mejoras de la tabla son amplias y permiten rechazos numéricos robustos.
Una certificación formal de esas desigualdades requeriría además cotas validadas
para sus integrales. Una vez completa la lista de candidatos y verificada su
optimalidad, las condiciones iniciales se clasifican mediante la línea de fase
de sus políticas, con la convención de continuación en los umbrales declarada
por el experimento.
