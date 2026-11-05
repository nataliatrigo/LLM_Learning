# Prueba de optimalidad de dos umbrales mediante programación lineal

Esta nota complementa [la demostración estructural](interval_policy_proof.md).
Se mantienen sus hipótesis: memoria llena, prior Beta(1,1), reemplazo
uniforme independiente del resultado nuevo, \(0<p_1<p_2<1\),
\(0\le c_1<c_2<R\), \(0<p_0<1\), \(0<\gamma<1\) y \(n\ge1\).

El resultado es: **la mejor política de intervalo es óptima entre todas
las políticas**, y su valor y ocupación proporcionan un certificado
primal-dual. El paso específico del modelo es demostrar que una mejora
greedy de cualquier política estacionaria siempre es un intervalo.
La formulación LP, por sí sola, no establece esta propiedad geométrica.

## 1. LP de Bellman en tiempo calendario

Sea \(P_i\) la matriz de transición condicional a una compra y
\(D=\operatorname{diag}(d_0,\ldots,d_n)\). Definimos

\[
K_i=I+D(P_i-I),\qquad \rho_i(s)=d_s(R-c_i).
\]

Así, \(K_i\) es la matriz de transición por período calendario,
incluyendo la opción externa, y \(\rho_i\) es el beneficio esperado
de ese período. Elija pesos \(\eta_s>0\) para todos los estados.
El LP exacto es

\[
\begin{aligned}
\min_{V\in\mathbb R^{n+1}}\quad&\eta^\top V,\\
\text{sujeto a}\quad &(I-\gamma K_i)V\ge\rho_i,
\qquad i=1,2.
\end{aligned}
\tag{P0}
\]

Todas las desigualdades son componente a componente. El valor óptimo
del MDP es la solución única: si \(V\) es factible, entonces
\(V\ge TV\), de donde, iterando el operador monótono y contractivo
de Bellman, \(V\ge V^*\). El vector \(V^*\) es factible; los pesos
estrictamente positivos hacen único al minimizador.

El dual es

\[
\begin{aligned}
\max_{x_1,x_2\ge0}\quad&\rho_1^\top x_1+\rho_2^\top x_2,\\
\text{sujeto a}\quad&
(I-\gamma K_1)^\top x_1+(I-\gamma K_2)^\top x_2=\eta.
\end{aligned}
\tag{D0}
\]

Cuando \(\sum_s\eta_s=1\), \(x_{s,i}\) representa el número esperado
descontado de visitas a \((s,i)\), sin multiplicar por \(1-\gamma\).
El LP no impone restricciones de intervalo sobre estas variables.

La formulación general y su equivalencia con Bellman aparecen en
[Ying y Zhu, sección 2](https://arxiv.org/html/2012.09417v1#S2).
El desarrollo siguiente aplica esa formulación a este modelo y usa
su propiedad particular de signos.

## 2. Transformación que elimina la acción de la recompensa

Use la notación

\[
\delta=\frac{1-\gamma}{\gamma},\quad
k=\frac{c_2-c_1}{\gamma(p_2-p_1)},\quad
A=p_1+\frac{(R-c_1)(p_2-p_1)}{c_2-c_1},\quad W_s=V_s-ks.
\]

Defina el generador por filas \(Q_i=K_i-I=D(P_i-I)\), y

\[
F_s=k\left[d_s\left(A-\frac{s}{n}\right)-\delta s\right].
\]

Sustituyendo \(V_s=W_s+ks\) en (P0) y dividiendo sus restricciones
por \(\gamma\), se obtiene

\[
\begin{aligned}
\min_{W\in\mathbb R^{n+1}}\quad&\eta^\top W,\\
\text{sujeto a}\quad& (\delta I-Q_i)W\ge F,
\qquad i=1,2.
\end{aligned}
\tag{P}
\]

El objetivo original solo difiere por la constante
\(k\sum_s\eta_s s\). Su dual es

\[
\begin{aligned}
\max_{z_1,z_2\ge0}\quad&F^\top(z_1+z_2),\\
\text{sujeto a}\quad&
(\delta I-Q_1)^\top z_1+(\delta I-Q_2)^\top z_2=\eta.
\end{aligned}
\tag{D}
\]

Los duales están relacionados mediante \(z_i=\gamma x_i\).

## 3. Certificado primal-dual para una política candidata

Para cualquier política estacionaria determinista \(\pi\), sea
\(Q_\pi\) la matriz que usa la fila \(s\) de \(Q_{\pi(s)}\).
Su valor transformado y su vector de ocupaciones transformadas son

\[
W^\pi=(\delta I-Q_\pi)^{-1}F,\qquad
m^\pi=(\delta I-Q_\pi)^{-\top}\eta>0.
\]

La no negatividad de la inversa y la positividad de su diagonal se
ven directamente de

\[
(\delta I-Q_\pi)^{-1}
=\gamma(I-\gamma K_\pi)^{-1}
=\gamma\sum_{t=0}^{\infty}\gamma^tK_\pi^t.
\tag{1}
\]

Asignando \(z_{s,\pi(s)}=m_s^\pi\) y cero a la otra acción,
obtenemos un punto factible en (D). Además,

\[
F^\top m^\pi=\eta^\top W^\pi.
\tag{2}
\]

Por tanto, **si \(W^\pi\) también es factible en (P), la política
es globalmente óptima**: los objetivos primal y dual coinciden, y
basta dualidad débil para certificarlo. Las restricciones de la acción
seleccionada se cumplen con igualdad, de modo que también se verifica
holgura complementaria.

Para una candidata de intervalo \(\pi_{L,U}\), defina

\[
B_s^\pi=(1-s/n)(W^\pi_{s+1}-W^\pi_s)
       +(s/n)(W^\pi_s-W^\pi_{s-1}),
\]

omitiendo términos exteriores. Como

\[
((Q_2-Q_1)W^\pi)_s=d_s(p_2-p_1)B_s^\pi,
\]

su factibilidad primal equivale a

\[
B_s^\pi\ge0\quad\text{si }L\le s\le U,\qquad
B_s^\pi\le0\quad\text{fuera del intervalo}.
\tag{3}
\]

Estas desigualdades son un certificado global para unos umbrales
dados; solo requieren evaluar la política mediante un sistema
tridiagonal y comprobar las restricciones de la acción alternativa.
Comprobar únicamente los dos extremos del intervalo no sustituye
la verificación de (3) en todos los estados.

## 4. Lema: la mejora greedy de cualquier política es un intervalo

**Lema.** Para cualquier política estacionaria determinista \(\pi\),
el conjunto \(\{s:B_s^\pi>0\}\) es un intervalo, posiblemente vacío.

**Prueba.** La prueba estructural anterior no necesitaba optimalidad de
la política al evaluar su valor. Para cualquier \(\pi\), se cumple

\[
\delta W^\pi=F+Q_\pi W^\pi.
\]

Poniendo \(w_j=W_j^\pi-W_{j-1}^\pi\), sus diferencias satisfacen

\[
(\delta+\mu_j+\lambda_{j-1})w_j
-\mu_{j-1}w_{j-1}-\lambda_jw_{j+1}=F_j-F_{j-1},
\]

con \(\lambda_s=d_sp_{\pi(s)}(1-s/n)\) y
\(\mu_s=d_s(1-p_{\pi(s)})s/n\). La demanda binomial implica que
los incrementos positivos de \(F\) forman un intervalo. La eliminación
tridiagonal con coeficientes no negativos preserva esa propiedad;
por ello, los \(w_j>0\) forman un intervalo. Promediar dos incrementos
vecinos como en \(B_s^\pi\) preserva también un solo intervalo positivo.
Estos pasos se demuestran íntegramente en las secciones 2–4 de
[la nota estructural](interval_policy_proof.md).

En consecuencia, la política que maximiza \((Q_iW^\pi)_s\), eligiendo
producto 1 en los empates, es siempre una política de intervalo.

## 5. La mejor política de intervalo es globalmente óptima

Sea \(\mathcal I\) el conjunto de políticas que eligen producto 2
en un intervalo entero y producto 1 fuera, incluyendo el intervalo
vacío. Tiene

\[
|\mathcal I|=1+\frac{(n+1)(n+2)}2
\]

elementos. Elijamos

\[
\bar\pi\in\arg\max_{\pi\in\mathcal I}\eta^\top W^\pi.
\tag{4}
\]

Esta selección existe por finitud. Todavía no se presupone que sea
óptima entre las políticas fuera de \(\mathcal I\).

Supongamos que \(W^{\bar\pi}\) viola alguna restricción de (P).
Sea \(\widehat\pi\) su política greedy y definamos el vector

\[
h_s=\max_i(Q_iW^{\bar\pi})_s
                  -(Q_{\bar\pi}W^{\bar\pi})_s.
\]

Tenemos \(h\ge0\), y la violación primal implica que alguna componente
es estrictamente positiva. Por el lema, \(\widehat\pi\in\mathcal I\).
Restando las ecuaciones de evaluación de políticas,

\[
W^{\widehat\pi}-W^{\bar\pi}
=(\delta I-Q_{\widehat\pi})^{-1}h.
\tag{5}
\]

Por (1), esta diferencia es no negativa y alguna componente es
estrictamente positiva. Como todos los pesos \(\eta_s\) son positivos,

\[
\eta^\top W^{\widehat\pi}>\eta^\top W^{\bar\pi},
\]

contradiciendo (4). Así, \(W^{\bar\pi}\) es factible en el LP
completo (P). Su ocupación dual, construida en la sección 3, es
factible y tiene el mismo objetivo. Esto demuestra que \(\bar\pi\)
es globalmente óptima, incluso al comparar con políticas aleatorizadas
o dependientes de la historia.

El argumento permite maximizar sobre dos umbrales sin imponer esa
restricción al problema original: el lema y la mejora estricta son los
que justifican la reducción.

## 6. Consecuencia algorítmica y alcance

La iteración de políticas de Howard entra en la familia de intervalos
después de una sola mejora greedy desde cualquier política estacionaria
determinista. Cada mejora posterior sigue siendo un intervalo. Si el
residual \(h\) es distinto de cero, el objetivo con pesos positivos
mejora estrictamente. Si \(h=0\), ya existe un certificado primal-dual
de optimalidad. En aritmética exacta y deteniéndose cuando \(h=0\), la
finitud de las políticas garantiza terminación.

Tanto este lema como el teorema de intervalos se refieren a memoria
llena. El LP también puede escribirse para las capas de llenado, pero
allí la mejora greedy no tiene necesariamente forma de intervalo,
como muestra el contraejemplo de la nota estructural.
