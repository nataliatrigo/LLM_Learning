# Política de intervalo bajo reemplazo aleatorio uniforme

Fecha: 3 de noviembre de 2026.

Esta nota se refiere exactamente al modelo de `code/model.py`: prior
Beta(1,1), demanda de Thompson, opción externa que no cambia la memoria y
eliminación uniforme de una observación antigua, independiente del resultado
nuevo. El resultado estructural es para **memoria llena**. La fase de llenado
se trata por separado al final.

## Teorema

Sean

\[
n\in\mathbb N,\quad n\ge1,\quad 0<p_0<1,\quad
0<p_1<p_2<1,\quad 0\le c_1<c_2<R,\quad 0<\gamma<1.
\]

Con memoria llena, existe una política óptima estacionaria tal que el
producto 2 se elige en un único intervalo de estados enteros:

\[
\{s:\pi^*(s)=2\}=\{L,L+1,\ldots,U\}\subseteq\{0,\ldots,n\}.
\]

Se permite el intervalo vacío. El intervalo puede incluir uno o ambos
extremos del espacio de estados. En los empates se elige el producto 1;
si no hay empates, la política óptima estacionaria es única.

No se necesita una condición adicional sobre el descuento, los costes,
las probabilidades de éxito o el tamaño de la memoria. Para \(\gamma=0\)
la conclusión también vale: siempre se elige el producto 1.

## 1. Bellman y transformación del valor

Escribamos \(r_i=R-c_i\), \(\Delta p=p_2-p_1\),
\(\Delta c=c_2-c_1\). La demanda con \(s\) éxitos es

\[
d_s=\Pr\{\operatorname{Beta}(s+1,n-s+1)>p_0\}
    =\sum_{j=0}^{s}\binom{n+1}{j}p_0^j(1-p_0)^{n+1-j}.
\]

El reemplazo aleatorio induce, condicional a una compra,

\[
P_i(s,s+1)=p_i(1-s/n),\qquad
P_i(s,s-1)=(1-p_i)s/n.
\]

La probabilidad restante corresponde a permanecer en \(s\). Definimos

\[
(\mathcal L_i v)_s
=p_i(1-s/n)(v_{s+1}-v_s)
 +(1-p_i)(s/n)(v_{s-1}-v_s),
\]

omitiendo términos que salen del dominio, cuyos coeficientes son cero.
El Bellman en tiempo calendario es

\[
(1-\gamma)V_s
=d_s\max_i\{r_i+\gamma(\mathcal L_iV)_s\}.
\tag{1}
\]

Este problema finito y descontado tiene un único valor óptimo y admite
una política óptima estacionaria determinista: su operador de Bellman
es una contracción de módulo \(\gamma\).

Introduzcamos

\[
\delta=\frac{1-\gamma}{\gamma}>0,\qquad
k=\frac{\Delta c}{\gamma\Delta p}>0,\qquad
A=p_1+\frac{r_1\Delta p}{\Delta c}>0,\qquad
W_s=V_s-ks.
\]

Como \(\mathcal L_i s=p_i-s/n\) y
\(r_i/\gamma+kp_i=kA\) para ambos productos, (1) equivale a

\[
\delta W_s=F_s+d_s\max_i(\mathcal L_i W)_s,
\qquad
F_s=k\left[d_s\left(A-\frac{s}{n}\right)-\delta s\right].
\tag{2}
\]

La recompensa transformada \(F_s\) **no depende de la acción**.
Esta transformación conserva exactamente las comparaciones entre
acciones; no es una aproximación fluida ni una expansión asintótica.

Sea \(w_j=W_j-W_{j-1}\), \(j=1,\ldots,n\), y definamos

\[
B_s=(1-s/n)w_{s+1}+(s/n)w_s,
\tag{3}
\]

con \(B_0=w_1\) y \(B_n=w_n\). La diferencia entre los valores de
elegir los productos 2 y 1, condicional a una compra, es exactamente

\[
\begin{aligned}
\mathcal A_s
&=-\Delta c+\gamma\Delta p
  \left[(1-s/n)(V_{s+1}-V_s)+(s/n)(V_s-V_{s-1})\right]\\
&=\gamma\Delta p\,B_s.
\end{aligned}
\tag{4}
\]

Por tanto, basta demostrar que \(\{s:B_s>0\}\) es un intervalo.

## 2. Los incrementos positivos de F forman un intervalo

Para \(j=1,\ldots,n\), escribamos

\[
b_j=d_j-d_{j-1}
 =\binom{n+1}{j}p_0^j(1-p_0)^{n+1-j}>0,
\]

\[
g_j=\frac{F_j-F_{j-1}}{k}
 =b_j\left(A-\frac jn\right)-\frac{d_{j-1}}n-\delta.
\tag{5}
\]

Si \(j\ge nA\), entonces \(g_j<0\). Solo hay que estudiar el prefijo
de índices \(j<nA\).

Para dos índices consecutivos de ese prefijo, la razón

\[
\frac{b_{j+1}}{b_j}
=\frac{n+1-j}{j+1}\frac{p_0}{1-p_0}
\]

da la identidad

\[
g_{j+1}-g_j
=\frac{b_j}{n(j+1)(1-p_0)}
 \left[(nA-j-1)((n+2)p_0-j-1)-2(j+1)(1-p_0)\right].
\tag{6}
\]

Pongamos \(t=nA\), \(b=(n+2)p_0\), \(q=1-p_0\) y

\[
H(x)=(t-x)(b-x)-2qx.
\]

En \(0<x<\min(t,b)\),

\[
H'(x)=2x-t-b-2q<0.
\]

Además, \(H(0)=tb>0\),
\(H(\min(t,b))=-2q\min(t,b)<0\), y si \(b\le x<t\) entonces
\(H(x)<0\). En consecuencia, en \(0<x<t\), el signo de \(H\)
solo puede pasar de positivo a negativo, una vez.

Por (6), la sucesión \(g_j\) primero crece y después decrece en el
prefijo \(j<t\), con cualquiera de esos tramos posiblemente vacío.
Fuera del prefijo es estrictamente negativa por (5). Por ello

\[
\{j:F_j-F_{j-1}>0\}\quad\text{es un intervalo, posiblemente vacío.}
\tag{7}
\]

No estamos afirmando que los incrementos sean unimodales en todo el
dominio: para la prueba basta la forma de su conjunto positivo.

## 3. El resolvente de una cadena de vecinos preserva esa propiedad

Fijemos una política óptima estacionaria \(\pi\), y escribamos sus tasas

\[
\lambda_s=d_s p_{\pi(s)}(1-s/n),\qquad
\mu_s=d_s(1-p_{\pi(s)})s/n.
\]

En particular, \(\lambda_n=\mu_0=0\). La ecuación (2), evaluada en
esa política, es

\[
\delta W_s=F_s+\lambda_s(W_{s+1}-W_s)
                     +\mu_s(W_{s-1}-W_s).
\]

Restando la ecuación en \(j-1\) de la ecuación en \(j\), obtenemos

\[
(\delta+\mu_j+\lambda_{j-1})w_j
 -\mu_{j-1}w_{j-1}-\lambda_jw_{j+1}=f_j,
\qquad f_j=F_j-F_{j-1}.
\tag{8}
\]

Los términos exteriores se omiten. Mostramos directamente que resolver
(8) preserva la propiedad de tener un único intervalo positivo.

Los pivotes de la eliminación tridiagonal son

\[
a_1=\delta+\mu_1+\lambda_0,
\qquad
a_j=\delta+\mu_j+\lambda_{j-1}
       -\frac{\mu_{j-1}\lambda_{j-1}}{a_{j-1}}.
\]

Todos son positivos. Más precisamente, \(a_1>\mu_1\), y si
\(a_{j-1}>\mu_{j-1}\), entonces

\[
a_j=\delta+\mu_j+
 \lambda_{j-1}\left(1-\frac{\mu_{j-1}}{a_{j-1}}\right)>\mu_j.
\]

La sustitución hacia delante y hacia atrás es

\[
y_1=f_1,\qquad
y_j=f_j+\frac{\mu_{j-1}}{a_{j-1}}y_{j-1},
\tag{9}
\]

\[
w_n=\frac{y_n}{a_n},\qquad
w_j=\frac{y_j+\lambda_jw_{j+1}}{a_j}.
\tag{10}
\]

Usamos una observación elemental: una recurrencia
\(z_j=u_j+\alpha_jz_{j-1}\), con \(\alpha_j\ge0\) y condición
inicial cero, preserva un único intervalo de entradas estrictamente
positivas. Antes de ese intervalo, la salida es no positiva. Dentro,
una vez que se hace positiva, no puede dejar de serlo. Después, una vez
que se hace no positiva, no puede volver a ser positiva. El mismo
argumento funciona recorriendo los índices en sentido inverso y
multiplicando las entradas o salidas por números positivos.

Por (7), las entradas \(f_j\) tienen un solo intervalo positivo. La
recurrencia (9) transmite esa propiedad a \(y_j\), y (10) la transmite
a \(w_j\). Así,

\[
\{j:w_j>0\}\quad\text{es un intervalo, posiblemente vacío.}
\tag{11}
\]

Este argumento no supone ninguna forma de la política \(\pi\), ni
monotonía de sus probabilidades de éxito en el estado.

## 4. De los incrementos del valor a la política

Si todos los \(w_j\) son no positivos, también lo son todos los
\(B_s\). En caso contrario, sea \(\{a,\ldots,b\}\) el intervalo
positivo de \(w\).

Por (3), \(B_s\le0\) fuera de \(\{a-1,\ldots,b\}\), y
\(B_s>0\) para \(s=a,\ldots,b-1\). Solo los dos estados vecinos
\(a-1\) y \(b\) pueden tener un signo adicional por determinar.
Si \(a=b\), los únicos candidatos son los dos estados consecutivos
\(a-1,a\). En todos los casos, \(\{s:B_s>0\}\) es un intervalo.

Por (4), elegir producto 2 exactamente cuando \(B_s>0\), y producto
1 en los demás estados, es óptimo y tiene la forma anunciada. Esto
prueba el teorema.

## 5. Aplicación a la calibración del proyecto

Para

\[
n=20,\quad p_0=\tfrac12,\quad p_1=\tfrac7{20},\quad
p_2=\tfrac45,\quad \gamma=\tfrac{49}{50},\quad
R=1,\quad c_1=\tfrac1{20},\quad c_2=\tfrac{13}{20},
\]

la política madura es producto 2 exactamente en \(s=5,\ldots,12\).
El teorema demuestra la forma de intervalo sin resolver numéricamente
el Bellman. Para **localizar** los extremos 5 y 12, el script
`code/verify_interval_theorem.py` evalúa esa política y comprueba todas
las desigualdades de Bellman con aritmética racional exacta.

## 6. La fase de llenado requiere distinguir otro resultado

Cuando todavía se recuerdan \(m<n\) observaciones, una compra lleva a
la capa \(m+1\). La ventaja es

\[
\mathcal A_{m,s}
=-\Delta c+\gamma\Delta p\left(V_{m+1,s+1}-V_{m+1,s}\right).
\tag{12}
\]

La prueba anterior no se aplica: el sistema ya no es una cadena de
vecinos en una única capa de tamaño fijo. De hecho, la afirmación
universal de intervalo es falsa en estas capas.

Un contraejemplo es

\[
n=30,\quad p_0=\tfrac9{20},\quad p_1=\tfrac1{50},\quad
p_2=\tfrac3{20},\quad R=1,\quad c_1=0,\quad
c_2=\tfrac{53}{100},\quad \gamma=\tfrac{19997}{20000}.
\]

En madurez es óptimo elegir siempre producto 1. Pero en la capa
\(m=3\), las ventajas son, aproximadamente,

\[
(\mathcal A_{3,0},\mathcal A_{3,1},\mathcal A_{3,2},\mathcal A_{3,3})
=(0.03720631,-0.00979137,-0.02709705,0.01320160).
\]

Por tanto, producto 2 es estrictamente óptimo en \(\{0,3\}\), que
no es un intervalo. El script adjunto verifica los signos mediante
fracciones exactas, evaluando la política madura y resolviendo después
todas las capas por inducción hacia atrás. Los decimales anteriores
solo sirven para mostrar los resultados.

### Condiciones suficientes que sí cubren todas las capas

También hay una familia explícita de parámetros, que contiene la
calibración del proyecto, en la que **cada capa** tiene una política
óptima de intervalo. Mantengamos

\[
n=20,\quad p_0=\tfrac12,\quad p_1=\tfrac7{20},\quad
p_2=\tfrac45,\quad \gamma=\tfrac{49}{50}.
\]

Si \(R>c_1\ge0\) y

\[
\boxed{\qquad
\frac{61}{100}\le
\theta:=\frac{c_2-c_1}{R-c_1}
\le\frac{16}{25},
\qquad}
\tag{13}
\]

la siguiente política es estrictamente óptima en todos los estados:

| Capa m | Estados s con producto 2 |
|---:|:---|
| 20 | 5–12 |
| 19, 18, 17 | 4–11 |
| 16, 15 | 3–10 |
| 14 | 3–9 |
| 13 | 2–9 |
| 12, 11 | 2–8 |
| 10, 9 | 1–7 |
| 8, 7 | 0–6 |
| 6, 5 | 0–5 |
| 4 | 0–4 |
| 3 | 0–3 |
| 2 | 0–2 |
| 1 | 0–1 |
| 0 | 0 |

Esta es una certificación finita exacta de condiciones suficientes,
distinta del teorema estructural de las secciones anteriores. Su
argumento es el siguiente. Normalizamos los beneficios por \(R-c_1\),
de modo que los dos beneficios son \(1\) y \(1-\theta\). Para la
política fija de la tabla, el valor en cada estado es afín en
\(\theta\), porque las transiciones no dependen de los costes y la
evaluación de políticas es lineal. Por tanto, cada ventaja normalizada
es también afín en \(\theta\).

El script verifica con fracciones exactas que, en **ambos** extremos
de (13), la ventaja con el signo de la acción prescrita es mayor que
\(1/2500\) en los 231 estados. Por afinidad, la misma desigualdad
vale en todo el intervalo. La política satisface entonces todas las
desigualdades estrictas de Bellman, lo que demuestra su optimalidad
única en toda esa familia.

Con \(R=1,c_1=0.05\), (13) equivale a
\(0.6295\le c_2\le0.658\), e incluye \(c_2=0.65\).

## Reproducción de las certificaciones exactas

Desde la raíz del repositorio:

```bash
python3 -B forgetting_random_replacement/code/verify_interval_theorem.py
```

El script usa únicamente la biblioteca estándar y `fractions.Fraction`.
Comprueba la banda madura del caso base, la familia suficiente de todas
las capas y el contraejemplo transitorio. La demostración estructural
de memoria llena es analítica y no depende de estas verificaciones.
