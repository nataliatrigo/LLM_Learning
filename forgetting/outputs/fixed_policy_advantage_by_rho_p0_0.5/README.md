# Ventaja respecto a la política fija de producto 1, p0=0.5

U_rho^1 es el valor de elegir siempre el producto 1. Las curvas muestran a_rho^1(z)=-delta_c+gamma*delta_p*[U_rho^1(T_success z)-U_rho^1(T_failure z)]. Un valor positivo favorece elegir producto 2 una vez y después producto 1 para siempre, frente a elegir siempre producto 1.

A_rho^1 es el máximo de esa curva sobre K_rho, el conjunto límite alcanzable. Se indica en cada panel, con un rombo en el estado del máximo, y se representa por separado frente a los ocho valores de rho. Las curvas por estado cubren la frontera z=(x,1-x), x=(1-rho)*S. Para rho<0.5, sólo un subconjunto de Cantor es límite alcanzable.

PNG y PDF en figures/. Curvas, máximos y verificaciones numéricas en data/ y manifest.json.

```bash
uv run python -B forgetting/Code/visualizations/plot_fixed_policy_advantage_by_rho.py
```
