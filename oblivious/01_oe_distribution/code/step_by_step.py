"""Ejemplo pequeño del OE mixto cerrado con la marginal del juego real.

Se conservan las decisiones del OE. Para cada política que probamos:
    política -> conjunta real estacionaria -> marginal del rival
             -> demanda promediada -> incentivos para cambiar la política.

El equilibrio requiere que esos incentivos sean compatibles con la política.
Este archivo explica los pasos y llama al mismo solver que el experimento.
No crea carpetas ni sustituye resultados guardados.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution.code.model import JointMarginalExperiment


def main():
    # 1. Parámetros que puedes cambiar. rho=.97 permite ilustrar mezcla en
    # esta grilla pequeña; el programa principal recorre los ocho rho.
    params = Parameters(rho=.97, c2=.65)
    n = 21
    experiment = JointMarginalExperiment(n=n, params=params, order=128)
    print(f"1. Dos vendedores iguales; rho={params.rho}, costo 2={params.c2}.")
    print(f"   {n} reputaciones por vendedor y {n*n} pares de reputaciones.")

    # 2. Punto de partida numérico: cualquier probabilidad entre 0 y 1.
    initial_alpha = np.full(n, .5)
    print("2. Comenzamos probando alpha=0.5: cada producto tiene probabilidad 50%.")
    print("   Es un inicio de búsqueda; todavía no afirmamos que sea óptimo.")

    # 3. Para cada alpha, el solver usa la verdadera selección TS entre A y B
    # y calcula la distribución conjunta. La marginal viene de sumar sobre A.
    print("3. Para cada política probada calculamos nu(xA,xB) en el juego real.")
    print("   Su marginal de B es mu(xB)=suma_xA nu(xA,xB).")

    # 4. El vendedor sigue optimizando como en el OE: depende de su propio x
    # y usa D@mu como demanda. Se ajusta alpha, permitiendo mezclas en empates.
    print("4. Usamos d(x)=suma_y D(x,y)*mu(y) para evaluar sus incentivos.")
    print("   Ajustamos alpha hasta que sea óptima frente a su propia marginal generada.")
    result = experiment.solve(initial_policy=initial_alpha)

    # 5. Estas sumas se calculan otra vez directamente con los datos devueltos.
    mu_from_joint = result["nu"].sum(axis=0)
    marginal_error = np.abs(result["mu"] - mu_from_joint).sum()
    demand_error = np.max(np.abs(result["demand"] - experiment.game.D @ mu_from_joint))
    print("5. Comprobamos TODA la distribución, no solo su media.")
    print(f"   Error marginal total: {marginal_error:.3g}")
    print(f"   Error de demanda respecto a esa marginal: {demand_error:.3g}")
    print(f"   Error de optimalidad: {result['dp_diagnostics']['conditional_regret']:.3g}")
    print(f"   Error de indiferencia en mezclas: {result['dp_diagnostics']['mixed_indifference_residual']:.3g}")
    print(f"   Resultado verificado en esta grilla: {result['verified']}")
    print("\nReputación | P(producto 2) | Marginal real de B | Ventaja del producto 2")
    for x, alpha, mass, advantage in zip(result["grid"], result["policy"], mu_from_joint, result["advantage"]):
        print(f"{x:10.3f} | {alpha:13.8f} | {mass:18.8f} | {advantage: .8f}")
    print("\nUna probabilidad interior requiere ventaja cero: ambos productos valen lo mismo.")
    print("Las probabilidades 0 y 1 también están permitidas dentro del modelo mixto.")
    print("La verificación es en la grilla; la estrategia sigue usando solo la reputación propia.")
    if not result["verified"]:
        raise SystemExit("Este intento no pasó todas las verificaciones; no debe llamarse equilibrio.")


if __name__ == "__main__":
    main()
