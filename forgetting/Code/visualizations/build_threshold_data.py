"""Recompute the original-model data used by the threshold explainer."""

from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, reachable_boundary_points, solve_boundary


def main():
    cases = [
        ("one", 0.7, 0.5, 0.65),
        ("two", 0.95, 0.5, 0.65),
        ("none", 0.999, 0.5, 0.65),
        ("gap", 0.45, 0.9, 0.6665),
        ("lower-cost", 0.45, 0.9, 0.65),
    ]
    output = {
        "method": "Original boundary Bellman; 8001 uniform nodes; linear interpolation; iteration tolerance 1e-12; Cantor endpoints depth 18. Not a rigorous discretization-error certificate.",
        "cases": [],
    }
    for name, rho, p0, c2 in cases:
        p = Primitives(p0=p0, c2=c2)
        sol = solve_boundary(p, rho, central_nodes=8001, outer_nodes=2,
                             half_widths=1e6, tolerance=1e-12)

        def advantage(x):
            return -p.delta_c + p.gamma * p.delta_p * (
                np.interp(rho*x+1-rho, sol.coordinate, sol.value)
                - np.interp(rho*x, sol.coordinate, sol.value)
            )

        # The interpolated value makes advantage piecewise linear. Include
        # all shifted interpolation knots, so its numerical roots/maxima are
        # not dependent on a coarse display grid.
        knots = np.unique(np.clip(np.concatenate([
            sol.coordinate/rho, (sol.coordinate-1+rho)/rho, [0., 1.]
        ]), 0, 1))
        a = advantage(knots)
        crossing = np.flatnonzero(a[:-1] * a[1:] < 0)
        roots = [float(brentq(advantage, knots[i], knots[i+1], xtol=1e-14))
                 for i in crossing]
        max_index = int(np.argmax(a))
        if rho < .5:
            pts = reachable_boundary_points(rho, depth=18)
            ac = advantage(pts)
            k = int(np.argmax(ac))
            a_reachable, x_reachable = float(ac[k]), float(pts[k])
        else:
            a_reachable, x_reachable = float(a[max_index]), float(knots[max_index])
        assert all(abs(float(advantage(x))) < 1e-10 for x in roots)
        assert np.min(np.diff(sol.value)) >= -1e-9
        item = {
            "id": name, "rho": rho, "primitives": p.as_dict(),
            "value": np.round(sol.value, 9).tolist(), "roots": roots,
            "maxFull": float(a[max_index]), "xMaxFull": float(knots[max_index]),
            "maxReachable": a_reachable, "xMaxReachable": x_reachable,
            "bellmanResidual": sol.bellman_residual,
            "cantorEndpointDepth": 18 if rho < .5 else None,
        }
        output["cases"].append(item)
        print(name, "roots", roots, "A_C", a_reachable,
              "max_full", float(a[max_index]), flush=True)
    destination = ROOT / "forgetting/outputs/threshold_visualizations/threshold-data.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print("Data bytes", destination.stat().st_size)


if __name__ == "__main__":
    main()
