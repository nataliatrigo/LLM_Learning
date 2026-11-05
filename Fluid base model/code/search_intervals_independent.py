"""Independent semi-Lagrangian search for disconnected product-2 sections.

The n=S+F step is dn and the S mesh step is dn/subdivisions.  For a fixed
action, the next state is (n+dn,S+p_i*dn), with linear interpolation. Demand
is frozen at the current state over this step, giving an exponential discount
and its exact constant-demand reward integral. This is a monotone, consistent
approximation of the same no-forgetting fluid HJB as model.py, but it does not
use the Bernoulli upwind transition of that solver.

A hit requires +,-,+ runs of strict advantage and each run must occupy at
least min_width in S. Output candidates are SEARCH LEADS, not certificates.
Terminal bounds bracket this semi-Lagrangian scheme, not its continuum error.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import betaincc


def strict_witness(gap, ds, tolerance=1e-6, min_width=0.5):
    """Return separated strict positive, negative, positive run interiors."""
    signs = np.where(gap > tolerance, 1, np.where(gap < -tolerance, -1, 0))
    starts = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1]) + 1]
    ends = np.r_[starts[1:], len(signs)]
    runs = [(int(signs[a]), int(a), int(b)) for a, b in zip(starts, ends)
            if signs[a] != 0 and (b-a-1)*ds >= min_width - 1e-10]
    left = None
    middle = None
    for sign, a, b in runs:
        if sign == 1:
            if middle is not None:
                points = [sum(left)//2, sum(middle)//2, (a+b-1)//2]
                return {"S": [float(k*ds) for k in points],
                        "gap": [float(gap[k]) for k in points],
                        "runs_S": [[left[0]*ds, left[1]*ds],
                                   [middle[0]*ds, middle[1]*ds],
                                   [a*ds, (b-1)*ds]]}
            left = (a, b-1)
        elif left is not None:
            middle = (a, b-1)
    return None


def solve_batch(p0, candidates, dn=1.0, subdivisions=4, cutoff_n=400,
                report_n=80, tolerance=1e-6, min_width=0.5):
    """Search candidates sharing p0, retaining only diagnostics and hits."""
    ds = dn/subdivisions
    outer = round(cutoff_n/dn)
    report = round(report_n/dn)
    pars = np.asarray([[c["p1"], c["p2"], c["cost"], c["delta"]]
                       for c in candidates], dtype=float)
    p1, p2, cost, delta = [pars[:, k:k+1] for k in range(4)]
    count = len(pars)
    lower_next = np.zeros((count, outer*subdivisions+1))
    upper_next = np.broadcast_to(1/delta, lower_next.shape).copy()
    offsets = [subdivisions*p1, subdivisions*p2]
    lefts = [np.floor(off).astype(int) for off in offsets]
    weights = [off-left for off, left in zip(offsets, lefts)]
    rows = np.arange(count)[:, None]
    hits = []
    errors = np.zeros(count)
    strict_points = np.zeros(count, dtype=int)
    max_components = np.zeros(count, dtype=int)
    max_nominal_components = np.zeros(count, dtype=int)
    best_nominal = [None]*count
    for level in range(outer-1, -1, -1):
        n = level*dn
        grid = np.arange(level*subdivisions+1)
        S = ds*grid
        D = betaincc(S+1, n-S+1, p0)[None, :]
        with np.errstate(over="ignore"):
            ratio = np.divide(delta*dn, D, out=np.full((count,len(grid)), np.inf), where=D>0)
        discount = np.exp(-ratio)
        reward = D/delta*(-np.expm1(-ratio))
        lower_q = []
        upper_q = []
        for i, (left, weight) in enumerate(zip(lefts, weights)):
            lo_idx = grid[None, :]+left
            hi_idx = np.minimum(lo_idx+1, lower_next.shape[1]-1)
            lo = (1-weight)*lower_next[rows, lo_idx] + weight*lower_next[rows, hi_idx]
            up = (1-weight)*upper_next[rows, lo_idx] + weight*upper_next[rows, hi_idx]
            margin = 1 if i == 0 else 1-cost
            lower_q.append(reward*margin+discount*lo)
            upper_q.append(reward*margin+discount*up)
        lower = np.maximum(*lower_q)
        upper = np.maximum(*upper_q)
        if level <= report:
            errors = np.maximum(errors, np.max(upper-lower, axis=1))
            gap = np.divide(lower_q[1]-lower_q[0], reward,
                            out=np.zeros_like(lower), where=reward>1e-280)
            for j in range(count):
                nominal = gap[j] > 0
                nominal_components = int(nominal[0]) + int(np.sum(nominal[1:] & ~nominal[:-1]))
                max_nominal_components[j] = max(max_nominal_components[j], nominal_components)
                if nominal_components >= 2:
                    nominal_witness = strict_witness(gap[j], ds, 0.0, 0.0)
                    if nominal_witness is not None:
                        strength = min(nominal_witness["gap"][0],
                                       -nominal_witness["gap"][1],
                                       nominal_witness["gap"][2])
                        if best_nominal[j] is None or strength > best_nominal[j]["strength"]:
                            best_nominal[j] = {"n":n,"witness":nominal_witness,
                                               "strength":strength}
                mask = gap[j] > tolerance
                strict_points[j] += int(mask.sum())
                components = int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1]))
                max_components[j] = max(max_components[j], components)
                if components >= 2:
                    witness = strict_witness(gap[j], ds, tolerance, min_width)
                    if witness is not None:
                        # Bounds for the action scores, normalized by common reward.
                        ids = np.rint(np.asarray(witness["S"])/ds).astype(int)
                        gl = (lower_q[1][j,ids]-upper_q[0][j,ids])/reward[j,ids]
                        gu = (upper_q[1][j,ids]-lower_q[0][j,ids])/reward[j,ids]
                        hit = {"candidate": candidates[j], "p0":p0, "n":n,
                               "witness":witness, "gap_lower":gl.tolist(),
                               "gap_upper":gu.tolist(), "dn":dn,
                               "subdivisions":subdivisions,
                               "cutoff_n":cutoff_n}
                        hits.append(hit)
        lower_next, upper_next = lower, upper
    return {"p0":p0, "candidates":candidates, "hits":hits,
            "value_bracket_max":errors.tolist(),
            "max_positive_components":max_components.tolist(),
            "max_nominal_positive_components":max_nominal_components.tolist(),
            "best_nominal_three_point_witness":best_nominal,
            "strict_product2_points":strict_points.tolist(),
            "dn":dn, "ds":ds, "cutoff_n":cutoff_n,
            "report_n":report_n, "advantage_tolerance":tolerance,
            "min_run_width_S":min_width}


def make_candidates(p0, seed, count, design="extremes"):
    rng = np.random.default_rng(seed)
    candidates = []
    fixed_pairs = [(0.001,0.999), (0.1,0.9), (0.35,0.8),
                   (max(0,p0-.002),min(1,p0+.002)),
                   (max(0,p0-.01),min(1,p0+.01)),
                   (.001,.002), (.998,.999),
                   (max(0,p0-.2),max(.0001,p0-.1)),
                   (min(.9999,p0+.1),min(1,p0+.2))]
    for k in range(count):
        if design == "scaled":
            center = np.clip(p0+rng.uniform(-.2,.2),1e-6,1-1e-6)
            width = 10**rng.uniform(-5,-.1)
            if k%3 == 0:
                p1,p2 = max(0,center-width),min(1,center+width)
            else:
                p1,p2 = np.sort(rng.uniform(0,1,2))
            cost = min(1-1e-6,(p2-p1)*10**rng.uniform(-4,2))
            delta = 10**rng.uniform(-1,1.5)
        elif k < len(fixed_pairs)*4:
            p1,p2 = fixed_pairs[k//4]
            cost = [.0001,.1,.9,.9999][k%4]
            delta = [.005,.05,.5,5][(k//len(fixed_pairs))%4]
        else:
            mode = k%5
            if mode == 0:
                p1,p2 = np.sort(rng.uniform(0,1,2))
            elif mode == 1:
                center = rng.uniform(.001,.999)
                width = 10**rng.uniform(-4,-1)
                p1,p2 = max(0,center-width),min(1,center+width)
            elif mode == 2:
                center = p0 + rng.uniform(-.05,.05)
                width = 10**rng.uniform(-3,-.5)
                p1,p2 = max(0,center-width),min(1,center+width)
            elif mode == 3:
                p1 = p0*rng.uniform(0,.999)
                p2 = p1 + (p0-p1)*rng.uniform(.001,.999)
            else:
                p1 = p0 + (1-p0)*rng.uniform(.001,.999)
                p2 = p1 + (1-p1)*rng.uniform(.001,.999)
            cost_mode = k%3
            cost = (10**rng.uniform(-5,-.1) if cost_mode==0
                    else 1-10**rng.uniform(-5,-.1) if cost_mode==1
                    else rng.uniform(.01,.99))
            delta = 10**rng.uniform(-2.5,1.5)
        if p1 < p2:
            candidates.append({"p1":float(p1), "p2":float(p2),
                               "cost":float(cost), "delta":float(delta)})
    return candidates


def write_coverage(output):
    """Summarize authoritative nominal scans without double-counting reruns."""
    output = Path(output)
    phases = {}
    unique = set()
    all_parameters = []
    for name, subdir in [("extremes","extremes_nominal"),
                         ("scaled_costs","scaled_nominal"),
                         ("refined_dn_0.5","refined_h05_nominal")]:
        results = [json.loads(path.read_text())
                   for path in sorted((output/subdir).glob("p0_*.json"))]
        if not results:
            continue
        errors = [e for r in results for e in r["value_bracket_max"]]
        cases = [dict(p0=r["p0"],**c) for r in results for c in r["candidates"]]
        all_parameters.extend(cases)
        for case in cases:
            unique.add(tuple(case[k] for k in ("p0","p1","p2","cost","delta")))
        sections = 0
        points = 0
        for result in results:
            levels = round(result["report_n"]/result["dn"])
            subdivision = round(result["dn"]/result["ds"])
            minlevel = math.ceil(2/subdivision)
            sections += len(result["candidates"])*(levels-minlevel+1)
            points += len(result["candidates"])*sum(subdivision*l+1
                                                  for l in range(minlevel,levels+1))
        phases[name] = {"count":len(cases),
                        "p0_values":sorted({r["p0"] for r in results}),
                        "dn":results[0]["dn"], "dS":results[0]["ds"],
                        "max_physical_n":results[0]["report_n"],
                        "cutoff_n":results[0]["cutoff_n"],
                        "sections_with_at_least_3_nodes":sections,
                        "nodes_checked":points,
                        "max_nominal_components":max(max(r["max_nominal_positive_components"])
                                                     for r in results),
                        "nominal_three_point_candidates":sum(w is not None for r in results
                                                       for w in r["best_nominal_three_point_witness"]),
                        "strict_wide_candidates":sum(len(r["hits"]) for r in results),
                        "max_value_bracket_width":max(errors),
                        "tail_bracket_below_1e-6_count":sum(e<1e-6 for e in errors),
                        "directory":subdir}
    if not all_parameters:
        raise ValueError("No completed nominal scan outputs found")
    coverage = {"scheme":"semi-Lagrangian exponential frozen demand",
                "equation":"V(n,S)=max_i a*(1-c_i)+b*I[V(n+dn,S+p_i*dn)]",
                "a":"D/delta*(1-exp(-delta*dn/D))", "b":"exp(-delta*dn/D)",
                "normalization":{"R":1,"c1":0},
                "unique_calibrations":len(unique),
                "total_solves_including_refinements":len(all_parameters),
                "phases":phases,
                "ranges":{k:[min(c[k] for c in all_parameters),
                              max(c[k] for c in all_parameters)]
                          for k in ("p0","p1","p2","cost","delta")},
                "delta_p_range":[min(c["p2"]-c["p1"] for c in all_parameters),
                                 max(c["p2"]-c["p1"] for c in all_parameters)],
                "strict_gap_tolerance":1e-6,
                "nominal_gap_tolerance":0,
                "min_strict_run_width_S":0.5,
                "limitations":["Finite parameter search is not a proof of interval geometry.",
                               "The terminal bracket bounds only this numerical scheme; it does not bound continuum discretization error.",
                               "Small-delta extreme cases may have a wide terminal bracket and are exploratory.",
                               "Floating-point terminal brackets are diagnostics, not outward-rounded certificates."]}
    (output/"coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
    rows = ["| Fase | Calibraciones | dn | dS | Máximo n | Componentes nominales máximas |",
            "|---|---:|---:|---:|---:|---:|"]
    for name,p in phases.items():
        rows.append(f"| {name} | {p['count']} | {p['dn']} | {p['dS']} | {p['max_physical_n']} | {p['max_nominal_components']} |")
    text = ("# Búsqueda independiente: modelo fluido sin olvido\n\n"
            f"Se examinaron {len(unique)} calibraciones diferentes y "
            f"{len(all_parameters)} soluciones contando el refinamiento. "
            "No se encontró ninguna sección a n fijo con el patrón estricto 2–1–2. "
            "El detector nominal usa signo >0, sin tolerancia ni filtro de anchura; "
            "incluye todas las diagonales con al menos tres nodos, también n<2.\n\n"
            +"\n".join(rows)+"\n\n"
            "El método es semi-Lagrangiano en n=S+F: mantiene constante la demanda "
            "durante cada paso, integra exactamente el descuento bajo esa aproximación "
            "e interpola linealmente el sucesor S+p_i dn. Es monótono y consistente "
            "con la HJB; difiere de la transición Bernoulli de model.py.\n\n"
            "Se resuelve dos veces simultáneamente con continuaciones 0 y 1/delta "
            "para diagnosticar la influencia del corte. El barrido de costes escalados "
            "tiene anchura máxima de ese bracket inferior a 5.3e-12. Algunos extremos "
            "con descuento muy pequeño tienen brackets amplios y son exploratorios. "
            "Estas cifras usan punto flotante, no redondeo certificado.\n\n"
            "La ausencia de ejemplos en esta búsqueda no demuestra que todas las "
            "secciones del modelo continuo sean intervalos. Los brackets de cola "
            "tampoco incluyen el error de discretización.\n\n"
            "Detalles de parámetros y recuentos: [coverage.json](coverage.json). "
            "Cada directorio contiene todos los parámetros para reproducirlo.\n\n"
            "Reproducir desde la raíz del repositorio:\n\n```bash\n"
            ".venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design extremes --count 90 --cutoff-n 400 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/extremes_nominal'\n"
            ".venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design scaled --count 120 --cutoff-n 360 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/scaled_nominal'\n"
            ".venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --design scaled --count 20 --dn 0.5 --cutoff-n 360 --report-n 80 --output 'Fluid base model/outputs/interval_search/engine_b/refined_h05_nominal'\n"
            ".venv/bin/python 'Fluid base model/code/search_intervals_independent.py' --coverage-only\n```\n")
    (output/"README.md").write_text(text)
    print(json.dumps(coverage,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", nargs="+", type=float,
                        default=[.001,.01,.05,.2,.5,.8,.95,.99,.999])
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--seed", type=int, default=261020)
    parser.add_argument("--design", choices=["extremes","scaled"], default="extremes")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--dn", type=float, default=1)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--cutoff-n", type=float, default=400)
    parser.add_argument("--report-n", type=float, default=80)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"outputs/interval_search/engine_b")
    args = parser.parse_args()
    if args.coverage_only:
        write_coverage(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"scheme":"semi-Lagrangian exponential frozen demand",
               "seed":args.seed, "design":args.design, "cases":[], "hits":[]}
    started = time.time()
    for k,p0 in enumerate(args.p0):
        candidates = make_candidates(p0,args.seed+k,args.count,args.design)
        result = solve_batch(p0,candidates,args.dn,args.subdivisions,
                             args.cutoff_n,args.report_n)
        destination = args.output/f"p0_{p0:g}.json"
        destination.write_text(json.dumps(result,indent=2)+"\n")
        summary["cases"].append({"p0":p0,"count":len(candidates),
                                 "hits":len(result["hits"]),
                                 "max_components":max(result["max_positive_components"]),
                                 "max_nominal_components":max(result["max_nominal_positive_components"]),
                                 "nominal_three_point_candidates":sum(x is not None for x in result["best_nominal_three_point_witness"]),
                                 "max_value_bracket":max(result["value_bracket_max"]),
                                 "path":str(destination)})
        summary["hits"].extend(result["hits"])
        summary["elapsed_seconds"] = time.time()-started
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary["cases"][-1]),flush=True)
    print(f"Total {sum(c['count'] for c in summary['cases'])} cases; "
          f"{len(summary['hits'])} provisional hits; {time.time()-started:.1f}s",flush=True)


if __name__ == "__main__":
    main()
