"""Fresh independent searches for the four uncovered initial-state cases.

Run from the repository root with the experiment's stable Python alias.
Original experiment outputs and plot selections are read only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.model import FixedRivalDemand, solve_response
from oblivious.fluid_oblivious.code.run_experiment import case_tag, numerical_sources, write_json
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import initial_conditions_for_equilibrium, interval_notation
from oblivious.fluid_oblivious_initial_conditions.code.independent_response import (
    excursion_then_hold_deviation, stationary_holding_value,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fresh_direct_scan(task):
    """Solve the HJB at every conjecture, without any FOC candidate filter."""
    config, p0, rho, points, n, order, folder = task
    params = Parameters(**config["parameters"], rho=rho)
    engine = FixedRivalDemand(params, p0, np.linspace(0, 1, n), order)
    rows = []
    cache = {}

    def evaluate(z):
        key = float(z)
        if key not in cache:
            response = solve_response(params, p0, key, n, order, prepared=engine)
            holding = stationary_holding_value(params, p0, key)["value"]
            gain = float(np.interp(key, response["x"], response["value"])-holding)
            cache[key] = dict(z=key, diagonal_gap=float(np.interp(key, response["x"], response["gap"])),
                              value_over_holding=gain, hjb_residual=response["residual"],
                              policy_roots=response["roots"], root_kinds=response["root_kinds"])
        return cache[key]

    grid = np.linspace(params.p1, params.p2, points)
    for index, z in enumerate(grid):
        rows.append(evaluate(z))
        if index % 100 == 0:
            print(f"Direct HJB scan p0={p0:g}, rho={rho:g}: {index+1}/{points}", flush=True)
    crossings = []
    for first, second in zip(rows, rows[1:]):
        if first["diagonal_gap"]*second["diagonal_gap"] < 0:
            z = brentq(lambda value:evaluate(value)["diagonal_gap"], first["z"], second["z"], xtol=1e-11)
            crossings.append(dict(evaluate(z), bracket=[first["z"],second["z"]]))
    result = dict(p0=p0, rho=rho, grid_nodes=n, quadrature_order=order,
                  conjecture_count=points, hjb_solves=len(cache), rows=rows, crossings=crossings,
                  maximum_hjb_residual=max(row["hjb_residual"] for row in cache.values()),
                  method="Fresh HJB for every rival conjecture; diagonal gap crossings refined independently of FOC")
    write_json(Path(folder)/(case_tag(p0,rho)+"_direct_scan.json"), result)
    return result


def fine_candidate_checks(config, p0, rho, candidates, grids, order):
    """Re-solve pure and mixed necessary candidates and refine policy consistency."""
    params = Parameters(**config["parameters"], rho=rho)
    lam, delta = -math.log(rho), -math.log(params.gamma)
    k = (params.c2-params.c1)/(lam*(params.p2-params.p1))
    records = []
    for n in grids:
        engine = FixedRivalDemand(params,p0,np.linspace(0,1,n),order)
        cache = {}

        def response(z):
            key = float(z)
            if key not in cache:
                cache[key] = solve_response(params,p0,key,n,order,prepared=engine)
            return cache[key]

        h = 1/(n-1)
        for index, candidate in enumerate(candidates):
            seed = float(candidate["z"])
            z = seed
            rr = response(z)
            refined = False
            if candidate["kind"] == "mixed":
                attractive = [r for r,kind in zip(rr["roots"],rr["root_kinds"]) if kind == "attractive"]
                if attractive and min(abs(root-z) for root in attractive) < .03:
                    def consistency(value):
                        result = response(value)
                        roots = [r for r,kind in zip(result["roots"],result["root_kinds"]) if kind == "attractive"]
                        return min(roots,key=lambda root:abs(root-seed))-value if roots else float("nan")
                    radius = max(.003,16*h)
                    lo,hi = max(params.p1+1e-10,z-radius),min(params.p2-1e-10,z+radius)
                    fl,fh = consistency(lo),consistency(hi)
                    if np.isfinite(fl) and np.isfinite(fh) and fl*fh < 0:
                        z = float(brentq(consistency,lo,hi,xtol=1e-11))
                        rr = response(z)
                        refined = True
            holding = stationary_holding_value(params,p0,z)
            value = float(np.interp(z,rr["x"],rr["value"]))
            gain = value-holding["value"]
            conditional = delta*gain/max(holding["demand"],np.finfo(float).tiny)
            gap = float(np.interp(z,rr["x"],rr["gap"]))
            distance = min((abs(root-z) for root in rr["roots"]),default=1.)
            if candidate["kind"] == "mixed":
                action = distance < 1e-7 and abs(gap) <= max(2e-5,8*h*(1+abs(k))*lam)
            elif candidate["kind"] == "pure_product1":
                action = gap <= 1e-8
            else:
                action = gap >= -1e-8
            value_tolerance = max(2e-6,50*h*h*(1+abs(k))*(1+abs(value)))
            conditional_tolerance = max(2e-5,10*h*h*(1+params.c2-params.c1+lam*abs(k)))
            valid = bool(action and abs(gain)<=value_tolerance and abs(conditional)<=conditional_tolerance
                         and rr["residual"]<=1e-8)
            policy = {key:rr[key] for key in ("roots","products","root_kinds","intervals","holding_points")}
            if valid and candidate["kind"] == "mixed":
                root_index = min(range(len(policy["roots"])),key=lambda i:abs(policy["roots"][i]-z))
                old_root = policy["roots"][root_index]
                policy["roots"][root_index] = z
                edges = [0.,*policy["roots"],1.]
                policy["intervals"] = [dict(lo=lo,hi=hi,product=product)
                                        for lo,hi,product in zip(edges,edges[1:],policy["products"])]
                policy["holding_points"] = [dict(x=z,alpha=holding["alpha"]) if abs(point["x"]-old_root)<1e-9
                                             else point for point in policy["holding_points"]]
                if not any(abs(point["x"]-z)<1e-12 for point in policy["holding_points"]):
                    policy["holding_points"].append(dict(x=z,alpha=holding["alpha"]))
            initial = initial_conditions_for_equilibrium(params,policy,z) if valid else None
            records.append(dict(p0=p0,rho=rho,candidate_id=index,kind=candidate["kind"],n=n,
                                analytic_seed=seed,x_bar=z,root_consistency_refined=refined,
                                holding_value=holding["value"],optimized_value=value,
                                holding_value_gap=gain,conditional_holding_defect=conditional,
                                hamiltonian_gap=gap,policy_root_distance=distance,
                                hjb_residual=rr["residual"],verified=valid,
                                initial_conditions=initial["intervals"] if initial else [],
                                initial_conditions_notation=initial["text"] if initial else "∅",
                                policy=policy if valid else None))
        print(f"Fresh candidate HJB checks p0={p0:g}, rho={rho:g}, n={n}: {len(cache)} solves",flush=True)
        for row in records:
            if row["n"] == n:
                row["grid_hjb_solves"] = len(cache)
    return records


def rejection_witness(config, p0, rho, row):
    params = Parameters(**config["parameters"],rho=rho)
    z = row["analytic_seed"]
    if row["kind"] == "pure_product1":
        target = .75
    elif row["kind"] == "pure_product2":
        target = .79
    else:
        target = z+(.1 if (p0,rho)!=(.9,.8) else -.1)
    holding = stationary_holding_value(params,p0,z)["value"]
    coarse = excursion_then_hold_deviation(params,p0,z,target,order=256,rtol=1e-10,atol=1e-12,max_step=10.)
    fine = excursion_then_hold_deviation(params,p0,z,target,order=512,rtol=1e-12,atol=1e-14,max_step=5.)
    change = abs(fine["value"]-coarse["value"])
    gain = fine["value"]-holding
    observed_margin = min(fine["value"],coarse["value"])-holding-max(1e-9,20*change)
    if observed_margin <= 0:
        raise ArithmeticError(f"No robust departure witness for {(p0,rho,z)}")
    return dict(p0=p0,rho=rho,z=z,kind=row["kind"],target=target,
                holding_value=holding,deviation_value=fine["value"],gain=gain,
                refinement_change=change,conservative_observed_margin=observed_margin,
                coarse=coarse,fine=fine,
                scope="Explicit feasible profitable departure; refinement margin is numerical, not interval-certified")


def write_report(results, output, protocol):
    lines = ["# Comprobación independiente de los intervalos sin equilibrio", "",
             "Se volvió a buscar en los cuatro casos con condiciones iniciales sin cubrir, sin utilizar los candidatos guardados del experimento 10.", "",
             "## Qué se comprobó", "",
             f'- HJB nuevo para {protocol["scan_points"]} conjeturas por caso en todo `[p1,p2]`, con {protocol["scan_nodes"]} nodos de estado.',
             '- Enumeración independiente de candidatos interiores mediante cuadratura Legendre, derivada analítica, refinamiento de la malla y búsqueda de extremos. También se cubren todos los intervalos de conjeturas con cotas analíticas de derivadas, condicionadas a los presupuestos numéricos de error de cuadratura.',
             f'- Verificación global de holding y consistencia en mallas nuevas {protocol["grids"]}.',
             '- Cada candidato rechazado tiene una desviación factible, calculada sin el HJB, que da un beneficio mayor que mantener la reputación propuesta.', "",
             "## Resultado por caso", "",
             "| p0 | rho | reputaciones de equilibrio, malla más fina | condiciones iniciales sin cubrir, malla más fina |",
             "| --- | --- | --- | --- |"]
    for result in results:
        accepted = result["accepted_finest"]
        lines.append(f'| {result["p0"]:g} | {result["rho"]:g} | '+", ".join(f'{row["x_bar"]:.9f}' for row in accepted)
                     +" | "+interval_notation(result["uncovered_finest"])+" |")
    lines.extend(["", "## Candidatos adicionales descartados", "",
                  "La última columna es la mejora de beneficio de una estrategia explícita: usar un producto puro hasta otra reputación y después mantenerla.", "",
                  "| p0 | rho | candidato x_bar | reputación de destino de la desviación | mejora de beneficio |",
                  "| --- | --- | --- | --- | --- |"])
    for result in results:
        for witness in result["rejection_witnesses"]:
            lines.append(f'| {result["p0"]:g} | {result["rho"]:g} | {witness["z"]:.9f} | {witness["target"]:.9f} | {witness["gain"]:.9f} |')
    lines.extend(["", "## Alcance de la conclusión", "",
                  "La nueva búsqueda no encontró equilibrios adicionales que cubran los huecos. Los candidatos descartados fallan optimalidad incluso al compararlos con estrategias factibles sencillas.", "",
                  "La exhaustividad de los candidatos tiene una comprobación numérica refinada. Las cotas de cuadratura y del HJB no son cotas certificadas por aritmética intervalar, por lo que este informe no se presenta como una prueba matemática formal de inexistencia en el modelo continuo.", "",
                  "Las fronteras de condiciones iniciales son aproximadas: se informa su cambio al refinar la malla. La inclusión exacta de los umbrales repulsivos conserva la convención de continuación del experimento.", "",
                  "El argumento de necesidad permite incluir candidatos aunque el valor óptimo no sea suave: ver [derivación](code/EXISTENCE_ARGUMENT.md). Los datos de la comprobación están en [data/existence_check/](data/existence_check/summary.json).", ""])
    (output/"EXISTENCE_CHECK.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-points",type=int,default=401)
    parser.add_argument("--scan-nodes",type=int,default=1001)
    parser.add_argument("--grids",type=int,nargs="+",default=[4001,8001,16001])
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--scan-only",action="store_true")
    parser.add_argument("--resume-scans",action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT/"config.json").read_text())
    old_cases = json.loads((ROOT/"data/equilibria.json").read_text())["cases"]
    targets = [(case["p0"],case["rho"]) for case in old_cases if case["uncovered_initial_states"]]
    folder = ROOT/"data/existence_check"
    folder.mkdir(exist_ok=True)
    tasks = [(config,p0,rho,args.scan_points,args.scan_nodes,128,str(folder)) for p0,rho in targets]
    scans = {}
    pending = []
    for task in tasks:
        path = folder/(case_tag(task[1],task[2])+"_direct_scan.json")
        if args.resume_scans and path.exists():
            saved = json.loads(path.read_text())
            if saved["conjecture_count"]==args.scan_points and saved["grid_nodes"]==args.scan_nodes:
                scans[(task[1],task[2])] = saved
                continue
        pending.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fresh_direct_scan,task) for task in pending]):
            scan = future.result()
            scans[(scan["p0"],scan["rho"])] = scan
    if args.scan_only:
        return
    from oblivious.fluid_oblivious_initial_conditions.code.root_certificate import enumerate_stationary_candidates
    from oblivious.fluid_oblivious_initial_conditions.code.run_experiment import correspondence
    results = []
    for p0,rho in targets:
        params = Parameters(**config["parameters"],rho=rho)
        roots = enumerate_stationary_candidates(params,p0,scan_points=4097,orders=(512,1024))
        if not roots["numerical_passed"]:
            raise ArithmeticError(f"Independent candidate enumeration failed for {(p0,rho)}")
        write_json(folder/(case_tag(p0,rho)+"_root_audit.json"),roots)
        candidates = [dict(z=params.p1,kind="pure_product1"),dict(z=params.p2,kind="pure_product2")]
        candidates.extend(dict(z=row["z"],kind="mixed") for row in roots["interior_roots"])
        checks = fine_candidate_checks(config,p0,rho,candidates,args.grids,256)
        finest = [row for row in checks if row["n"]==args.grids[-1]]
        accepted = [row for row in finest if row["verified"]]
        witnesses = [rejection_witness(config,p0,rho,row) for row in finest if not row["verified"]]
        coarse_accepted = {row["candidate_id"]:row for row in checks if row["n"]==args.grids[-2] and row["verified"]}
        refinement = []
        for row in accepted:
            previous = coarse_accepted.get(row["candidate_id"])
            if previous is None:
                raise ArithmeticError("An accepted candidate did not persist on the previous grid")
            old,new = previous["initial_conditions"],row["initial_conditions"]
            topology = len(old)==len(new) and all(
                first["left_closed"]==second["left_closed"] and first["right_closed"]==second["right_closed"]
                for first,second in zip(old,new))
            change = max((abs(first[key]-second[key]) for first,second in zip(old,new)
                          for key in ("lo","hi")),default=0.) if topology else None
            refinement.append(dict(candidate_id=row["candidate_id"],x_bar_change=abs(previous["x_bar"]-row["x_bar"]),
                                   same_initial_set_topology=topology,boundary_change=change))
            if not topology or change>config["initial_conditions_refinement_absolute"]:
                raise ArithmeticError("Fresh initial conditions failed grid refinement")
        if len(coarse_accepted)!=len(accepted):
            raise ArithmeticError("Fresh accepted candidate count changed on the finest pair")
        adapted = [dict(equilibrium_id=index,verified=True,initial_conditions=row["initial_conditions"])
                   for index,row in enumerate(accepted)]
        uncovered = [part for part in correspondence(adapted) if part["equilibrium_count"]==0]
        record = dict(p0=p0,rho=rho,direct_scan=scans[(p0,rho)],root_audit=roots,
                      candidate_refinements=checks,accepted_finest=accepted,
                      rejection_witnesses=witnesses,uncovered_finest=uncovered,finest_pair_refinement=refinement)
        write_json(folder/(case_tag(p0,rho)+"_verification.json"),record)
        results.append(record)
        print(f"Independent verification p0={p0:g}, rho={rho:g}: {len(accepted)} accepted; {len(witnesses)} explicit rejections",flush=True)
    protocol = dict(parameters=config["parameters"],scan_points=args.scan_points,scan_nodes=args.scan_nodes,scan_order=128,
                    grids=args.grids,candidate_order=256,root_scan_points=[4097,8193],root_orders=[512,1024])
    summary = dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
                   case_count=len(results),results=results,
                   numerical_source_sha256=numerical_sources(),
                   source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sorted((ROOT/"code").glob("*.py"))},
                   scope="Fresh independent numerical existence check, including explicit rejection witnesses; not a formal interval proof")
    write_json(folder/"summary.json",summary)
    rows = [dict(p0=case["p0"],rho=case["rho"],**{k:v for k,v in row.items()
                                              if k not in {"p0","rho","policy","initial_conditions"}})
            for case in results for row in case["candidate_refinements"]]
    pd.DataFrame(rows).to_csv(folder/"candidate_refinements.csv",index=False)
    write_report(results,ROOT,protocol)
    write_json(folder/"manifest.json",dict(status="complete",protocol=protocol,
               source_sha256=summary["source_sha256"],numerical_source_sha256=numerical_sources(),
               generated_file_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(folder.iterdir())
                                      if p.is_file() and p.name!="manifest.json"}))
    print(f"Independent existence check complete: {ROOT/'EXISTENCE_CHECK.md'}",flush=True)


if __name__ == "__main__":
    main()
