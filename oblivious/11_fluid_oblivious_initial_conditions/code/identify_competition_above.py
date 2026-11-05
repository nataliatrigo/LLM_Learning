"""Check the reverse long-run comparison for every saved experiment-10 policy."""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.run_experiment import case_tag, write_json
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import phase_limit, interval_notation

SOURCE = Path(__file__).resolve().parents[2]/"10_fluid_oblivious"
PROJECT = SOURCE.parents[1]


def compare(params, monopoly, competition, conjecture):
    edges = sorted({0.,1.,params.p1,params.p2,*monopoly["roots"],*competition["roots"]})
    atoms = []
    for index,point in enumerate(edges):
        atoms.append(dict(lo=point,hi=point,left_closed=True,right_closed=True))
        if index+1<len(edges):
            atoms.append(dict(lo=point,hi=edges[index+1],left_closed=False,right_closed=False))
    result = []
    for atom in atoms:
        x0 = (atom["lo"]+atom["hi"])/2
        m = phase_limit(params,monopoly,x0)
        c = phase_limit(params,competition,x0)
        am,ac = [(value-params.p1)/(params.p2-params.p1) for value in (m,c)]
        if ac <= am+1e-8:
            continue
        row = dict(atom,monopoly_limit=m,competition_limit=c,
                   monopoly_product2_fraction=am,competition_product2_fraction=ac,
                   competition_self_confirming=bool(abs(c-conjecture)<1e-9),
                   competition_100_monopoly_0=bool(ac>1-1e-8 and am<1e-8))
        if (result and result[-1]["hi"]==row["lo"]
                and (result[-1]["right_closed"] or row["left_closed"])
                and all(result[-1][key]==row[key] for key in (
                    "monopoly_limit","competition_limit","competition_self_confirming"))):
            result[-1].update(hi=row["hi"],right_closed=row["right_closed"])
        else:
            result.append(row)
    return result


def main():
    config_path = SOURCE/"config.json"
    monopoly_path = SOURCE/"data/monopoly_diagnostics.json"
    config = json.loads(config_path.read_text())
    monop = {(row["p0"],row["rho"]):row["policy"] for row in json.loads(monopoly_path.read_text())["cases"]}
    sources = [config_path,monopoly_path,Path(__file__),Path(__file__).with_name("initial_conditions.py")]
    rows,checked = [],[]
    policy_count = 0
    for p0 in config["p0s"]:
        for rho in config["rhos"]:
            params = Parameters(**config["parameters"],rho=rho)
            checkpoint = SOURCE/"data/checkpoints"/(case_tag(p0,rho)+".json.gz")
            sources.append(checkpoint)
            with gzip.open(checkpoint,"rt") as handle:
                saved = json.load(handle)
            finest = saved["grids"][-1]
            count = 0
            for index,eq in enumerate(finest["result"]["equilibria"]):
                if not eq["verified"]:
                    continue
                policy_count += 1
                found = compare(params,monop[(p0,rho)],eq["policy"],eq["x_star"])
                rows.extend(dict(p0=p0,rho=rho,equilibrium_id=index,n=finest["n"],
                                 competition_conjecture=eq["x_star"],**row) for row in found)
                count += len(found)
            checked.append(dict(p0=p0,rho=rho,competition_above_interval_count=count))
    valid = [row for row in rows if row["competition_self_confirming"]]
    extreme = [row for row in valid if row["competition_100_monopoly_0"]]
    summary = dict(case_count=len(checked),policy_count=policy_count,
                   competition_above_pairs=len({(row["p0"],row["rho"]) for row in valid}),
                   competition_100_monopoly_0_pairs=len({(row["p0"],row["rho"]) for row in extreme}),
                   intervals=rows,checked_cases=checked,
                   method="Exact phase-line partition of all switch points and pure stationary reputations; boundaries evaluated separately")
    folder = SOURCE/"data/competition_above_monopoly"
    folder.mkdir(exist_ok=True)
    write_json(folder/"summary.json",summary)
    pd.DataFrame(rows).to_csv(folder/"intervals.csv",index=False)
    pd.DataFrame(checked).to_csv(folder/"checked_cases.csv",index=False)
    lines = ["# Competencia por encima de monopolio desde el mismo inicio", "",
             f'Se revisaron {len(checked)} combinaciones y {policy_count} políticas competitivas del experimento 10.', "",
             "## Competencia 100% y monopolio 0%", "",
             "No aparece ningún caso en la grilla activa, ni usando otras políticas guardadas cuya trayectoria no confirme su conjetura.", "",
             "## Competencia con una proporción mayor de producto 2", "",
             f'Sí aparece en {summary["competition_above_pairs"]} combinaciones. En todas las filas siguientes la competencia alcanza su propia conjetura; varios resultados son mixtos, no 100% frente a 0%.', "",
             "| p0 | rho | condiciones iniciales x0 | producto 2 monopolio | producto 2 competencia |",
             "| --- | --- | --- | --- | --- |"]
    for row in valid:
        lines.append(f'| {row["p0"]:g} | {row["rho"]:g} | {interval_notation([row],digits=12)} | '
                     f'{100*row["monopoly_product2_fraction"]:.4f}% | {100*row["competition_product2_fraction"]:.4f}% |')
    lines.extend(["", "Se comparan destinos asintóticos de las políticas guardadas de 4001 nodos, con idéntico inicio en monopolio y en ambos vendedores competitivos. Las fronteras siguen el selector guardado y son aproximaciones numéricas. Las reputaciones finales determinan la proporción de producto 2 mediante `(x-p1)/(p2-p1)`.", "",
                  "Datos y revisión de todos los casos: [summary.json](data/competition_above_monopoly/summary.json).", ""])
    report = SOURCE/"COMPETITION_ABOVE_MONOPOLY.md"
    report.write_text("\n".join(lines))
    generated = [report,*[path for path in folder.iterdir() if path.is_file() and path.name!="manifest.json"]]
    write_json(folder/"manifest.json",dict(status="complete",
               source_sha256={str(path.relative_to(PROJECT)):sha256(path.read_bytes()).hexdigest() for path in sources},
               generated_file_sha256={str(path.relative_to(SOURCE)):sha256(path.read_bytes()).hexdigest() for path in generated}))
    print(f'Checked reverse direction: {summary["competition_above_pairs"]} parameter pairs with higher competition use; '
          f'{summary["competition_100_monopoly_0_pairs"]} with competition100%/monopoly0%.',flush=True)


if __name__=="__main__":
    main()
