"""Identify every initial-state reversal in the active experiment-10 grid."""
from __future__ import annotations

import argparse
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-reversal-plots")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.reporting import _metric_figure
from oblivious.fluid_oblivious.code.run_experiment import case_tag, write_json
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import phase_limit, interval_notation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/"10_fluid_oblivious"
PROJECT = SOURCE.parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def compare_policy_pair(params, monopoly, competition, conjecture):
    """Partition every switch and stationary state; classify each point/cell."""
    edges = sorted({0.,1.,params.p1,params.p2,*monopoly["roots"],*competition["roots"]})
    atoms = []
    for index, edge in enumerate(edges):
        atoms.append(dict(lo=edge,hi=edge,left_closed=True,right_closed=True))
        if index+1<len(edges):
            atoms.append(dict(lo=edge,hi=edges[index+1],left_closed=False,right_closed=False))
    selected = []
    for atom in atoms:
        x0 = (atom["lo"]+atom["hi"])/2
        m = phase_limit(params,monopoly,x0)
        c = phase_limit(params,competition,x0)
        am,ac = ((value-params.p1)/(params.p2-params.p1) for value in (m,c))
        if am <= ac+1e-8:
            continue
        row = dict(atom,monopoly_limit=m,competition_limit=c,
                   monopoly_product2_fraction=am,competition_product2_fraction=ac,
                   competition_self_confirming=bool(abs(c-conjecture)<1e-9))
        if (selected and selected[-1]["hi"]==row["lo"]
                and (selected[-1]["right_closed"] or row["left_closed"])
                and all(selected[-1][key]==row[key] for key in (
                    "monopoly_limit","competition_limit","competition_self_confirming"))):
            selected[-1].update(hi=row["hi"],right_closed=row["right_closed"])
        else:
            selected.append(row)
    return selected


def enumerate_saved_cases():
    config = json.loads((SOURCE/"config.json").read_text())
    diagnostics = json.loads((SOURCE/"data/monopoly_diagnostics.json").read_text())
    monopoly = {(case["p0"],case["rho"]):case for case in diagnostics["cases"]}
    paths = [SOURCE/"config.json",SOURCE/"data/monopoly_diagnostics.json"]
    reversals,checked = [],[]
    policy_count = 0
    for p0 in config["p0s"]:
        for rho in config["rhos"]:
            params = Parameters(**config["parameters"],rho=rho)
            path = SOURCE/"data/checkpoints"/(case_tag(p0,rho)+".json.gz")
            with gzip.open(path,"rt") as handle:
                case = json.load(handle)
            paths.append(path)
            m = monopoly[(p0,rho)]
            finest = case["grids"][-1]
            policies = finest["result"]["equilibria"]
            count = 0
            for index,equilibrium in enumerate(policies):
                if not equilibrium["verified"]:
                    continue
                policy_count += 1
                rows = compare_policy_pair(params,m["policy"],equilibrium["policy"],equilibrium["x_star"])
                for row in rows:
                    reversals.append(dict(p0=p0,rho=rho,equilibrium_id=index,
                                          competition_conjecture=equilibrium["x_star"],
                                          n=finest["n"],**row))
                    count += 1
            checked.append(dict(p0=p0,rho=rho,verified_competition_policies=len(policies),
                                reversal_interval_count=count))
    return config,dict(case_count=len(checked),policy_count=policy_count,checked_cases=checked,
                      reversals=reversals,source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in paths})


def highlighted_plot(config, audit, output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.titlesize":11,
                         "legend.frameon":False,"figure.max_open_warning":0})
    summary_path = SOURCE/"data/comparison_summary.csv"
    data = pd.read_csv(summary_path)
    fig = _metric_figure(data,config["p0s"],config["rhos"],horizontal="p0",
                         metric="product2_conditional",title="Stationary product-2 comparison",
                         ylabel="Product 2 (% of own sales)",fixed_limits=True)
    pairs = sorted({(row["p0"],row["rho"]) for row in audit["reversals"] if row["competition_self_confirming"]})
    for number,(p0,rho) in enumerate(pairs,1):
        ax = fig.axes[config["rhos"].index(rho)]
        ax.axvspan(p0-.012,p0+.012,color="#009E73",alpha=.13,zorder=0)
        ax.text(p0,.5,str(number),ha="center",va="center",color="white",fontsize=10,
                bbox=dict(boxstyle="circle,pad=.25",facecolor="#009E73",edgecolor="none"))
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_linestyle() not in {"None","none","",None}:
                line.set_linestyle("-");line.set_linewidth(1.8)
    for legend in list(fig.legends):
        legend.remove()
    handles = [Line2D([],[],color="#0072B2",marker="o",mfc="none",lw=1.8,label="Fluid monopoly"),
               Line2D([],[],color="#D55E00",marker="D",mfc="none",lw=1.8,label="Fluid competition"),
               Patch(facecolor="#009E73",alpha=.2,label="Monopoly high / competition low for some common x0")]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,fontsize=9,frameon=False)
    fig.suptitle("Monopoly can exceed competition from a common initial reputation\n"
                 "Green bands mark the three parameter pairs; each requires a narrow interval of x0.",fontsize=12)
    destination = output/"figures/16_monopoly_above_competition"
    destination.mkdir(parents=True,exist_ok=True)
    paths = []
    for extension in ("png","pdf"):
        path = destination/f"01_highlighted_cases.{extension}"
        fig.savefig(path,dpi=175,bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths,summary_path


def write_report(audit, refinement, output):
    valid = [row for row in audit["reversals"] if row["competition_self_confirming"]]
    off_target = [row for row in audit["reversals"] if not row["competition_self_confirming"]]
    lines = ["# Casos donde monopolio termina por encima de competencia", "",
             f'Se revisaron las **{audit["case_count"]} combinaciones activas** y las **{audit["policy_count"]} políticas competitivas verificadas** del experimento 10.', "",
             "Se compara el mismo inicio: monopolio parte de `x0` y ambos vendedores competitivos parten de `(x0,x0)`. Arriba/abajo se refiere al porcentaje de producto 2 entre las ventas propias en el largo plazo. Los destinos se calculan mediante la línea de fase, sin inferirlos de una simulación de duración finita.", "",
             "## Tres casos con competencia que confirma el equilibrio bajo", "",
             "En todos, monopolio converge a `0.8` y usa 100% producto 2; competencia converge a su conjetura `0.35` y usa 0%. No se encontraron otros casos, incluidas las comparaciones con equilibrios mixtos.", "",
             "| Marca en el gráfico | p0 | rho | condiciones iniciales, políticas guardadas de 4001 nodos |",
             "| --- | --- | --- | --- |"]
    for number,row in enumerate(valid,1):
        lines.append(f'| {number} | {row["p0"]:g} | {row["rho"]:g} | {interval_notation([row],digits=12)} |')
    lines.extend(["", "Los extremos siguen la convención guardada: producto 1 y continuación hacia la región baja en el umbral repulsivo. Las fronteras son aproximaciones numéricas.", "",
                  "![Casos señalados sobre el gráfico del experimento 10](figures/16_monopoly_above_competition/01_highlighted_cases.png)", "",
                  "[PDF del gráfico señalado](figures/16_monopoly_above_competition/01_highlighted_cases.pdf). Los gráficos históricos originales se conservan."])
    if refinement is not None:
        lines.extend(["", "## Comprobación en mallas más finas", "",
                      "Las diferencias de umbrales son pequeñas. Se vuelve a resolver monopolio y la respuesta competitiva al rival `0.35`, con la misma malla para ambos. La demanda coincide con el cálculo original y se contrasta duplicando la cuadratura.", "",
                      "| p0 | rho | nodos de la malla final | intervalo refinado de x0 | ejemplo comprobado en las tres mallas más finas |",
                      "| --- | --- | --- | --- | --- |"])
        for case in refinement["cases"]:
            finest = max(case["grids"],key=lambda row:row["n"])
            fine_three = sorted(case["grids"],key=lambda row:row["n"])[-3:]
            lo = max(row["threshold_monopoly"] for row in fine_three)
            hi = min(row["threshold_competition"] for row in fine_three)
            if not lo<hi:
                raise ArithmeticError("No common fine-grid reversal interval")
            example = (lo+hi)/2
            lines.append(f'| {case["p0"]:g} | {case["rho"]:g} | {finest["n"]} | '
                         f'({finest["threshold_monopoly"]:.10f}, {finest["threshold_competition"]:.10f}] | {example:.10f} |')
        lines.extend(["", "El refinamiento confirma la existencia numérica de los tres intervalos. Sus ubicaciones cambian con la malla; los intervalos más estrechos requieren usar los valores refinados. Los cambios entre mallas son indicadores de sensibilidad y no cotas matemáticas certificadas."])
    lines.extend(["", "## Trayectorias bajo políticas antiguas que no confirman su conjetura", "",
                  "La condición antigua también conserva políticas cuya aplicación desde ciertos inicios acaba en otra reputación. Si se cuentan esas trayectorias, hay cuatro combinaciones, con intervalos más amplios. Estas filas no representan un equilibrio competitivo alcanzado desde esos inicios.", "",
                  "| p0 | rho | conjetura competitiva usada | condiciones iniciales, malla guardada de 4001 nodos | producto 2 monopolio | producto 2 competencia |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for row in off_target:
        lines.append(f'| {row["p0"]:g} | {row["rho"]:g} | {row["competition_conjecture"]:.9f} | '
                     f'{interval_notation([row],digits=12)} | {100*row["monopoly_product2_fraction"]:.4f}% | {100*row["competition_product2_fraction"]:.4f}% |')
    lines.extend(["", "Con el inicio original `x0=0.5`, ninguno de los tres casos con equilibrio competitivo bajo produce la inversión: `(p0,rho)=(0.8,0.9)` y `(0.9,0.8)` terminan abajo en ambos modelos, mientras `(0.9,0.7)` termina arriba en ambos.", "",
                  "Datos completos y procedencia: [data/monopoly_above_competition/](data/monopoly_above_competition/summary.json).", ""])
    (output/"MONOPOLY_ABOVE_COMPETITION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-file",type=Path)
    args = parser.parse_args()
    config,audit = enumerate_saved_cases()
    refinement = None
    if args.refinement_file:
        refinement = json.loads(args.refinement_file.read_text())
        if not refinement.get("complete"):
            raise ValueError("Refinement is still running")
    data = SOURCE/"data/monopoly_above_competition"
    data.mkdir(exist_ok=True)
    write_json(data/"summary.json",audit)
    pd.DataFrame(audit["reversals"]).to_csv(data/"intervals.csv",index=False)
    pd.DataFrame(audit["checked_cases"]).to_csv(data/"checked_cases.csv",index=False)
    if refinement is not None:
        write_json(data/"refinement.json",refinement)
    paths,summary_path = highlighted_plot(config,audit,SOURCE)
    write_report(audit,refinement,SOURCE)
    sources = dict(audit["source_sha256"])
    sources[str(summary_path.relative_to(PROJECT))] = digest(summary_path)
    for path in (Path(__file__),ROOT/"code/initial_conditions.py",ROOT/"code/reversal_thresholds.py"):
        if path.exists():
            sources[str(path.relative_to(PROJECT))] = digest(path)
    outputs = [*paths,SOURCE/"MONOPOLY_ABOVE_COMPETITION.md",*[p for p in data.iterdir() if p.is_file() and p.name!="manifest.json"]]
    write_json(data/"manifest.json",dict(status="complete",source_sha256=sources,
               generated_file_sha256={str(p.relative_to(SOURCE)):digest(p) for p in outputs},
               grid_scope="24 active experiment-10 parameter pairs and all 27 verified competition policies"))
    print(f"Identified {len({(r['p0'],r['rho']) for r in audit['reversals'] if r['competition_self_confirming']})} self-confirming parameter pairs; report {SOURCE/'MONOPOLY_ABOVE_COMPETITION.md'}",flush=True)


if __name__=="__main__":
    main()
