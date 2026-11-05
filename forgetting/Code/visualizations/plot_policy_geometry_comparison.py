"""Render the saved dense policy-geometry experiment and write its report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd

from compare_policy_geometry import DEST, MODELS

BLUE, ORANGE = "#2468a0", "#d47a24"
COLORS = {"exact":BLUE, "logistic":ORANGE}
STYLES = {"exact":"-", "logistic":"--"}
NAMES = {"exact":"Exact Beta", "logistic":"Logistic"}


def setup(ax):
    ax.grid(axis="y", color="#e0e5ea", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["right","top"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#b4bdc7")
    ax.set_xlabel(r"Forgetting parameter $\rho$", labelpad=9)


def save(fig, destination, stem):
    for extension in ("png", "pdf"):
        fig.savefig(destination / "figures" / f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def frame(title, caption):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"width_ratios":[1.2,1]})
    fig.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.21, wspace=0.23)
    fig.suptitle(title + r"  |  $p_0=0.5$", y=0.963, fontsize=20)
    fig.text(0.52, 0.889, r"$R=1,\ c_1=0.05,\ c_2=0.65,\ p_1=0.35,\ p_2=0.80,\ \gamma=0.98$", ha="center", fontsize=12)
    fig.text(0.52, 0.054, caption, ha="center", fontsize=11)
    for ax in axes:
        setup(ax)
    return fig, axes


def model_legend(fig, extra=False):
    handles = [Line2D([0],[0],color=COLORS[m],ls=STYLES[m],lw=2.3,label=NAMES[m]) for m in MODELS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52,0.828), ncols=2, frameon=False, fontsize=12)


def root_limits(thresholds, kind, pad):
    roots = thresholds[f"rho_{kind}"].dropna().to_numpy()
    if not len(roots):
        return (0.98,0.9995)
    return max(0.5,float(roots.min())-pad), min(0.9995,float(roots.max())+pad)


def mark_roots(ax, thresholds, kind, zero_mark=False):
    for model in MODELS:
        root = thresholds.loc[model, f"rho_{kind}"]
        if pd.notna(root):
            ax.axvline(root, color=COLORS[model], lw=1, alpha=0.5, ls=":")
            if zero_mark:
                ax.scatter([root],[0],s=42,facecolor="white",edgecolor=COLORS[model],zorder=5)


def plot_boundaries(table, thresholds, destination):
    fig, axes = frame("Product-2 policy boundaries", "Endpoints describe strict-positive regions; zero-crossing endpoints are excluded. Blank curves mean an empty region.")
    handles = []
    for bound, color in (("xL",BLUE),("xH",ORANGE)):
        for model in MODELS:
            label = rf"$x_{{{'L' if bound=='xL' else 'H'}}}$ · {NAMES[model]}"
            handles.append(Line2D([0],[0],color=color,ls=STYLES[model],lw=2,label=label))
            for ax in axes:
                ax.plot(table.rho, table[f"{model}_{bound}"], color=color, ls=STYLES[model], lw=2.1)
    axes[0].set_xlim(0.5,0.9995)
    axes[0].set_ylim(-0.025,0.69)
    axes[0].set_title("Full scanned range", pad=12)
    axes[1].set_xlim(0.99,0.9995)
    axes[1].set_ylim(0.39,0.57)
    axes[1].set_title("Near extinction", pad=12)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    for ax in axes:
        ax.set_ylabel(r"Boundary state $x=(1-\rho)S$")
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(0.52,0.828),ncols=4,frameon=False,fontsize=11)
    save(fig,destination,"01_policy_boundaries")


def plot_scalar(table, thresholds, destination, *, field, kind, title, ylabel, stem, padding):
    roots_text = "    ".join(rf"$\rho_{{{kind}}}^{{\mathrm{{{model}}}}}={thresholds.loc[model, f'rho_{kind}']:.8f}$" for model in MODELS)
    fig, axes = frame(title, roots_text)
    model_legend(fig)
    lo, hi = root_limits(thresholds, kind, padding)
    for ax in axes:
        for model in MODELS:
            ax.plot(table.rho,table[f"{model}_{field}"],color=COLORS[model],ls=STYLES[model],lw=2.2)
        ax.axhline(0,color="#46505b",lw=1,ls=":")
        ax.set_ylabel(ylabel)
        mark_roots(ax,thresholds,kind,zero_mark=True)
    axes[0].set_xlim(0.5,0.9995)
    axes[0].set_title("Full scanned range",pad=12)
    axes[1].set_xlim(lo,hi)
    selected = table[(table.rho>=lo)&(table.rho<=hi)]
    values = np.concatenate([selected[f"{model}_{field}"].to_numpy() for model in MODELS]+[np.array([0.0])])
    span = max(float(np.ptp(values)),1e-6)
    axes[1].set_ylim(float(values.min())-0.1*span,float(values.max())+0.1*span)
    axes[1].set_title(rf"Refined threshold $\rho_{kind}$",pad=12)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.5f" if kind=="H" else "%.3f"))
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    save(fig,destination,stem)


def plot_components(table, thresholds, destination):
    fig, axes = frame("Number of connected positive regions", "Counts come from all detected components. Step lines connect numerical samples; dotted lines mark refined extinction thresholds.")
    model_legend(fig)
    for ax in axes:
        for model in MODELS:
            ax.step(table.rho,table[f"{model}_num_positive_components"],where="post",color=COLORS[model],ls=STYLES[model],lw=2.3)
        ax.set_ylabel("Number of positive components")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        max_count=max(table.exact_num_positive_components.max(),table.logistic_num_positive_components.max())
        ax.set_ylim(-0.08,max_count+0.15)
        mark_roots(ax,thresholds,"H")
    axes[0].set_xlim(0.5,0.9995)
    axes[0].set_title("Full scanned range",pad=12)
    axes[1].set_xlim(*root_limits(thresholds,"H",3e-5))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_title("Existence disagreement near extinction",pad=12)
    save(fig,destination,"04_positive_components")


def numerical_report(table, thresholds, manifest, convergence, destination):
    joint=(table.exact_num_positive_components==1)&(table.logistic_num_positive_components==1)
    comparison=table[["rho","exact_num_positive_components","logistic_num_positive_components","exact_xL","exact_xH","logistic_xL","logistic_xH"]].copy()
    comparison["both_single_component"]=joint
    comparison["abs_xL_difference"]=(comparison.exact_xL-comparison.logistic_xL).abs().where(joint)
    comparison["abs_xH_difference"]=(comparison.exact_xH-comparison.logistic_xH).abs().where(joint)
    comparison["existence_disagreement"]=(comparison.exact_num_positive_components>0)!=(comparison.logistic_num_positive_components>0)
    comparison["connected_vs_multiple_disagreement"]=((comparison.exact_num_positive_components==1)&(comparison.logistic_num_positive_components>1))|((comparison.logistic_num_positive_components==1)&(comparison.exact_num_positive_components>1))
    comparison.to_csv(destination/"data/boundary_comparison.csv",index=False)
    comparison[comparison.existence_disagreement].to_csv(destination/"data/existence_disagreements.csv",index=False)
    final_checks=convergence.sort_values("fine_nodes").groupby(["rho","model"]).tail(1)
    assert final_checks.stable.all() and len(final_checks)==2*len(table)
    stats=dict(sampled_rho_count=len(table), sampled_domain=[float(table.rho.min()),float(table.rho.max())],
        exact_max_positive_components=int(table.exact_num_positive_components.max()),
        logistic_max_positive_components=int(table.logistic_num_positive_components.max()),
        connected_vs_multiple_disagreements=int(comparison.connected_vs_multiple_disagreement.sum()),
        existence_disagreements=int(comparison.existence_disagreement.sum()),
        maximum_final_root_movement=float(final_checks.root_movement.max()),
        all_final_geometry_checks_passed=bool(final_checks.stable.all()),
        maximum_bellman_residual=max(float(table[f"{model}_bellman_residual"].max()) for model in MODELS))
    for endpoint in ("xL","xH"):
        column=f"abs_{endpoint}_difference"
        index=comparison[column].idxmax()
        stats[f"maximum_{endpoint}_difference"]=float(comparison.loc[index,column])
        stats[f"rho_at_maximum_{endpoint}_difference"]=float(comparison.loc[index,"rho"])
        stats[f"RMSE_{endpoint}_difference_over_sampled_joint_region"]=float(np.sqrt(np.nanmean(comparison[column]**2)))
    for kind in ("B","H"):
        stats[f"rho_{kind}_logistic_minus_exact"]=float(thresholds.loc["logistic",f"rho_{kind}"]-thresholds.loc["exact",f"rho_{kind}"])
    for model in MODELS:
        structures=table[f"{model}_structure"].tolist()
        sequence=[s for i,s in enumerate(structures) if i==0 or s!=structures[i-1]]
        stats[f"{model}_observed_sequence"]=sequence
    (destination/"comparison_metrics.json").write_text(json.dumps(stats,indent=2)+"\n")
    lines=["# Geometría de la política: demanda Beta y logística", "",
        f"Se compararon {len(table)} valores de rho entre 0.5 y 0.9995: {manifest['base_rho_count']} de la malla inicial y {manifest['additional_rho_count']} adicionales cerca de los umbrales. "
        "El paso inicial es 0.001, con paso 0.0001 desde 0.98. Los umbrales se refinan mediante búsqueda de raíces con intervalos de cambio de signo y comparación entre mallas de estados.", "",
        "Parámetros: R=1, c1=0.05, c2=0.65, p1=0.35, p2=0.80, gamma=0.98, p0=0.5. "
        "Sólo cambia la demanda. Se utiliza el mismo solver Bellman, interpolación lineal, valor inicial cero y tolerancia 1e-12 en ambos modelos. "
        "La demanda exacta usa scipy.stats.beta.sf y la logística la lambda indicada, calculada con expit.", "",
        "## Umbrales estimados", "", "| Modelo | rho_B | rho_H |", "|---|---:|---:|"]
    for model in MODELS:
        lines.append(f"| {NAMES[model]} | {thresholds.loc[model,'rho_B']:.10f} | {thresholds.loc[model,'rho_H']:.10f} |")
    lines += ["",f"Diferencia logística menos Beta: rho_B = {stats['rho_B_logistic_minus_exact']:.10f}; rho_H = {stats['rho_H_logistic_minus_exact']:.10f}.", "",
        "La tabla data/critical_rates.csv incluye los intervalos de signo verificados, la resolución final en x y el cambio de cada estimación al refinar esa resolución. "
        "data/critical_events.json conserva las evaluaciones de ambos modelos realizadas durante cada búsqueda.", "",
        "## Conectividad y secuencia", "",
        f"Máximo número de componentes positivos observado: Beta {stats['exact_max_positive_components']}; logística {stats['logistic_max_positive_components']}.", "",
        "Secuencia Beta: " + " → ".join(stats["exact_observed_sequence"]) + ".", "",
        "Secuencia logística: " + " → ".join(stats["logistic_observed_sequence"]) + ".", "",
        "left-anchored significa que la región incluye x=0; interior excluye ambos extremos; empty significa que ningún estado tiene ventaja estrictamente positiva. "
        "Todos los ceros y componentes se detectan sobre los nudos de la ventaja numérica, incluidos los nudos desplazados por las dos transiciones. No se impone un número de intervalos.", "",
        f"Casos con una región conexa en un modelo y varias componentes en el otro: {stats['connected_vs_multiple_disagreements']}.", "",
        "Estas observaciones son evidencia numérica sobre la malla y el rango indicados. No demuestran conectividad para cada rho real mayor o igual que 0.5 ni para valores fuera del rango estudiado.", "",
        "## Distancias entre fronteras", "",
        "Las comparaciones de xL y xH se realizan sólo cuando ambos modelos tienen una única región positiva. "
        "Cuando uno es vacío no se le asignan fronteras artificiales; si hubiera varias componentes, se conservarían todas sin resumirlas como un solo intervalo.", "",
        f"Máxima diferencia absoluta de xL: {stats['maximum_xL_difference']:.8f}, en rho={stats['rho_at_maximum_xL_difference']:.10f}.", "",
        f"Máxima diferencia absoluta de xH: {stats['maximum_xH_difference']:.8f}, en rho={stats['rho_at_maximum_xH_difference']:.10f}.", "",
        "data/boundary_comparison.csv contiene las distancias para cada rho. Los RMSE en comparison_metrics.json ponderan por igual cada punto muestreado, por lo que la zona refinada tiene mayor peso.", "",
        "## Desacuerdo sobre la existencia de producto 2", "",
        f"Se encontraron {stats['existence_disagreements']} valores muestreados donde sólo uno de los modelos tiene una región positiva. "
        "Se enumeran en data/existence_disagreements.csv.", ""]
    if comparison.existence_disagreement.any():
        samples=comparison[comparison.existence_disagreement]
        lines.append(f"Los casos detectados van de rho={samples.rho.min():.10f} a rho={samples.rho.max():.10f}. "
            "La diferencia entre los umbrales de extinción delimita el intervalo de desacuerdo estimado; los extremos son numéricos, no cotas analíticas.")
    lines += ["", "## Validación", "",
        "Cada valor de rho se resuelve inicialmente con 5001 puntos y se compara con 10001; cerca de cambios de signo, raíces próximas a cero o varias componentes se exige al menos 20001. "
        "Se sigue refinando si cambian los números de raíces o componentes, la inclusión de x=0, la existencia de región positiva o si los ceros se desplazan más de 2e-5.", "",
        "Los umbrales rho_B y rho_H se buscan desde 20001 puntos y se repiten con 40001, ampliando la malla si el cambio en rho supera 5e-8. "
        "La precisión de la búsqueda de raíces en rho y el error de discretización en x se registran por separado.", "",
        f"Máximo desplazamiento de un cero en la última comparación de mallas: {stats['maximum_final_root_movement']:.3e}. "
        f"Máximo residual de Bellman en las filas de política: {stats['maximum_bellman_residual']:.3e}. "
        "Los controles de malla son comprobaciones de estabilidad, no certificados rigurosos del error continuo.", "",
        "## Archivos", "",
        "- figures/01_policy_boundaries: xL y xH, rango completo y detalle de extinción.",
        "- figures/02_maximum_advantage: máximo de ventaja, con detalle de rho_H.",
        "- figures/03_advantage_at_zero: ventaja en x=0, con detalle de rho_B.",
        "- figures/04_positive_components: número de componentes, con detalle del desacuerdo de existencia.",
        "- data/policy_summary.csv: todas las columnas solicitadas, todos los ceros y todos los intervalos; una fila por rho.",
        "- data/critical_rates.csv: resumen de rho_B y rho_H y su estabilidad.",
        "- data/all_policy_regions.json: intervalos con inclusión o exclusión explícita de extremos.",
        "- data/grid_convergence.csv: historial de refinamiento de cada caso.",
        "", "Las figuras se guardan en PNG y PDF vectorial. Los extremos con ventaja cero no pertenecen a la región estrictamente positiva. "
        "En policy_summary.csv, touches_zero significa a(0)>0. xL/xH son vacíos cuando no hay una única componente, y all_components conserva la geometría completa.", "",
        "Reproducir:", "", "```bash", "uv run python -B forgetting/Code/visualizations/compare_policy_geometry.py --workers 4", "```", "",
        "Regenerar sólo las figuras y el informe:", "", "```bash", "uv run python -B forgetting/Code/visualizations/plot_policy_geometry_comparison.py", "```", ""]
    (destination/"report.md").write_text("\n".join(lines))
    print(json.dumps(stats,indent=2),flush=True)


def generate(destination=DEST):
    table=pd.read_csv(destination/"data/policy_summary.csv").sort_values("rho")
    thresholds=pd.read_csv(destination/"data/critical_rates.csv").set_index("model")
    manifest=json.loads((destination/"manifest.json").read_text())
    convergence=pd.read_csv(destination/"data/grid_convergence.csv")
    plt.rcParams.update({"font.size":11,"axes.titlesize":14,"axes.labelsize":12})
    plot_boundaries(table,thresholds,destination)
    plot_scalar(table,thresholds,destination,field="max_advantage",kind="H",title="Maximum product-2 advantage",
        ylabel=r"$A_\rho=\max_x a_\rho(x)$",stem="02_maximum_advantage",padding=0.0004)
    plot_scalar(table,thresholds,destination,field="advantage_at_zero",kind="B",title="Advantage at the lower boundary",
        ylabel=r"$a_\rho(0)$",stem="03_advantage_at_zero",padding=0.015)
    plot_components(table,thresholds,destination)
    numerical_report(table,thresholds,manifest,convergence,destination)
    print(f"Saved four figures (PNG/PDF) and report to {destination}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEST)
    generate(parser.parse_args().output_dir)
