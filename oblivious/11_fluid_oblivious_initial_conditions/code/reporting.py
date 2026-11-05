"""Experiment-10 presentation for equilibria and sets of initial conditions."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-oblivious-initial-conditions-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from oblivious.fluid_oblivious.code.reporting import _subplots, _style, _legend, _rho_axis
from oblivious.fluid_oblivious_initial_conditions.code.comparison_figures import POLICY_COLORS, POLICY_LINEWIDTH, comparison_figures
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import contains


COMPETITION_COLOR = "#D55E00"
CURVE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#332288", "#666666")


def _tag(value):
    return f"{value:g}".replace(".", "p")


def _equilibria(case):
    return sorted((eq for eq in case["equilibria"] if eq["verified"]), key=lambda eq: eq["x_bar"])


def _case_lookup(cases):
    return {(float(case["p0"]), float(case["rho"])):case for case in cases}


def _initial_set_overview(cases, p0s, rhos):
    lookup = _case_lookup(cases)
    fig, axes = _subplots(len(p0s))
    for ax, p0 in zip(axes, p0s, strict=True):
        for position, rho in enumerate(rhos):
            for branch, eq in enumerate(_equilibria(lookup[(p0, rho)])):
                color = CURVE_COLORS[branch % len(CURVE_COLORS)]
                for interval in eq["initial_conditions"]:
                    lo, hi = interval["lo"], interval["hi"]
                    ax.plot([position, position], [lo, hi], color=color, ls="-", lw=2.)
                    for endpoint, included in ((lo, interval["left_closed"]), (hi, interval["right_closed"])):
                        ax.plot(position, endpoint, marker="o", ms=5,
                                mfc=color if included else "white",
                                mec=color, linestyle="none", zorder=4)
                ax.plot(position, eq["x_bar"], marker="D", ms=5, mfc="none",
                        mec=color, linestyle="none", zorder=5)
        ax.set_title(rf"$p_0={p0:g}$", loc="left")
        ax.set(ylabel=r"Admissible common initial reputation $x_0$", ylim=(-.025,1.025),
               yticks=np.arange(0., 1.01, .2))
        _rho_axis(ax, rhos)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30); tick.set_horizontalalignment("right")
        _style(ax)
    handles = [Line2D([], [], color=CURVE_COLORS[0], ls="-", lw=2., label="Lower or only equilibrium"),
               Line2D([], [], color=CURVE_COLORS[1], ls="-", lw=2., label="Higher equilibrium, when present"),
               Line2D([], [], color=COMPETITION_COLOR, marker="o", mfc=COMPETITION_COLOR,
                      ls="none", ms=5, label="Included endpoint"),
               Line2D([], [], color=COMPETITION_COLOR, marker="o", mfc="white", ls="none",
                      ms=5, label="Excluded endpoint"),
               Line2D([], [], color=COMPETITION_COLOR, marker="D", mfc="none", ls="none",
                      ms=5, label=r"Consistent reputation $\bar x$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Equilibrium sets of initial conditions\n"
                 "Each interval contains the common initial reputations that converge to its conjecture.", fontsize=12)
    return fig


def _coverage_overview(cases, p0s, rhos):
    lookup = _case_lookup(cases)
    fig, axes = _subplots(len(p0s))
    maximum = 1
    for ax, p0 in zip(axes, p0s, strict=True):
        for ordinal, rho in enumerate(rhos):
            eqs = _equilibria(lookup[(p0, rho)])
            edges = sorted({0., 1., *(x for eq in eqs for interval in eq["initial_conditions"]
                                      for x in (interval["lo"],interval["hi"]))})
            samples = []
            for left, right in zip(edges, edges[1:]):
                count = sum(contains(eq["initial_conditions"], (left+right)/2) for eq in eqs)
                maximum = max(maximum, count)
                samples.extend([(left,count),(right,count),(np.nan,np.nan)])
            xx, yy = zip(*samples)
            color = CURVE_COLORS[ordinal % len(CURVE_COLORS)]
            ax.plot(xx, yy, color=color, ls="-", lw=1.5+.9*(len(rhos)-ordinal-1),
                    label=rf"$\rho={rho:g}$", zorder=2+ordinal)
            for point in edges:
                count = sum(contains(eq["initial_conditions"], point) for eq in eqs)
                maximum = max(maximum, count)
                ax.scatter(point, count, s=16, facecolors="none", edgecolors=color, zorder=8+ordinal)
        ax.set_title(rf"$p_0={p0:g}$", loc="left")
        ax.set(xlabel=r"Common initial reputation $x_0$", ylabel="Number of consistent equilibria", xlim=(0,1))
        _style(ax)
    for ax in axes:
        ax.set(ylim=(-.05,maximum+.05), yticks=range(maximum+1))
    _legend(fig, axes)
    fig.suptitle("Equilibria by common initial reputation\n"
                 "The complete correspondence is computed from the sets of initial conditions.", fontsize=12)
    return fig


def _detail(cases, p0, rhos):
    lookup = _case_lookup(cases)
    fig, axes = _subplots(len(rhos))
    for ax, rho in zip(axes, rhos, strict=True):
        eqs = _equilibria(lookup[(p0, rho)])
        for branch, eq in enumerate(eqs):
            policy = eq["policy"]
            edges = [0., *policy["roots"], 1.]
            color = POLICY_COLORS[branch]
            label = rf"$\rho={rho:g}$, $\bar x={eq['x_bar']:.6f}$"
            xx = np.repeat(edges, 2)[1:-1]
            yy = np.repeat([float(product==2) for product in policy["products"]], 2)
            ax.plot(xx,yy,color=color,ls="-",lw=POLICY_LINEWIDTH,
                    label=label,zorder=2+branch)
            ax.scatter(eq["x_bar"],eq["alpha"],marker="D",s=30,facecolors="none",
                       edgecolors=color,linewidths=1.4,zorder=8+branch)
            description = rf"$\bar x={eq['x_bar']:.6f}$; $x_0\in$ "+eq["initial_conditions_notation"]
            ax.text(.035,.53-.13*branch,description,transform=ax.transAxes,fontsize=8.3,color=color,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=2))
        ax.set(title=rf"$\rho={rho:g}$",xlabel="Own normalized reputation x",
               ylabel=r"$\sigma(x)$: product 2 probability",xlim=(0,1),ylim=(-.03,1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); _style(ax)
    _legend(fig,axes)
    fig.suptitle(rf"Fluid competition: optimal policies and initial conditions | outside option $p_0={p0:g}$"+
                 "\nLegend: persistence and equilibrium reputation; diamonds: stationary product mix.",fontsize=12)
    return fig


def make_figures(cases, output_dir, *, p0s, rhos, parameters):
    """Use the original experiment-10 renderers and add the new initial sets."""
    cases = list(cases)
    p0s, rhos = list(map(float,p0s)), list(map(float,rhos))
    lookup = _case_lookup(cases)
    if len(lookup)!=len(cases) or set(lookup)!={(p0,rho) for p0 in p0s for rho in rhos}:
        raise ValueError("Require one case for every requested p0/rho pair")
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    old_manifest = root/"report_manifest.json"
    previous = json.loads(old_manifest.read_text()) if old_manifest.exists() else {}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"figure.max_open_warning":0,
                         "axes.titlesize":11,"legend.frameon":False})
    figures,comparison = comparison_figures(cases,root,p0s=p0s,rhos=rhos,parameters=parameters)
    figures.extend([
        ("initial_conditions","01_equilibrium_initial_conditions",_initial_set_overview(cases,p0s,rhos)),
        ("initial_conditions","02_equilibrium_multiplicity",_coverage_overview(cases,p0s,rhos)),
    ])
    figures.extend(("policies_and_initial_conditions",f"policies_initial_conditions_p0_{_tag(p0)}",
                    _detail(cases,p0,rhos)) for p0 in p0s)
    (root/"overview").mkdir(exist_ok=True)
    combined_path = root/"overview/equilibria_and_initial_conditions.pdf"
    comparison_pdf = root/"overview/fluid_monopoly_vs_competition.pdf"
    paths = []
    with PdfPages(combined_path) as combined, PdfPages(comparison_pdf) as original:
        for index,(folder,stem,fig) in enumerate(figures):
            destination = root/folder
            destination.mkdir(parents=True,exist_ok=True)
            for extension in ("png","pdf"):
                path = destination/f"{stem}.{extension}"
                fig.savefig(path,dpi=175,bbox_inches="tight")
                paths.append(path)
            combined.savefig(fig,bbox_inches="tight")
            if index < comparison["figure_count"]:
                original.savefig(fig,bbox_inches="tight")
            plt.close(fig)
    paths.extend([combined_path,comparison_pdf])
    report = dict(status="complete",case_count=len(cases),
                  equilibrium_count=sum(len(_equilibria(case)) for case in cases),
                  p0s=p0s,rhos=rhos,figure_count=len(figures),format_reference="Experiment 10; original renderers reused",
                  comparison=comparison,initial_condition_method="Complete phase-line sets, not a sampled time horizon",
                  reporting_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  generated_file_sha256={str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest() for path in paths})
    for name,checksum in previous.get("generated_file_sha256",{}).items():
        old = root/name
        if name not in report["generated_file_sha256"] and old.is_file() and sha256(old.read_bytes()).hexdigest()==checksum:
            old.unlink()
    old_manifest.write_text(json.dumps(report,indent=2)+"\n")
    policy_links = [f"- [All policies and initial conditions: p0={p0:g}](policies_and_initial_conditions/policies_initial_conditions_p0_{_tag(p0)}.pdf)." for p0 in p0s]
    (root/"README.md").write_text(
        "# Equilibrios y conjuntos de condiciones iniciales\n\n"
        "Se conservan los paneles, la tipografía y los ejes del experimento 10. "
        "Las curvas son sólidas y usan colores diferentes. En las comparaciones, el monopolio es azul "
        "y la competencia naranja. En las figuras de políticas se usan sólo tres colores: azul para "
        "monopolio, naranja para el equilibrio único o inferior, y verde para el superior cuando hay dos. "
        "La leyenda al pie indica explícitamente rho y x_bar. Los conjuntos de condiciones "
        "iniciales se muestran dentro de cada panel. "
        "Todas las curvas de políticas tienen el mismo grosor.\n\n"
        "Las métricas y trayectorias comparativas usan la condición inicial común x0=0.5. "
        "Las figuras de 11_policies y de conjuntos de condiciones iniciales incluyen las 27 políticas "
        "verificadas, con todos los equilibrios de cada caso.\n\n"
        "- [Todas las figuras](overview/equilibria_and_initial_conditions.pdf).\n"
        "- [Comparación en el formato del experimento 10](overview/fluid_monopoly_vs_competition.pdf).\n"
        "- [Producto 2 entre ventas propias](overview/02_primary_product2_given_sale_by_rho.png).\n"
        "- [Conjuntos de condiciones iniciales](initial_conditions/01_equilibrium_initial_conditions.png).\n"
        "- [Número de equilibrios por condición inicial](initial_conditions/02_equilibrium_multiplicity.png).\n\n"
        +"\n".join(policy_links)+"\n\n"
        "Los extremos llenos están incluidos y los vacíos están excluidos. Los rombos huecos indican "
        "la reputación consistente. Los umbrales son aproximaciones numéricas y la inclusión de una "
        "frontera repulsiva sigue la convención de continuación del experimento.\n")
    return paths,report
