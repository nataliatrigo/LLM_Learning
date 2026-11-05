"""Calendar Monte Carlo comparison of saved experiment-08 policies at rho=.7."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oe08-monte-carlo-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import MODELS
from oblivious.oe_distribution_fixed_p0.code.mc_fixed_p0_engine import simulate_calendar_paths
from oblivious.oe_distribution_fixed_p0.code.mc_fixed_p0_inputs import load_cases, exact_reference


P0S = [.1, .3, .5, .7, .8, .9]
RHO = .7
METRICS = ["own_sale_rate", "own_product2_rate", "product2_given_sale", "customer_product2_rate"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+"\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def case_name(case):
    return "p0_"+str(case["p0"]).replace(".", "p")+"_"+case["model"]


def validate_monte_carlo(periods, exact, n_paths):
    rows = []
    for period in [0, 1, 9, 24, 49, 99]:
        if period >= len(periods) or period >= len(exact):
            continue
        observed, expected = periods.iloc[period], exact.iloc[period]
        for metric, count_field in [
            ("own_sale_rate", "own_sale_count"),
            ("own_product2_rate", "own_product2_count"),
            ("customer_product2_rate", "customer_product2_count"),
            ("product2_given_sale", "own_product2_count"),
        ]:
            denominator = int(observed.own_sale_count) if metric == "product2_given_sale" else n_paths
            probability = float(expected[metric])
            if denominator == 0 or not np.isfinite(probability):
                rows.append(dict(period=period, metric=metric, count=int(observed[count_field]),
                                 trials=denominator, expected_probability=probability,
                                 p_value=np.nan, passed=True, scope="No observed conditioning events"))
                continue
            if not -1e-12 <= probability <= 1+1e-12:
                raise ValueError("Exact transient produced an invalid probability")
            probability = min(1., max(0., probability))
            p_value = float(binomtest(int(observed[count_field]), denominator, probability).pvalue)
            rows.append(dict(period=period, metric=metric, count=int(observed[count_field]),
                             trials=denominator, expected_probability=probability,
                             monte_carlo_probability=float(observed[count_field])/denominator,
                             p_value=p_value, passed=p_value >= 1e-8,
                             scope="Exact binomial check at selected dates; stringent multiple-check threshold"))
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise ValueError("Monte Carlo differs from the exact finite-grid transient: " +
                         result[~result.passed].to_json(orient="records"))
    return result


def style(ax):
    ax.set(xlabel="Calendar period", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def caption(fig, title, config):
    fig.suptitle(title + "\n" +
        rf"$\rho=0.7$, $c_2=0.65$ | {config['paths']:,} paths | saved n=101 policies | common mature start $x=0.5$" +
        f"\n{config['bin_width']}-period blocks; no burn-in removed; grid interpolation after observed outcomes.", fontsize=11)


def render(data, figures, manifest):
    figures.mkdir(parents=True, exist_ok=True)
    config = manifest["configuration"]
    stationary = pd.DataFrame([case["stationary"] for case in manifest["completed"]])
    lookup = {(float(row.p0), row.model): row for row in stationary.itertuples()}
    frames = {(p0, model): pd.read_csv(data/("p0_"+str(p0).replace(".", "p")+"_"+model)/"binned.csv")
              for p0 in config["p0s"] for model, *_ in MODELS}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.frameon": False})
    outputs = []
    pdf_path = figures/"monte_carlo_rho_0p7_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        def save(fig, name):
            png = figures/(name+".png")
            fig.savefig(png, dpi=190, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            outputs.append(png)
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True, layout="constrained")
        colors = plt.colormaps["viridis"](np.linspace(.05, .9, len(config["p0s"])))
        for column, (model, label, *_) in enumerate(MODELS):
            for p0, color in zip(config["p0s"], colors, strict=True):
                frame = frames[p0, model]
                for row, metric in enumerate(["own_sale_rate", "own_product2_rate"]):
                    axes[row, column].plot(frame.bin_center+1, frame[metric], color=color, lw=1.5,
                                           label=rf"$p_0={p0:g}$")
            axes[0, column].set_title(label)
            for ax in axes[:, column]:
                style(ax)
                ax.set_xlim(1, config["periods"])
        axes[0, 0].set_ylabel("Seller A demand\n(% of calendar periods)")
        axes[1, 0].set_ylabel("Seller A selected AND product 2\n(% of calendar periods)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(config["p0s"]))
        caption(fig, "Monte Carlo over time: monopoly, marginal OE and conditional OE", config)
        save(fig, "01_monopoly_style_demand_and_product2")

        for metric, prefix, title, ylabel in [
            ("product2_given_sale", "02_primary_product2_given_sale",
             "Primary metric over time: product 2 among own sales", "Product 2 (% of own sales)"),
            ("customer_product2_rate", "03_complementary_customer_product2",
             "Complementary metric over time: product 2 received by the customer", "Customer receives product 2\n(% of calendar periods)"),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
            for ax, p0 in zip(axes.flat, config["p0s"], strict=True):
                for model, label, color, linestyle, marker in MODELS:
                    frame = frames[p0, model]
                    ax.plot(frame.bin_center+1, frame[metric], color=color, ls=linestyle, lw=1.6, label=label)
                    ax.fill_between(frame.bin_center+1, frame[metric+"_ci_lower"], frame[metric+"_ci_upper"],
                                    color=color, alpha=.12, linewidth=0)
                    level = float(getattr(lookup[p0, model], metric))
                    ax.axhline(level, color=color, ls=":", lw=.9, alpha=.8)
                style(ax)
                ax.set(title=rf"$p_0={p0:g}$", ylabel=ylabel, xlim=(1, config["periods"]))
            handles, labels = axes.flat[0].get_legend_handles_labels()
            handles.append(Line2D([], [], color="#555555", ls=":"))
            labels.append("Saved stationary reference")
            fig.legend(handles, labels, loc="outside lower center", ncol=4)
            caption(fig, title+"\nShading: nominal 95% path-cluster intervals; dotted lines are stationary references.", config)
            save(fig, prefix)
    outputs.append(pdf_path)
    manifest["figure_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in outputs}
    manifest["rendering_source_sha256"] = digest(__file__)
    write_json(data/"manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=20000)
    parser.add_argument("--periods", type=int, default=3000)
    parser.add_argument("--bin-width", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--name", default="monte_carlo_rho_0p7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in [".", ".."]:
        parser.error("Use a simple output folder name")
    data, figures = HOME/"data"/args.name, HOME/"figures"/args.name
    if args.render_only:
        manifest = json.loads((data/"manifest.json").read_text())
        render(data, figures, manifest)
        return
    config = dict(rho=RHO, p0s=P0S, n=101, paths=args.paths, periods=args.periods,
                  bin_width=args.bin_width, seed=args.seed, initial_mature_reputation=.5,
                  exact_reference_periods=min(100, args.periods), burn_in_periods=0,
                  parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98),
                  common_random_numbers="Same four-channel seed across p0 values and models",
                  numerical_model="Saved finite-grid game with interpolation lotteries after own sales")
    if data.exists() and not args.resume:
        raise ValueError("Output already exists; use --resume or a fresh name")
    cases, provenance = load_cases(P0S, rho=RHO, n=101)
    data.mkdir(parents=True, exist_ok=True)
    if args.resume and (data/"manifest.json").exists():
        manifest = json.loads((data/"manifest.json").read_text())
        if manifest["configuration"] != config:
            raise ValueError("Resume configuration differs")
    else:
        manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                        configuration=config, numerical_inputs=provenance, completed=[])
        (data/"provenance").mkdir(exist_ok=True)
        for name in ["monte_carlo_rho_0p7.py", "mc_fixed_p0_engine.py", "mc_fixed_p0_inputs.py"]:
            source = Path(__file__).with_name(name)
            shutil.copyfile(source, data/"provenance"/name)
        manifest["executed_source_sha256"] = {name: digest(data/"provenance"/name) for name in
                                              ["monte_carlo_rho_0p7.py", "mc_fixed_p0_engine.py", "mc_fixed_p0_inputs.py"]}
    write_json(data/"configuration.json", config)
    done = {case["name"] for case in manifest["completed"]}
    summary_rows = []
    for case in cases:
        name = case_name(case)
        if name in done:
            record = next(r for r in manifest["completed"] if r["name"] == name)
            for filename, expected in record["data_sha256"].items():
                if digest(data/name/filename) != expected:
                    raise ValueError("Cached simulation data changed: "+name+"/"+filename)
            continue
        start = time.monotonic()
        print("START", name, flush=True)
        folder = data/name
        folder.mkdir(exist_ok=True)
        params = case["params"]
        if any(getattr(params, field) != value for field, value in config["parameters"].items()):
            raise ValueError("Saved economic calibration differs")
        result = simulate_calendar_paths(grid=case["grid"], policy=case["policy"], selection=case["selection"],
            rho=RHO, p1=params.p1, p2=params.p2, n_paths=args.paths, periods=args.periods,
            seed=args.seed, bin_width=args.bin_width, initial_index=len(case["grid"])//2)
        exact = exact_reference(case, periods=config["exact_reference_periods"])
        if isinstance(exact, dict):
            exact = exact["by_period"]
        validation = validate_monte_carlo(result["by_period"], exact, args.paths)
        tables = {"by_period": result["by_period"], "binned": result["binned"], "paths": result["paths"],
                  "exact_first100": exact, "monte_carlo_checks": validation}
        for filename, frame in tables.items():
            frame.to_csv(folder/(filename+".csv"), index=False)
        saved = case["outcome"]
        sellers = case["selection"].shape[0]-1
        stationary = dict(p0=case["p0"], rho=RHO, model=case["model"],
            own_sale_rate=saved["sale_rate"], own_product2_rate=saved["product2_calendar"],
            product2_given_sale=saved["product2_conditional"], customer_product2_rate=sellers*saved["product2_calendar"])
        record = dict(name=name, p0=case["p0"], model=case["model"], stationary=stationary,
                      seconds=time.monotonic()-start, diagnostics=result["diagnostics"],
                      exact_binomial_checks_passed=bool(validation.passed.all()),
                      data_sha256={filename+".csv": digest(folder/(filename+".csv")) for filename in tables})
        manifest["completed"].append(record)
        write_json(data/"manifest.json", manifest)
        print("DONE", name, f"{record['seconds']:.1f}s", flush=True)
    for record in manifest["completed"]:
        period = pd.read_csv(data/record["name"]/"by_period.csv")
        for first in [0, max(0, args.periods-1000)]:
            window = period[period.period >= first]
            sale_count, p2_count = window.own_sale_count.sum(), window.own_product2_count.sum()
            trials = len(window)*args.paths
            summary_rows.append(dict(p0=record["p0"], rho=RHO, model=record["model"],
                first_period=first+1, last_period=args.periods,
                own_sale_rate=sale_count/trials, own_product2_rate=p2_count/trials,
                product2_given_sale=p2_count/sale_count if sale_count else np.nan,
                customer_product2_rate=window.customer_product2_count.sum()/trials))
    pd.DataFrame(summary_rows).to_csv(data/"summary.csv", index=False)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    scope="Finite-horizon Monte Carlo from a common mature state, not a claim that 3000 periods reach stationarity",
                    initialization="Unlike the old monopoly paper plot, these saved equilibrium policies start at mature x=.5, not zero observations")
    write_json(data/"manifest.json", manifest)
    render(data, figures, manifest)
    (figures/"README.md").write_text("\n".join([
        "# Monte Carlo: rho=0.7", "",
        "Comparación de monopolio, OE marginal y OE condicional con c2=0.65 y los seis p0 del experimento 08.",
        "20,000 trayectorias por caso y 3,000 períodos; cada seller comienza en la misma reputación madura x=0.5.",
        "Se simulan las políticas auditadas de 101 nodos sin resolver nuevos equilibrios. La elección del customer usa las reputaciones reales de ambos sellers y la opción externa.",
        "Solo se actualiza el seller que vende; comprar a la opción externa deja ambos estados iguales.",
        "La lotería entre nodos vecinos reproduce exactamente la transición del modelo en grilla.", "",
        "**Inicio:** no son las políticas de aprendizaje desde cero observaciones del antiguo Monte Carlo del monopolio. Se usa el mismo estado maduro inicial en los tres modelos para comparar los equilibrios guardados.", "",
        "- [Formato del monopolio: demanda propia y producto 2 propio por período](01_monopoly_style_demand_and_product2.png).",
        "- [Métrica principal: producto 2 entre ventas propias](02_primary_product2_given_sale.png).",
        "- [Complementaria: producto 2 recibido por el customer por período](03_complementary_customer_product2.png).",
        "- [Los tres gráficos en PDF](monte_carlo_rho_0p7_comparison.pdf).", "",
        "Los gráficos usan bloques de 25 períodos. Las bandas de las dos métricas muestran intervalos nominales del 95% por trayectoria; las líneas punteadas son las referencias estacionarias guardadas.",
        "El cociente condicional de cada bloque divide el total de ventas de producto 2 por el total de ventas propias.",
        "El flujo del customer cuenta efectivamente las entregas de A más las de B; no duplica por aproximación la realización de A.",
        "Se contrastan fechas seleccionadas de los primeros 100 períodos con propagación determinista exacta del mismo kernel.",
        "Los datos por período, por bloque y por trayectoria, las referencias exactas y los chequeos están en ../../data/"+args.name+"/.",
        "Las curvas son transitorias: no se declara estacionariedad por observar una meseta o por simular 3,000 períodos.", ""]))
    print("COMPLETE", figures, flush=True)


if __name__ == "__main__":
    main()
