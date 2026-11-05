"""Run true-quality consistency without modifying the existing experiments.

Example: python -m oblivious.oe_following_monopoly.code.run_experiment --name quality_sales_n51
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import (
    case_tag, load_bundle, save_bundle, select_candidate, now, write_json, write_csv,
)
from .model import QualityExperiment, LABEL
from .diagnostics import run_diagnostics, stationary_predictions


ROOT = Path(__file__).resolve().parents[3]
RHOS = [.6, .7, .8, .9, .95, .97, .99]


def select_mixed_equilibrium(case):
    """Use the same certified mixed-strategy selection as the OE p0 model."""
    candidates = [e for e in case["candidates"]
                  if e.get("candidate_verified") and e.get("verified")
                  and e.get("dp_diagnostics", {}).get("verified")
                  and e.get("stationary_diagnostics", {}).get("verified")
                  and e.get("monopoly_certificate", {}).get("verified")]
    if not candidates:
        diagnostic = case.get("selected")
        if diagnostic is not None and diagnostic.get("candidate_verified"):
            diagnostic = dict(diagnostic, candidate_verified=False)
        return diagnostic, "No verified candidate; diagnostic only"
    return select_candidate(candidates), "Mixed-strategy equilibrium selection: certified interior candidate when available; lowest p0 within that group; pure endpoints remain admissible"


def numerical_hashes():
    paths = [ROOT/"oblivious/03_oe_following_monopoly/code"/name for name in ("model.py", "event_chain.py", "diagnostics.py")]
    paths += [ROOT/"oblivious/02_oe_p0/code"/name for name in ("core.py", "game.py", "policy.py", "primitives.py")]
    paths += [ROOT/"forgetting/Code/model.py", ROOT/"oblivious/06_equilibrium_comparison/code/simulation.py",
              ROOT/"oblivious/06_equilibrium_comparison/code/cross_rho.py", ROOT/"oblivious/06_equilibrium_comparison/code/analysis.py"]
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="quality_sales_n51")
    parser.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    parser.add_argument("--grids", nargs="+", type=int, default=[51])
    parser.set_defaults(weighting="purchase")
    parser.add_argument("--p0-points", type=int, default=41)
    parser.add_argument("--consistency-tol", type=float, default=1e-7)
    parser.add_argument("--order", type=int, default=128)
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--initial-p0", type=float, default=.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--solve-only", action="store_true")
    for key, value in asdict(Parameters()).items():
        if key != "rho":
            parser.add_argument("--"+key, type=float, default=value)
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ("", ".", "..")
            or len(set(args.rhos)) != len(args.rhos) or len(set(args.grids)) != len(args.grids)
            or any(not .5 < rho < 1 for rho in args.rhos)
            or any(n < 3 or n % 2 != 1 for n in args.grids)
            or args.p0_points < 3 or args.replications < 2 or args.periods < 1 or args.burn_in < 0
            or not 0 < args.initial_p0 < 1 or args.order < 8 or not 0 < args.consistency_tol < .01):
        parser.error("Require a valid name, distinct rates, odd grids >=3, and positive numerical settings")
    return args


def export_tables(cases, reports, directory):
    summary, candidates, history, outcomes = [], [], [], []
    for case in sorted(cases, key=lambda c:(c["rho"], c["n"])):
        e = case["selected"]
        summary.append(dict(rho=case["rho"], n=case["n"],
            verified=bool(e and e.get("candidate_verified")),
            candidates=len(case["candidates"]),
            p0=None if e is None else e["p0"], F=None if e is None else e["F"],
            residual=None if e is None else abs(e["G"]),
            mixed_nodes=None if e is None else e["mixed_nodes"],
            pure_selector_iteration_status=case["iteration_status"],selection_reason=case.get("selection_reason")))
        for index, candidate in enumerate(case["candidates"]):
            prediction = case["candidate_predictions"][index]
            candidates.append(dict(rho=case["rho"], n=case["n"], candidate=index,
                p0=candidate["p0"], F=candidate["F"], residual=abs(candidate["G"]),
                verified=candidate["candidate_verified"], mixed_nodes=candidate["mixed_nodes"],
                **prediction))
        for attempt in case["iteration_attempts"]:
            for row in attempt["rows"]:
                history.append(dict(rho=case["rho"], n=case["n"], status=attempt["status"],
                                    scope="Auxiliary pure-selector diagnostic; not mixed-equilibrium iterations",
                                    **{k:v for k,v in row.items() if k != "policy_signature"}))
    for report in reports:
        for metric, estimate in report["simulation"]["summary"].items():
            outcomes.append(dict(rho=report["rho"], n=report["n"], metric=metric,
                stationary=report["predicted"].get(metric), **estimate))
    for name, rows in (("summary",summary), ("candidates",candidates),
                       ("iteration_history",history), ("simulation_summary",outcomes)):
        write_csv(directory/(name+".csv"), rows)


def write_findings(cases, directory, config):
    lines = ["OE quality: calidad media entre las ventas del rival", "",
        "El vendedor resuelve el problema de monopolio original de forgetting para un p0 fijo.",
        "Los dos vendedores juegan después con demanda TS real e independiente.",
        "Consistencia: p0 = suma(nu_ij D_B(i,j) q(x_B)) / suma(nu_ij D_B(i,j)).",
        "q(x)=p1+(p2-p1)alpha(x); se promedia sobre ventas, no sobre todos los períodos.",
        "La verificación final usa la conjunta real, optimalidad mixta y el Bellman de forgetting.",
        "La búsqueda rápida usa una cadena exacta en tiempos de venta; no reemplaza la verificación conjunta.",
        "El equilibrio permite alpha(x) entre 0 y 1; una mezcla interior exige indiferencia verificada.",
        "Selección como en OE p0: se prefiere un candidato interior verificado; entre alternativas, menor p0.",
        "Las políticas puras son extremos admisibles del mismo espacio de estrategias mixtas.",
        "La actualización con selector puro es un diagnóstico auxiliar y no selecciona el equilibrio mixto.",
        "Sus ciclos no describen una iteración del equilibrio mixto ni demuestran que este no exista.",
        "Candidatos verificados en la grilla; no se demuestra unicidad, exhaustividad, MPE ni convergencia continua.",
        "", "rho | n | p0 | error | candidatos | diagnóstico con selector puro"]
    for case in sorted(cases,key=lambda c:(c["rho"],c["n"])):
        e=case["selected"]
        lines.append(f"{case['rho']:g} | {case['n']} | "+
            (f"{e['p0']:.9f} | {abs(e['G']):.3g}" if e else "sin resultado | —")+
            f" | {len(case['candidates'])} | {case['iteration_status']}")
    lines += ["",f"Simulaciones: {config['replications']} trayectorias independientes, "+
              f"{config['periods']} períodos retenidos y {config['burn_in']} de burn-in, inicio (0.5,0.5).",
              "Los intervalos MC no incluyen sesgo de inicialización; rho altos pueden mezclar lentamente.",
              "Los experimentos existentes conservan sus resultados."]
    (directory/"RESULTADOS.txt").write_text("\n".join(lines)+"\n")


def main():
    args = arguments()
    data=ROOT/"oblivious/03_oe_following_monopoly/data"/args.name
    figures=ROOT/"oblivious/03_oe_following_monopoly/figures"/args.name
    path=data/"manifest.json"
    if args.resume or args.plots_only:
        manifest=json.loads(path.read_text())
        config=manifest["configuration"]
        if not args.plots_only and numerical_hashes()!=manifest["numerical_source_sha256"]:
            raise SystemExit("Numerical sources changed; use a new run name")
    else:
        if data.exists() or figures.exists():
            raise SystemExit("Existing results protected; use --resume, --plots-only or a new name")
        config={k:v for k,v in vars(args).items() if k not in ("resume","plots_only","solve_only")}
        data.mkdir(parents=True)
        (data/"arrays").mkdir()
        manifest=dict(status="incomplete",started_utc=now(),configuration=config,
                      numerical_source_sha256=numerical_hashes(),completed_cases=[],completed_reports=[])
        write_json(path,manifest)
    figures.mkdir(parents=True,exist_ok=True)
    params=Parameters(**{k:config[k] for k in asdict(Parameters()) if k!="rho"})
    cases,reports=[],[]
    ordered=([.95] if .95 in config["rhos"] else [])+[rho for rho in sorted(config["rhos"]) if rho!=.95]
    try:
        for rho in ordered:
            for n in sorted(config["grids"]):
                tag=case_tag(rho,n)
                if tag in manifest["completed_cases"]:
                    case=load_bundle(data/"arrays"/tag)
                    print(f"Reusing {tag}",flush=True)
                elif args.plots_only:
                    raise RuntimeError(f"Missing {tag}; finish the experiment first")
                else:
                    started=time.monotonic()
                    experiment=QualityExperiment(n=n,params=replace(params,rho=rho),
                                                 weighting=config["weighting"],order=config["order"])
                    print(f"Solving {LABEL}: rho={rho:g}, n={n}",flush=True)
                    search=experiment.search(p0_points=config["p0_points"],
                        consistency_tol=config["consistency_tol"],progress=lambda s:print("  "+s,flush=True))
                    selected=select_candidate(search["candidates"]) if search["candidates"] else None
                    if selected is None and search["best_diagnostic"] is not None:
                        best=search["best_diagnostic"]
                        selected=experiment.evaluate_mixed(best["p0"],best["policy"],config["consistency_tol"])
                        selected["candidate_verified"]=False
                    iterations=[experiment.iteration_history(config["initial_p0"],damping=d,
                                 tolerance=config["consistency_tol"]) for d in (1.,.5)]
                    case=dict(rho=rho,n=n,selected=selected,candidates=search["candidates"],
                        configuration=experiment.configuration(),demand_audit=experiment.demand_audit(),
                        candidate_predictions=[stationary_predictions(experiment,c) for c in search["candidates"]],
                        map_rows=[{k:e.get(k) for k in ("p0","F","G","strategy_mode","exploratory","full_game_verified")}
                                  for e in search["evaluations"]],
                        search_summary=search["summary"],search_log=search["search_log"],
                        failures=search["failures"],mixing_branches=search["mixing_branches"],
                        iteration_attempts=iterations,iteration_history=iterations[0]["rows"],
                        iteration_status=iterations[0]["status"],elapsed_seconds=time.monotonic()-started)
                    case["selected"], case["selection_reason"] = select_mixed_equilibrium(case)
                    selected = case["selected"]
                    save_bundle(data/"arrays"/tag,case)
                    manifest["completed_cases"].append(tag)
                    write_json(path,manifest)
                    print(f"  Saved {len(case['candidates'])} verified candidates; "+
                          (f"p0={selected['p0']:.9f}, residual={abs(selected['G']):.3g}" if selected else "no verified candidate"),flush=True)
                selected, reason = select_mixed_equilibrium(case)
                if case.get("selection_reason") != reason:
                    case["selected"], case["selection_reason"] = selected, reason
                    case["iteration_scope"] = "Auxiliary pure-selector diagnostic only; mixed equilibrium is solved separately"
                    if not args.plots_only:
                        save_bundle(data/"arrays"/tag,case)
                cases.append(case)
                export_tables(cases,reports,data)
        chosen={rho:max((c for c in cases if c["rho"]==rho),
                    key=lambda c:(bool(c["selected"] and c["selected"].get("candidate_verified")),c["n"]))
                for rho in sorted(config["rhos"])}
        if not args.solve_only:
            for rho,case in chosen.items():
                e=case["selected"]
                if not e or not e.get("candidate_verified"):
                    continue
                tag=case_tag(rho,case["n"])+"_diagnostics"
                report=load_bundle(data/"arrays"/tag) if tag in manifest["completed_reports"] else None
                policy_hash=sha256(e["policy"].tobytes()).hexdigest()
                reusable=bool(report and abs(report["p0"]-e["p0"]) <= 1e-12
                    and report.get("policy_sha256",policy_hash)==policy_hash)
                if reusable:
                    print(f"Reusing diagnostics {tag}",flush=True)
                elif args.plots_only:
                    raise RuntimeError(f"Missing diagnostics {tag}")
                else:
                    experiment=QualityExperiment(n=case["n"],params=replace(params,rho=rho),
                        weighting=config["weighting"],order=config["order"])
                    print(f"Actual TS simulation: rho={rho:g}, n={case['n']}",flush=True)
                    report=run_diagnostics(experiment,e,config,
                        progress=lambda done,total:print(f"  {done:,}/{total:,} periods",flush=True))
                    report["policy_sha256"]=policy_hash
                    save_bundle(data/"arrays"/tag,report)
                    if tag not in manifest["completed_reports"]:
                        manifest["completed_reports"].append(tag)
                    write_json(path,manifest)
                reports.append(report)
        export_tables(cases,reports,data)
        write_findings(cases,data,config)
        from .figures import make_figures
        make_figures(cases,reports,figures)
        if not args.plots_only and numerical_hashes()!=manifest["numerical_source_sha256"]:
            raise RuntimeError("Numerical sources changed during the experiment")
        manifest.update(status="solved" if args.solve_only else "complete",finished_utc=now(),
                        all_rates_verified=all(c["selected"] is not None and c["selected"].get("candidate_verified") for c in chosen.values()),
                        selected_cases={str(rho):case_tag(rho,c["n"]) for rho,c in chosen.items()},
                        selection_rule="Same as OE p0: certified interior candidate when available, then lowest p0; pure endpoints admissible",
                        strategy_space="Mixed own-reputation policies alpha(x) in [0,1]; interior mixing requires verified action indifference",
                        runner_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                        figure_source_sha256=sha256((Path(__file__).parent/"figures.py").read_bytes()).hexdigest(),
                        checkpoint_sha256={str(p.relative_to(data)):sha256(p.read_bytes()).hexdigest()
                            for p in sorted((data/"arrays").glob("*")) if p.suffix in (".json",".npz")})
        write_json(path,manifest)
        print(f"Complete: {figures}",flush=True)
    except BaseException as exc:
        if not args.plots_only:
            manifest.update(status="incomplete",last_error=dict(type=type(exc).__name__,message=str(exc),time=now()))
            write_json(path,manifest)
        raise


if __name__=="__main__":
    main()
