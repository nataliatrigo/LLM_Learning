"""Run mixed oblivious optimization with actual-joint marginal consistency.

From the project root:
    .venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment
    .venv/bin/python -B -m oblivious.oe_distribution.code.run_experiment --resume
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.core import ScalarExperiment, posterior_mean
from oblivious.oe_p0.code.run_experiment import (
    save_bundle, load_bundle, write_csv, write_json, clean, now, case_tag,
)
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution.code.model import JointMarginalExperiment
from oblivious.oe_distribution.code.diagnostics import run_diagnostics

RHOS = [.6, .7, .8, .9, .95, .97, .99]


def source_hashes():
    names = list((ROOT / "oblivious/01_oe_distribution/code").glob("*.py"))
    names += list((ROOT / "oblivious/02_oe_p0/code").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(names)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="rho_forgetting")
    parser.add_argument("--output", type=Path, default=ROOT / "oblivious/01_oe_distribution/data")
    parser.add_argument("--figures-output", type=Path,
                        default=ROOT / "oblivious/01_oe_distribution/figures")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-diagnostics", action="store_true",
                        help="With --resume, regenerate reports after a reporting-only correction")
    parser.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    parser.add_argument("--grids", nargs="+", type=int, default=[21, 31, 51])
    parser.add_argument("--order", type=int, default=128)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--dp-tol", type=float, default=1e-10)
    parser.add_argument("--max-evaluations", type=int, default=1500)
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261017)
    for key, value in asdict(Parameters()).items():
        if key != "rho":
            parser.add_argument("--" + key, type=float, default=value)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use one directory name for --name")
    if args.resume:
        allowed = {"--resume", "--refresh-diagnostics", "--name", "--output"}
        if any(token.split("=")[0] not in allowed for token in sys.argv[1:]
               if token.startswith("--")):
            parser.error("Resume uses saved settings; only resume/report flags, --name, --output are allowed")
    elif args.refresh_diagnostics:
        parser.error("--refresh-diagnostics requires --resume")
    elif (len(set(args.rhos)) != len(args.rhos) or len(set(args.grids)) != len(args.grids)
          or min(args.grids) < 3 or args.order < 8 or args.max_evaluations < 1
          or args.periods < 2 or args.burn_in < 0 or args.trajectories < 2
          or not 0 < args.tol < .01 or not 0 < args.dp_tol < .01):
        parser.error("Invalid or repeated grids/rhos, or invalid numerical settings")
    return args


def prepare_resume(manifest, directory, refresh_diagnostics=False):
    """Preserve solved cases; a reporting correction must rebuild every report."""
    current = source_hashes()
    previous = manifest.get("diagnostic_source_sha256", manifest["source_sha256"])
    changed = sorted(key for key in set(previous) | set(current)
                     if previous.get(key) != current.get(key))
    entry = dict(time=now(), refresh_diagnostics=refresh_diagnostics,
                 changed_sources=changed)
    if refresh_diagnostics:
        config = manifest["configuration"]
        expected = {case_tag(rho, n) for rho in config["rhos"] for n in config["grids"]}
        if set(manifest["completed_cases"]) != expected:
            raise SystemExit("Finish every equilibrium case before refreshing diagnostics")
        reporting_files = {"oblivious/01_oe_distribution/code/" + name for name in
                           ("diagnostics.py", "run_experiment.py", "test_model.py")}
        solver_changed = {key for key in set(manifest["source_sha256"]) | set(current)
                          if manifest["source_sha256"].get(key) != current.get(key)} - reporting_files
        if solver_changed:
            raise SystemExit("Numerical solver sources changed; use a new run name")
        case_hashes = {}
        for tag in sorted(expected):
            for suffix in ("json", "npz"):
                path = directory / "arrays" / (tag + "." + suffix)
                case_hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get("case_bundle_sha256", case_hashes) != case_hashes:
            raise SystemExit("Saved equilibrium cases changed since the previous report refresh")
        manifest["case_bundle_sha256"] = case_hashes
        prior_reports = {}
        for tag in manifest["completed_diagnostics"]:
            for suffix in ("json", "npz"):
                path = directory / "arrays" / (tag + "." + suffix)
                prior_reports[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry.update(previous_source_sha256=previous, new_source_sha256=current,
                     superseded_report_sha256=prior_reports,
                     interpretation="Equilibrium case bundles retained unchanged; all diagnostics and figures regenerated")
        manifest["completed_diagnostics"] = []
        manifest["diagnostic_source_sha256"] = current
    elif changed:
        raise SystemExit("Code changed since this run. Use a new run name, or --refresh-diagnostics for reporting-only changes after all cases finish.")
    manifest.setdefault("resume_history", []).append(entry)


def seed_profiles(experiment, previous):
    seeds = [dict(name="product_1", policy=np.zeros(experiment.n)),
             dict(name="half", policy=np.full(experiment.n, .5))]
    if previous is not None:
        seeds.append(dict(name="previous_grid", policy=np.interp(
            experiment.grid, previous["grid"], previous["policy"])))
    else:
        seeds.append(dict(name="product_2", policy=np.ones(experiment.n)))
    return seeds


def candidate_score(candidate):
    diagnostics = candidate.get("dp_diagnostics", {})
    return max(float(candidate.get("marginal_residual", np.inf)),
               float(diagnostics.get("conditional_regret", np.inf)),
               float(diagnostics.get("mixed_indifference_residual", np.inf)))


def marginal_cdf_gap(first, second):
    # Merge only floating-point noise at mathematically coincident grid nodes.
    ga, gb = np.round(first["grid"], 14), np.round(second["grid"], 14)
    support = np.unique(np.concatenate((ga, gb)))
    ca = np.r_[0., np.cumsum(first["mu_b"])][np.searchsorted(ga, support, side="right")]
    cb = np.r_[0., np.cumsum(second["mu_b"])][np.searchsorted(gb, support, side="right")]
    return float(np.max(np.abs(ca-cb)))


def candidate_row(candidate, params):
    alpha, grid = candidate["policy"], candidate["grid"]
    nu, mu = candidate["nu"], candidate["mu_b"]
    mixed = (alpha > 0) & (alpha < 1)
    row = dict(verified=bool(candidate["verified"]),
               marginal_residual=candidate["marginal_residual"],
               mixed_nodes=int(mixed.sum()),
               mixed_reputations=grid[mixed].tolist(),
               mixing_probabilities=alpha[mixed].tolist(),
               mean_reputation=float(mu @ grid),
               mean_TS=float(mu @ posterior_mean(grid, params)),
               total_joint_probability=float(nu.sum()))
    for prefix, source in (("dp", candidate.get("dp_diagnostics", {})),
                           ("stationary", candidate.get("stationary_diagnostics", {}))):
        row.update({prefix + "_" + key: value for key, value in source.items()
                    if isinstance(value, (str, int, float, bool, np.generic)) or value is None})
    return row


def export_tables(cases, folder):
    summaries, roots, starts, history, refinement = [], [], [], [], []
    for case in cases:
        base = dict(rho=case["rho"], grid_size=case["n"])
        params = Parameters(**case["configuration"]["parameters"])
        summaries.append({**base, "candidates_found": len(case["candidates"]),
            "selected_start": case["selected_start"], "elapsed_seconds": case["elapsed_seconds"],
            **(candidate_row(case["selected"], params) if case["selected"] is not None else
               {"verified": False, "status": "No evaluable candidate"})})
        for index, candidate in enumerate(case["candidates"]):
            roots.append({**base, "candidate": index, **candidate_row(candidate, params)})
        for attempt in case["attempts"]:
            value = attempt.get("candidate")
            starts.append({**base, "start": attempt["start"], "error": attempt.get("error"),
                           **(candidate_row(value, params) if value is not None else {"verified": False})})
            if value is not None:
                history.extend({**base, "start": attempt["start"], **row}
                               for row in value.get("history", []))
    for rho in sorted({case["rho"] for case in cases}):
        group = sorted((case for case in cases if case["rho"] == rho), key=lambda c: c["n"])
        for low, high in zip(group[:-1], group[1:]):
            a, b = low["selected"], high["selected"]
            row = dict(rho=rho, base_grid=low["n"], refined_grid=high["n"],
                       both_verified=bool(a is not None and b is not None
                                          and a["verified"] and b["verified"]))
            if a is not None and b is not None:
                row.update(mean_reputation_change=float(b["mu_b"] @ b["grid"] - a["mu_b"] @ a["grid"]),
                    policy_max_interpolated_difference=float(np.max(np.abs(
                        b["policy"] - np.interp(b["grid"], a["grid"], a["policy"])))),
                    marginal_cdf_max_difference=marginal_cdf_gap(a, b),
                    marginal_wasserstein_distance=float(wasserstein_distance(
                        a["grid"], b["grid"], u_weights=a["mu_b"], v_weights=b["mu_b"])))
            refinement.append(row)
    for name, rows in (("summary", summaries), ("candidates", roots), ("initializations", starts),
                       ("solver_history", history), ("grid_refinement", refinement)):
        write_csv(folder / (name + ".csv"), rows)


def export_diagnostics(reports, folder):
    sections = {key: [] for key in ("simulation", "demand", "deviations")}
    for report in reports:
        base = {key: report[key] for key in ("rho", "n", "candidate", "verified")}
        for key in sections:
            sections[key].append({**base, **report[key]["summary"]})
    for key, rows in sections.items():
        write_json(folder / (key + "_summary.json"), rows)
        write_csv(folder / (key + "_summary.csv"), rows)


def selected_case(cases, rho):
    group = [case for case in cases if case["rho"] == rho]
    valid = [case for case in group if case["candidates"]]
    return max(valid or group, key=lambda case: case["n"])


def panel_grid(rhos):
    columns = min(2, len(rhos))
    rows = (len(rhos) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.7 * rows),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[len(rhos):]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def save_figure(fig, folder, name):
    for suffix in ("png", "pdf"):
        fig.savefig(folder / (name + "." + suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(cases, reports, folder):
    rhos = sorted({case["rho"] for case in cases})
    chosen = {rho: selected_case(cases, rho) for rho in rhos}
    diagnostics = {}
    for rho, case in chosen.items():
        matching = [r for r in reports if r["rho"] == rho and r["n"] == case["n"]]
        if matching:
            diagnostics[rho] = matching[0]
    for kind in ("policy", "marginal", "joint", "demand", "dependence", "advantage"):
        fig, axes = panel_grid(rhos)
        for rho, ax in zip(rhos, axes):
            case, report = chosen[rho], diagnostics.get(rho)
            e = case["selected"]
            if e is None:
                ax.text(.5, .5, "No evaluable result", ha="center", transform=ax.transAxes)
                continue
            grid, alpha, nu = e["grid"], e["policy"], e["nu"]
            status = "verified on grid" if e["verified"] else "diagnostic; unverified"
            ax.set_title(f"ρ={rho:g}, n={case['n']} | {status}")
            if kind == "policy":
                ax.plot(grid, alpha, ".-", ms=3, color="#009E73", label="OE distribution")
                mixed = (alpha > 0) & (alpha < 1)
                ax.scatter(grid[mixed], alpha[mixed], marker="D", color="#D55E00", s=35,
                           label="Mixing at a grid point" if mixed.any() else None, zorder=3)
                ax.set(xlabel="Own reputation", ylabel="Probability of product 2", ylim=(-.04, 1.04))
            elif kind == "advantage":
                ax.plot(grid, e["advantage"], ".-", ms=3, color="#009E73",
                        label="Product 2 − product 1")
                ax.axhline(0, color="black", ls="--", lw=.8, label="Indifference")
                mixed = (alpha > 0) & (alpha < 1)
                ax.scatter(grid[mixed], e["advantage"][mixed], marker="D", color="#D55E00", s=35,
                           label="Mixing at a grid point" if mixed.any() else None, zorder=3)
                ax.set(xlabel="Own reputation", ylabel="Product 2 advantage\nwhen selected")
            elif kind == "marginal":
                ax.step(grid, np.cumsum(e["mu"]), where="post", lw=2, label="Used to choose the strategy")
                ax.step(grid, np.cumsum(e["mu_b"]), where="post", ls="--", label="From the real two-seller game")
                ax.set(xlabel="Reputation", ylabel="Cumulative probability", ylim=(0, 1))
            elif kind in ("joint", "dependence"):
                z = nu if kind == "joint" else nu - np.outer(e["mu_a"], e["mu_b"])
                options = {} if kind == "joint" else dict(vmin=-np.max(np.abs(z)), vmax=np.max(np.abs(z)))
                im = ax.pcolormesh(grid, grid, z.T, shading="auto",
                                   cmap="viridis" if kind == "joint" else "RdBu_r", **options)
                fig.colorbar(im, ax=ax, label="Joint probability" if kind == "joint" else "Difference from independence")
                ax.set(xlabel="Seller A's reputation", ylabel="Seller B's reputation")
            elif report:
                arrays = report["demand"]["arrays"]
                ax.plot(grid, arrays["d_optimized"], color="#0072B2",
                        label="Used for product choice: rival marginal")
                ax.plot(grid, arrays["d_real_marginal"], "--", color="#D55E00",
                        label="Real game: unconditional rival distribution")
                ax.plot(grid, arrays["d_conditional"], ":", lw=2, color="#009E73",
                        label="Real game: conditional on own reputation")
                ax.set(xlabel="Own reputation", ylabel="Probability seller A is selected")
            if kind not in ("joint", "dependence"):
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
        names = dict(policy=("01_policy", "Product choice: OE distribution"),
                     marginal=("02_marginal_consistency", "OE distribution: reputation distributions"),
                     joint=("03_joint_stationary", "OE distribution: long-run joint reputation distribution"),
                     demand=("04_demand", "Demand comparison: OE distribution"),
                     dependence=("05_joint_dependence", "OE distribution: dependence between sellers' reputations"),
                     advantage=("06_advantage", "OE distribution: product 2 advantage\n"
                                "Above zero: product 2 preferred; below zero: product 1 preferred\n"
                                "Points are solved nodes; lines interpolate between them"))
        name, title = names[kind]
        fig.suptitle(title, fontsize=12)
        save_figure(fig, folder, name)


def write_findings(cases, folder):
    lines = ["OE distribution", "",
        "Optimización: demanda promediada sobre la marginal incondicional del rival, como en el OE distribution.",
        "Consistencia: esa marginal debe ser la marginal de la distribución conjunta real.",
        "La mezcla interior exige indiferencia. Los extremos 0 y 1 siguen permitidos.",
        "Verificado significa verificado en la grilla; no demuestra precisión del modelo continuo ni MPE.",
        "La búsqueda con varios inicios no demuestra unicidad ni exhaustividad.", "",
        "rho | puntos por vendedor | candidatos verificados | mezcla en seleccionado | error marginal L1"]
    for case in sorted(cases, key=lambda c: (c["rho"], c["n"])):
        e = case["selected"]
        lines.append(f"{case['rho']:g} | {case['n']} | {len(case['candidates'])} | " +
                     (f"{int(np.sum((e['policy']>0)&(e['policy']<1)))} | {e['marginal_residual']:.9g}"
                      if e is not None else "sin candidato evaluable"))
    lines += ["", "En el solver reducido la marginal se obtiene de la conjunta en cada evaluación.",
              "Por eso el error marginal puede ser cero por construcción; además debe pasar la optimalidad.",
              "Un intento sin optimalidad verificada se conserva como diagnóstico, no como equilibrio.",
              "Consulte candidates.csv para todas las soluciones y initializations.csv para todos los intentos.",
              "Las simulaciones interpolan la política entre nodos y no certifican optimalidad fuera de ellos."]
    (folder / "RESULTADOS.txt").write_text("\n".join(lines) + "\n")


def main():
    args = arguments()
    directory = args.output / args.name
    manifest_path = directory / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] == "complete":
            raise SystemExit("Run already complete; use a new name")
        refresh_diagnostics = args.refresh_diagnostics
        args = argparse.Namespace(**manifest["configuration"], resume=True)
        args.output = resolve_saved_path(args.output, project_root=ROOT)
        args.figures_output = resolve_saved_path(args.figures_output, project_root=ROOT)
        prepare_resume(manifest, directory, refresh_diagnostics)
    else:
        if directory.exists() or (args.figures_output / args.name).exists():
            raise SystemExit("Existing results protected; use --resume or a new --name")
        # Validate all economics before creating output directories.
        base = {key: getattr(args, key) for key in asdict(Parameters()) if key != "rho"}
        for rho in args.rhos:
            Parameters(**base, rho=rho)
        directory.mkdir(parents=True)
        manifest = dict(status="incomplete", started_utc=now(),
            configuration={key: clean(value) for key, value in vars(args).items() if key != "resume"},
            source_sha256=source_hashes(), completed_cases=[], completed_diagnostics=[],
            model="Own-state mixed optimization against the unconditional rival marginal; full marginal equals that of the actual joint stationary game",
            interpretation="Finite-grid marginal consistency and oblivious optimality; not MPE or a continuous-state existence theorem")
    (directory / "arrays").mkdir(exist_ok=True)
    figures = args.figures_output / args.name
    figures.mkdir(parents=True, exist_ok=args.resume)
    write_json(manifest_path, manifest)
    params = Parameters(**{key: getattr(args, key) for key in asdict(Parameters()) if key != "rho"})
    cases, reports = [], []
    ordered = ([.95] if .95 in args.rhos else []) + [r for r in sorted(args.rhos) if r != .95]
    try:
        for rho in ordered:
            p = replace(params, rho=rho)
            previous = None
            for n in sorted(args.grids):
                tag = case_tag(rho, n)
                path = directory / "arrays" / tag
                if tag in manifest["completed_cases"]:
                    case = load_bundle(path)
                    print(f"Reusing ρ={rho:g}, n={n}", flush=True)
                else:
                    start = time.monotonic()
                    experiment = JointMarginalExperiment(n=n, params=p, order=args.order,
                                                         tol=args.tol, dp_tol=args.dp_tol)
                    seeds = seed_profiles(experiment, previous)
                    attempts, candidates = [], []
                    for seed in seeds:
                        print(f"Solving ρ={rho:g}, n={n}, start={seed['name']}", flush=True)
                        try:
                            result = experiment.solve(initial_policy=seed["policy"],
                                max_evaluations=args.max_evaluations,
                                progress=lambda message: print("  " + str(message), flush=True))
                            attempts.append(dict(start=seed["name"], candidate=result))
                            if result["verified"] and not any(np.max(np.abs(result["policy"]-old["policy"])) <= 1e-6 for old in candidates):
                                candidates.append(result)
                            print(f"  verified={result['verified']}, marginal L1={result['marginal_residual']:.3g}", flush=True)
                        except (ArithmeticError, RuntimeError, ValueError) as exc:
                            attempts.append(dict(start=seed["name"], error=str(exc)))
                            print(f"  Numerical failure: {exc}", flush=True)
                    evaluable = [a for a in attempts if "candidate" in a]
                    selected = candidates[0] if candidates else (min(evaluable, key=lambda a: candidate_score(a["candidate"]))["candidate"] if evaluable else None)
                    selected_start = (next((a["start"] for a in attempts if a.get("candidate") is selected), None)
                                      if selected is not None else None)
                    # Reuse only the established TS quadrature audit, not scalar optimization.
                    audit = ScalarExperiment(n=n, params=p, order=args.order).demand_audit()
                    case = dict(rho=rho, n=n, configuration=dict(parameters=asdict(p), n=n,
                        order=args.order, tolerance=args.tol, dp_tolerance=args.dp_tol),
                        initializations=seeds, attempts=attempts, candidates=candidates,
                        selected=selected, selected_start=selected_start,
                        demand_audit=audit, elapsed_seconds=time.monotonic()-start)
                    save_bundle(path, case)
                    manifest["completed_cases"].append(tag)
                    write_json(manifest_path, manifest)
                cases.append(case)
                previous = case["selected"]
                export_tables(cases, directory)
            preferred = selected_case(cases, rho)
            for index, candidate in enumerate(preferred["candidates"]):
                n = preferred["n"]
                tag = case_tag(rho, n) + f"_diagnostics_{index}"
                path = directory / "arrays" / tag
                if tag in manifest["completed_diagnostics"]:
                    report = load_bundle(path)
                else:
                    print(f"Actual TS simulation and diagnostics ρ={rho:g}, n={n}, candidate={index}", flush=True)
                    experiment = JointMarginalExperiment(n=n, params=p, order=args.order,
                                                         tol=args.tol, dp_tol=args.dp_tol)
                    report = dict(rho=rho, n=n, candidate=index, verified=True,
                        **run_diagnostics(experiment.game, candidate, periods=args.periods,
                            burn_in=args.burn_in, trajectories=args.trajectories,
                            seed=args.seed + (RHOS.index(rho) if rho in RHOS else ordered.index(rho))))
                    save_bundle(path, report)
                    manifest["completed_diagnostics"].append(tag)
                    write_json(manifest_path, manifest)
                reports.append(report)
        export_tables(cases, directory)
        export_diagnostics(reports, directory)
        make_figures(cases, reports, figures)
        write_findings(cases, directory)
        if "last_error" in manifest:
            manifest.setdefault("recovered_errors", []).append(manifest.pop("last_error"))
        manifest.update(status="complete", finished_utc=now(), final_source_sha256=source_hashes(),
                        verified_cases=sum(bool(case["candidates"]) for case in cases),
                        total_cases=len(cases))
        write_json(manifest_path, manifest)
        print(f"Complete. Data: {directory}\nFigures: {figures}", flush=True)
    except BaseException as exc:
        manifest["last_error"] = dict(type=type(exc).__name__, message=str(exc), time=now())
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
