"""Certified stationary monopoly sweep near a high outside benchmark.

Run from the repository root with ``.venv/bin/python -B -m`` followed by
``forgetting.Code.experiments.monopoly_high_p0_stationary_check --output DIR``.
Historical results are never used as writable checkpoints. The output contains
finite-grid certificates, not a continuous-state error bound or a claim of
global unimodality in the outside benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "monopoly_high_p0_mpl"))

import numpy as np
import scipy

from oblivious.oe_distribution_fixed_p0.code.run_experiment import numerical_sources, solve_monopoly
from oblivious.oe_p0.code.primitives import Parameters


ROOT = Path(__file__).resolve().parents[3]
REQUESTED = (0.8, 0.81039, 0.85, 0.9)
PARAMETERS = Parameters(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98, rho=.7)
DP_TOLERANCE = 1e-10


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def bands(grid, active):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [dict(first_index=int(group[0]), last_index=int(group[-1]),
                 first_node=float(grid[group[0]]), last_node=float(grid[group[-1]]),
                 node_count=len(group)) for group in groups]


def tag(p0, n):
    return f"p0_{format(p0, '.8g').replace('.', 'p')}_n{n}"


def summarize(p0, n, solution, seconds):
    grid, alpha, eta, mu, demand = (solution[key] for key in
                                   ("grid", "policy", "sale_stationary", "mu", "demand"))
    diagnostics, event = solution["diagnostics"], solution["diagnostics"]["sale_time"]
    sale_rate = float(mu @ demand)
    product2_conditional = float(eta @ alpha)
    product2_calendar = float(mu @ (demand * alpha))
    intervals = bands(grid, alpha > .5)
    mixed = (alpha > 1e-12) & (alpha < 1 - 1e-12)
    if (not solution["verified"] or np.any(mu <= 0) or np.any(eta <= 0)
            or np.any(mixed) or abs(product2_calendar - sale_rate * product2_conditional) > 1e-12):
        raise ArithmeticError("Unverified, nonpositive, mixed or inconsistent monopoly result")
    return dict(
        p0=p0, rho=PARAMETERS.rho, n=n, requested_case=p0 in REQUESTED, verified=True,
        product2_conditional=product2_conditional, product2_calendar=product2_calendar,
        product2_calendar_occupancy=float(mu @ alpha), sale_rate=sale_rate,
        outside_rate=1-sale_rate, profit_calendar=float(mu @ (demand *
            (PARAMETERS.R-PARAMETERS.c1-alpha*(PARAMETERS.c2-PARAMETERS.c1)))),
        value_initial=float(solution["value"][n//2]),
        quality_conditional=float(event["generated_quality"]),
        mean_reputation_at_sale=float(event["mean_reputation_at_sale"]),
        mean_reputation_calendar=float(mu @ grid),
        product2_node_count=int(np.sum(alpha > .5)), product2_band_count=len(intervals),
        product2_first_node=intervals[0]["first_node"] if intervals else "",
        product2_last_node=intervals[-1]["last_node"] if intervals else "",
        product2_node_bands=json.dumps(intervals, separators=(",", ":")),
        mixed_node_count=int(mixed.sum()),
        advantage_min=float(solution["advantage"].min()),
        advantage_max=float(solution["advantage"].max()),
        minimum_demand=float(demand.min()), minimum_sale_mass=float(eta.min()),
        minimum_calendar_mass=float(mu.min()),
        bellman_residual=float(diagnostics["bellman_residual"]),
        conditional_regret=float(diagnostics["conditional_regret"]),
        original_value_max_difference=float(diagnostics["original_value_max_difference"]),
        original_bellman_residual=float(diagnostics["original_bellman_residual"]),
        stationary_residual=float(diagnostics["stationary_residual"]),
        independent_stationary_l1=float(diagnostics["independent_stationary_l1"]),
        sale_law_l1=float(diagnostics["sale_law_l1"]),
        sale_time_stationary_residual=float(event["stationary_residual"]),
        sale_time_reversed_order_l1=float(event["stability_l1"]),
        quality_reputation_difference=float(event["quality_reputation_difference"]),
        elapsed_seconds=seconds, artifact=f"arrays/{tag(p0,n)}.npz",
    )


def comparisons(rows):
    lookup = {(row["p0"], row["n"]): row for row in rows}
    result = []
    for p0 in sorted({row["p0"] for row in rows}):
        for coarse, fine in ((201,401), (401,801)):
            if (p0,coarse) not in lookup or (p0,fine) not in lookup:
                continue
            a, b = lookup[p0,coarse], lookup[p0,fine]
            row = dict(p0=p0, rho=PARAMETERS.rho, coarse_n=coarse, fine_n=fine,
                       both_verified=bool(a["verified"] and b["verified"]))
            for metric in ("product2_conditional", "product2_calendar", "sale_rate",
                           "profit_calendar", "value_initial"):
                row[metric+"_coarse"] = a[metric]
                row[metric+"_fine"] = b[metric]
                row[metric+"_absolute_difference"] = abs(a[metric]-b[metric])
                row[metric+"_relative_difference"] = abs(a[metric]-b[metric])/max(abs(b[metric]),1e-300)
            for metric in ("product2_first_node", "product2_last_node", "product2_band_count"):
                row[metric+"_coarse"], row[metric+"_fine"] = a[metric], b[metric]
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)
    source_hashes = numerical_sources()
    source_hashes[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__).read_bytes()).hexdigest()
    sweep = sorted({float(Decimal(".78") + k*Decimal(".0025")) for k in range(57)} | {.81039})
    cases = [(p0,n) for p0 in REQUESTED for n in (401,801)]
    cases += [(p0,n) for p0 in sweep for n in (201,401,801) if (p0,n) not in cases]
    configuration = dict(parameters=asdict(PARAMETERS), requested_p0=list(REQUESTED),
        requested_grids=[401,801], sweep_p0=sweep, sweep_grids=[201,401,801],
        dp_tolerance=DP_TOLERANCE, calendar_stationary_tolerance=1e-11,
        sale_stationary_tolerance=1e-11, independent_order_stability_tolerance=1e-9,
        method="Existing solve_monopoly: independent mature Bellman solver; positive GTH sale-time "
               "invariants in both state orders; log-domain calendar holding-time reweighting; "
               "independent positive GTH calendar invariant. No signed-law clipping.",
        policy_geometry="Contiguous product-2 grid-node bands; endpoints are saved nodes, not exact thresholds.",
        probability_measures=dict(product2_conditional="Fraction of seller sales using product 2",
            product2_calendar="Fraction of calendar periods with a product-2 sale",
            product2_calendar_occupancy="Calendar probability of a state whose policy uses product 2"),
        scope="Certified finite-grid mature monopoly. Grid comparisons are numerical sensitivity checks; "
              "no continuous-state error guarantee or global unimodality claim.")
    manifest = dict(status="running", created_utc=now(), updated_utc=now(),
        configuration=configuration, numerical_sources_sha256=source_hashes,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__),
        planned_cases=len(cases), completed_cases=0, artifact_sha256={})
    write_json(output / "manifest.json", manifest)
    rows = []
    try:
        for p0, n in cases:
            started = time.monotonic()
            solution = solve_monopoly(PARAMETERS, n, p0, dp_tol=DP_TOLERANCE)
            row = summarize(p0,n,solution,time.monotonic()-started)
            artifact = output / row["artifact"]
            np.savez_compressed(artifact, **{key:solution[key] for key in
                ("grid","policy","value","demand","advantage","mu","sale_stationary")})
            rows.append(row)
            ordered = sorted(rows,key=lambda r:(r["p0"],r["n"]))
            write_csv(output / "summary.csv", ordered)
            write_csv(output / "grid_comparison.csv", comparisons(ordered))
            manifest["artifact_sha256"][row["artifact"]] = sha256(artifact.read_bytes()).hexdigest()
            manifest.update(completed_cases=len(rows), updated_utc=now())
            write_json(output / "manifest.json", manifest)
            print(f"[{len(rows)}/{len(cases)}] p0={p0:.8g} n={n} "
                  f"P2|sale={row['product2_conditional']:.10g} "
                  f"sale_rate={row['sale_rate']:.10g} "
                  f"bands={row['product2_node_bands']} seconds={row['elapsed_seconds']:.2f}", flush=True)
        for source, expected in source_hashes.items():
            if sha256((ROOT/source).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Numerical source changed during run: {source}")
        for name in ("summary.csv", "grid_comparison.csv"):
            manifest["artifact_sha256"][name] = sha256((output/name).read_bytes()).hexdigest()
        manifest.update(status="complete", updated_utc=now())
        write_json(output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", updated_utc=now(), error=repr(error))
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
