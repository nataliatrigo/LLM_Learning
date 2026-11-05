"""Add verified Conditional OE policies to the existing comparison snapshot.

Only reads saved solutions. Does not solve, simulate, or refresh other models.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-comparison-mpl")
PROJECT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = PROJECT / "oblivious/04_oe_conditional_distribution/data/rho_forgetting"
sys.path.insert(0, str(PROJECT))

import numpy as np
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, write_csv, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    comparison = json.loads((DATA / "comparison.json").read_text())
    baseline = json.loads((DATA / "manifest.json").read_text())
    for name, digest in baseline["data_files_sha256"].items():
        require(sha256((DATA / name).read_bytes()).hexdigest() == digest,
                f"Comparison snapshot changed: {name}")
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete", "Conditional OE run is incomplete")
    n = comparison["grid"]
    require(n == 51, "Expected the common 51-point comparison grid")
    sources = {str(manifest_path.relative_to(PROJECT)):
               sha256(manifest_path.read_bytes()).hexdigest()}
    rows, checks = [], []
    with np.load(DATA / "stationary_inputs.npz", allow_pickle=False) as reference:
        for rho in comparison["rhos"]:
            tag = case_tag(rho, n)
            require(tag in manifest["completed_cases"], f"Missing Conditional OE case: {tag}")
            for suffix in ("json", "npz"):
                path = SOURCE / "arrays" / f"{tag}.{suffix}"
                digest = sha256(path.read_bytes()).hexdigest()
                require(digest == manifest["checkpoint_sha256"][f"arrays/{tag}.{suffix}"],
                        f"Conditional OE source changed: {path}")
                sources[str(path.relative_to(PROJECT))] = digest
            case = load_bundle(SOURCE / "arrays" / tag)
            selected = case["selected"]
            require(case["rho"] == rho and case["n"] == n, f"Case mismatch: {tag}")
            require(case["configuration"]["parameters"] == {**comparison["parameters"], "rho":rho},
                    f"Primitives differ: {tag}")
            require(selected is not None and all((
                selected["verified"], selected["candidate_verified"],
                selected["dp_diagnostics"]["verified"],
                selected["stationary_diagnostics"]["verified"],
                selected["conditional_diagnostics"]["verified"], case["demand_audit"]["verified"])),
                f"Unverified Conditional OE policy: {tag}")
            grid, alpha = selected["grid"], selected["policy"]
            require(np.array_equal(grid, reference[f"{tag}/grid"]), f"Grid differs: {tag}")
            require(alpha.shape == (n,) and np.isfinite(alpha).all()
                    and np.all((alpha >= 0) & (alpha <= 1)), f"Invalid policy: {tag}")
            mixed = (alpha > 1e-9) & (alpha < 1-1e-9)
            rows.extend(dict(rho=rho, n=n, model="conditional", label="Conditional OE",
                             reputation=float(x), product2_probability=float(a), mixed=bool(m))
                        for x, a, m in zip(grid, alpha, mixed))
            checks.append(dict(rho=rho, n=n, verified=True, mixed_nodes=int(mixed.sum()),
                bellman_residual=selected["dp_diagnostics"]["bellman_residual"],
                stationary_residual=selected["stationary_diagnostics"]["residual"],
                conditional_residual=selected["conditional_residual"],
                conditional_order_row_l1=selected["conditional_diagnostics"]["independent_order_supported_row_l1_max"]))
    output = DATA / "conditional_policies.csv"
    write_csv(output, rows)
    write_json(DATA / "conditional_policy_manifest.json", dict(
        status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Conditional OE policy comparison only; existing model and outcome snapshots retained",
        model=dict(key="conditional", label="Conditional OE", color="#0072B2", linestyle="-."),
        rhos=comparison["rhos"], grid=n, parameters=comparison["parameters"],
        no_new_simulations=True, no_equilibrium_solved=True, verification=checks,
        source_files_sha256=sources,
        comparison_manifest_sha256=sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        data_files_sha256={output.name:sha256(output.read_bytes()).hexdigest()},
        prepare_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    print(f"Saved {len(rows)} policy rows: eight verified Conditional OE policies, n={n}.")


if __name__ == "__main__":
    main()
