"""Collect and check the saved n=51 results; never solve or simulate a model."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-comparison-mpl")
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, write_csv, write_json
from oblivious.equilibrium_comparison.code.analysis import interval


SOURCES = {
    "oe": ROOT / "oblivious/01_oe_distribution/data/rho_forgetting",
    "p0": ROOT / "oblivious/02_oe_p0/data/rho_forgetting_mixed",
    "monopoly": ROOT / "oblivious/03_oe_following_monopoly/data/quality_sales_n51",
}
COMPARISON = ROOT / "oblivious/06_equilibrium_comparison/data/cross_rho_n51"
MODELS = {
    "oe": {"label": "OE distribution", "color": "#009E73", "linestyle": "-"},
    "p0": {"label": "OE p0", "color": "#D55E00", "linestyle": "--"},
    "monopoly": {"label": "Following monopoly", "color": "#7B3294", "linestyle": ":"},
}
METRICS = ("product2_market", "seller_profit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    files = {}

    def record(path):
        files[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()

    def manifest(directory):
        path = directory / "manifest.json"
        record(path)
        result = json.loads(path.read_text())
        require(result["status"] == "complete", f"Incomplete source: {path}")
        return result

    def bundle(base):
        for suffix in (".json", ".npz"):
            record(base.with_suffix(suffix))
        return load_bundle(base)

    manifests = {key: manifest(path) for key, path in SOURCES.items()}
    comparison_manifest = manifest(COMPARISON)
    config = comparison_manifest["configuration"]
    rhos = sorted(config["rhos"])
    n = config["grid"]
    require(n == 51 and config["mode"] == "grid", "Expected the common n=51 grid.")
    for key, item in manifests.items():
        require(sorted(item["configuration"]["rhos"]) == rhos, f"Rho mismatch: {key}")
        require(n in item["configuration"]["grids"], f"Grid unavailable: {key}")

    policies, outcomes, replications, checks, snapshots = [], [], [], [], {}
    parameters = None
    for rho in rhos:
        tag = case_tag(rho, n)
        comparison = bundle(COMPARISON / "checkpoints" / tag)
        require(comparison["verified"] and comparison["audit"]["verified"],
                f"Unverified comparison: {tag}")
        grid = comparison["grid"]
        params = comparison["parameters"]
        common_params = {key: value for key, value in params.items() if key != "rho"}
        if parameters is None:
            parameters = common_params
        require(common_params == parameters, f"Primitives differ across rho: {tag}")
        game = Game(grid, Parameters(**params), order=comparison["audit"]["order"])
        snapshots[f"{tag}/grid"] = grid
        snapshots[f"{tag}/demand_A"] = game.D

        provenance = comparison["simulation_provenance"]["metadata"]
        # Reused simulations may wrap their metadata one level deeper.
        metadata = provenance.get("metadata", provenance)
        expected_metadata = {
            "params": params, "mode": "grid", "initial": [.5, .5],
            **{key: config[key] for key in ("periods", "burn_in", "replications", "seed")},
        }
        require(all(metadata.get(key) == value for key, value in expected_metadata.items()),
                f"Incompatible comparison simulation: {tag}")

        for key, directory in SOURCES.items():
            case = bundle(directory / "arrays" / tag)
            selected = case["selected"]
            require(case["rho"] == rho and case["n"] == n,
                    f"Incompatible case: {key}, {tag}")
            require(case["configuration"]["parameters"] == params,
                    f"Primitives differ: {key}, {tag}")
            require(case["configuration"]["order"] == game.order,
                    f"Demand integration differs: {key}, {tag}")
            statuses = {
                "candidate": selected["candidate_verified"],
                "verified": selected["verified"],
                "dp": selected["dp_diagnostics"]["verified"],
                "stationary": selected["stationary_diagnostics"]["verified"],
                "demand": case["demand_audit"]["verified"],
            }
            require(all(value is True for value in statuses.values()),
                    f"Unverified selected candidate: {key}, {tag}")
            require(np.array_equal(selected["grid"], grid), f"Grid differs: {key}, {tag}")
            alpha, nu = selected["policy"], selected["nu"]
            require(alpha.shape == (n,) and np.isfinite(alpha).all()
                    and np.all((alpha >= 0) & (alpha <= 1)),
                    f"Invalid policy: {key}, {tag}")
            require(nu.shape == (n, n) and np.isfinite(nu).all()
                    and nu.min() >= 0 and abs(nu.sum() - 1) < 1e-10,
                    f"Invalid stationary law: {key}, {tag}")

            if key == "monopoly":
                require(selected["monopoly_certificate"]["verified"],
                        f"Unverified monopoly Bellman equation: {tag}")
                require(case["configuration"]["quality_weighting"] == "purchase",
                        "Following monopoly must use quality conditional on rival sales.")
                report = bundle(directory / "arrays" / f"{tag}_diagnostics")
                require(report["candidate_verified"] and report["rho"] == rho
                        and report["n"] == n and report["p0"] == selected["p0"],
                        f"Diagnostics do not match selected monopoly candidate: {tag}")
                simulation = report["simulation"]
                require(all(simulation["metadata"].get(k) == v
                            for k, v in expected_metadata.items()),
                        f"Incompatible monopoly simulation: {tag}")
                prediction = report["predicted"]
            else:
                simulation = comparison["profiles"][key]
                require(np.array_equal(simulation["policy"], alpha),
                        f"Cached simulation uses another policy: {key}, {tag}")
                require(np.array_equal(simulation["stationary_joint"], nu),
                        f"Cached stationary law differs: {key}, {tag}")
                prediction = simulation["predicted"]

            sigma_a = np.broadcast_to(alpha[:, None], (n, n))
            sigma_b = sigma_a.T
            transition = game.transition(sigma_a, sigma_b)
            residual = float(np.max(np.abs(nu.ravel() @ transition - nu.ravel())))
            require(residual < 1e-10, f"Stationarity failed: {key}, {tag}")
            usage = float(np.sum(nu * (game.D * sigma_a + game.D.T * sigma_b)))
            profit = float(np.sum(game.rewards(sigma_a, sigma_b) * nu[None, :, :]) / 2)
            formula_profit = .5 * (params["R"] - params["c1"]
                                   - (params["c2"] - params["c1"]) * usage)
            require(abs(profit - formula_profit) < 1e-12, "Profit accounting failed.")
            predicted = {"product2_market": usage, "seller_profit": profit}
            require(all(abs(predicted[m] - prediction[m]) < 1e-10 for m in METRICS),
                    f"Saved predictions disagree with joint law: {key}, {tag}")
            reps = simulation["replications"]
            require(len(reps) == config["replications"], f"Replication count differs: {tag}")
            require(sorted(row["replication"] for row in reps) == list(range(len(reps))),
                    f"Invalid replication identifiers: {key}, {tag}")
            for row in reps:
                identity = .5 * (params["R"] - params["c1"]
                                - (params["c2"] - params["c1"]) * row["product2_market"])
                require(abs(row["seller_profit"] - identity) < 1e-12,
                        f"Simulation profit accounting failed: {key}, {tag}")
                replications.append({
                    "rho": rho, "model": key, "replication": row["replication"],
                    **{metric: row[metric] for metric in METRICS},
                })
            for metric in METRICS:
                estimate = interval([row[metric] for row in reps])
                saved = simulation["summary"][metric]
                require(all(np.isclose(estimate[field], saved[field], rtol=0, atol=1e-12)
                            for field in estimate), f"MC summary mismatch: {key}, {tag}")
                outcomes.append({
                    "rho": rho, "n": n, "model": key, "label": MODELS[key]["label"],
                    "metric": metric, "stationary": predicted[metric],
                    **{f"simulation_{field}": value for field, value in estimate.items()},
                })
            mixed = (alpha > 1e-9) & (alpha < 1 - 1e-9)
            for index, x in enumerate(grid):
                policies.append({
                    "rho": rho, "n": n, "model": key, "label": MODELS[key]["label"],
                    "reputation": float(x), "product2_probability": float(alpha[index]),
                    "mixed": bool(mixed[index]),
                })
            snapshots[f"{tag}/{key}/policy"] = alpha
            snapshots[f"{tag}/{key}/stationary_joint"] = nu
            checks.append({
                "rho": rho, "n": n, "model": key, "verified": True,
                "stationary_residual_max": residual, "mixed_nodes": int(mixed.sum()),
                "p0": selected.get("p0"), "source": str(directory.relative_to(ROOT)),
            })
        print(f"rho={rho:g}: three verified policies; stationary and simulation checks passed.")

    data = OUTPUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_csv(data / "policies.csv", policies)
    write_csv(data / "outcomes.csv", outcomes)
    write_csv(data / "replications.csv", replications)
    write_csv(data / "verification.csv", checks)
    np.savez_compressed(data / "stationary_inputs.npz", **snapshots)
    write_json(data / "comparison.json", {
        "rhos": rhos, "grid": n, "models": MODELS, "parameters": parameters,
        "simulation": {key: config[key] for key in
                       ("periods", "burn_in", "replications", "seed", "mode")},
        "initial": [.5, .5],
        "following_monopoly_public_name": "OE quality",
        "definitions": {
            "policy": "alpha(x) = P(product 2 | seller selected, own reputation x).",
            "product2_market": "sum_ij nu_ij [D_ij alpha_i + D_ji alpha_j]; share of all purchases.",
            "seller_profit": "Mean profit of A and B per calendar period, including unselected periods: "
                             "0.5 * [R-c1-(c2-c1)*product2_market].",
            "uncertainty": "Pointwise 95% Student-t intervals across 20 independent path means; "
                           "initialization bias is excluded.",
            "optimality": "Saved selected policies are verified for each model's own decision problem "
                          "on the common finite grid; this is not an MPE certificate.",
        },
    })
    write_json(data / "manifest.json", {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "no_new_simulations": True, "no_equilibrium_solved": True,
        "all_24_model_rho_cases_verified": True,
        "source_files_sha256": files,
        "data_files_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.iterdir())
            if path.name in {"policies.csv", "outcomes.csv", "replications.csv",
                             "verification.csv", "stationary_inputs.npz", "comparison.json"}
        },
        "code_files_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    })
    print(f"Saved comparison data in {data}")


if __name__ == "__main__":
    main()
