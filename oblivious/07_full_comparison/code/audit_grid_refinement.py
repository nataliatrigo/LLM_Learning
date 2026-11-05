"""Audit saved 51/101 results; never solve an equilibrium or simulate a path.

Examples, from the repository root::

    .venv/bin/python 'oblivious/07_full_comparison/code/audit_grid_refinement.py'
    .venv/bin/python 'oblivious/07_full_comparison/code/audit_grid_refinement.py' \
        --oe-retry 'oblivious/07_full_comparison/data/oe099_retry_n101/solution'

Inputs are read only. The only writes are the requested CSV and JSON report.
Certificates are checked as saved; this script does not recertify model equations.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oblivious.layout import resolve_saved_path

DATA = ROOT / "oblivious/07_full_comparison/data"
SOURCES = {
    "oe": ROOT / "oblivious/01_oe_distribution/data/rho_forgetting",
    "p0": ROOT / "oblivious/02_oe_p0/data/rho_forgetting_mixed",
    "monopoly": ROOT / "oblivious/03_oe_following_monopoly/data/quality_sales_n51",
    "conditional": ROOT / "oblivious/04_oe_conditional_distribution/data/rho_forgetting",
}


def certified(candidate):
    if not candidate or not candidate.get("verified") or not candidate.get("candidate_verified"):
        return False
    diagnostics = ["dp_diagnostics", "stationary_diagnostics"]
    diagnostics += [name for name in ("conditional_diagnostics", "monopoly_certificate")
                    if name in candidate]
    return all(candidate.get(name, {}).get("verified") for name in diagnostics)


def exact_policy_l1(grid_a, policy_a, grid_b, policy_b):
    """Integrate the difference of the piecewise linear policies exactly."""
    grid = np.union1d(grid_a, grid_b)
    difference = np.interp(grid, grid_b, policy_b) - np.interp(grid, grid_a, policy_a)
    left, right, width = difference[:-1], difference[1:], np.diff(grid)
    area = width * (abs(left) + abs(right)) / 2
    crossing = left * right < 0
    area[crossing] = (width[crossing] * (left[crossing] ** 2 + right[crossing] ** 2)
                      / (2 * (abs(left[crossing]) + abs(right[crossing]))))
    return float(area.sum())


def wasserstein(grid_a, marginal_a, grid_b, marginal_b):
    """Exact one-dimensional W1 between finite probability distributions."""
    grid = np.union1d(grid_a, grid_b)
    cdf_a = np.r_[0., np.cumsum(marginal_a)]
    cdf_b = np.r_[0., np.cumsum(marginal_b)]
    diff = (cdf_a[np.searchsorted(grid_a, grid[:-1], side="right")]
            - cdf_b[np.searchsorted(grid_b, grid[:-1], side="right")])
    return float(np.sum(np.diff(grid) * abs(diff)))


class Inputs:
    def __init__(self):
        self.hashes = {}

    def name(self, path):
        path = Path(path).resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def record(self, path, expected=None):
        path = Path(path)
        actual = sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"Saved input hash differs: {path}")
        self.hashes[self.name(path)] = actual
        return path

    def json(self, path, expected=None):
        return json.loads(self.record(path, expected).read_text())

    def bundle(self, base, expected=None, checked=False):
        base = Path(base)
        if checked:
            expected = self.json(base.with_name(base.name + "_sha256.json"))
        metadata = self.json(base.with_suffix(".json"), (expected or {}).get(".json"))
        path = self.record(base.with_suffix(".npz"), (expected or {}).get(".npz"))
        with np.load(path, allow_pickle=False) as arrays:
            def restore(value):
                if isinstance(value, dict):
                    if set(value) == {"__array__"}:
                        return arrays[value["__array__"]].copy()
                    return {key: restore(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [restore(item) for item in value]
                return value
            return restore(metadata)


def state_summary(candidate, n):
    grid = np.asarray(candidate["grid"])
    policy = np.asarray(candidate["policy"])
    joint = np.asarray(candidate["nu"])
    if (not np.array_equal(grid, np.linspace(0, 1, n)) or policy.shape != (n,)
            or joint.shape != (n, n) or not np.isfinite(joint).all()
            or joint.min() < -1e-14 or abs(joint.sum() - 1) > 1e-10
            or not np.isfinite(policy).all() or np.any((policy < 0) | (policy > 1))):
        raise ValueError(f"Invalid saved policy/distribution on n={n}")
    marginal = .5 * (joint.sum(axis=0) + joint.sum(axis=1))
    mixed = np.flatnonzero((policy > 1e-9) & (policy < 1 - 1e-9))
    return dict(grid=grid, policy=policy, marginal=marginal,
                mean=float(grid @ marginal), mixed=[dict(x=float(grid[i]), alpha=float(policy[i]))
                                                   for i in mixed])


def normalize_base(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "solution"
    if path.suffix in (".json", ".npz"):
        path = path.with_suffix("")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="refinement_101_201")
    parser.add_argument("--oe-retry", type=Path,
                        help="Optional isolated checked OE rho=.99/n=101 solution or directory")
    parser.add_argument("--output", type=Path, default=DATA / "grid_audit_51_101")
    args = parser.parse_args(argv)
    if Path(args.campaign).name != args.campaign:
        parser.error("--campaign must be a directory name under oblivious/07_full_comparison/data")
    inputs = Inputs()
    original = inputs.json(DATA / "manifest.json")
    conditional = inputs.json(DATA / "conditional_policy_manifest.json")
    conditional_outcomes = inputs.json(DATA / "conditional_outcomes_manifest.json")
    campaign = DATA / args.campaign
    manifest = inputs.json(campaign / "manifest.json")
    config = manifest["configuration"]
    design = inputs.json(DATA / "comparison.json", original["data_files_sha256"]["comparison.json"])
    if config["parameters"] != design["parameters"]:
        raise ValueError("51 and 101 use different economic primitives")
    rhos = sorted(config["rhos"])
    if set(config["models"]) != set(SOURCES) or .999 in rhos:
        raise ValueError("Expected the four-model campaign without rho=.999")
    outcome_path = inputs.record(DATA / "outcomes.csv", original["data_files_sha256"]["outcomes.csv"])
    conditional_path = inputs.record(DATA / "conditional_outcomes.csv",
                                    conditional_outcomes["data_files_sha256"]["conditional_outcomes.csv"])
    outcomes_51 = {}
    for path in (outcome_path, conditional_path):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["metric"] == "product2_market":
                    outcomes_51[row["model"], float(row["rho"])] = row

    rows = []
    for model, source in SOURCES.items():
        for rho in rhos:
            tag = "rho_" + str(rho).replace(".", "p")
            old_base = source / "arrays" / (tag + "_n051")
            source_hashes = (conditional if model == "conditional" else original)["source_files_sha256"]
            expected = {suffix: source_hashes[inputs.name(old_base.with_suffix(suffix))]
                        for suffix in (".json", ".npz")}
            old = inputs.bundle(old_base, expected=expected)
            new_base = campaign / model / tag / "n101/solution"
            retry = model == "oe" and rho == .99 and args.oe_retry is not None
            if retry:
                new_base = normalize_base(args.oe_retry)
                retry_manifest = inputs.json(new_base.parent / "manifest.json")
                retry_config = retry_manifest["configuration"]
                if (retry_manifest.get("source_campaign_fingerprint") != manifest["fingerprint"]
                        or retry_config.get("model") != "oe" or retry_config.get("rho") != .99
                        or retry_config.get("n") != 101
                        or retry_config.get("parameters") != config["parameters"]
                        or any(retry_config.get(key) != config[key] for key in ("order", "tol", "dp_tol"))
                        or resolve_saved_path(retry_config["source_campaign"], project_root=ROOT).resolve() != campaign.resolve()):
                    raise ValueError("OE retry manifest differs from the source campaign/economics")
                expected_retry_fingerprint = sha256(json.dumps(
                    dict(configuration=retry_config, sources=retry_manifest["source_sha256"]),
                    sort_keys=True).encode()).hexdigest()
                if (retry_manifest.get("fingerprint") != expected_retry_fingerprint
                        or retry_config.get("input_sha256") != retry_manifest["input_sha256"]):
                    raise ValueError("OE retry fingerprint or declared input hashes differ")
                for collection in (retry_manifest["input_sha256"], retry_manifest["source_sha256"]):
                    for name, expected_hash in collection.items():
                        path = resolve_saved_path(name, project_root=ROOT)
                        inputs.record(path, expected_hash)
            new = inputs.bundle(new_base, checked=True)
            if (new.get("rho") != rho or new.get("n") != 101
                    or new["configuration"]["parameters"] != dict(config["parameters"], rho=rho)):
                raise ValueError(f"Incompatible 101 checkpoint: {new_base}")
            expected_fingerprint = retry_manifest["fingerprint"] if retry else manifest["fingerprint"]
            if new.get("campaign_fingerprint") != expected_fingerprint:
                raise ValueError(f"Campaign fingerprint mismatch: {new_base}")
            if retry and new.get("source_campaign_fingerprint") != manifest["fingerprint"]:
                raise ValueError("OE retry solution does not identify the original campaign")
            old_valid = certified(old.get("selected")) and old.get("demand_audit", {}).get("verified", False)
            new_valid = certified(new.get("selected")) and new.get("demand_audit", {}).get("verified", False)
            if not old_valid:
                raise ValueError(f"The original 51 result lacks saved certificates: {old_base}")
            a = state_summary(old["selected"], 51)
            old_outcome = outcomes_51[model, rho]
            old_usage = float(old_outcome["stationary"])
            row = dict(model=model, rho=rho, n51_certified=bool(old_valid), n101_certified=bool(new_valid),
                       n51_candidate_count=len(old.get("candidates", [])),
                       n101_candidate_count=len(new.get("candidates", [])),
                       n51_certified_candidate_count=sum(certified(c) for c in old.get("candidates", [])),
                       n101_certified_candidate_count=sum(certified(c) for c in new.get("candidates", [])),
                       n51_usage=old_usage, n101_usage=None, usage_change_pp=None,
                       n51_mean_reputation=a["mean"], n101_mean_reputation=None,
                       mean_reputation_change=None, marginal_wasserstein_1=None, policy_l1=None,
                       n51_selected_p0=old["selected"].get("p0"),
                       n101_selected_p0=(new.get("selected") or {}).get("p0"),
                       n51_mixed_nodes=json.dumps(a["mixed"], separators=(",", ":")),
                       n101_mixed_nodes=None,
                       n51_simulation_usage=float(old_outcome["simulation_mean"]),
                       n101_simulation_usage=None, n101_simulation_stationary_gap_pp=None,
                       source_n51=inputs.name(old_base), source_n101=inputs.name(new_base),
                       isolated_oe_retry=bool(retry),
                       branch_tracking="not established; detected candidates may differ across grids")
            if new_valid:
                b = state_summary(new["selected"], 101)
                usage = float(new["predicted"]["product2_market"])
                row.update(n101_usage=usage, usage_change_pp=100 * (usage - old_usage),
                           n101_mean_reputation=b["mean"], mean_reputation_change=b["mean"] - a["mean"],
                           marginal_wasserstein_1=wasserstein(a["grid"], a["marginal"], b["grid"], b["marginal"]),
                           policy_l1=exact_policy_l1(a["grid"], a["policy"], b["grid"], b["policy"]),
                           n101_mixed_nodes=json.dumps(b["mixed"], separators=(",", ":")))
                simulation_base = new_base.parent / "simulation_grid"
                if simulation_base.with_name("simulation_grid_sha256.json").exists():
                    simulation = inputs.bundle(simulation_base, checked=True)
                    policy_hash = sha256(b["policy"].tobytes()).hexdigest()
                    metadata = simulation["metadata"]
                    if (simulation.get("campaign_fingerprint") != expected_fingerprint
                            or (retry and simulation.get("source_campaign_fingerprint") != manifest["fingerprint"])
                            or simulation.get("policy_sha256") != policy_hash
                            or metadata.get("params") != new["configuration"]["parameters"]
                            or metadata.get("mode") != "grid"
                            or metadata.get("initial") != design["initial"]
                            or any(metadata.get(key) != config[key]
                                   for key in ("periods", "burn_in", "replications", "seed"))):
                        raise ValueError("Saved simulation uses different policy, settings or provenance")
                    for metric in ("product2_market", "seller_profit"):
                        if abs(simulation["predicted"][metric] - new["predicted"][metric]) > 1e-12:
                            raise ValueError("Saved solution and simulation predictions differ")
                    sim_usage = float(simulation["summary"]["product2_market"]["mean"])
                    row.update(n101_simulation_usage=sim_usage,
                               n101_simulation_stationary_gap_pp=100 * (sim_usage - usage))
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(scope="Saved finite-grid certificates and outcomes only; no new solves or simulations",
                  comparison="51 to 101 selected equilibria; rho=.999 excluded", models=list(SOURCES), rhos=rhos,
                  all_101_certified=all(row["n101_certified"] for row in rows),
                  continuous_state_convergence_established=False,
                  caveats=["Two grids do not establish convergence to a continuous-state equilibrium.",
                           "Multiple detected candidates can change selection; continuation of the same branch is not established.",
                           "Monte Carlo gaps include initialization and slow mixing; they are separate from grid differences.",
                           "W1 uses the mean of the two seller reputation marginals, in reputation units.",
                           "No result from an incomplete 201 run is used."],
                  numpy_version=np.__version__, input_sha256=inputs.hashes,
                  code_sha256={inputs.name(Path(__file__)): sha256(Path(__file__).read_bytes()).hexdigest()},
                  output_sha256={csv_path.name: sha256(csv_path.read_bytes()).hexdigest()}, rows=rows)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} comparisons in {args.output}; certified 101: "
          f"{sum(row['n101_certified'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
