"""Read certified fixed-p0 policies and reconstruct their actual grid games.

No equilibrium, dynamic program, or stationary distribution is solved here.
The module never writes files. Historical report dependencies on removed
figures are recorded separately from the verified numerical dependencies.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fixed-p0-mc-mpl")

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.core import fixed_demand
from oblivious.oe_p0.code.policy import transition_matrix
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_distribution_fixed_p0.code.multi_p0_report import validate_saved_case
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle


P0S = (.1, .3, .5, .7, .8, .9)
RHOS = (.6, .7, .8)
MODELS = ("monopoly", "competition", "conditional_competition")
CALIBRATION = dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98)
TOLERANCES = dict(row_sum=1e-12, probability_mass=1e-10,
                  stationary_l1=1e-10, outcome=1e-10, saved_demand=1e-12)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class _Inputs:
    def __init__(self):
        self.verified = {}
        self.missing_visual = {}

    def verify(self, declared):
        _require(isinstance(declared, dict), "Malformed input fingerprint map")
        for name, expected in declared.items():
            _require(isinstance(name, str) and isinstance(expected, str),
                     "Malformed input fingerprint")
            path = ROOT / name
            _require(not Path(name).is_absolute()
                     and path.resolve().is_relative_to(ROOT.resolve()),
                     "Input must be inside the repository: " + name)
            previous = self.verified.get(name, self.missing_visual.get(name))
            _require(previous is None or previous == expected,
                     "Conflicting input fingerprints: " + name)
            if not path.is_file():
                # These archived report figures were moved/removed after the
                # numerical publication. Missing data, code, or audits never
                # qualify for this exception, including other JSON files.
                prefix = str(HOME.relative_to(ROOT)) + "/figures/"
                if name.startswith(prefix) and path.suffix.lower() in (".png", ".pdf", ".json"):
                    self.missing_visual[name] = expected
                    continue
                raise ValueError("Required numerical input is missing: " + name)
            if name not in self.verified:
                _require(_digest(path) == expected, "Input fingerprint differs: " + name)
                self.verified[name] = expected

    def read_json(self, path):
        path = Path(path)
        name = str(path.relative_to(ROOT))
        encoded = path.read_bytes()
        self.verify({name: sha256(encoded).hexdigest()})
        return json.loads(encoded)

    def recheck(self):
        for name, expected in self.verified.items():
            _require(_digest(ROOT/name) == expected, "Input changed while loading: " + name)


def _read_publication(folder, rho, inputs):
    report_path = folder / "comparison_report_manifest.json"
    report = inputs.read_json(report_path)
    retained = report.get("common_retained_rhos", report.get("retained_rhos", []))
    _require(report.get("displayed_n") == 101 and rho in retained,
             "Requested case is outside the published n101 comparison")
    _require(report.get("independent_audit", {}).get("verified") is True,
             "Source report has no verified numerical independent review")
    for field in ("source_input_sha256", "reporting_source_sha256",
                  "generated_data_sha256", "figure_files"):
        declared = report.get(field)
        _require(bool(declared), "Source report has no " + field)
        inputs.verify(declared)
    audit_path = ROOT / report["independent_audit"]["path"]
    audit = inputs.read_json(audit_path)
    _require(audit.get("verified") is True and rho in audit.get("rhos", []),
             "Independent numerical review is unverified or lacks the requested rho")
    inputs.verify(audit["audited_input_sha256"])
    if audit.get("audited_bundle_sha256"):
        inputs.verify(audit["audited_bundle_sha256"])
    for source_field, hash_field in (("audit_source", "audit_source_sha256"),
                                    ("audit_entrypoint", "audit_entrypoint_sha256")):
        if audit.get(source_field):
            inputs.verify({audit[source_field]: audit[hash_field]})
    return report, audit, dict(
        report_path=str(report_path.relative_to(ROOT)), report_sha256=_digest(report_path),
        audit_path=str(audit_path.relative_to(ROOT)), audit_sha256=_digest(audit_path),
        audit_verified=True, audited_p0s=audit["p0s"], audited_rhos=audit["rhos"],
        audited_bundle_count=audit["audited_bundle_count"])


def _source(p0, rho, n, closure):
    rho_tag = "rho_" + str(float(rho)).replace(".", "p")
    stem = f"{rho_tag}_n{n:03d}"
    if p0 == .5:
        campaign = "p0_0p5_common_grid" if closure == "marginal" else "p0_0p5_conditional_fixed"
        return HOME/"data"/campaign/rho_tag, stem
    p0_tag = "p0_" + str(float(p0)).replace(".", "p")
    return HOME/"data/p0_sweep"/p0_tag/rho_tag, stem + "_" + closure


def _load_bundle(p0, rho, n, closure, baseline, audit, inputs):
    directory, stem = _source(p0, rho, n, closure)
    manifest_path = directory / "manifest.json"
    manifest = inputs.read_json(manifest_path)
    _require(manifest.get("status") != "running" and manifest.get("current_grid") is None
             and manifest.get("current_model") is None and stem in manifest["completed"],
             "Checkpoint is not a committed, inactive source: " + str(directory))
    for field in ("numerical_sources", "reference_input_sha256"):
        if manifest.get(field):
            inputs.verify(manifest[field])
    for field in ("worker_archive_sha256", "diagnostic_artifact_sha256"):
        if manifest.get(field):
            inputs.verify({str((directory/name).relative_to(ROOT)): expected
                           for name, expected in manifest[field].items()})
    for suffix in (".json", ".npz"):
        path = directory/"arrays"/(stem+suffix)
        inputs.verify({str(path.relative_to(ROOT)): manifest["checkpoint_sha256"][path.name]})
    bundle = check_bundle(directory, stem, manifest)
    config, declared = bundle["configuration"], manifest["configuration"]
    _require(config["parameters"] == dict(CALIBRATION, rho=rho),
             "The saved economic calibration differs from the c2=.65 comparison")
    _require(declared["p0"] == p0 and declared["rho"] == rho
             and config["order"] == declared["order"]
             and config["tolerance"] == declared["tol"]
             and config["dp_tolerance"] == declared["dp_tol"]
             and declared["parameters"] == CALIBRATION,
             "Source manifest/checkpoint configuration mismatch")
    validate_saved_case(bundle, p0, rho, n, closure, baseline)
    if bundle.get("reference_input_sha256"):
        inputs.verify(bundle["reference_input_sha256"])
    # The original p0=.5 baseline predates the multi-p0 audit. It retains its
    # own certified bundles and is covered by the publication's input hashes.
    independently_audited = p0 in audit.get("p0s", [])
    if independently_audited:
        matches = [case for case in audit["cases"] if case.get("p0") == p0
                   and case.get("rho") == rho and case.get("n") == n
                   and case.get("closure") == closure]
        _require(len(matches) == 1 and matches[0].get("verified") is True,
                 "Dedicated independent review does not certify this bundle")
    source = dict(p0=p0, rho=rho, n=n, closure=closure,
                  manifest=str(manifest_path.relative_to(ROOT)),
                  checkpoint=str((directory/"arrays"/stem).relative_to(ROOT)),
                  independent_multi_p0_audit_record=independently_audited,
                  certification="Stored verified policy/demand certificate; fresh actual-kernel checks below")
    return bundle, source


def _flows(selection, policy):
    outside, own_sale = selection[0], selection[1]
    own_alpha = policy if own_sale.ndim == 1 else policy[:, None]
    own_p2 = own_sale * own_alpha
    b_sale = np.zeros_like(own_sale) if len(selection) == 2 else selection[2]
    b_p2 = np.zeros_like(own_sale) if len(selection) == 2 else b_sale * policy[None, :]
    return dict(own_sale_rate=own_sale, own_product2_rate=own_p2,
                customer_product2_rate=own_p2+b_p2, outside_rate=outside,
                b_sale_rate=b_sale, b_product2_rate=b_p2,
                customer_sale_rate=own_sale+b_sale)


def _validate_case(case):
    law = case["saved_law"].ravel()
    transition, selection = case["transition"], case["selection"]
    _require(np.isfinite(law).all() and np.all(law >= 0)
             and abs(law.sum()-1) <= TOLERANCES["probability_mass"], "Invalid saved stationary law")
    _require(np.isfinite(selection).all() and np.all(selection >= 0), "Invalid selection tensor")
    selection_error = float(abs(selection.sum(axis=0)-1).max())
    row_error = float(abs(np.asarray(transition.sum(axis=1)).ravel()-1).max())
    stationary_error = float(abs(transition.T@law-law).sum())
    _require(np.isfinite(transition.data).all() and np.all(transition.data >= 0)
             and row_error <= TOLERANCES["row_sum"]
             and selection_error <= TOLERANCES["row_sum"]
             and stationary_error <= TOLERANCES["stationary_l1"],
             "Fresh actual-grid transition/stationarity check failed")
    flows = _flows(selection, case["policy"])
    values = {key: float(law@value.ravel()) for key, value in flows.items()}
    _require(values["own_sale_rate"] > 0, "No representable stationary own sales")
    values["product2_given_sale"] = values["own_product2_rate"]/values["own_sale_rate"]
    row = case["outcome"]
    sellers = len(selection)-1
    expected = dict(own_sale_rate=row["sale_rate"], own_product2_rate=row["product2_calendar"],
                    product2_given_sale=row["product2_conditional"], outside_rate=row["outside_rate"],
                    customer_product2_rate=sellers*row["product2_calendar"],
                    customer_sale_rate=sellers*row["sale_rate"],
                    b_sale_rate=(sellers-1)*row["sale_rate"],
                    b_product2_rate=(sellers-1)*row["product2_calendar"])
    errors = {key: abs(values[key]-float(value)) for key, value in expected.items()}
    _require(max(errors.values()) <= TOLERANCES["outcome"],
             "Fresh stationary actual-flow check differs from saved outcomes")
    case["flow_rewards"] = flows
    case["stationary_metrics"] = values
    case["validation"] = dict(transition_row_sum_max=row_error,
        selection_partition_max=selection_error, stationary_l1=stationary_error,
        outcome_absolute_errors=errors, verified=True)


def load_cases(p0s, rho=.7, n=101):
    """Return actual monopoly/marginal/conditional grid games and provenance.

    Supports the published common n101 cases, not the paused n201 campaign.
    Case model names are ``monopoly``, ``competition`` (marginal OE), and
    ``conditional_competition``. Selection axes are [outside, A, B], with B
    omitted in monopoly. A case's ``saved_law`` is the actual calendar law.
    """
    p0s = list(map(float, p0s))
    rho = float(rho)
    _require(p0s and len(set(p0s)) == len(p0s) and set(p0s).issubset(P0S),
             "Use unique p0 values from the six published cases")
    _require(n == 101 and rho in RHOS, "Only published common rho cases on n101 are supported")
    inputs = _Inputs()
    main_report, main_audit, main_summary = _read_publication(HOME/"data/p0_sweep", rho, inputs)
    reports = [main_summary]
    extension = None
    if .9 in p0s:
        extension = _read_publication(HOME/"data/p0_sweep/p0_0p9", rho, inputs)
        _require(extension[0]["shared_economic_configuration"] == main_report["shared_economic_configuration"],
                 "The p0=.9 extension has a different calibration")
        reports.append(extension[2])
    baseline = main_report["shared_economic_configuration"]
    cases, sources, games = [], [], {}
    for p0 in p0s:
        audit = extension[1] if p0 == .9 else main_audit
        marginal, source = _load_bundle(p0, rho, n, "marginal", baseline, audit, inputs)
        conditional, conditional_source = _load_bundle(p0, rho, n, "conditional", baseline, audit, inputs)
        sources.extend((source, conditional_source))
        for model, bundle, result_key, source_record in (
                ("monopoly", marginal, "monopoly", source),
                ("competition", marginal, "competition", source),
                ("conditional_competition", conditional, "competition", conditional_source)):
            result = bundle[result_key]
            grid = np.asarray(result["grid"], dtype=float).copy()
            policy = np.asarray(result["policy"], dtype=float).copy()
            config = bundle["configuration"]
            params = Parameters(**config["parameters"])
            if model == "monopoly":
                demand = fixed_demand(grid, p0, params)
                _require(float(abs(demand-result["demand"]).max()) <= TOLERANCES["saved_demand"],
                         "Saved monopoly demand differs from the Beta selection rule")
                selection = np.stack((1-demand, demand))
                transition = transition_matrix(grid, demand, policy, params)
                law = np.asarray(result["mu"], dtype=float).copy()
            else:
                if p0 not in games:
                    games[p0] = OutsideOptionGame(grid, params, p0, config["order"])
                game = games[p0]
                selection = np.stack((game.outside, game.D, game.D.T))
                shape = (n,n)
                transition = game.transition(np.broadcast_to(policy[:,None], shape),
                                             np.broadcast_to(policy[None,:], shape))
                law = np.asarray(result["nu"], dtype=float).copy()
            outcome = [row for row in bundle["outcomes"] if row["model"] == model]
            _require(len(outcome) == 1, "Missing or duplicate saved model outcome")
            case = dict(p0=p0, rho=rho, n=n, model=model, grid=grid, policy=policy,
                        params=params, selection=selection, transition=transition.tocsr(),
                        saved_law=law, outcome=dict(outcome[0]), configuration=config,
                        source=dict(source_record))
            _validate_case(case)
            cases.append(case)
    own_source = Path(__file__).resolve()
    inputs.verify({str(own_source.relative_to(ROOT)): _digest(own_source)})
    inputs.recheck()
    provenance = dict(
        verified_numerical_inputs=True, p0s=p0s, rho=rho, n=n,
        parameters=dict(CALIBRATION, rho=rho), models=list(MODELS),
        source_input_sha256=dict(sorted(inputs.verified.items())), source_publications=reports,
        checkpoint_sources=sources, tolerances=TOLERANCES.copy(),
        missing_legacy_visual_inputs=[dict(path=name, expected_sha256=expected,
            reason="Missing historical figure dependency after figure relocation/removal")
            for name, expected in sorted(inputs.missing_visual.items())],
        historical_graphics_audit="Not claimed; missing legacy figure dependencies are diagnostics, not numerical inputs",
        baseline_p0_half_audit="Original certified p0=.5 bundles and their publication input hashes; "
                               "the dedicated multi-p0 independent review covers the other p0 values",
        kernel_scope="Actual finite-grid calendar game; full joint rival state, not perceived one-dimensional OE demand",
        reconstruction="No equilibrium, dynamic program, or stationary distribution solved; no source or output files written",
        initial_state="Mature reputation .5 for every strategic seller, not zero-observation prior learning",
        validation=[dict(p0=case["p0"], model=case["model"], **case["validation"]) for case in cases])
    return cases, provenance


def exact_reference(case, periods=100):
    """Forward-propagate from mature .5 and return pre-transition flows.

    ``period`` is zero-based for matching the original Monte Carlo tables;
    ``calendar_period`` is its one-based display label. Ratios are formed from
    expected product-2 and expected own-sale flows, not statewise ratios.
    """
    _require(isinstance(periods, (int, np.integer)) and periods >= 1,
             "Use a positive integer number of periods")
    grid = case["grid"]
    half = int(np.argmin(abs(grid-.5)))
    _require(abs(grid[half]-.5) <= 1e-14, "The mature initial state .5 is not on the saved grid")
    shape = case["saved_law"].shape
    law = np.zeros(int(np.prod(shape)))
    law[np.ravel_multi_index((half,)*len(shape), shape)] = 1.
    rewards = case.get("flow_rewards", _flows(case["selection"], case["policy"]))
    keys = list(rewards)
    reward_matrix = np.vstack([rewards[key].ravel() for key in keys])
    forward = case["transition"].T.tocsr()
    records = []
    for period in range(periods):
        values = reward_matrix@law
        row = dict(p0=case["p0"], rho=case["rho"], n=case["n"], model=case["model"],
                   period=period, calendar_period=period+1, **dict(zip(keys, map(float, values))))
        row["product2_given_sale"] = (row["own_product2_rate"]/row["own_sale_rate"]
                                          if row["own_sale_rate"] > 0 else np.nan)
        row["probability_mass_error"] = abs(float(law.sum())-1)
        _require(np.isfinite(law).all() and np.all(law >= 0)
                 and row["probability_mass_error"] <= TOLERANCES["probability_mass"],
                 "Exact forward propagation lost probability mass or positivity")
        records.append(row)
        law = forward@law
    return pd.DataFrame(records)

