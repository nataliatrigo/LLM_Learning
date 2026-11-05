"""Deterministic finite-horizon validation of already saved policy kernels.

This module never solves or modifies an equilibrium. It propagates the initial
joint law through the existing transition matrix and separates initial-state
bias from Monte Carlo error. Continuous policy extensions are outside its scope.
"""

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from oblivious.equilibrium_comparison.code.inputs import predictions


LINEAR_METRICS = (
    "mean_reputation_A", "mean_reputation_B", "mean_posterior_A", "mean_posterior_B",
    "market_share_A", "market_share_B", "product2_calendar_A", "product2_calendar_B",
    "product2_market", "profit_A", "profit_B", "industry_profit", "success_rate",
    "failure_rate", "expected_success_probability",
)


def _metrics(pair, key, law):
    """Linear expectations; no claim about expectation of sample variances."""
    original = pair["profiles"][key]
    alternate = dict(pair, profiles={key: dict(original, nu=law)})
    all_metrics = predictions(alternate, key)
    result = {name: all_metrics[name] for name in LINEAR_METRICS}
    grid = pair["grid"]
    for seller, marginal in (("A", law.sum(axis=1)), ("B", law.sum(axis=0))):
        result[f"mean_ts_{seller}"] = result[f"mean_posterior_{seller}"]
        result[f"mean_error_{seller}"] = (
            result[f"mean_reputation_{seller}"] - original["predicted"][f"mean_reputation_{seller}"])
        for threshold in (.2, .3, .4):
            suffix = str(threshold).replace(".", "p")
            result[f"rare_mass_le_{suffix}_{seller}"] = float(marginal[grid <= threshold].sum())
    if key == "p0":
        result["p0_mean_error"] = result["mean_posterior_B"] - original["p0"]
        result["p0_ts_error"] = result["mean_ts_B"] - original["p0"]
    return result


def validate_transients(pair, windows, checkpoints=(1000, 10000, 100000, 200000, 220000)):
    """Return exact finite-grid expected outcomes for windows ``name:(start,length)``.

    Windows use zero-based pre-selection calendar states, exactly as raw
    simulation records do. ``profile_keys`` can be restricted by passing a pair
    whose profiles dictionary contains only one profile; otherwise both are
    computed. The horizon is the maximum requested window end/checkpoint.
    """
    if not windows:
        raise ValueError("At least one analysis window is required.")
    normalized = {}
    for name, specification in windows.items():
        if not isinstance(name, str) or len(specification) != 2:
            raise ValueError("Windows must map names to (start, length).")
        start, length = specification
        if (isinstance(start, bool) or isinstance(length, bool) or int(start) != start
                or int(length) != length or start < 0 or length < 1):
            raise ValueError("Window starts must be nonnegative integers and lengths positive integers.")
        normalized[name] = (int(start), int(length))
    if any(isinstance(point, bool) or int(point) != point or point < 0 for point in checkpoints):
        raise ValueError("Checkpoints must be nonnegative integer times.")
    checkpoints = sorted(set(map(int, checkpoints)))
    cuts = {0} | {edge for start, length in normalized.values() for edge in (start, start + length)}
    horizon = max(cuts | set(checkpoints))
    grid = np.asarray(pair["grid"])
    center = np.flatnonzero(grid == .5)
    if len(center) != 1:
        raise ValueError("The common mature initial state .5 must be an exact grid node.")
    n = len(grid)
    output = dict(
        method="Repeated sparse propagation q_(t+1)=P.T @ q_t; expected pre-selection window law=sum(q_t)/length.",
        initial_state=[.5, .5], horizon=horizon, parameters=asdict(pair["params"]),
        grid_points=n, windows={name: dict(start=start, length=length) for name, (start, length) in normalized.items()},
        checkpoints=checkpoints, profiles={}, no_equilibrium_solved=True,
        numerical_scope="Existing finite-grid transition matrices; selection integration error is documented in the original input audit. These are deterministic expectations, not Monte Carlo estimates or continuous-state equilibrium claims.",
        source_input_audit=pair.get("audit", {}),
        validation_module_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for key, profile in pair["profiles"].items():
        started = time.monotonic()
        transition = profile["transition"].T.tocsr()
        law = np.zeros(n * n)
        law[int(center[0]) * n + int(center[0])] = 1.
        stationary = profile["nu"]
        total = np.zeros_like(law)
        prefixes = {0: total.copy()}
        snapshots = {}

        def snapshot(point):
            shaped = law.reshape(n, n)
            snapshots[str(point)] = dict(
                time=point, probability_sum=float(law.sum()), minimum_probability=float(law.min()),
                l1_distance_to_stationary=float(np.abs(shaped - stationary).sum()),
                metrics=_metrics(pair, key, shaped),
            )

        if 0 in checkpoints:
            snapshot(0)
        for period in range(horizon):
            total += law
            law = transition @ law
            point = period + 1
            if point in cuts:
                prefixes[point] = total.copy()
            if point in checkpoints:
                snapshot(point)
        if not np.isfinite(law).all() or law.min() < -1e-13 or abs(law.sum() - 1) > 1e-8:
            raise ArithmeticError(f"Probability mass was not preserved for {key}.")
        stationarity_residual = float(np.max(np.abs(transition @ stationary.ravel() - stationary.ravel())))
        record = dict(label=profile["label"], stationary_metrics=_metrics(pair, key, stationary),
                      checkpoints=snapshots, windows={}, saved_stationarity_residual_max=stationarity_residual,
                      final_probability_sum=float(law.sum()), final_minimum_probability=float(law.min()),
                      final_l1_distance_to_stationary=float(np.abs(law - stationary.ravel()).sum()),
                      elapsed_seconds=time.monotonic() - started)
        for name, (start, length) in normalized.items():
            average = ((prefixes[start + length] - prefixes[start]) / length).reshape(n, n)
            record["windows"][name] = dict(
                start=start, length=length, probability_sum=float(average.sum()),
                l1_distance_to_stationary=float(np.abs(average - stationary).sum()),
                expected_metrics=_metrics(pair, key, average),
            )
        output["profiles"][key] = record
    return output


def write_validation(pair, analysis_summary_rows, output_dir, windows, *,
                     checkpoints=(1000, 10000, 100000, 200000, 220000)):
    """Write deterministic JSON/CSV and compare with replication-based grid CIs.

    Returns the validation dictionary. It reads the supplied summary rows only;
    it never changes the raw simulations, saved equilibria, or previous report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_transients(pair, windows, checkpoints=checkpoints)
    lookup = {(row["profile"], row["window"], row["metric"]): row
              for row in analysis_summary_rows if row.get("mode") == "grid"}
    comparisons = []
    for key, record in validation["profiles"].items():
        stationary = record["stationary_metrics"]
        for window, result in record["windows"].items():
            for metric, expected in result["expected_metrics"].items():
                simulated = lookup.get((key, window, metric), {})
                mean, low, high = (simulated.get(field) for field in ("mean", "ci_low", "ci_high"))
                interval_available = low is not None and high is not None
                comparisons.append(dict(
                    profile=key, window=window, start=result["start"], length=result["length"],
                    metric=metric, expected_finite_window=expected, stationary_prediction=stationary[metric],
                    initial_condition_bias=expected - stationary[metric],
                    simulation_mean=mean, simulation_se=simulated.get("se"),
                    simulation_ci_low=low, simulation_ci_high=high, replications=simulated.get("n_valid"),
                    simulation_minus_finite_expectation=None if mean is None else mean - expected,
                    ci_contains_finite_expectation=None if not interval_available else bool(low <= expected <= high),
                    ci_contains_stationary_prediction=None if not interval_available else bool(low <= stationary[metric] <= high),
                ))
    validation["monte_carlo_comparisons"] = comparisons
    (output_dir / "transient_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
    with (output_dir / "transient_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "FINITE-HORIZON VERSUS STATIONARY VALIDATION", "",
        "No equilibrium was solved. The already saved joint transition matrices were propagated",
        "deterministically from (.5,.5). Every reported window expectation is free of Monte Carlo noise.",
        "The calculation concerns the discrete grid kernel; continuous policy extensions remain separate.", "",
    ]
    for key, record in validation["profiles"].items():
        lines += [record["label"],
                  f"Saved stationary-law residual (maximum coordinate): {record['saved_stationarity_residual_max']:.9g}.",
                  f"Final law L1 distance from stationarity at t={validation['horizon']}: {record['final_l1_distance_to_stationary']:.9g}.",
                  f"Final probability sum: {record['final_probability_sum']:.14g}."]
        for window, result in record["windows"].items():
            metrics = result["expected_metrics"]
            lines += [f"  {window}: start={result['start']}, retained={result['length']}.",
                      f"    Expected mean reputation B {metrics['mean_reputation_B']:.9g}; stationary {record['stationary_metrics']['mean_reputation_B']:.9g}.",
                      f"    Expected posterior mean B {metrics['mean_posterior_B']:.9g}; stationary {record['stationary_metrics']['mean_posterior_B']:.9g}.",
                      f"    Expected mass X_B<=.2: {metrics['rare_mass_le_0p2_B']:.9g}; stationary {record['stationary_metrics']['rare_mass_le_0p2_B']:.9g}."]
            for metric in ("mean_reputation_B", "mean_posterior_B", "product2_market"):
                row = next(row for row in comparisons if row["profile"] == key and row["window"] == window and row["metric"] == metric)
                lines.append(f"    {metric} finite-window minus stationary: {row['initial_condition_bias']:.9g}.")
                if row["simulation_mean"] is not None:
                    lines.append(f"      Monte Carlo {row['simulation_mean']:.9g}; 95% CI [{row['simulation_ci_low']:.9g}, {row['simulation_ci_high']:.9g}]; contains finite-window expectation: {row['ci_contains_finite_expectation']}; contains stationary prediction: {row['ci_contains_stationary_prediction']}.")
        lines.append("")
    lines += [
        "INTERPRETATION",
        "A nonzero deterministic window bias is an initial-condition/slow-convergence effect,",
        "not a failed stationary fixed-point equation. A small stationary residual does not imply fast mixing.",
        "Replication intervals describe finite-horizon sampling uncertainty and do not remove initial-state bias.",
        "Rare, persistent low-reputation episodes can make the distribution of replication means skewed.",
        "An interval excluding the exact finite-window expectation can occur through sampling variation or",
        "limited accuracy of the replication-t approximation; it does not by itself diagnose an equilibrium error.",
        "Check reputation and scalar-consistency diagnostics separately from product use/profit/success:",
        "flow outcomes may stabilize much sooner than the calendar-time reputation distribution.",
    ]
    (output_dir / "transient_validation.txt").write_text("\n".join(lines) + "\n")
    return validation
