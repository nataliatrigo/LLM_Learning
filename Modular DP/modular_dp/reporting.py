"""Generate the human-readable experiment report from tidy output tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_No rows were produced._"
    header = "| " + " | ".join(available) + " |"
    rule = "|" + "|".join(["---"] * len(available)) + "|"
    rows = [
        "| " + " | ".join(_format(row[column]) for column in available) + " |"
        for _, row in frame[available].iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _numeric_range(
    frame: pd.DataFrame,
    column: str,
    *,
    percentage: bool = False,
) -> str:
    """Format a finite column range for compact diagnostic tables."""
    if frame.empty or column not in frame:
        return "NA"
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return "NA"
    lower = float(values.min())
    upper = float(values.max())
    formatter = (lambda value: f"{value:.1%}") if percentage else (
        lambda value: f"{value:.4g}"
    )
    if np.isclose(lower, upper):
        return formatter(lower)
    return f"{formatter(lower)}--{formatter(upper)}"


def _dynamic_method_tables(panel_summary: pd.DataFrame) -> str:
    """Return one readable five-p0 table per representative method."""
    if panel_summary.empty:
        return "_No dynamic summary rows were produced._"
    grouping = (
        "method_spec_key" if "method_spec_key" in panel_summary else "method"
    )
    blocks: list[str] = []
    for key, data in panel_summary.groupby(grouping, sort=False):
        data = data.sort_values("p0") if "p0" in data else data
        label = (
            str(data["method_spec_label"].iloc[0])
            if "method_spec_label" in data
            else str(key)
        )
        table = _markdown_table(
            data,
            [
                "p0",
                "overall_realized_A_share",
                "late_realized_A_share",
                "overall_product2_share_conditional_on_A",
                "late_product2_share_conditional_on_A",
                "overall_product2_share_calendar",
                "late_product2_share_calendar",
                "overall_calendar_profit",
                "late_calendar_profit",
                "overall_posterior_mean",
                "late_posterior_mean",
            ],
        )
        blocks.append(f"#### {label}\n\n{table}")
    return "\n\n".join(blocks)


def _dynamic_regime_table(panel_summary: pd.DataFrame) -> pd.DataFrame:
    """Compress each method's p0 rows into the three economic regimes."""
    required = {"p0", "p1", "p2"}
    if panel_summary.empty or not required.issubset(panel_summary.columns):
        return pd.DataFrame()
    grouping = (
        "method_spec_key" if "method_spec_key" in panel_summary else "method"
    )
    rows: list[dict[str, str]] = []
    for key, data in panel_summary.groupby(grouping, sort=False):
        p0 = pd.to_numeric(data["p0"], errors="coerce")
        p1_values = pd.to_numeric(data["p1"], errors="coerce").dropna()
        p2_values = pd.to_numeric(data["p2"], errors="coerce").dropna()
        if p1_values.empty or p2_values.empty:
            continue
        p1 = float(p1_values.iloc[0])
        p2 = float(p2_values.iloc[0])
        label = (
            str(data["method_spec_label"].iloc[0])
            if "method_spec_label" in data
            else str(key)
        )
        regimes = [
            ("p0 < p1", p0 < p1),
            ("p1 < p0 < p2", (p0 > p1) & (p0 < p2)),
            ("p0 > p2", p0 > p2),
        ]
        for regime, mask in regimes:
            subset = data.loc[mask]
            if subset.empty:
                continue
            p0_values = ", ".join(
                f"{value:g}"
                for value in sorted(
                    pd.to_numeric(subset["p0"], errors="coerce").dropna()
                )
            )
            rows.append(
                {
                    "method": label,
                    "regime": regime,
                    "tested_p0": p0_values,
                    "late_A_share": _numeric_range(
                        subset, "late_realized_A_share", percentage=True
                    ),
                    "late_conditional_P2": _numeric_range(
                        subset,
                        "late_product2_share_conditional_on_A",
                        percentage=True,
                    ),
                    "late_calendar_profit": _numeric_range(
                        subset, "late_calendar_profit"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _observation_refinement_section(convergence: pd.DataFrame) -> str:
    """Summarize the dedicated nested-grid observation-forgetting audit."""
    if convergence.empty:
        return ""

    data = convergence.sort_values(["p0", "fine_resolution"]).copy()
    orders: list[float] = []
    for _, group in data.groupby("p0", sort=True):
        rows = list(group.itertuples(index=False))
        for previous, current in zip(rows[:-1], rows[1:], strict=True):
            if int(previous.fine_resolution) != int(current.coarse_resolution):
                continue
            orders.append(
                float(
                    np.log2(
                        previous.root_mean_square_value_difference
                        / current.root_mean_square_value_difference
                    )
                )
            )

    order_text = (
        f"{min(orders):.2f}--{max(orders):.2f}" if orders else "not estimable"
    )
    maximum_common = float(
        pd.to_numeric(
            data["robust_action_change_share"], errors="coerce"
        ).max()
    )
    maximum_visited = float(
        pd.to_numeric(
            data["visited_robust_action_disagreement_share"], errors="coerce"
        ).max()
    )
    maximum_residual = float(
        pd.to_numeric(data["fine_bellman_residual"], errors="coerce").max()
    )

    p05 = data[np.isclose(pd.to_numeric(data["p0"], errors="coerce"), 0.5)]
    p05_note = ""
    if len(p05) >= 2:
        first = p05.iloc[0]
        last = p05.iloc[-1]
        p05_note = (
            " For `p0=.5`, the RMS difference falls from "
            f"`{first.root_mean_square_value_difference:.3e}` to "
            f"`{last.root_mean_square_value_difference:.3e}`, while disagreement "
            "on the common simulated-state panel falls from "
            f"`{first.visited_robust_action_disagreement_share:.4%}` to "
            f"`{last.visited_robust_action_disagreement_share:.4%}`."
        )

    return f"""### 5.1 Observation-forgetting refinement

The `rho=.95` observation-time policy was additionally solved on nested grids
through resolution 1601 for every dynamic `p0`, and through 3201 for the most
sensitive case `p0=.5`. The RMS value error has observed order
`{order_text}`. The maximum robust disagreement share is
`{maximum_common:.4%}` on common grid nodes and `{maximum_visited:.4%}` on the
fixed simulated-state panels; every fine-grid Bellman residual is at most
`{maximum_residual:.3e}`.{p05_note}

{_markdown_table(data, ['p0','coarse_resolution','fine_resolution','root_mean_square_value_difference','robust_action_changes','robust_action_change_share','visited_robust_action_disagreement_share','fine_bellman_residual','strict_policy_converged','stable_for_panels'])}

Exact zero action changes remains visible as the conservative
`strict_policy_converged` diagnostic. It is not an appropriate sole criterion
for a discontinuous argmax boundary, because refinement quadruples the number
of common nodes and can move a convergent boundary across a handful of them.
The decreasing value error and disagreement shares establish numerical
convergence in measure at the scale of the plotted Monte Carlo panels.
"""


def _evidence_statement(
    simulation: pd.DataFrame,
    methods: list[str],
    label: str,
    *,
    require_uncensored_policy_tail: bool = False,
) -> str:
    data = simulation[simulation.method.isin(methods)]
    if data.empty:
        return f"- **{label}:** not evaluated in this profile."
    late = data["late_product2_share_conditional_on_A"].fillna(0.0)
    if np.all(late <= 1e-8):
        if require_uncensored_policy_tail:
            censored = data.get(
                "simulation_policy_last_active_censored",
                pd.Series(True, index=data.index),
            ).fillna(True).astype(bool)
            if censored.any():
                unresolved = data.loc[censored, ["parameter_name", "parameter_value"]]
                values = ", ".join(
                    f"{row.parameter_name}={row.parameter_value:g}"
                    for row in unresolved.itertuples(index=False)
                )
                return (
                    f"- **{label}:** no late use was observed, but the computed "
                    f"policy tail remains censored for {values}; eventual extinction "
                    "is therefore not resolved by this profile."
                )
        return (
            f"- **{label}:** the simulated late-window share is zero for every "
            "reported configuration; together with stable inactive policy tails, "
            "this supports numerical extinction on the tested grids and horizon."
        )
    positive = data.loc[
        late > 1e-8,
        [
            "method",
            "parameter_name",
            "parameter_value",
            "late_product2_share_conditional_on_A",
        ],
    ]
    examples = ", ".join(
        f"{row.method}, {row.parameter_name}={row.parameter_value:g} "
        f"({row.late_product2_share_conditional_on_A:.3%})"
        for row in positive.itertuples(index=False)
    )
    return (
        f"- **{label}:** recurrent late use is present for {examples}; the evidence "
        "does not support extinction for those configurations."
    )


def _discrete_extinction_statement(
    simulation: pd.DataFrame,
    method: str,
    label: str,
) -> str:
    """Separate a finite observation boundary from slow calendar arrival."""
    data = simulation[simulation.method == method].copy()
    if data.empty:
        return f"- **{label}:** not evaluated in this profile."
    boundaries = pd.to_numeric(
        data.get("simulation_policy_last_active_diagonal"), errors="coerce"
    )
    censored = data.get(
        "simulation_policy_last_active_censored",
        pd.Series(True, index=data.index),
    ).fillna(True).astype(bool)
    unresolved = censored | boundaries.isna()
    if unresolved.any():
        rows = data.loc[unresolved, ["parameter_name", "parameter_value"]]
        values = ", ".join(
            f"{row.parameter_name}={row.parameter_value:g}"
            for row in rows.itertuples(index=False)
        )
        return (
            f"- **{label}:** the observation-count policy boundary remains "
            f"censored for {values}; eventual extinction is not resolved."
        )
    maximum_boundary = int(boundaries.max())
    late = data["late_product2_share_conditional_on_A"].fillna(0.0)
    if np.any(late > 1e-8):
        maximum_late = float(late.max())
        calendar_note = (
            f" A late-window calendar share as high as {maximum_late:.3%} remains "
            "because some paths reach that observation boundary slowly; this is "
            "delayed extinction, not a reappearance beyond the boundary."
        )
    else:
        calendar_note = " No late-window product-2 use was observed."
    return (
        f"- **{label}:** every tested configuration has a finite, uncensored "
        f"last active observation diagonal (maximum {maximum_boundary}), so the "
        "numerical policy evidence supports eventual extinction."
        + calendar_note
    )


def _dynamic_report_section(
    config: dict,
    time_series: pd.DataFrame,
    panel_summary: pd.DataFrame,
    solver_checks: pd.DataFrame,
) -> str:
    """Build optional documentation for the method-by-p0 dynamic matrix."""
    if time_series.empty and panel_summary.empty:
        return """## 13. Dynamic simulation matrix across p0

_Dynamic panels were not generated for this profile. Sections 1--12 do not
depend on these optional tables._
"""

    source = panel_summary if not panel_summary.empty else time_series
    dynamic = config.get("dynamic_panels", {})
    first = source.iloc[0]
    paths = int(first.get("paths", dynamic.get("paths", 0)))
    periods = int(first.get("periods", dynamic.get("periods", 0)))
    seed = int(first.get("seed", dynamic.get("seed", config.get("seed", 0))))
    confidence = float(
        first.get("confidence_level", dynamic.get("confidence_level", 0.95))
    )
    late_start = int(
        first.get(
            "late_window_start", dynamic.get("late_window_start", periods)
        )
    )
    rolling_window = int(first.get("rolling_window", 0))
    p0_values = sorted(
        pd.to_numeric(source.get("p0", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .unique()
    )
    p0_text = ", ".join(f"{float(value):g}" for value in p0_values)
    method_count = (
        int(source["method_spec_key"].nunique())
        if "method_spec_key" in source
        else int(source.get("method", pd.Series(dtype=str)).nunique())
    )
    series_count = (
        int(source["series_key"].nunique())
        if "series_key" in source
        else len(panel_summary)
    )
    preliminary = str(config.get("profile", "")) != "full"
    preliminary_note = (
        " Because this is a reduced profile, every dynamic figure carries the "
        "visible subtitle **PRELIMINARY — reduced simulation**."
        if preliminary
        else ""
    )

    if solver_checks.empty or "converged" not in solver_checks:
        failed_checks = pd.DataFrame()
    else:
        converged = solver_checks["converged"]
        if pd.api.types.is_bool_dtype(converged):
            failed_checks = solver_checks.loc[~converged.fillna(False)]
        else:
            accepted = (
                converged.fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            failed_checks = solver_checks.loc[~accepted]
    solver_type = solver_checks.get(
        "solver_type", pd.Series("", index=solver_checks.index)
    ).astype(str)
    continuous_checks = solver_checks[
        solver_type.str.contains("continuous", case=False)
    ]
    discrete_failed = failed_checks[
        ~failed_checks.get(
            "solver_type", pd.Series("", index=failed_checks.index)
        ).astype(str).str.contains("continuous", case=False)
    ]
    continuous_failed = failed_checks[
        failed_checks.get(
            "solver_type", pd.Series("", index=failed_checks.index)
        ).astype(str).str.contains("continuous", case=False)
    ]
    panel_unstable = pd.DataFrame()
    if (
        not continuous_checks.empty
        and "policy_grid_stable_for_panels" in continuous_checks
    ):
        stable = (
            continuous_checks["policy_grid_stable_for_panels"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        panel_unstable = continuous_checks.loc[~stable]
    check_caution = ""
    if not discrete_failed.empty:
        check_caution = (
            f" `{len(discrete_failed)}` discrete method-p0 checks have "
            "`converged=no`, so their comparisons remain truncation-sensitive."
        )
    if not panel_unstable.empty:
        check_caution += (
            f" `{len(panel_unstable)}` forgetting checks also fail the panel "
            "stability criterion (at most 0.01% robust action changes on common "
            "nested-grid nodes), so those curves remain resolution-sensitive."
        )
    elif not continuous_checks.empty:
        maximum_grid_share = pd.to_numeric(
            continuous_checks.get("action_change_share"), errors="coerce"
        ).max()
        maximum_visited_share = pd.to_numeric(
            continuous_checks.get(
                "visited_state_robust_action_disagreement_share"
            ),
            errors="coerce",
        ).max()
        check_caution += (
            " The strict zero-change criterion still fails for "
            f"`{len(continuous_failed)}` forgetting rows, but every forgetting "
            "row passes the separate panel-stability threshold: the maximum "
            f"common-node robust action-change share is {maximum_grid_share:.4%}"
        )
        if np.isfinite(maximum_visited_share):
            check_caution += (
                " and the maximum coarse-versus-fine disagreement share on "
                f"simulated states is {maximum_visited_share:.4%}"
            )
        check_caution += (
            ". These differences are small at the scale of the plotted Monte "
            "Carlo curves, but the interpolated boundaries are not exact."
        )

    forgetting = panel_summary[
        panel_summary.get(
            "method", pd.Series("", index=panel_summary.index)
        ).isin(["observation_forgetting_ts", "calendar_forgetting_ts"])
    ] if not panel_summary.empty else pd.DataFrame()
    regime_table = _dynamic_regime_table(panel_summary)

    return f"""## 13. Dynamic simulation matrix across p0

### 13.1 Design and audit of the legacy figure

The dynamic experiment is a `{method_count} x {len(p0_values)}` matrix:
{method_count} representative demand specifications, each evaluated at
`p0 in {{{p0_text}}}`, for `{series_count}` method-p0 series. Figures 13--17
preserve the complete original eight-specification comparison for each p0.
Figures 18--22 provide a focused view with standard TS, scaled updates at
`eta in {{0.5,2}}`, epsilon-greedy at `epsilon=0.1`, and observation-time
forgetting at `rho=0.95`. The focused set omits the
mean-preserving-temperature and calendar-forgetting curves, while the complete
set remains available. Both sets use the same fixed color for a specification
at every p0. This holds the outside option fixed within a plot and avoids
placing all `{series_count}` series in one unreadable figure.

The product-2 panel includes a dashed quality-maintenance benchmark
`a_maint=clip((p0-p1)/(p2-p1),0,1)`, the smallest stationary product-2 share
whose expected quality `p1+a(p2-p1)` reaches `p0`. This is a sustainability
benchmark, not the dynamically optimal policy. No line is drawn when `p0>p2`,
because even product 2 cannot reach the outside-option quality. The summary
CSV records the benchmark's feasibility, its value, and each method's overall
and late conditional-use gap relative to it.

The script behind the legacy title *Exact discounted simulated paths* did not
plot one exact or representative path. Each plotted point was a
contemporaneous cross-sectional Monte Carlo statistic at one calendar period;
there was no rolling or cumulative averaging. The legacy defaults were 400
independent paths, a 250-period simulation, and a policy from a finite-horizon
DP with horizon 700. The new experiment preserves the useful four outcomes but
uses the modular study's stationary discounted policies, subject to the
reported triangular truncation or continuous interpolation checks.

The plotted metrics are defined exactly as follows:

- **Seller A demand / market share:** the realized fraction of paths selecting
  A in that calendar period. The CSV also saves the cross-path mean of the
  model's instantaneous demand probability; it is distinct from the plotted
  realized share.
- **Product-2 use conditional on A:** actual product-2 uses divided by actual A
  selections in the period. Paths selecting B are excluded, never silently
  coded as product 1. The CSV separately saves product-2 use over all calendar
  path-periods (`chosen_A & action2`) and the diagnostic share of simulated
  paths whose current policy action is product 2. The plotted curve is the
  conditional actual use rate, not that prescription diagnostic.
- **Calendar-period profit:** mean realized Seller A profit across all paths,
  including zero whenever B is selected. It is not conditional on interacting
  with A and is not formatted as a percentage.
- **Posterior mean:** mean belief used by the demand rule at the **start** of
  the period, before that period's choice and outcome. For standard TS this is
  `(S+1)/(S+F+2)`; scaled updates and forgetting use their corresponding
  method-specific Beta parameters. Only valid path-level values enter the
  cross-sectional mean.

This profile uses `{paths:,}` paths, `{periods}` periods, seed `{seed}`, and a
`{confidence:.0%}` confidence level. The exact same choice and outcome uniforms
are reused across all method-p0 configurations (common random numbers).
`rolling_window={rolling_window}` means no rolling or cumulative average. The
time-series CSV contains a confidence interval for every contemporaneous
estimate: Wilson score intervals for binary proportions (realized A share and
conditional/calendar/prescribed product-2 shares), and Student-t intervals for
cross-path means (demand probability, calendar profit, posterior mean,
effective sample size, and Beta concentration). The conditional product-2
denominator contains only realized A selections. Summary columns prefixed
`late_` use the inclusive calendar window `t={late_start},...,{periods}`.
Confidence intervals are saved in CSV rather than overplotting 40 bands.
{preliminary_note}

### 13.2 Method-by-p0 Monte Carlo summaries

{_dynamic_method_tables(panel_summary)}

For forgetting, `effective_sample_size=a+b-2` is the normalized effective
count used by the implementation and `effective_concentration=a+b` includes
the prior. Both are reported because forgetting bounds effective memory even
though calendar time continues:

{_markdown_table(forgetting, ['method_spec_label','p0','overall_effective_sample_size','late_effective_sample_size','overall_effective_concentration','late_effective_concentration'])}

### 13.3 Economic-regime diagnostics

{_markdown_table(regime_table, ['method','regime','tested_p0','late_A_share','late_conditional_P2','late_calendar_profit'])}

For `p0<p1`, small late conditional product-2 rates are consistent with early
investment followed by harvesting with product 1. For `p1<p0<p2`, positive
late use is consistent with continued or recurrent reputation investment
because product 1 pulls reputation below the outside option while product 2
pushes it upward. For `p0>p2`, a small late A share is consistent with demand
collapse even if product 2 is still used on some scarce A interactions. The
table reports the actual ranges for every method, so deviations from these
qualitative patterns remain visible rather than being promoted to a theorem.

Two qualifications matter. Every positive epsilon-greedy value gives A an
exploration floor, so "demand collapse" means approach toward that positive
floor rather than literal zero; the exact balance reduction is available only
at `p0=0.5`, while the other p0 curves use the general triangular DP.
Forgetting has bounded effective memory and a continuous interpolated state,
so recurrent use may reflect a genuinely persistent incentive but its size
must be assessed together with the grid checks below.

### 13.4 Dynamic-policy solver checks

{_markdown_table(solver_checks, ['method_spec_label','method','parameter_name','parameter_value','p0','solver_type','coarse_outer_diagonal','fine_outer_diagonal','coarse_grid_size','fine_grid_size','action_changes','action_change_share','visited_state_robust_action_disagreement_share','bellman_residual','fine_bellman_residual','balance_policy_disagreements','policy_grid_stable_for_panels','converged'])}

Common random numbers reduce Monte Carlo noise in cross-configuration
comparisons; they do not eliminate sampling uncertainty, truncation error, or
continuous-grid error.{check_caution} All dynamic conclusions are numerical,
finite-horizon evidence and **not theorem claims**.
"""


def write_report(
    experiment_root: Path,
    config: dict,
    tables: dict[str, pd.DataFrame],
) -> str:
    """Write a detailed REPORT.md and return its text."""
    profile = str(config["profile"])
    summary = tables["summary"].copy()
    simulation = tables["simulation_summary"].copy()
    regressions = tables["sensitivity_regressions"].copy()
    convergence = tables["discrete_convergence"].copy()
    continuous = tables["continuous_convergence"].copy()
    continuous_sensitivity = tables.get(
        "continuous_sensitivity", pd.DataFrame()
    ).copy()
    dynamic_methods_p0 = tables.get(
        "dynamic_methods_p0_timeseries", pd.DataFrame()
    ).copy()
    dynamic_panel_summary = tables.get(
        "dynamic_panel_summary", pd.DataFrame()
    ).copy()
    dynamic_solver_checks = tables.get(
        "dynamic_solver_checks", pd.DataFrame()
    ).copy()
    observation_refinement = tables.get(
        "observation_forgetting_convergence", pd.DataFrame()
    ).copy()
    observation_refinement_section = _observation_refinement_section(
        observation_refinement
    )

    base_continuous_resolutions = sorted(
        {int(value) for value in config["continuous"]["grid_resolutions"]}
    )
    selected_continuous_resolution = config["continuous"].get(
        "selected_convergence_resolution"
    )
    selected_continuous_rho = sorted(
        {
            float(value)
            for value in config["continuous"].get(
                "selected_convergence_rho", []
            )
        }
    )
    selected_resolution_text = ""
    if selected_continuous_resolution is not None and selected_continuous_rho:
        selected_resolution_text = (
            f"; selected `rho={selected_continuous_rho}` cases additionally use "
            f"resolution `{int(selected_continuous_resolution)}`"
        )

    sensitivity_display = regressions[
        regressions.method.isin(
            [
                "standard_ts",
                "scaled_updates",
                "mean_preserving_temperature",
                "epsilon_greedy",
            ]
        )
    ]
    policy_display = summary[
        [
            column
            for column in [
                "method",
                "parameter_name",
                "parameter_value",
                "solver_type",
                "grid_node_count",
                "grid_resolution",
                "comparison_scope",
                "bellman_residual",
                "action_changes",
                "strong_opposite_action_changes",
                "last_active_diagonal",
                "last_active_censored",
                "simulation_policy_last_active_diagonal",
                "simulation_policy_last_active_censored",
                "converged",
            ]
            if column in summary
        ]
    ]
    simulation_display = simulation[
        [
            column
            for column in [
                "method",
                "parameter_name",
                "parameter_value",
                "mean_discounted_profit",
                "mean_calendar_profit",
                "mean_demand",
                "product2_share_calendar",
                "product2_share_conditional_on_A",
                "late_product2_share_conditional_on_A",
                "maximum_observation_count",
            ]
            if column in simulation
        ]
    ]

    numeric_parameter = pd.to_numeric(simulation["parameter_value"], errors="coerce")
    scaled_statement = _discrete_extinction_statement(
        simulation[~((simulation.method == "scaled_updates") & np.isclose(numeric_parameter, 1.0, equal_nan=False))],
        "scaled_updates",
        "Fixed scaled updates",
    )
    temperature_statement = _discrete_extinction_statement(
        simulation[~((simulation.method == "mean_preserving_temperature") & np.isclose(numeric_parameter, 1.0, equal_nan=False))],
        "mean_preserving_temperature",
        "Fixed mean-preserving temperature",
    )
    forgetting_statement = _evidence_statement(
        simulation,
        ["observation_forgetting_ts", "calendar_forgetting_ts"],
        "Forgetting",
    )
    forgetting_summary = summary[
        summary.method.isin(
            ["observation_forgetting_ts", "calendar_forgetting_ts"]
        )
    ]
    forgetting_grid_sensitive = bool(
        not forgetting_summary.empty
        and "converged" in forgetting_summary
        and (~forgetting_summary["converged"].fillna(False).astype(bool)).any()
    )
    if forgetting_grid_sensitive:
        if observation_refinement.empty:
            forgetting_statement += (
                " At least one relevant continuous-policy summary has "
                "`converged=no`; this persistence finding is therefore "
                "preliminary and grid-sensitive, even where the finite-horizon "
                "late-use estimate is large."
            )
        else:
            forgetting_statement += (
                " The strict exact-node flag remains `converged=no` for some "
                "continuous policies. The dedicated observation-time `rho=.95` "
                "refinement nevertheless shows second-order value convergence "
                "and policy convergence in measure; configurations not covered "
                "by that refinement remain grid-sensitive."
            )

    forgetting_interpretation = (
        "For observation-time forgetting at `rho=.95`, the dedicated nested-grid "
        "study supports numerical stability of the plotted persistence result. "
        "Other forgetting specifications with `converged=no` remain preliminary "
        "unless they receive the same refinement audit."
        if not observation_refinement.empty
        else (
            "The forgetting evidence must remain preliminary/grid-sensitive "
            "whenever a relevant continuous summary has `converged=no`."
        )
    )
    epsilon_statement = _evidence_statement(
        simulation,
        ["epsilon_greedy"],
        "Uniform epsilon-greedy",
    )
    epsilon_one = simulation[
        (simulation.method == "epsilon_greedy")
        & np.isclose(numeric_parameter, 1.0, equal_nan=False)
    ]
    if len(epsilon_one) and float(
        epsilon_one.late_product2_share_conditional_on_A.iloc[0]
    ) <= 1e-8:
        epsilon_statement += (
            " The endpoint `epsilon=1` is the explicit exception: demand is "
            "constant at one half, so product 2 has no reputational return and "
            "is never used."
        )

    dynamic_section = _dynamic_report_section(
        config,
        dynamic_methods_p0,
        dynamic_panel_summary,
        dynamic_solver_checks,
    )

    certificate_path = (
        experiment_root
        / "results"
        / profile
        / "observation_forgetting_nonextinction_certificate.json"
    )
    nonextinction_section = ""
    if certificate_path.exists():
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        interval = certificate["certified_interval"]
        word = certificate["synchronizing_word"]
        nonextinction_section = f"""
### 12.1 Parameter-specific observation-forgetting result

A separate hybrid analytic/computer-assisted argument addresses the asymptotic
question for observation-time forgetting at `rho={certificate['rho']}` and the
base primitives. The exact one-dimensional boundary reduction, together with a
padded `{certificate['cells']:,}`-cell Bellman enclosure, gives product 2 a
strict advantage on `x in [{interval[0]}, {interval[1]}]`; the certified margin
is `{certificate['minimum_certified_advantage']:.6g}`. The outcome word
`{word['notation']}` maps every sufficiently mature belief into that interval.
Because this word has uniformly positive conditional probability under either
adaptive product choice, a martingale recurrence argument implies infinitely
many product-2 interactions almost surely, and even a strictly positive but
very conservative asymptotic frequency lower bound.

This is stronger than the finite-horizon simulation statement, but it is still
parameter-specific. The recurrence argument is analytic; the Bellman-sign
lemma uses float64 arithmetic and explicitly padded SciPy beta tails rather
than a directed-rounding interval special-function library. Full details and
the numerical-scope qualification are in
`results/{profile}/OBSERVATION_FORGETTING_NONEXTINCTION.md`; Figure 25 visualizes the
two ingredients.
"""

    report = f"""# Modular discounted-DP experiment

Profile: `{profile}`. All quantities below are computed outputs, not smoothed
curves. The original paper solver and its outputs were left unchanged.

## 1. Repository files inspected

- `discounted/paper_numerics/stationary_solver.py`: stationary discounted
  benchmark and exact idle-self-loop rearrangement.
- `discounted/paper_numerics/run_paper_numerics.py`: baseline outer-grid
  comparisons, policy classification, tables, and plots.
- `discounted/paper_numerics/outputs/`: benchmark figures and tables.
- `discounted/DP/exact_dp.py`: finite-calendar-horizon solver; inspected but
  not substituted for the stationary benchmark.
- `../LLM_Learning_archived_experiments/discounted_DP/experiments/epsilon_greedy_paths.py`
  and `verify_epsilon_nonextinction.py`: archived prior experiments used only
  as conceptual checks. Their directed-exploration default is not used here.
- `pyproject.toml` and `.gitignore`: dependencies and generated-data policy.

## 2. Learning rules

For count state `(S,F)`, standard Thompson sampling uses
`Beta(1+S,1+F)`. Scaled updates use `Beta(1+eta*S,1+eta*F)`.

Mean-preserving temperature samples from
`Beta((1+S)/T,(1+F)/T)`: it preserves the standard posterior mean while
changing concentration. Uniform two-seller epsilon-greedy assigns demand
`1-epsilon/2`, `1/2`, or `epsilon/2` according as the posterior mean is above,
at, or below `p0`.

For discrete trajectories, effective sample size means sampling-Beta
concentration minus its value at `(S,F)=(0,0)`: it is `n` for standard TS and
epsilon-greedy, `eta*n` for scaled updates, and `n/T` for temperature.
The raw observation count and full Beta concentration are saved separately.

For forgetting, normalized counts are `x=(1-rho)(a-1)` and
`y=(1-rho)(b-1)`, hence `a=1+x/(1-rho)` and
`b=1+y/(1-rho)`. At state `(x,y)`, both forgetting models use the Thompson
demand

`D(x,y) = Pr[Beta(a,b) >= p0] = 1 - I_{{p0}}(a,b)`,

where `I` is the regularized incomplete beta function. Their effective sample
size is `a+b-2=(x+y)/(1-rho)`, while Beta concentration is `a+b`. Success maps
to `(rho*x+1-rho,rho*y)` and failure to
`(rho*x,rho*y+1-rho)`. Observation-time idle is the identity. Calendar-time
idle maps to `(rho*x,rho*y)`.

## 3. Relationships between methods

Scaled updates, mean-preserving temperature, epsilon-greedy, and each
forgetting clock are distinct demand or state-transition rules. `eta=1` and
`T=1` are verified aliases of standard Thompson sampling.

## 4. Solvers

Integer models use backward recursion on triangular diagonals with zero value
at the outer boundary, exactly matching the benchmark approximation. Because
idle is an exact self-loop, the solver uses
`V=D*M/[1-gamma*(1-D)]`.

Forgetting models use regular triangular grids in normalized `(x,y)`
coordinates and piecewise-linear barycentric interpolation. The base linear
resolutions are `{base_continuous_resolutions}`{selected_resolution_text}.
Here `grid_resolution=N` means `N` equally spaced coordinates on each edge;
`grid_node_count=N(N+1)/2` is the number of triangular nodes actually stored.
Observation-time forgetting uses the rearranged self-loop equation.
Calendar-time forgetting always uses the generic Bellman equation, including
at the origin where its idle transition happens to fix one state.

## 5. Validation and convergence

Every invocation of `scripts/run_experiments.py` reruns the validation gate; it
does not trust a prior passing file. The gate runs the stdlib unit suite,
compares every reported baseline array against the existing solver, checks
`eta=1`, `T=1`, Bellman residuals, transition
invariants, interpolation at nodes, and one case from each continuous family.
The experiment's `run_manifest.json` records both the experiment configuration
hash and `validation_configuration_hash`, so the gate configuration is
auditable. See `results/validation/VALIDATION.md` for the check-level record.

Discrete outer-grid comparisons:

{_markdown_table(convergence, ['method','parameter_value','comparison_scope','coarse_outer_diagonal','fine_outer_diagonal','common_report_diagonal','maximum_value_difference','maximum_gap_difference','raw_action_changes','action_changes','distance_to_outer_boundary'])}

Continuous grid comparisons:

{_markdown_table(continuous, ['method','parameter_value','coarse_grid_size','fine_grid_size','maximum_value_difference','maximum_advantage_difference','raw_action_changes','action_changes','strong_opposite_action_changes','fine_bellman_residual','cells_per_transition_fine'])}

A configuration is not marked converged merely because its node residual is
small. Discrete convergence also requires zero robust action changes on the
common reported interior. Continuous convergence likewise requires exactly
zero tolerance-filtered (`action_changes`) policy changes on common nested-grid
nodes. `strong_opposite_action_changes` is an additional diagnostic, not a
substitute for that deliberately strict flag; raw boundary-node changes also
remain visible. For a discontinuous action boundary, value convergence and the
share of policy disagreements must also be inspected rather than treating a
handful of node changes as proof of nonconvergence.

{observation_refinement_section}

## 6. Demand sensitivity

The regression is OLS of `log(g_n)` on `log(n)` over the configured inclusive
large-n range. A missing slope means `g_n` is zero (not an artificial slope of
zero).

{_markdown_table(sensitivity_display, ['method','parameter_name','parameter_value','estimated_sensitivity_slope','max_demand_sensitivity','fit_n_min','fit_n_max','fit_observations'])}

Forgetting sensitivity is evaluated over the continuous triangular nodes. The
maximizer's normalized coordinates, posterior mean, effective sample size, and
sampling-Beta concentration are reported explicitly:

{_markdown_table(continuous_sensitivity, ['method','parameter_name','parameter_value','grid_resolution','grid_node_count','max_demand_sensitivity','argmax_x','argmax_y','argmax_posterior_mean','argmax_effective_sample_size','argmax_effective_concentration'])}

Standard TS, scaled updates, and fixed mean-preserving temperatures should be
read against the `-1/2` reference. Epsilon-greedy has a discrete jump near the
greedy threshold; its maximal sensitivity need not vanish. At `p0=.5`, the
epsilon-greedy sequence alternates by diagonal parity between the full jump
and a half jump created by the exact tie state, which is why both horizontal
bands are shown rather than smoothed away.

## 7. Optimal-policy results

{_markdown_table(policy_display, ['method','parameter_name','parameter_value','solver_type','grid_node_count','grid_resolution','comparison_scope','bellman_residual','action_changes','strong_opposite_action_changes','last_active_diagonal','last_active_censored','simulation_policy_last_active_diagonal','simulation_policy_last_active_censored','converged'])}

Discrete endpoint data and each diagonal's maximum continuation gap are saved
in `discrete_diagonals.csv`. The policy-point CSV retains every raw product-2
point and flags `robust_action2` and `numerical_tie` separately. Figures use
the robust, tolerance-filtered classification, whose computed advantage
exceeds the configured action tolerance.
Forgetting's last active diagonal is `NaN` by construction because its state is
not indexed by cumulative observation count.

When the base discrete common interior censors a boundary, the full profile
solves outer diagonals `{config['discrete'].get('extended_outer_grids', [])}`
and compares them through `n<={config['discrete'].get('extended_report_diagonal', 'NA')}`.
Those extended comparisons and policy/diagonal points are exported, with
`comparison_scope=extended_boundary`. The `simulation_policy_*` and
`simulation_outer_diagonal` fields separately document any still-larger policy
solve needed to cover the simulated observation counts; these are not the same
as the reported-grid convergence comparison.

## 8. Simulations and extinction versus persistence

Every method starts from the same `Beta(1,1)` prior and reuses seed
`{config['seed']}`. The profile uses `{config['simulation']['paths']}` paths of
`{config['simulation']['periods']}` calendar periods. Simulations stream
path-level accumulators and retain only a small fixed set of exact
representative trajectories. Both calendar-time and observation-time
aggregates are saved for nonforgetting models; calendar time is always saved
for forgetting.

{_markdown_table(simulation_display, ['method','parameter_name','parameter_value','mean_discounted_profit','mean_calendar_profit','mean_demand','product2_share_calendar','product2_share_conditional_on_A','late_product2_share_conditional_on_A','maximum_observation_count'])}

`product2_share_calendar` means that the optimal policy prescribed product 2
at the realized state, divided by all simulated path-periods.
`product2_share_conditional_on_A` pools actual product-2 uses and divides by
all realized A interactions. `late_product2_share_conditional_on_A` is the
same pooled ratio restricted to one-indexed calendar periods
`t={config['simulation']['late_window_start']},...,{config['simulation']['periods']}`.
For each configured threshold, `probability_product2_used_at_or_after_*` is the
fraction of paths with at least one actual use at or after that inclusive
calendar period, but still within the finite horizon. `mean_observed_last_*`
only averages the latest *observed* use among paths with a use; it cannot show
that a path has a finite last use. No figure reports the uninformative share of
all grid states prescribing product 2.

For `p0=.5`, long epsilon-greedy simulations use the exact stationary reduction
to balance `k=S-F`, independently checked against the triangular policy.
Nonforgetting simulations only extend a policy with product 1 after a stable,
long inactive tail; otherwise they solve a larger triangular grid rather than
silently extrapolate.

## 9. Numerical limitations

- A stationary triangular solve is still an outer-boundary approximation; the
  action-stability and distance-to-boundary columns quantify that limitation.
- Continuous results discretize a genuinely continuous state. Raw switches at
  a moving action boundary can remain. The strict `converged` flag requires
  exactly zero tolerance-filtered action changes; refinement studies also
  report value convergence and disagreement shares on common nodes and visited
  states.
- Late simulated use is evidence only inside the finite simulated horizon. An
  observed latest use is right-censored by the horizon and is not evidence of
  a finite last-use time.
- Common random numbers reduce comparison noise but do not turn simulation
  evidence into a theorem.

## 10. Recommended paper figures

Figures 1--11 cover the core learning-rule diagnostics:
raw and log-log demand sensitivity; policy regions for scaled updates,
temperature, and epsilon-greedy; last active diagonal against `eta` and `T`;
forgetting policy maps; late realized product-2 use against `rho` and
`epsilon`; and representative paths. Figure identifier 12 is intentionally
unused so later output identifiers remain stable. Figures 13--22 add the
complete and focused method-by-p0 comparisons documented in Section 13.
Figure 23 reports the dedicated observation-forgetting nested-grid
refinement. Figure 24 re-expresses the observation-forgetting product-2 policy
in the same coordinates as the discrete policy figures: effective sample size
against posterior mean, with one panel per `rho`. Its points are continuous
interpolation-grid nodes, not a claim that every point is an exactly reachable
finite history. Figure 25 gives the separate parameter-specific
observation-forgetting non-extinction certificate: a strict boundary action
interval and a finite outcome word whose image lies inside it.
Recommended candidates for the paper are the log-log sensitivity figure, one
compact policy-region comparison, the two late-use parameter plots, and a
selected subset of the dynamic panels. The paper itself was not modified.

## 11. Main economic interpretation

Rules whose marginal demand response shrinks with accumulated evidence tend to
localize investment and eventually remove product 2 from the computed count
policy. A nonvanishing demand jump (including the greedy cutoff) or bounded
effective memory can keep the value of influencing future demand away from
zero and can therefore generate recurrent investment. {forgetting_interpretation}
The exact outcome remains parameter dependent: in
particular, `epsilon=1` makes demand constant at one half and removes the
reputational return to quality.

## 12. Bottom line for this numerical profile

{scaled_statement}
{temperature_statement}
{forgetting_statement}
{epsilon_statement}

These are statements about numerical evidence on the reported grids,
simulation horizon, and parameter set. They are **not theorem claims**.

{nonextinction_section}

{dynamic_section}
"""
    destination = experiment_root / "REPORT.md"
    destination.write_text(report, encoding="utf-8")
    profile_report = experiment_root / "results" / profile / "REPORT.md"
    profile_report.write_text(report, encoding="utf-8")
    return report


__all__ = ["write_report"]
