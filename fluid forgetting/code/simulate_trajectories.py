"""Calendar-time trajectories of the mature monopoly fluid with forgetting.

Reuse the project's HJB solver, then integrate smooth fixed-product segments.
Threshold events stop integration before changing action. At attractive roots
the state is held with the tied product mixture; no Euler-step chattering is
introduced. All numerical claims concern this continuous fluid model.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from forgetting.Code.model import Primitives, boundary_demand
from forgetting.Code.fluid_convergence.numerics import solve_fluid

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PiecewisePolicy:
    primitives: Primitives
    rho: float
    roots: list[float]
    products: list[int]
    demand_function: Callable | None = None

    def __post_init__(self):
        if len(self.products) != len(self.roots) + 1:
            raise ValueError("one product is required for each policy interval")
        if any(a >= b for a, b in zip(self.roots, self.roots[1:])):
            raise ValueError("policy roots must be increasing")
        if any(product not in (1, 2) for product in self.products):
            raise ValueError("interval products must be 1 or 2")

    @property
    def lam(self):
        return -math.log(self.rho)

    def demand(self, x):
        if self.demand_function is not None:
            return np.asarray(self.demand_function(np.asarray(x)), dtype=float)
        return boundary_demand(x, self.rho, self.primitives.p0)

    def success_probability(self, product):
        return self.primitives.p1 if product == 1 else self.primitives.p2

    def velocity(self, x, product):
        return self.lam * self.demand(x) * (self.success_probability(product) - x)

    def root_kind(self, index):
        x = self.roots[index]
        left = float(self.velocity(x, self.products[index]))
        right = float(self.velocity(x, self.products[index + 1]))
        if left > 0.0 and right < 0.0:
            return "attractive"
        if left < 0.0 and right > 0.0:
            return "repelling"
        return "crossing"

    def holding_alpha(self, x):
        p = self.primitives
        return (x - p.p1) / p.delta_p

    def initial_region(self, x):
        """Resolve a state exactly at a root without displacing that state.

        At a repelling tie choose product 1. At a crossing choose the outgoing
        interval. Attractive roots are handled by simulate_path before this.
        """
        close = [i for i, root in enumerate(self.roots) if abs(x-root) <= 1e-13]
        if not close:
            return int(np.searchsorted(self.roots, x, side="right"))
        index = close[0]
        if self.root_kind(index) == "repelling":
            return index if self.products[index] == 1 else index + 1
        left = float(self.velocity(x, self.products[index]))
        right = float(self.velocity(x, self.products[index+1]))
        if left >= 0.0 and right > 0.0:
            return index + 1
        if left < 0.0 and right <= 0.0:
            return index
        return index if self.products[index] == 1 else index + 1

    def action_at(self, x):
        for index, root in enumerate(self.roots):
            if abs(x-root) <= 1e-13 and self.root_kind(index) == "attractive":
                return 0, self.holding_alpha(root)
        region = self.initial_region(x)
        product = self.products[region]
        return product, float(product == 2)

    def intervals(self):
        edges = [0.0, *self.roots, 1.0]
        return [dict(lo=a, hi=b, product=product)
                for a, b, product in zip(edges, edges[1:], self.products)]

    def holding_points(self):
        return [dict(x=root, alpha=self.holding_alpha(root))
                for i, root in enumerate(self.roots)
                if self.root_kind(i) == "attractive"]


def policy_from_solution(primitives, rho, solution):
    roots = solution["roots"]
    edges = [0.0, *roots, 1.0]
    products = [2 if np.interp((a+b)/2, solution["x"], solution["gap"]) > 0
                else 1 for a, b in zip(edges, edges[1:])]
    return PiecewisePolicy(primitives, rho, list(roots), products)


def simulate_path(policy, x0, times, *, rtol=1e-10, atol=1e-12, max_step=2.0):
    """Integrate a stationary piecewise policy with exact threshold events."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must increase strictly from zero")
    if not 0 <= x0 <= 1:
        raise ValueError("initial state must lie in [0,1]")
    horizon = float(times[-1])
    states = np.full(times.shape, np.nan)
    alphas = np.full(times.shape, np.nan)
    holding_time = None
    events = []
    t, x = 0.0, float(x0)
    region = policy.initial_region(x)
    evaluations = 0
    for _ in range(len(policy.roots) + 3):
        holding = next((root for i, root in enumerate(policy.roots)
                        if abs(x-root) <= 1e-13
                        and policy.root_kind(i) == "attractive"), None)
        if holding is not None:
            holding_time = float(t)
            mask = times >= t
            states[mask] = holding
            alphas[mask] = policy.holding_alpha(holding)
            break
        product = policy.products[region]
        probability = policy.success_probability(product)
        drift = float(policy.velocity(x, product))
        if abs(probability - x) <= 1e-14:
            mask = times >= t
            states[mask] = x
            alphas[mask] = float(product == 2)
            break
        direction = 1 if drift > 0 else -1
        candidate = (region if direction > 0 else region-1)
        target = policy.roots[candidate] if 0 <= candidate < len(policy.roots) else None
        # A fixed-product equilibrium before the next threshold is approached
        # asymptotically; that threshold cannot be reached along this segment.
        can_hit = (target is not None
                   and direction*(target-x) > 1e-13
                   and direction*(probability-target) > 1e-13)
        event = None
        if can_hit:
            def event(_t, y):
                return y[0]-target
            event.terminal = True
            event.direction = direction
        solution = solve_ivp(
            lambda _t, y: [float(policy.velocity(y[0], product))],
            (t, horizon), [x], method="DOP853", dense_output=True,
            events=event, rtol=rtol, atol=atol, max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        evaluations += solution.nfev
        end = float(solution.t[-1])
        mask = (times >= t) & (times <= end)
        states[mask] = solution.sol(times[mask])[0]
        alphas[mask] = float(product == 2)
        if not can_hit or len(solution.t_events[0]) == 0:
            break
        t, x = end, float(target)
        kind = policy.root_kind(candidate)
        event_record = dict(time=t, x=x, kind=kind, product_before=product)
        if kind == "attractive":
            event_record["alpha_after"] = policy.holding_alpha(x)
        else:
            region += direction
            event_record["product_after"] = policy.products[region]
        events.append(event_record)
    else:
        raise RuntimeError("unexpected repeated policy events")
    if np.any(~np.isfinite(states)) or np.any(~np.isfinite(alphas)):
        raise RuntimeError("trajectory samples were not filled")
    p = policy.primitives
    drift = policy.lam*policy.demand(states)*(p.p1+p.delta_p*alphas-states)
    if holding_time is not None:
        drift[times >= holding_time] = 0.0
    return dict(x0=float(x0), x=states, alpha=alphas, drift=drift,
                holding_time=holding_time, events=events, nfev=evaluations)


def _finite_json(value):
    """Compact significant digits, retaining small scientific-notation drifts."""
    if isinstance(value, np.ndarray):
        return [_finite_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return float(f"{value:.12g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


def verify_path(policy, path, times):
    x, alpha = path["x"], path["alpha"]
    if x.min() < -1e-10 or x.max() > 1+1e-10:
        raise AssertionError("state left its invariant interval")
    if alpha.min() < 0 or alpha.max() > 1:
        raise AssertionError("invalid product fraction")
    if abs(x[0]-path["x0"]) > 1e-12:
        raise AssertionError("initial condition changed")
    motion = np.diff(x)
    # Autonomous one-dimensional paths cannot reverse direction. Very small
    # differences at a limiting equilibrium are below the integration tolerance.
    if np.any(motion > 2e-10) and np.any(motion < -2e-10):
        raise AssertionError("trajectory reverses direction")
    if path["holding_time"] is not None:
        mask = times >= path["holding_time"]
        if np.any(mask) and np.ptp(x[mask]) > 1e-12:
            raise AssertionError("attractive threshold was not held")
        if np.any(np.abs(path["drift"][mask]) > 1e-12):
            raise AssertionError("holding drift is not zero")
    for event in path["events"]:
        if min(abs(event["x"]-root) for root in policy.roots) > 1e-12:
            raise AssertionError("event is not at a threshold")


def analytic_constant_demand_check():
    p = Primitives()
    lam, demand = 0.2, 0.4
    policy = PiecewisePolicy(p, math.exp(-lam), [], [1],
                             lambda x: np.full_like(x, demand))
    times = np.linspace(0, 100, 401)
    path = simulate_path(policy, 0.9, times)
    exact = p.p1+(0.9-p.p1)*np.exp(-lam*demand*times)
    error = float(np.max(np.abs(path["x"]-exact)))
    if error > 2e-9:
        raise AssertionError(f"constant-demand analytic error: {error}")
    return error


def make_figures(dataset, output, *, pdf_pages=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    count = len(dataset["cases"])
    columns = min(3, count)
    rows = math.ceil(count/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.35*columns, 3.45*rows+1.9),
                             sharex=True, sharey=True, squeeze=False)
    palette = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00",
               "#56B4E9", "#000000", "#777777"]
    colors = [palette[index % len(palette)] for index in range(len(dataset["initial_states"]))]
    times = dataset["times"]
    phase_time = .98*dataset["horizon"]
    has_fixed_equilibrium = False
    for ax, case in zip(axes.flat, dataset["cases"]):
        for root in case["roots"]:
            ax.axhline(root, color="#59636E", lw=.9, ls="--", alpha=.7)
        for color, path in zip(colors, case["paths"]):
            ax.plot(times, path["x"], lw=1.7, color=color)
        # Arrow direction on the vertical phase line is constant on pieces
        # separated by policy roots and fixed-product equilibria.
        edges = sorted(set([0.0, 1.0, *case["roots"],
                            dataset["parameters"]["p1"], dataset["parameters"]["p2"]]))
        policy = PiecewisePolicy(Primitives.from_mapping(dataset["parameters"]),
                                 case["rho"], case["roots"],
                                 [interval["product"] for interval in case["intervals"]])
        for lo, hi in zip(edges, edges[1:]):
            x = (lo+hi)/2
            product, _ = policy.action_at(x)
            sign = np.sign(policy.velocity(x, product))
            if hi-lo > .025:
                ax.annotate("", xy=(phase_time, x+sign*min(.025, (hi-lo)/4)),
                            xytext=(phase_time, x-sign*min(.025, (hi-lo)/4)),
                            arrowprops=dict(arrowstyle="->", color="#7B848D", lw=.9))
        for equilibrium in (policy.primitives.p1, policy.primitives.p2):
            product, _ = policy.action_at(equilibrium)
            if product and abs(policy.success_probability(product)-equilibrium) < 1e-13:
                has_fixed_equilibrium = True
                ax.scatter([phase_time], [equilibrium], s=22, marker="o",
                           facecolors="white", edgecolors="#59636E", linewidths=1, zorder=5)
        ax.set_title(rf"$\rho={case['rho']:g}$", loc="left", fontsize=12)
        ax.set_xlim(0, dataset["horizon"])
        ax.set_ylim(-.015, 1.015)
        ax.grid(axis="y", alpha=.16)
        ax.set_xlabel("Continuous calendar time, t")
        ax.set_ylabel("Normalized reputation, x(t)")
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=color, lw=2, label=f"x(0) = {x0:g}")
               for color, x0 in zip(colors, dataset["initial_states"])]
    handles += [Line2D([], [], color="#59636E", lw=.9, ls="--", label="Policy threshold")]
    if has_fixed_equilibrium:
        handles += [Line2D([], [], color="#59636E", lw=0, marker="o", markerfacecolor="white",
                           label="Fixed-product stationary state")]
    legend_rows = math.ceil(len(handles)/3)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .055))
    fig.suptitle("Fluid monopoly trajectories with forgetting", fontsize=17, x=.07, ha="left")
    p = dataset["parameters"]
    caption = (rf"$p_0={p['p0']:g}$, $p_1={p['p1']:g}$, $p_2={p['p2']:g}$, "
               rf"$c_1={p['c1']:g}$, $c_2={p['c2']:g}$, $R={p['revenue']:g}$, "
               rf"$\gamma={p['gamma']:g}$; mature boundary and numerical HJB policy.")
    fig.text(.07, .926, caption, fontsize=10)
    fig.text(.07, .025, "At attractive thresholds, a holding mixture keeps x constant. "
             "Low demand can make movement very slow; arrows show its direction.", fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.87, bottom=.12+.035*legend_rows,
                        hspace=.31, wspace=.23)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"trajectories-overview.{extension}", dpi=180)
    if pdf_pages is not None:
        pdf_pages.savefig(fig)
    plt.close(fig)


def run(config_path, output, *, p0=None, pdf_pages=None):
    config = json.loads(Path(config_path).read_text())
    parameters = dict(config["parameters"])
    if p0 is not None:
        parameters["p0"] = p0
    p = Primitives.from_mapping(parameters)
    times = np.linspace(0, config["horizon"], config["time_samples"])
    output.mkdir(parents=True, exist_ok=True)
    dataset = dict(schema_version=1, parameters=p.as_dict(),
                   initial_states=config["initial_states"], horizon=config["horizon"],
                   times=times, state="x=(1-rho)*S on the mature boundary",
                   time="continuous calendar time; demand multiplies the state drift",
                   policy="numerical fluid optimum with holding mixtures at attractive roots",
                   repelling_tie="choose product 1 at an exact repelling threshold",
                   cases=[])
    rows = []
    analytic_error = analytic_constant_demand_check()
    max_ode_change, max_root_change = 0.0, 0.0
    for rho in config["rhos"]:
        solution = solve_fluid(p, rho, config["policy_nodes"])
        refined = solve_fluid(p, rho, config["refinement_nodes"])
        if len(solution["roots"]) != len(refined["roots"]):
            raise AssertionError("policy topology changes under grid refinement")
        root_change = max((abs(a-b) for a, b in zip(solution["roots"], refined["roots"])), default=0.0)
        max_root_change = max(max_root_change, root_change)
        if root_change > 5/(config["policy_nodes"]-1):
            raise AssertionError("policy thresholds changed by more than five grid cells")
        policy = policy_from_solution(p, rho, solution)
        nodes = np.unique(np.r_[np.linspace(0, 1, config["field_nodes"]),
                                 solution["roots"], p.p1, p.p2])
        field = []
        for x in nodes:
            product, alpha = policy.action_at(float(x))
            drift = policy.lam*float(policy.demand(x))*(p.p1+p.delta_p*alpha-x)
            if product == 0:
                drift = 0.0
            field.append(dict(x=x, drift=drift, product=product, alpha=alpha))
        case = dict(rho=rho, roots=solution["roots"], intervals=policy.intervals(),
                    root_kinds=[policy.root_kind(i) for i in range(len(policy.roots))],
                    holding_points=policy.holding_points(), field=field, paths=[])
        case_change, hold_time_change = 0.0, 0.0
        for x0 in config["initial_states"]:
            ode = config["ode"]
            path = simulate_path(policy, x0, times, rtol=ode["rtol"], atol=ode["atol"],
                                 max_step=ode["max_step"])
            checked = simulate_path(policy, x0, times, rtol=ode["check_rtol"],
                                    atol=ode["check_atol"], max_step=ode["check_max_step"])
            verify_path(policy, path, times)
            verify_path(policy, checked, times)
            change = float(np.max(np.abs(path["x"]-checked["x"])))
            case_change = max(case_change, change)
            if (path["holding_time"] is None) != (checked["holding_time"] is None):
                raise AssertionError("holding event changed under ODE refinement")
            if path["holding_time"] is not None:
                hold_time_change = max(hold_time_change, abs(path["holding_time"]-checked["holding_time"]))
            if change > 2e-8 or hold_time_change > 2e-6:
                raise AssertionError("ODE refinement sensitivity is too large")
            for index, t in enumerate(times):
                rows.append(dict(rho=rho, x0=x0, time=t, x=path["x"][index],
                                 alpha_product2=path["alpha"][index], drift=path["drift"][index]))
            case["paths"].append(path)
        max_ode_change = max(max_ode_change, case_change)
        case["diagnostics"] = dict(
            policy_nodes=config["policy_nodes"], refinement_nodes=config["refinement_nodes"],
            hjb_residual=solution["residual"], refined_hjb_residual=refined["residual"],
            refined_roots=refined["roots"], maximum_root_change=root_change,
            maximum_ode_state_change=case_change, maximum_holding_time_change=hold_time_change,
            invariant_interval_and_monotonicity_passed=True,
        )
        dataset["cases"].append(case)
        print(f"p0={p.p0:g}, rho={rho:g}: roots={solution['roots']}; "
              f"max ODE change={case_change:.3g}", flush=True)
    dataset["diagnostics"] = dict(
        path_count=len(rows)//len(times), samples_per_path=len(times),
        constant_demand_analytic_maximum_error=analytic_error,
        maximum_root_change=max_root_change, maximum_ode_state_change=max_ode_change,
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        parameter_overrides={} if p0 is None else dict(p0=p.p0),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Refinement differences measure numerical sensitivity; they are not rigorous spatial error bounds.",
    )
    (output/"trajectory-data.json").write_text(json.dumps(_finite_json(dataset), separators=(",", ":"), allow_nan=False)+"\n")
    with (output/"trajectories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rho", "x0", "time", "x", "alpha_product2", "drift"])
        writer.writeheader()
        writer.writerows(rows)
    (output/"verification.json").write_text(json.dumps(_finite_json(dict(
        diagnostics=dataset["diagnostics"],
        cases=[dict(rho=case["rho"], roots=case["roots"], root_kinds=case["root_kinds"],
                    holding_points=case["holding_points"], diagnostics=case["diagnostics"])
               for case in dataset["cases"]])), indent=2)+"\n")
    make_figures(dataset, output, pdf_pages=pdf_pages)
    return dataset


def run_sweep(config_path, output, p0s):
    """Keep each outside-option experiment separate and collect its figures."""
    from matplotlib.backends.backend_pdf import PdfPages

    config = json.loads(Path(config_path).read_text())
    p0s = list(map(float, p0s))
    if not p0s or len(set(p0s)) != len(p0s):
        raise ValueError("Supply distinct outside-option probabilities")
    for p0 in p0s:
        Primitives.from_mapping(dict(config["parameters"], p0=p0))
    output.mkdir(parents=True, exist_ok=True)
    summary, experiments = [], []
    with PdfPages(output/"trajectories-all-p0.pdf") as pdf_pages:
        metadata = pdf_pages.infodict()
        metadata["Title"] = "Fluid monopoly trajectories across outside options"
        metadata["Subject"] = "Mature-state trajectories under optimal HJB policies"
        for p0 in p0s:
            tag = "p0_"+str(p0).replace(".", "p")
            dataset = run(config_path, output/tag, p0=p0, pdf_pages=pdf_pages)
            for case in dataset["cases"]:
                for path in case["paths"]:
                    summary.append(dict(
                        p0=p0, rho=case["rho"], x0=path["x0"],
                        final_time=dataset["horizon"], final_x=float(path["x"][-1]),
                        final_product2_fraction=float(path["alpha"][-1]),
                        final_drift=float(path["drift"][-1]),
                        holding_time=path["holding_time"],
                    ))
            experiments.append(dict(
                p0=p0, directory=tag, diagnostics=dataset["diagnostics"],
                maximum_hjb_residual=max(case["diagnostics"]["hjb_residual"]
                                         for case in dataset["cases"]),
                maximum_refined_hjb_residual=max(case["diagnostics"]["refined_hjb_residual"]
                                                 for case in dataset["cases"]),
            ))
    with (output/"trajectory-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    verification = dict(
        p0s=p0s, rhos=config["rhos"], initial_states=config["initial_states"],
        horizon=config["horizon"], path_count=len(summary), experiments=experiments,
        note="Finite-horizon mature-state paths; a nearly flat path need not be stationary. "
             "Grid refinement measures numerical sensitivity, not rigorous spatial error bounds.",
    )
    (output/"verification.json").write_text(
        json.dumps(_finite_json(verification), indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(p0s)} outside-option figures and {len(summary)} paths in {output}", flush=True)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs")
    parser.add_argument("--p0s", type=float, nargs="+",
                        help="Run one plot per outside option, plus a combined PDF, in --output-dir")
    args = parser.parse_args()
    if args.p0s is None:
        run(args.config, args.output_dir)
    else:
        run_sweep(args.config, args.output_dir, args.p0s)
