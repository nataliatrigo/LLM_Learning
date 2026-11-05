"""Evaluate and plot total discounted seller profit from reputation 0.5.

Competition follows saved OE p0 or following-monopoly (OE quality) policies
in the actual two-seller game.
Monopoly follows the saved optimal forgetting policy against fixed p0(rho).
No simulation, burn-in, stationary averaging, or new equilibrium solve is used.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monopoly-comparison-mpl")
OUTPUT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.policy import evaluate_policy, transition_matrix
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle, write_csv, write_json
from plot_comparison import STYLE

DATA = OUTPUT / "data"
FIGURES = OUTPUT / "Figures"
SOURCE = ROOT / "oblivious/02_oe_p0/data/rho_forgetting_mixed/arrays"
FOLLOWING = ROOT / "oblivious/03_oe_following_monopoly/data/quality_sales_n51"
INITIAL = .5
TOLERANCE = 1e-12
VALIDATION_PERIODS = 2000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forward_total(matrix, rewards, initial_index, gamma):
    """Independent finite discounted sum, propagating the initial state law."""
    rewards = np.asarray(rewards).reshape(matrix.shape[0], -1)
    mass = np.zeros(matrix.shape[0])
    mass[initial_index] = 1.
    total = np.zeros(rewards.shape[1])
    discount = 1.
    transpose = matrix.T.tocsr()
    for _ in range(VALIDATION_PERIODS):
        total += discount * (mass @ rewards)
        mass = transpose @ mass
        discount *= gamma
    tail_bound = discount * np.max(np.abs(rewards), axis=0) / (1-gamma)
    require(abs(mass.sum()-1) < 1e-10, "Forward probability mass drift")
    return total, tail_bound


def plot(rows, gamma):
    rhos = [row["rho"] for row in rows]
    x = np.arange(len(rows))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        fig.subplots_adjust(left=.095, right=.98, top=.75, bottom=.25)
        for key, label, color, marker in (
                ("competition", "Competition (OE p0 policy)", "#D55E00", "o"),
                ("monopoly", "Monopoly (forgetting)", "#0072B2", "s"),
                ("following_monopoly", "Following monopoly (OE quality)", "#7B3294", "D")):
            ax.plot(x, [row[f"{key}_discounted_profit"] for row in rows],
                    color=color, marker=marker, ms=6, lw=1.8, label=label,
                    ls="--" if key == "following_monopoly" else "-",
                    mfc="white" if key == "following_monopoly" else color)
        ax.set_xticks(x, [f"{rho:g}" for rho in rhos])
        ax.set(xlabel=r"Forgetting factor $\rho$ (discrete values)",
               ylabel="Expected total discounted profit per seller",
               xlim=(-.2, len(rows)-.8), ylim=(0, 50))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=.7)
        fig.suptitle("Total discounted profit: monopoly versus competition",
                     fontsize=14, y=.96)
        fig.text(.53, .885,
                 rf"$V=\mathbb{{E}}[\sum_{{t=0}}^{{\infty}}\gamma^t\pi_t]$,  "
                 rf"$\gamma={gamma:g}$; initial reputation $0.5$ for every seller",
                 ha="center", fontsize=11)
        fig.legend(*ax.get_legend_handles_labels(), ncol=3, loc="upper center",
                   bbox_to_anchor=(.53, .855), frameon=False)
        fig.text(.53, .14,
                 r"Monopoly: fixed OE p0 benchmark. OE p0 and following monopoly: actual two-seller demand under each saved policy.",
                 ha="center", fontsize=8.5)
        fig.text(.53, .095,
                 "Discounted calendar-period profits from the initial state; includes periods with no own sale.",
                 ha="center", fontsize=8.5)
        fig.text(.53, .05,
                 "Direct policy evaluation on the common 51-point grid. Rho values are equally spaced; lines guide the eye.",
                 ha="center", fontsize=8.5)
        paths = []
        FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            path = FIGURES / f"04_total_discounted_profit_across_rho.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    config = json.loads((DATA / "comparison.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["data_files_sha256"].items():
        require(sha256((DATA/name).read_bytes()).hexdigest() == digest,
                f"Comparison input changed: {name}")
    inputs = {}

    def record(path):
        inputs[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()

    for name in ("comparison.json", "manifest.json"):
        record(DATA/name)
    following_manifest_path = FOLLOWING/"manifest.json"
    record(following_manifest_path)
    following_manifest = json.loads(following_manifest_path.read_text())
    require(following_manifest["status"] == "complete"
            and following_manifest["configuration"]["rhos"] == config["rhos"]
            and following_manifest["configuration"]["grids"] == [config["grid"]],
            "Following-monopoly experiment is incomplete or incompatible")
    rows, diagnostics, arrays = [], [], {}
    for rho in config["rhos"]:
        tag = case_tag(rho, config["grid"])
        base = DATA/"arrays"/tag
        source_base = SOURCE/tag
        for path in (base.with_suffix(".json"), base.with_suffix(".npz"),
                     source_base.with_suffix(".json"), source_base.with_suffix(".npz")):
            record(path)
            if path.parent == SOURCE:
                expected = manifest["source_files_sha256"][str(path.relative_to(ROOT))]
                require(inputs[str(path.relative_to(ROOT))] == expected,
                        f"Original OE p0 source changed: {path.name}")
        bundle = load_bundle(base)
        source = load_bundle(source_base)
        following_base = FOLLOWING/"arrays"/tag
        for suffix in (".json", ".npz"):
            path = following_base.with_suffix(suffix)
            record(path)
            require(inputs[str(path.relative_to(ROOT))] ==
                    following_manifest["checkpoint_sha256"][str(path.relative_to(FOLLOWING))],
                    f"Following-monopoly source changed: {path.name}")
        following_case = load_bundle(following_base)
        following = following_case["selected"]
        params = Parameters(**bundle["parameters"])
        grid = bundle["grid"]
        require(bundle["verified"] and params.rho == rho and params.gamma == .98,
                "Expected verified case with the agreed discount factor")
        require(np.array_equal(grid, source["selected"]["grid"])
                and np.array_equal(bundle["competition"]["policy"], source["selected"]["policy"])
                and bundle["p0"] == source["selected"]["p0"], "Source mismatch")
        initial_nodes = np.flatnonzero(np.isclose(grid, INITIAL, rtol=0, atol=1e-14))
        require(len(initial_nodes) == 1, "Initial reputation must be a grid node")
        i = int(initial_nodes[0])
        n = len(grid)
        require(following_case["rho"] == rho and following_case["n"] == n
                and following_case["configuration"]["parameters"] == bundle["parameters"]
                and following_case["configuration"]["order"] == source["configuration"]["order"]
                and following_case["configuration"]["quality_weighting"] == "purchase"
                and np.array_equal(following["grid"], grid), "Following-monopoly primitives differ")
        require(following["verified"] and following["candidate_verified"]
                and following["dp_diagnostics"]["verified"]
                and following["stationary_diagnostics"]["verified"]
                and following["monopoly_certificate"]["verified"]
                and following_case["demand_audit"]["verified"], "Unverified following-monopoly source")

        # Re-evaluate the original pure monopoly optimum in calendar time.
        demand, policy = bundle["monopoly"]["demand"], bundle["policy"]
        mono = evaluate_policy(grid, demand, policy, params)
        mono_P = transition_matrix(grid, demand, policy, params)
        mono_reward = demand*(params.R-params.c1-(params.c2-params.c1)*policy)
        mono_forward, mono_tail = forward_total(mono_P, mono_reward, i, params.gamma)
        mono_value = float(mono["value"][i])
        saved_gap = float(np.max(abs(mono["value"]-bundle["monopoly"]["independent_solver_value"])))
        perceived_gap = float(np.max(abs(mono["value"]-source["selected"]["value"])))
        mono_optimality = float(np.max(np.maximum(mono["advantage"], 0)-policy*mono["advantage"]))

        # Evaluate the existing OE p0 policy, without changing its actions,
        # under actual two-seller demand on the joint reputation grid.
        game = Game(grid, params, order=source["configuration"]["order"])
        sigma = np.broadcast_to(bundle["competition"]["policy"][:, None], (n,n))
        comp = game.evaluate(sigma, sigma.T, tol=TOLERANCE)
        comp_P = game.transition(sigma, sigma.T)
        comp_reward = game.rewards(sigma, sigma.T).reshape(2, -1).T
        comp_forward, comp_tail = forward_total(comp_P, comp_reward, i*n+i, params.gamma)
        comp_values = comp["values"][:, i, i]
        symmetry_error = float(np.max(abs(comp["values"][0]-comp["values"][1].T)))
        comp_forward_error = float(np.max(abs(comp_values-comp_forward)))
        mono_forward_error = float(abs(mono_value-mono_forward[0]))
        row_error = max(float(np.max(abs(np.asarray(comp_P.sum(axis=1)).ravel()-1))),
                        float(np.max(abs(np.asarray(mono_P.sum(axis=1)).ravel()-1))))
        require(mono["residual"] < 1e-11 and comp["residuals"].max() <= TOLERANCE,
                "Discounted Bellman residual too large")
        require(saved_gap < 1e-9 and perceived_gap < 1e-8 and mono_optimality < 1e-10,
                "Monopoly optimal value does not match the original solvers")
        require(comp_forward_error < 1e-9 and mono_forward_error < 1e-9
                and symmetry_error < 1e-9 and row_error < 1e-12,
                "Forward discounted sum or symmetry check failed")
        if np.all(sigma == 0):
            require(np.max(abs(comp_values-(params.R-params.c1)/(2*(1-params.gamma)))) < 1e-9,
                    "Symmetric all-product-1 analytical value check failed")

        # Following monopoly is a second saved competitive policy, with its
        # own quality-consistent benchmark, evaluated under the same real game.
        following_sigma = np.broadcast_to(following["policy"][:, None], (n,n))
        following_eval = game.evaluate(following_sigma, following_sigma.T, tol=TOLERANCE)
        following_P = game.transition(following_sigma, following_sigma.T)
        following_reward = game.rewards(following_sigma, following_sigma.T).reshape(2, -1).T
        following_forward, following_tail = forward_total(
            following_P, following_reward, i*n+i, params.gamma)
        following_values = following_eval["values"][:, i, i]
        following_forward_error = float(np.max(abs(following_values-following_forward)))
        following_symmetry = float(np.max(abs(following_eval["values"][0]-following_eval["values"][1].T)))
        following_row_error = float(np.max(abs(np.asarray(following_P.sum(axis=1)).ravel()-1)))
        rival_sales = following["nu"]*game.D.T
        rival_quality = params.p1+(params.p2-params.p1)*following_sigma.T
        following_consistency = float(abs((rival_sales*rival_quality).sum()/rival_sales.sum()-following["p0"]))
        require(following_eval["residuals"].max() <= TOLERANCE
                and following_forward_error < 1e-9 and following_symmetry < 1e-9
                and following_row_error < 1e-12 and following_consistency < 1e-7,
                "Following-monopoly discounted evaluation or equilibrium consistency failed")
        if np.all(following_sigma == 0):
            require(np.max(abs(following_values-(params.R-params.c1)/(2*(1-params.gamma)))) < 1e-9,
                    "Following-monopoly all-product-1 analytical value check failed")

        row = dict(rho=rho, n=n, p0=bundle["p0"], gamma=params.gamma,
                   initial_reputation=INITIAL,
                   competition_discounted_profit=float(comp_values.mean()),
                   monopoly_discounted_profit=mono_value,
                   competition_seller_a=float(comp_values[0]),
                   competition_seller_b=float(comp_values[1]),
                   oe_p0_perceived_bellman_value=float(source["selected"]["value"][i]),
                   following_monopoly_p0=float(following["p0"]),
                   following_monopoly_discounted_profit=float(following_values.mean()),
                   following_monopoly_seller_a=float(following_values[0]),
                   following_monopoly_seller_b=float(following_values[1]),
                   following_monopoly_perceived_bellman_value=float(following["value"][i]))
        rows.append(row)
        diagnostics.append(dict(rho=rho, verified=True,
            monopoly_bellman_residual=mono["residual"],
            competition_bellman_residuals=comp["residuals"].tolist(),
            monopoly_residual_value_error_bound=mono["residual"]/(1-params.gamma),
            competition_residual_value_error_bounds=(comp["residuals"]/(1-params.gamma)).tolist(),
            monopoly_optimality_gap=mono_optimality,
            original_forgetting_solver_max_value_gap=saved_gap,
            native_oe_p0_solver_max_value_gap=perceived_gap,
            monopoly_forward_sum_error=mono_forward_error,
            competition_forward_sum_error=comp_forward_error,
            forward_sum_tail_bounds=dict(monopoly=mono_tail.tolist(), competition=comp_tail.tolist()),
            full_state_seller_symmetry_error=symmetry_error,
            transition_row_sum_error=row_error,
            competition_evaluation_method=comp["method"],
            following_monopoly=dict(
                bellman_residuals=following_eval["residuals"].tolist(),
                residual_value_error_bounds=(following_eval["residuals"]/(1-params.gamma)).tolist(),
                forward_sum_error=following_forward_error,
                forward_sum_tail_bounds=following_tail.tolist(),
                full_state_seller_symmetry_error=following_symmetry,
                transition_row_sum_error=following_row_error,
                quality_equilibrium_consistency_error=following_consistency,
                evaluation_method=following_eval["method"])))
        arrays[tag+"_grid"] = grid
        arrays[tag+"_monopoly_value"] = mono["value"]
        arrays[tag+"_competition_values"] = comp["values"]
        arrays[tag+"_oe_p0_perceived_value"] = source["selected"]["value"]
        arrays[tag+"_following_monopoly_values"] = following_eval["values"]
        arrays[tag+"_following_monopoly_policy"] = following["policy"]
        arrays[tag+"_following_monopoly_perceived_value"] = following["value"]
        print(f"rho={rho:g}: competition={row['competition_discounted_profit']:.9f}, "
              f"monopoly={mono_value:.9f}, following_monopoly={following_values.mean():.9f}; verified", flush=True)

    csv_path = DATA/"total_discounted_profit.csv"
    arrays_path = DATA/"total_discounted_profit_arrays.npz"
    write_csv(csv_path, rows)
    np.savez_compressed(arrays_path, **arrays)
    figures = plot(rows, config["parameters"]["gamma"])
    for path in (Path(__file__), Path(__file__).with_name("plot_comparison.py"),
                 ROOT/"oblivious/02_oe_p0/code/game.py", ROOT/"oblivious/02_oe_p0/code/policy.py",
                 ROOT/"oblivious/02_oe_p0/code/primitives.py", ROOT/"oblivious/02_oe_p0/code/run_experiment.py"):
        record(path)
    write_json(DATA/"total_discounted_profit_manifest.json", dict(
        status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        definition="Expected sum from t=0 to infinity of gamma^t times calendar-period seller profit, before the initial selection; not multiplied by 1-gamma.",
        gamma=config["parameters"]["gamma"], initial_monopoly=INITIAL,
        initial_competition=[INITIAL, INITIAL], rhos=config["rhos"], grid=config["grid"],
        competition="Value of saved OE p0 policy with actual two-seller demand; per seller, averaging equal initial A and B values. Policy evaluation only, not a full-state equilibrium solve.",
        monopoly="Value of original forgetting optimal pure policy with fixed saved equilibrium p0(rho).",
        following_monopoly="Value of saved following-monopoly (OE quality) policy under actual two-seller demand, per seller from (.5,.5). Its own benchmark equals rival quality conditional on rival sales; no new equilibrium or monopoly solve.",
        following_monopoly_native_value="Stored separately as following_monopoly_perceived_bellman_value; the plotted curve uses actual two-seller competition, not fixed-benchmark perceived value.",
        native_oe_p0_value="Saved separately as oe_p0_perceived_bellman_value; its fixed-benchmark environment is the monopoly environment, and it is not the plotted competition outcome.",
        method="Discounted linear policy evaluation; validated by 2000-period forward expected discounted sums and a remaining-tail bound. No Monte Carlo or stationary averaging.",
        validation_periods=VALIDATION_PERIODS, diagnostics=diagnostics,
        scope="Finite 51-point grid and saved OE p0 and OE quality equilibria; residual bounds exclude grid, demand quadrature, and equilibrium-benchmark approximation error.",
        input_files_sha256=inputs,
        output_files_sha256={str(path.relative_to(OUTPUT)):sha256(path.read_bytes()).hexdigest()
                             for path in [csv_path, arrays_path, *figures]}))
    print(f"Saved total discounted profit data and figures in {OUTPUT}")


if __name__ == "__main__":
    main()
