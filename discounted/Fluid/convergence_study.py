"""Numerical convergence study for the discounted fluid-HJB policy.

This script deliberately calls ``solve_fluid_hjb`` without changing its
semi-Lagrangian recursion.  It varies only:

* the uniform state-grid spacing;
* the terminal-tail tolerance; and
* the stored/computed domain extent.

Every solver run is cached as a CSV plus JSON metadata so that an interrupted
study can be resumed.  Policy comparisons are made on a common lattice at the
finest requested spacing using the solver's nearest-grid feedback convention.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/tmp") / "llm_learning_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from fluid_model import (
    PRODUCT1_COLOR,
    PRODUCT2_COLOR,
    FluidParams,
    configure_plot_style,
    posterior_cutoff_curve,
    solution_frame,
    solve_fluid_hjb,
)


DEFAULT_GRID_STEPS = (1.0, 0.5, 0.25)
DEFAULT_P0_VALUES = (0.10, 0.30)


@dataclass(frozen=True)
class RunSpec:
    p0: float
    label: str
    grid_step: float
    max_count: float
    tail_tolerance: float

    @property
    def stem(self) -> str:
        p0_tag = f"{round(100 * self.p0):03d}"
        return f"p0_{p0_tag}_{self.label}"


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Expected a nonempty comma-separated list.")
    return result


def grid_label(grid_step: float) -> str:
    return f"refine_h_{grid_step:g}".replace(".", "p")


def output_paths(
    outputs_dir: Path,
    spec: RunSpec,
) -> tuple[Path, Path]:
    return (
        outputs_dir / "data" / f"{spec.stem}.csv.gz",
        outputs_dir / "metadata" / f"{spec.stem}.json",
    )


def run_one(
    spec: RunSpec,
    params: FluidParams,
    outputs_dir: Path,
    force: bool,
) -> dict[str, float | str | bool]:
    """Run and persist one unchanged fluid-HJB solve."""
    data_path, metadata_path = output_paths(outputs_dir, spec)
    if data_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        metadata["cached"] = True
        return metadata

    print(
        (
            f"START {spec.stem}: h={spec.grid_step:g}, "
            f"N={spec.max_count:g}, tail_tol={spec.tail_tolerance:.0e}"
        ),
        flush=True,
    )
    started = time.perf_counter()
    solution = solve_fluid_hjb(
        p0=spec.p0,
        params=params,
        grid_step=spec.grid_step,
        plot_max_count=spec.max_count,
        tail_tolerance=spec.tail_tolerance,
    )
    frame = solution_frame(solution)
    elapsed = time.perf_counter() - started

    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False, compression="gzip")
    metadata: dict[str, float | str | bool] = {
        **asdict(spec),
        **{f"parameter_{key}": value for key, value in asdict(params).items()},
        "terminal_count": solution.terminal_count,
        "tail_discount_bound": solution.tail_discount_bound,
        "terminal_value_boundary": 0.0,
        "stored_state_count": len(frame),
        "product2_state_count": int(frame["product"].eq(2).sum()),
        "value_at_origin": float(solution.values[0][0]),
        "elapsed_seconds": elapsed,
        "data_path": str(data_path),
        "cached": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        (
            f"DONE  {spec.stem}: {elapsed:.1f}s, "
            f"terminal={solution.terminal_count:g}, "
            f"product2_nodes={metadata['product2_state_count']}"
        ),
        flush=True,
    )
    return metadata


def frame_to_action_array(
    frame: pd.DataFrame,
    grid_step: float,
    max_count: float,
) -> np.ndarray:
    steps = int(round(max_count / grid_step))
    actions = np.zeros((steps + 1, steps + 1), dtype=np.int8)
    s = frame["s"].to_numpy(dtype=float)
    f = frame["f"].to_numpy(dtype=float)
    layer = np.rint((s + f) / grid_step).astype(int)
    success_index = np.rint(s / grid_step).astype(int)
    actions[layer, success_index] = frame["product"].to_numpy(dtype=np.int8)
    return actions


def reference_lattice(
    max_count: float,
    reference_step: float,
) -> pd.DataFrame:
    steps = int(round(max_count / reference_step))
    layer, success_index = np.tril_indices(steps + 1)
    return pd.DataFrame(
        {
            "layer": layer,
            "success_index": success_index,
            "s": reference_step * success_index,
            "f": reference_step * (layer - success_index),
            "n": reference_step * layer,
        }
    )


def evaluate_policy(
    frame: pd.DataFrame,
    grid_step: float,
    max_count: float,
    reference: pd.DataFrame,
) -> np.ndarray:
    """Apply the production solver's nearest-grid feedback convention."""
    actions = frame_to_action_array(frame, grid_step, max_count)
    layer = np.rint(reference["n"].to_numpy(dtype=float) / grid_step).astype(int)
    layer = np.clip(layer, 0, actions.shape[0] - 1)
    success_index = np.rint(
        reference["s"].to_numpy(dtype=float) / grid_step
    ).astype(int)
    success_index = np.clip(success_index, 0, layer)
    return actions[layer, success_index]


def extent_row(
    spec: RunSpec,
    policy: np.ndarray,
    reference: pd.DataFrame,
    reference_step: float,
) -> dict[str, float | str | int]:
    is_product2 = policy == 2
    row: dict[str, float | str | int] = {
        "p0": spec.p0,
        "run": spec.label,
        "grid_step": spec.grid_step,
        "max_count": spec.max_count,
        "tail_tolerance": spec.tail_tolerance,
        "reference_step": reference_step,
        "reference_nodes": len(reference),
        "product2_nodes": int(is_product2.sum()),
        "product2_share": float(is_product2.mean()),
        "product2_area_estimate": float(is_product2.sum() * reference_step**2),
    }
    for column in ("s", "f", "n"):
        values = reference.loc[is_product2, column]
        row[f"{column}_min"] = float(values.min()) if len(values) else np.nan
        row[f"{column}_max"] = float(values.max()) if len(values) else np.nan
        row[f"{column}_centroid"] = float(values.mean()) if len(values) else np.nan
    s_zero = is_product2 & np.isclose(reference["s"].to_numpy(dtype=float), 0.0)
    f_at_s_zero = reference.loc[s_zero, "f"]
    row["s_zero_product2_nodes"] = int(s_zero.sum())
    row["s_zero_f_min"] = (
        float(f_at_s_zero.min()) if len(f_at_s_zero) else np.nan
    )
    row["s_zero_f_max"] = (
        float(f_at_s_zero.max()) if len(f_at_s_zero) else np.nan
    )
    return row


def comparison_row(
    p0: float,
    coarse_name: str,
    fine_name: str,
    coarse_policy: np.ndarray,
    fine_policy: np.ndarray,
    reference_step: float,
) -> dict[str, float | str | int]:
    coarse_indicator = (coarse_policy == 2).astype(np.int8)
    fine_indicator = (fine_policy == 2).astype(np.int8)
    absolute_change = np.abs(fine_indicator - coarse_indicator)
    return {
        "p0": p0,
        "run_a": coarse_name,
        "run_b": fine_name,
        "reference_step": reference_step,
        "compared_nodes": len(absolute_change),
        "maximum_absolute_policy_change": int(absolute_change.max(initial=0)),
        "mismatched_nodes": int(absolute_change.sum()),
        "mismatch_share": float(absolute_change.mean()),
        "symmetric_difference_area_estimate": float(
            absolute_change.sum() * reference_step**2
        ),
        "product2_nodes_a": int(coarse_indicator.sum()),
        "product2_nodes_b": int(fine_indicator.sum()),
        "product2_node_change": int(
            fine_indicator.sum() - coarse_indicator.sum()
        ),
    }


def load_run(outputs_dir: Path, spec: RunSpec) -> pd.DataFrame:
    data_path, _ = output_paths(outputs_dir, spec)
    return pd.read_csv(data_path)


def policy_image(
    policy: np.ndarray,
    reference: pd.DataFrame,
    max_count: float,
    reference_step: float,
) -> np.ma.MaskedArray:
    steps = int(round(max_count / reference_step))
    image = np.full((steps + 1, steps + 1), np.nan)
    row = np.rint(reference["f"].to_numpy(dtype=float) / reference_step).astype(int)
    column = np.rint(
        reference["s"].to_numpy(dtype=float) / reference_step
    ).astype(int)
    image[row, column] = policy
    return np.ma.masked_invalid(image)


def plot_refinements(
    specs: list[RunSpec],
    frames: dict[str, pd.DataFrame],
    p0_values: tuple[float, ...],
    grid_steps: tuple[float, ...],
    plot_max_count: float,
    reference_step: float,
    output_path: Path,
) -> None:
    configure_plot_style()
    reference = reference_lattice(plot_max_count, reference_step)
    fig, axes = plt.subplots(
        len(p0_values),
        len(grid_steps),
        figsize=(4.15 * len(grid_steps), 4.0 * len(p0_values)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    cmap.set_bad("white")
    norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)
    lookup = {(spec.p0, spec.grid_step): spec for spec in specs}

    for row, p0 in enumerate(p0_values):
        for column, grid_step in enumerate(grid_steps):
            ax = axes[row, column]
            spec = lookup[(p0, grid_step)]
            policy = evaluate_policy(
                frames[spec.stem],
                spec.grid_step,
                spec.max_count,
                reference,
            )
            image = policy_image(
                policy,
                reference,
                plot_max_count,
                reference_step,
            )
            ax.imshow(
                image,
                origin="lower",
                interpolation="nearest",
                extent=(
                    -reference_step / 2.0,
                    plot_max_count + reference_step / 2.0,
                    -reference_step / 2.0,
                    plot_max_count + reference_step / 2.0,
                ),
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            cutoff_s, cutoff_f = posterior_cutoff_curve(p0, plot_max_count)
            ax.plot(
                cutoff_s,
                cutoff_f,
                color="#475569",
                linestyle="--",
                linewidth=1.0,
            )
            ax.set_title(f"$p_0={p0:.2f}$, $h={grid_step:g}$", loc="left")
            ax.set_aspect("equal", adjustable="box")
            if row == len(p0_values) - 1:
                ax.set_xlabel("Fluid successes, $s$")
            if column == 0:
                ax.set_ylabel("Fluid failures, $f$")
            ax.set_xlim(-1.0, plot_max_count + 1.0)
            ax.set_ylim(-1.0, plot_max_count + 1.0)

    fig.legend(
        handles=[
            Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
            Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
            plt.Line2D(
                [0],
                [0],
                color="#475569",
                linestyle="--",
                linewidth=1.1,
                label="Posterior mean $=p_0$",
            ),
        ],
        loc="outside lower center",
        ncols=3,
    )
    fig.suptitle(
        "Discounted fluid-policy convergence under uniform grid refinement",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_pair(
    baseline_specs: dict[float, RunSpec],
    alternative_specs: dict[float, RunSpec],
    frames: dict[str, pd.DataFrame],
    p0_values: tuple[float, ...],
    reference_step: float,
    plot_max_count: float,
    alternative_title: str,
    output_path: Path,
) -> None:
    configure_plot_style()
    reference = reference_lattice(plot_max_count, reference_step)
    fig, axes = plt.subplots(
        len(p0_values),
        3,
        figsize=(12.2, 4.0 * len(p0_values)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    policy_cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    policy_cmap.set_bad("white")
    policy_norm = BoundaryNorm([0.5, 1.5, 2.5], policy_cmap.N)
    difference_cmap = ListedColormap(["#7c3aed"])
    difference_cmap.set_bad("white")

    for row, p0 in enumerate(p0_values):
        baseline = baseline_specs[p0]
        alternative = alternative_specs[p0]
        baseline_policy = evaluate_policy(
            frames[baseline.stem],
            baseline.grid_step,
            baseline.max_count,
            reference,
        )
        alternative_policy = evaluate_policy(
            frames[alternative.stem],
            alternative.grid_step,
            alternative.max_count,
            reference,
        )
        for column, (policy, title) in enumerate(
            (
                (baseline_policy, "Baseline"),
                (alternative_policy, alternative_title),
            )
        ):
            image = policy_image(
                policy,
                reference,
                plot_max_count,
                reference_step,
            )
            axes[row, column].imshow(
                image,
                origin="lower",
                interpolation="nearest",
                extent=(
                    -reference_step / 2.0,
                    plot_max_count + reference_step / 2.0,
                    -reference_step / 2.0,
                    plot_max_count + reference_step / 2.0,
                ),
                cmap=policy_cmap,
                norm=policy_norm,
                rasterized=True,
            )
            axes[row, column].set_title(
                f"$p_0={p0:.2f}$: {title}",
                loc="left",
            )

        changed = baseline_policy != alternative_policy
        difference = np.where(changed, 1.0, np.nan)
        difference_image = policy_image(
            difference,
            reference,
            plot_max_count,
            reference_step,
        )
        axes[row, 2].imshow(
            difference_image,
            origin="lower",
            interpolation="nearest",
            extent=(
                -reference_step / 2.0,
                plot_max_count + reference_step / 2.0,
                -reference_step / 2.0,
                plot_max_count + reference_step / 2.0,
            ),
            cmap=difference_cmap,
            vmin=0.5,
            vmax=1.5,
            rasterized=True,
        )
        axes[row, 2].set_title(
            f"Changed nodes: {int(changed.sum()):,}",
            loc="left",
        )
        if not changed.any():
            axes[row, 2].text(
                0.5,
                0.5,
                "No policy changes",
                transform=axes[row, 2].transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#334155",
            )
        for column in range(3):
            ax = axes[row, column]
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.0, plot_max_count + 1.0)
            ax.set_ylim(-1.0, plot_max_count + 1.0)
            if row == len(p0_values) - 1:
                ax.set_xlabel("Fluid successes, $s$")
            if column == 0:
                ax.set_ylabel("Fluid failures, $f$")

    fig.legend(
        handles=[
            Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
            Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
            Patch(facecolor="#7c3aed", label="Policy differs"),
        ],
        loc="outside lower center",
        ncols=3,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_extended_domain(
    extended_specs: dict[float, RunSpec],
    frames: dict[str, pd.DataFrame],
    p0_values: tuple[float, ...],
    reference_step: float,
    extended_max_count: float,
    output_path: Path,
) -> None:
    """Show the extended policies over their full N=160 state triangle."""
    configure_plot_style()
    reference = reference_lattice(extended_max_count, reference_step)
    fig, axes = plt.subplots(
        1,
        len(p0_values),
        figsize=(6.0 * len(p0_values), 5.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    cmap.set_bad("white")
    norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)

    for ax, p0 in zip(axes, p0_values, strict=True):
        spec = extended_specs[p0]
        policy = evaluate_policy(
            frames[spec.stem],
            spec.grid_step,
            spec.max_count,
            reference,
        )
        image = policy_image(
            policy,
            reference,
            extended_max_count,
            reference_step,
        )
        ax.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            extent=(
                -reference_step / 2.0,
                extended_max_count + reference_step / 2.0,
                -reference_step / 2.0,
                extended_max_count + reference_step / 2.0,
            ),
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        cutoff_s, cutoff_f = posterior_cutoff_curve(p0, extended_max_count)
        ax.plot(
            cutoff_s,
            cutoff_f,
            color="#475569",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axline(
            (0.0, 80.0),
            slope=-1.0,
            color="#7c3aed",
            linestyle=":",
            linewidth=1.3,
        )
        ax.set_title(f"$p_0={p0:.2f}$, extended $N=160$", loc="left")
        ax.set_xlabel("Fluid successes, $s$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.0, extended_max_count + 2.0)
        ax.set_ylim(-2.0, extended_max_count + 2.0)
    axes[0].set_ylabel("Fluid failures, $f$")

    fig.legend(
        handles=[
            Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
            Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
            plt.Line2D(
                [0],
                [0],
                color="#475569",
                linestyle="--",
                linewidth=1.1,
                label="Posterior mean $=p_0$",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#7c3aed",
                linestyle=":",
                linewidth=1.3,
                label="Original display edge $s+f=80$",
            ),
        ],
        loc="outside lower center",
        ncols=4,
    )
    fig.suptitle(
        "Discounted fluid policy over the doubled state domain",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_run_config(
    outputs_dir: Path,
    params: FluidParams,
    p0_values: tuple[float, ...],
    grid_steps: tuple[float, ...],
    max_count: float,
    extended_max_count: float,
    tail_tolerance: float,
    tight_tail_tolerance: float,
) -> None:
    config = {
        "parameters": asdict(params),
        "p0_values": p0_values,
        "grid_steps": grid_steps,
        "max_count": max_count,
        "extended_max_count": extended_max_count,
        "tail_tolerance": tail_tolerance,
        "tight_tail_tolerance": tight_tail_tolerance,
        "scheme": (
            "unchanged monotone backward semi-Lagrangian recursion from "
            "fluid_model.solve_fluid_hjb"
        ),
        "policy_comparison": (
            "nearest-grid feedback evaluated on the finest common lattice"
        ),
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


def analyze(
    outputs_dir: Path,
    params: FluidParams,
    p0_values: tuple[float, ...],
    grid_steps: tuple[float, ...],
    max_count: float,
    extended_max_count: float,
    tail_tolerance: float,
    tight_tail_tolerance: float,
    run_metadata: list[dict[str, float | str | bool]],
) -> None:
    reference_step = min(grid_steps)
    refinement_specs = [
        RunSpec(
            p0=p0,
            label=grid_label(grid_step),
            grid_step=grid_step,
            max_count=max_count,
            tail_tolerance=tail_tolerance,
        )
        for p0 in p0_values
        for grid_step in grid_steps
    ]
    baseline_specs = {
        p0: RunSpec(
            p0=p0,
            label=grid_label(grid_steps[0]),
            grid_step=grid_steps[0],
            max_count=max_count,
            tail_tolerance=tail_tolerance,
        )
        for p0 in p0_values
    }
    tight_specs = {
        p0: RunSpec(
            p0=p0,
            label="tight_tail",
            grid_step=grid_steps[0],
            max_count=max_count,
            tail_tolerance=tight_tail_tolerance,
        )
        for p0 in p0_values
    }
    extended_specs = {
        p0: RunSpec(
            p0=p0,
            label="extended_domain",
            grid_step=grid_steps[0],
            max_count=extended_max_count,
            tail_tolerance=tail_tolerance,
        )
        for p0 in p0_values
    }
    all_specs = (
        refinement_specs
        + list(tight_specs.values())
        + list(extended_specs.values())
    )
    unique_specs = {spec.stem: spec for spec in all_specs}
    frames = {
        stem: load_run(outputs_dir, spec)
        for stem, spec in unique_specs.items()
    }

    common_reference = reference_lattice(max_count, reference_step)
    policies: dict[str, np.ndarray] = {}
    extent_rows: list[dict[str, float | str | int]] = []
    for stem, spec in unique_specs.items():
        policy = evaluate_policy(
            frames[stem],
            spec.grid_step,
            spec.max_count,
            common_reference,
        )
        policies[stem] = policy
        extent_rows.append(
            extent_row(spec, policy, common_reference, reference_step)
        )

    comparison_rows: list[dict[str, float | str | int]] = []
    for p0 in p0_values:
        for coarse_h, fine_h in zip(grid_steps[:-1], grid_steps[1:], strict=True):
            coarse = next(
                spec
                for spec in refinement_specs
                if spec.p0 == p0 and spec.grid_step == coarse_h
            )
            fine = next(
                spec
                for spec in refinement_specs
                if spec.p0 == p0 and spec.grid_step == fine_h
            )
            comparison_rows.append(
                comparison_row(
                    p0,
                    coarse.label,
                    fine.label,
                    policies[coarse.stem],
                    policies[fine.stem],
                    reference_step,
                )
            )

    sensitivity_rows: list[dict[str, float | str | int]] = []
    for p0 in p0_values:
        baseline = baseline_specs[p0]
        for alternative, test_name in (
            (tight_specs[p0], "tail_tolerance"),
            (extended_specs[p0], "domain_extension"),
        ):
            row = comparison_row(
                p0,
                baseline.label,
                alternative.label,
                policies[baseline.stem],
                policies[alternative.stem],
                reference_step,
            )
            row["test"] = test_name
            sensitivity_rows.append(row)

    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_metadata).sort_values(
        ["p0", "grid_step", "max_count", "tail_tolerance"]
    ).to_csv(tables_dir / "run_metadata.csv", index=False)
    pd.DataFrame(extent_rows).sort_values(["p0", "run"]).to_csv(
        tables_dir / "policy_extents_common_lattice.csv",
        index=False,
    )
    pd.DataFrame(comparison_rows).to_csv(
        tables_dir / "refinement_policy_changes.csv",
        index=False,
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        tables_dir / "sensitivity_policy_changes.csv",
        index=False,
    )

    plots_dir = outputs_dir / "plots"
    plot_refinements(
        refinement_specs,
        frames,
        p0_values,
        grid_steps,
        max_count,
        reference_step,
        plots_dir / "policy_grid_refinement.png",
    )
    plot_sensitivity_pair(
        baseline_specs,
        tight_specs,
        frames,
        p0_values,
        reference_step,
        max_count,
        r"Tail tolerance $10^{-6}$",
        plots_dir / "policy_tail_tolerance_comparison.png",
    )
    plot_sensitivity_pair(
        baseline_specs,
        extended_specs,
        frames,
        p0_values,
        reference_step,
        max_count,
        r"Extended domain $N=160$",
        plots_dir / "policy_domain_extension_comparison.png",
    )
    plot_extended_domain(
        extended_specs,
        frames,
        p0_values,
        reference_step,
        extended_max_count,
        plots_dir / "policy_extended_domain_full.png",
    )

    extended_extent_rows: list[dict[str, float | str | int]] = []
    extended_reference = reference_lattice(extended_max_count, reference_step)
    for p0 in p0_values:
        spec = extended_specs[p0]
        policy = evaluate_policy(
            frames[spec.stem],
            spec.grid_step,
            spec.max_count,
            extended_reference,
        )
        extended_extent_rows.append(
            extent_row(spec, policy, extended_reference, reference_step)
        )
    pd.DataFrame(extended_extent_rows).to_csv(
        tables_dir / "extended_domain_policy_extents.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    defaults = FluidParams()
    parser = argparse.ArgumentParser(
        description="Run grid, tail-tolerance, and domain convergence checks."
    )
    parser.add_argument("--p0-values", default="0.10,0.30")
    parser.add_argument("--grid-steps", default="1.0,0.5,0.25")
    parser.add_argument("--max-count", type=float, default=80.0)
    parser.add_argument("--extended-max-count", type=float, default=160.0)
    parser.add_argument("--tail-tolerance", type=float, default=1e-4)
    parser.add_argument("--tight-tail-tolerance", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs_gamma_0999"
        / "convergence_study",
    )
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", type=float, default=defaults.revenue)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p0_values = parse_float_list(args.p0_values)
    grid_steps = tuple(
        sorted(parse_float_list(args.grid_steps), reverse=True)
    )
    if len(grid_steps) < 3:
        raise ValueError("At least three grid steps are required.")
    for coarse, fine in zip(grid_steps[:-1], grid_steps[1:], strict=True):
        ratio = coarse / fine
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("Grid refinements must be nested integer ratios.")
    params = FluidParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        gamma=args.gamma,
    )
    outputs_dir = args.outputs_dir.resolve()
    write_run_config(
        outputs_dir,
        params,
        p0_values,
        grid_steps,
        args.max_count,
        args.extended_max_count,
        args.tail_tolerance,
        args.tight_tail_tolerance,
    )

    refinement_specs = [
        RunSpec(
            p0=p0,
            label=grid_label(grid_step),
            grid_step=grid_step,
            max_count=args.max_count,
            tail_tolerance=args.tail_tolerance,
        )
        for grid_step in sorted(grid_steps)
        for p0 in p0_values
    ]
    tight_specs = [
        RunSpec(
            p0=p0,
            label="tight_tail",
            grid_step=grid_steps[0],
            max_count=args.max_count,
            tail_tolerance=args.tight_tail_tolerance,
        )
        for p0 in p0_values
    ]
    extended_specs = [
        RunSpec(
            p0=p0,
            label="extended_domain",
            grid_step=grid_steps[0],
            max_count=args.extended_max_count,
            tail_tolerance=args.tail_tolerance,
        )
        for p0 in p0_values
    ]
    specs = refinement_specs + tight_specs + extended_specs
    print(
        f"Running {len(specs)} solves with {args.workers} worker(s). "
        "Existing complete runs will be reused.",
        flush=True,
    )

    run_metadata: list[dict[str, float | str | bool]] = []
    if args.workers == 1:
        for spec in specs:
            run_metadata.append(
                run_one(spec, params, outputs_dir, args.force)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one,
                    spec,
                    params,
                    outputs_dir,
                    args.force,
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                run_metadata.append(future.result())

    analyze(
        outputs_dir,
        params,
        p0_values,
        grid_steps,
        args.max_count,
        args.extended_max_count,
        args.tail_tolerance,
        args.tight_tail_tolerance,
        run_metadata,
    )
    print(f"Study outputs saved under: {outputs_dir}", flush=True)


if __name__ == "__main__":
    main()
