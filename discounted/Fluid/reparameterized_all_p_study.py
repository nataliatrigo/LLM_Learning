"""All-p0 analysis of the discounted fluid model in (n, m) coordinates.

This script uses only the main production grid and compares every p0 panel
used in the paper. It reads the original fluid grids and optimal paths, applies
the exact coordinate change, and writes all outputs to a separate
``reparameterized_all_p`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
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
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from fluid_model import (
    PRODUCT1_COLOR,
    PRODUCT2_COLOR,
    FluidParams,
    configure_plot_style,
)
from reparameterized_utils import (
    InputSpec,
    extract_boundaries,
    regular_nm_field,
    transform_solution,
)


DEFAULT_P0_VALUES = (0.10, 0.30, 0.50, 0.70, 0.90)


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Expected a nonempty comma-separated list.")
    return result


def panel_layout(count: int) -> tuple[int, int]:
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    return rows, columns


def regime_label(p0: float, params: FluidParams) -> str:
    if p0 < params.p1:
        return r"$p_0<p_1$"
    if p0 <= params.p2:
        return r"$p_1\leq p_0\leq p_2$"
    return r"$p_0>p_2$"


def load_and_transform(
    source_dir: Path,
    outputs_dir: Path,
    p0_values: tuple[float, ...],
    params: FluidParams,
    grid_step: float,
) -> tuple[
    dict[float, pd.DataFrame],
    dict[float, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
    grids: dict[float, pd.DataFrame] = {}
    paths: dict[float, pd.DataFrame] = {}
    boundary_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, float | int]] = []
    data_dir = outputs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for p0 in p0_values:
        p0_tag = f"{round(100 * p0):03d}"
        grid_path = source_dir / f"fluid_solution_grid_p0_{p0_tag}.csv"
        path_path = source_dir / f"fluid_solution_path_p0_{p0_tag}.csv"
        if not grid_path.exists() or not path_path.exists():
            raise FileNotFoundError(
                f"Missing production grid or path for p0={p0:.2f}."
            )
        print(f"Transforming production solution for p0={p0:.2f}...", flush=True)
        source_grid = pd.read_csv(grid_path)
        spec = InputSpec(p0=p0, grid_step=grid_step, path=grid_path)
        transformed, diagnostics = transform_solution(
            source_grid,
            spec,
            params,
        )
        boundaries = extract_boundaries(transformed, spec)
        path = pd.read_csv(path_path).copy()
        path["n"] = path["s"] + path["f"]
        path["m"] = (path["s"] + 1.0) / (path["n"] + 2.0)
        path["product2"] = path["product"].eq(2).astype(np.int8)
        path["regime"] = regime_label(p0, params)

        transformed.to_csv(
            data_dir / f"reparameterized_grid_p0_{p0_tag}.csv.gz",
            index=False,
            compression="gzip",
        )
        path.to_csv(
            data_dir / f"reparameterized_path_p0_{p0_tag}.csv.gz",
            index=False,
            compression="gzip",
        )
        grids[p0] = transformed
        paths[p0] = path
        boundary_frames.append(boundaries)
        diagnostics["regime"] = regime_label(p0, params)
        diagnostic_rows.append(diagnostics)

    return (
        grids,
        paths,
        pd.concat(boundary_frames, ignore_index=True),
        pd.DataFrame(diagnostic_rows),
    )


def summarize_all_p(
    grids: dict[float, pd.DataFrame],
    paths: dict[float, pd.DataFrame],
    boundaries: pd.DataFrame,
    diagnostics: pd.DataFrame,
    p0_values: tuple[float, ...],
    params: FluidParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    path_rows: list[dict[str, float | int | str]] = []
    for p0 in p0_values:
        grid = grids[p0]
        p0_boundaries = boundaries[
            np.isclose(boundaries["p0"], p0)
        ].sort_values("n")
        present = p0_boundaries[p0_boundaries["product2_present"].eq(1)]
        last = p0_boundaries.iloc[-1]
        diagnostic = diagnostics[
            np.isclose(diagnostics["p0"], p0)
        ].iloc[0]
        max_width_row = present.loc[present["m_width"].idxmax()]
        summary_rows.append(
            {
                "p0": p0,
                "regime": regime_label(p0, params),
                "product2_state_share": float(grid["product"].eq(2).mean()),
                "first_n_with_product2": float(present["n"].min()),
                "last_n_with_product2": float(present["n"].max()),
                "n_layers_with_product2": len(present),
                "maximum_m_width": float(max_width_row["m_width"]),
                "n_at_maximum_m_width": float(max_width_row["n"]),
                "m_lower_at_n80": float(last["m_lower"]),
                "m_upper_at_n80": float(last["m_upper"]),
                "m_width_at_n80": float(last["m_width"]),
                "p0_inside_band_at_n80": int(
                    last["m_lower"] <= p0 <= last["m_upper"]
                ),
                "layers_touching_s_zero": int(present["touches_s_zero"].sum()),
                "layers_touching_f_zero": int(present["touches_f_zero"].sum()),
                "hjb_residual_rmse": float(diagnostic["hjb_residual_rmse"]),
                "continuous_policy_mismatch_share": float(
                    diagnostic["continuous_policy_mismatch_share"]
                ),
            }
        )

        path = paths[p0].sort_values("n")
        product2 = path["product2"].to_numpy(dtype=np.int8)
        transitions = int(np.sum(product2[1:] != product2[:-1]))
        product2_path = path[path["product2"].eq(1)]
        path_rows.append(
            {
                "p0": p0,
                "regime": regime_label(p0, params),
                "path_product2_share_by_observations": float(product2.mean()),
                "path_product_switches": transitions,
                "first_n_using_product2": (
                    float(product2_path["n"].min())
                    if len(product2_path)
                    else np.nan
                ),
                "last_n_using_product2": (
                    float(product2_path["n"].max())
                    if len(product2_path)
                    else np.nan
                ),
                "terminal_n": float(path["n"].iloc[-1]),
                "terminal_m": float(path["m"].iloc[-1]),
                "terminal_product": int(path["product"].iloc[-1]),
                "terminal_calendar_time": float(
                    path["calendar_time"].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def plot_policy_all_p(
    grids: dict[float, pd.DataFrame],
    p0_values: tuple[float, ...],
    grid_step: float,
    params: FluidParams,
    output_path: Path,
) -> None:
    configure_plot_style()
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    cmap.set_bad("white")
    norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)
    for ax, p0 in zip(axes, p0_values, strict=False):
        n_grid, m_grid, field = regular_nm_field(
            grids[p0],
            grid_step,
            "product",
        )
        ax.imshow(
            field,
            origin="lower",
            interpolation="nearest",
            extent=(n_grid[0], n_grid[-1], m_grid[0], m_grid[-1]),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.axhline(p0, color="#475569", linestyle="--", linewidth=1.0)
        ax.set_title(
            f"$p_0={p0:.2f}$  ({regime_label(p0, params)})",
            loc="left",
        )
        ax.set_xlabel("Observations, $n$")
        ax.set_ylabel("Posterior mean, $m$")
        ax.set_ylim(0.0, 1.0)
    for ax in axes[len(p0_values) :]:
        ax.set_axis_off()
    if len(axes) > len(p0_values):
        axes[len(p0_values)].legend(
            handles=[
                Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
                Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
                plt.Line2D(
                    [0],
                    [0],
                    color="#475569",
                    linestyle="--",
                    label="$m=p_0$",
                ),
            ],
            loc="center",
            fontsize=11,
        )
    fig.suptitle(
        "Discounted fluid policy in reparameterized coordinates: all $p_0$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_boundaries_all_p(
    boundaries: pd.DataFrame,
    p0_values: tuple[float, ...],
    params: FluidParams,
    output_path: Path,
) -> None:
    configure_plot_style()
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, p0 in zip(axes, p0_values, strict=False):
        data = boundaries[
            np.isclose(boundaries["p0"], p0)
            & boundaries["product2_present"].eq(1)
        ].sort_values("n")
        ax.fill_between(
            data["n"],
            data["m_lower"],
            data["m_upper"],
            color=PRODUCT2_COLOR,
            alpha=0.22,
        )
        ax.plot(data["n"], data["m_lower"], color=PRODUCT2_COLOR)
        ax.plot(data["n"], data["m_upper"], color=PRODUCT2_COLOR)
        ax.axhline(p0, color="#475569", linestyle="--", linewidth=1.0)
        ax.axhline(params.p1, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.axhline(params.p2, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.set_title(
            f"$p_0={p0:.2f}$  ({regime_label(p0, params)})",
            loc="left",
        )
        ax.set_xlabel("Observations, $n$")
        ax.set_ylabel("Posterior mean, $m$")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(p0_values) :]:
        ax.set_axis_off()
    if len(axes) > len(p0_values):
        axes[len(p0_values)].legend(
            handles=[
                Patch(
                    facecolor=PRODUCT2_COLOR,
                    alpha=0.22,
                    label="Product-2 band",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#475569",
                    linestyle="--",
                    label="$m=p_0$",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#94a3b8",
                    linestyle=":",
                    label="$p_1,p_2$",
                ),
            ],
            loc="center",
            fontsize=11,
        )
    fig.suptitle(
        "Product-2 switching bands across all benchmark qualities",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phi_margin_all_p(
    grids: dict[float, pd.DataFrame],
    p0_values: tuple[float, ...],
    grid_step: float,
    params: FluidParams,
    output_path: Path,
) -> None:
    fields: dict[float, tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]] = {}
    finite_values: list[np.ndarray] = []
    for p0 in p0_values:
        field = regular_nm_field(
            grids[p0],
            grid_step,
            "continuous_switch_margin",
        )
        fields[p0] = field
        finite_values.append(field[2].compressed())
    scale = float(
        np.quantile(np.abs(np.concatenate(finite_values)), 0.90)
    )
    scale = max(scale, 1e-8)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)

    configure_plot_style()
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.8 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    image = None
    for ax, p0 in zip(axes, p0_values, strict=False):
        n_grid, m_grid, field = fields[p0]
        image = ax.imshow(
            field,
            origin="lower",
            interpolation="nearest",
            extent=(n_grid[0], n_grid[-1], m_grid[0], m_grid[-1]),
            aspect="auto",
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        ax.contour(
            n_grid,
            m_grid,
            field.filled(np.nan),
            levels=[0.0],
            colors=["#111827"],
            linewidths=0.9,
        )
        ax.axhline(p0, color="#475569", linestyle="--", linewidth=0.9)
        ax.set_title(
            f"$p_0={p0:.2f}$  ({regime_label(p0, params)})",
            loc="left",
        )
        ax.set_xlabel("Observations, $n$")
        ax.set_ylabel("Posterior mean, $m$")
        ax.set_ylim(0.0, 1.0)
    for ax in axes[len(p0_values) :]:
        ax.set_axis_off()
    colorbar = fig.colorbar(
        image,
        ax=axes[: len(p0_values)],
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label(r"$w_m/(n+2)-\Delta c/\Delta p$")
    fig.suptitle(
        "Continuous switching margin for every $p_0$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paths_all_p(
    paths: dict[float, pd.DataFrame],
    p0_values: tuple[float, ...],
    params: FluidParams,
    output_path: Path,
) -> None:
    configure_plot_style()
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.4 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, p0 in zip(axes, p0_values, strict=False):
        path = paths[p0]
        ax.fill_between(
            path["n"],
            0.0,
            1.0,
            where=path["product2"].eq(1),
            transform=ax.get_xaxis_transform(),
            color=PRODUCT2_COLOR,
            alpha=0.16,
            step="post",
        )
        ax.plot(path["n"], path["m"], color="#2563eb", linewidth=1.8)
        ax.axhline(p0, color="#475569", linestyle="--", linewidth=1.0)
        ax.axhline(params.p1, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.axhline(params.p2, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.set_title(
            f"$p_0={p0:.2f}$  ({regime_label(p0, params)})",
            loc="left",
        )
        ax.set_xlabel("Observations, $n$")
        ax.set_ylabel("Optimal posterior path, $m(n)$")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes[len(p0_values) :]:
        ax.set_axis_off()
    if len(axes) > len(p0_values):
        axes[len(p0_values)].legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    color="#2563eb",
                    linewidth=1.8,
                    label="$m(n)$",
                ),
                Patch(
                    facecolor=PRODUCT2_COLOR,
                    alpha=0.16,
                    label="Product 2 on path",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="#475569",
                    linestyle="--",
                    label="$m=p_0$",
                ),
            ],
            loc="center",
            fontsize=11,
        )
    fig.suptitle(
        "Optimal observation-time paths for every $p_0$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected_n_intervals(
    boundaries: pd.DataFrame,
    p0_values: tuple[float, ...],
    selected_n: tuple[float, ...],
    output_path: Path,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        len(selected_n),
        figsize=(4.7 * len(selected_n), 4.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, n_value in zip(axes, selected_n, strict=True):
        for p0 in p0_values:
            row = boundaries[
                np.isclose(boundaries["p0"], p0)
                & np.isclose(boundaries["n"], n_value)
            ]
            if row.empty or not int(row.iloc[0]["product2_present"]):
                continue
            record = row.iloc[0]
            ax.plot(
                [record["m_lower"], record["m_upper"]],
                [p0, p0],
                color=PRODUCT2_COLOR,
                linewidth=7,
                solid_capstyle="butt",
            )
            ax.scatter(
                [p0],
                [p0],
                marker="D",
                color="#111827",
                s=24,
                zorder=3,
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="#94a3b8", linestyle="--")
        ax.set_title(f"$n={n_value:g}$", loc="left")
        ax.set_xlabel("Posterior mean, $m$")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Benchmark quality, $p_0$")
    axes[0].set_yticks(p0_values)
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=PRODUCT2_COLOR,
                linewidth=7,
                label="Product-2 interval",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="#111827",
                linestyle="none",
                label="$m=p_0$",
            ),
        ],
        loc="outside lower center",
        ncols=2,
    )
    fig.suptitle(
        "Cross-$p_0$ comparison of product-2 intervals",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyze all paper p0 panels in (n, m) coordinates."
    )
    parser.add_argument("--p0-values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--grid-step", type=float, default=1.0)
    parser.add_argument("--selected-n", default="10,40,80")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_dir / "outputs_gamma_0999" / "data",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=project_dir
        / "outputs_gamma_0999"
        / "reparameterized_all_p",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    if outputs_dir == source_dir or source_dir in outputs_dir.parents:
        raise ValueError("outputs-dir must be separate from the source data.")
    p0_values = parse_float_list(args.p0_values)
    selected_n = parse_float_list(args.selected_n)
    params = FluidParams()
    grids, paths, boundaries, diagnostics = load_and_transform(
        source_dir,
        outputs_dir,
        p0_values,
        params,
        args.grid_step,
    )
    if boundaries["product2_runs"].max() > 1:
        raise AssertionError(
            "At least one n-layer has a disconnected product-2 set."
        )
    summary, path_summary = summarize_all_p(
        grids,
        paths,
        boundaries,
        diagnostics,
        p0_values,
        params,
    )

    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "all_p_policy_summary.csv", index=False)
    path_summary.to_csv(
        tables_dir / "all_p_optimal_path_summary.csv",
        index=False,
    )
    boundaries.to_csv(
        tables_dir / "all_p_product2_boundaries_nm.csv",
        index=False,
    )
    diagnostics.to_csv(
        tables_dir / "all_p_hjb_diagnostics.csv",
        index=False,
    )
    config = {
        "source_directory": str(source_dir),
        "output_directory": str(outputs_dir),
        "parameters": {
            "p1": params.p1,
            "p2": params.p2,
            "c1": params.c1,
            "c2": params.c2,
            "revenue": params.revenue,
            "gamma": params.gamma,
        },
        "p0_values": p0_values,
        "grid_step": args.grid_step,
        "selected_n": selected_n,
        "scope": (
            "Cross-p0 structural analysis on the main production grid; "
            "not a grid-convergence diagnostic."
        ),
    }
    (outputs_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    plot_policy_all_p(
        grids,
        p0_values,
        args.grid_step,
        params,
        plots_dir / "policy_nm_all_p.png",
    )
    plot_boundaries_all_p(
        boundaries,
        p0_values,
        params,
        plots_dir / "product2_boundaries_nm_all_p.png",
    )
    plot_phi_margin_all_p(
        grids,
        p0_values,
        args.grid_step,
        params,
        plots_dir / "phi_switching_margin_nm_all_p.png",
    )
    plot_paths_all_p(
        paths,
        p0_values,
        params,
        plots_dir / "optimal_paths_nm_all_p.png",
    )
    plot_selected_n_intervals(
        boundaries,
        p0_values,
        selected_n,
        plots_dir / "product2_intervals_selected_n_all_p.png",
    )
    print(f"Saved all-p study under: {outputs_dir}", flush=True)


if __name__ == "__main__":
    main()
