"""Plot the selected OE p0 and following-monopoly equilibrium benchmarks."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from plot_comparison import plt, STYLE


OUTPUT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[4]
DATA, FIGURES = OUTPUT / "data", OUTPUT / "Figures"
SOURCE = PROJECT / "oblivious/02_oe_p0/data/rho_forgetting_mixed/arrays"
FOLLOWING = PROJECT / "oblivious/03_oe_following_monopoly/data/quality_sales_n51"


def main():
    config_path = DATA / "comparison.json"
    config = json.loads(config_path.read_text())
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    if digest(config_path) != manifest["data_files_sha256"]["comparison.json"]:
        raise ValueError("Comparison configuration changed since verification")

    rows, inputs = [], {}
    following_manifest_path = FOLLOWING / "manifest.json"
    following_manifest = json.loads(following_manifest_path.read_text())
    if not (following_manifest["status"] == "complete"
            and following_manifest["configuration"]["rhos"] == config["rhos"]
            and following_manifest["configuration"]["grids"] == [config["grid"]]
            and following_manifest["configuration"]["weighting"] == "purchase"):
        raise ValueError("Following-monopoly experiment is incomplete or incompatible")
    for rho in config["rhos"]:
        tag = f"rho_{str(rho).replace('.', 'p')}_n{config['grid']:03d}"
        source_path = SOURCE / f"{tag}.json"
        source_name = str(source_path.relative_to(PROJECT))
        inputs[source_name] = digest(source_path)
        if inputs[source_name] != manifest["source_files_sha256"][source_name]:
            raise ValueError(f"Equilibrium source changed: {source_name}")
        source = json.loads(source_path.read_text())
        selected = source["selected"]
        p0 = config["p0_by_rho"][str(rho)]
        if not (selected["verified"] and selected["candidate_verified"]
                and selected["p0"] == p0 and source["rho"] == rho
                and source["n"] == config["grid"] and 0 < p0 < 1):
            raise ValueError(f"Unverified or mismatched equilibrium at rho={rho}")
        following_path = FOLLOWING / "arrays" / f"{tag}.json"
        following_name = str(following_path.relative_to(PROJECT))
        inputs[following_name] = digest(following_path)
        if inputs[following_name] != following_manifest["checkpoint_sha256"][str(following_path.relative_to(FOLLOWING))]:
            raise ValueError(f"Following-monopoly source changed: {following_name}")
        following_case = json.loads(following_path.read_text())
        following = following_case["selected"]
        if not (following_case["rho"] == rho and following_case["n"] == config["grid"]
                and following_case["configuration"]["parameters"] == source["configuration"]["parameters"]
                and following_case["configuration"]["quality_weighting"] == "purchase"
                and following["verified"] and following["candidate_verified"]
                and following["dp_diagnostics"]["verified"]
                and following["stationary_diagnostics"]["verified"]
                and following["monopoly_certificate"]["verified"]
                and 0 < following["p0"] < 1):
            raise ValueError(f"Invalid following-monopoly equilibrium at rho={rho}")
        rows.append(dict(rho=rho, n=config["grid"], p0=p0,
                         selection="Saved selected OE p0 equilibrium",
                         source=source_name,
                         following_monopoly_p0=following["p0"],
                         following_monopoly_selection="Saved selected OE quality equilibrium; quality conditional on rival sales",
                         following_monopoly_source=following_name))

    csv_path = DATA / "equilibrium_p0_across_rho.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        fig.subplots_adjust(left=.09, right=.98, top=.76, bottom=.28)
        x = list(range(len(rows)))
        for key, label, color, marker, offset in (
                ("p0", "OE p0", "#D55E00", "o", (0, -17)),
                ("following_monopoly_p0", "Following monopoly (OE quality)", "#7B3294", "D", (0, 11))):
            values = [row[key] for row in rows]
            ax.plot(x, values, color=color, marker=marker, ms=6, lw=1.8,
                    ls="--" if key == "following_monopoly_p0" else "-", label=label)
            for pos, value in zip(x, values):
                ax.annotate(f"{value:.4f}", (pos, value), xytext=offset,
                            textcoords="offset points", ha="center", fontsize=9,
                            color=color, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_xticks(x, [f"{row['rho']:g}" for row in rows])
        ax.set(xlabel=r"Forgetting factor $\rho$ (discrete values)",
               ylabel=r"Equilibrium benchmark $p_0$",
               xlim=(-.3, len(rows)-.7), ylim=(0, .9))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=.7)
        fig.suptitle(r"Equilibrium $p_0$ across $\rho$", fontsize=15, y=.97)
        fig.text(.53, .885, "Selected equilibria on the common 51-point grid",
                 ha="center", fontsize=10)
        fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="upper center",
                   bbox_to_anchor=(.53, .855), frameon=False)
        fig.text(.53, .17, "OE p0: mean rival TS score. Following monopoly: mean rival quality conditional on sales.",
                 ha="center", fontsize=8.5)
        fig.text(.53, .115, "OE p0 at rho = 0.97 and 0.99 remains sensitive to grid refinement.",
                 ha="center", fontsize=8.5)
        fig.text(.53, .065, "Rho values are equally spaced; lines guide the eye.",
                 ha="center", fontsize=8.5)
        for suffix in ("png", "pdf"):
            path = FIGURES / f"05_equilibrium_p0_across_rho.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            paths.append(path)
        plt.close(fig)

    for path in (config_path, manifest_path, following_manifest_path, Path(__file__),
                 Path(__file__).with_name("plot_comparison.py")):
        inputs[str(path.relative_to(PROJECT))] = digest(path)
    result = dict(
        status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
        definition="Saved selected equilibrium p0 for OE p0 and following monopoly (OE quality), at every rho on the same 51-point grid.",
        model_definitions={"p0":"Stationary mean rival TS score", "following_monopoly_p0":"Stationary true rival quality conditional on rival sales"},
        grid=config["grid"], rhos=config["rhos"],
        caveat="These finite-grid selected equilibria are not asserted unique. Rho 0.97 and 0.99 remain sensitive to grid refinement.",
        input_files_sha256=inputs,
        output_files_sha256={str(path.relative_to(OUTPUT)):digest(path)
                             for path in [csv_path, *paths]})
    (DATA / "equilibrium_p0_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Verified all {len(rows)} source benchmarks; saved PNG, PDF, and CSV in {OUTPUT}")


if __name__ == "__main__":
    main()
