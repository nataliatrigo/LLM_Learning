"""Collect primary and complementary plots in figures; group individual p0s."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.customer_product2_report import (
    GRID, HOME, P0S, RHOS, ROOT, digest, load_sources, verify_inputs,
)


def relocate_with_legacy_link(source, destination):
    """Keep archived manifest paths valid while moving the actual directories."""
    if source.is_symlink():
        if source.resolve() != destination.resolve() or not destination.is_dir():
            raise ValueError("Unexpected legacy directory link: " + str(source))
        return
    if not source.is_dir() or destination.exists():
        raise ValueError("Unsafe individual-folder relocation: " + str(source))
    source.rename(destination)
    source.symlink_to(destination.relative_to(source.parent), target_is_directory=True)


def read_report(folder, inputs):
    path = folder / "manifest.json"
    report = json.loads(path.read_text())
    if (report.get("status") != "complete" or report.get("displayed_n") != GRID
            or report.get("p0s") != P0S or report.get("rhos") != RHOS
            or report["shared_economic_configuration"]["parameters"]["c2"] != .65):
        raise ValueError("Source plot report differs from the agreed common comparison")
    for field in ["source_input_sha256", "reporting_source_sha256"]:
        verify_inputs(report[field])
        inputs.update(report[field])
    generated = {str(folder/rel): expected for rel, expected in report["generated_file_sha256"].items()}
    verify_inputs(generated)
    inputs.update(generated)
    inputs[str(path)] = digest(path)
    return report


def main():
    figures = HOME / "figures"
    primary = HOME / "seller_product2_given_sale"
    complementary = HOME / "customer_product2_per_period"
    inputs = {}
    first, second = (read_report(folder, inputs) for folder in [primary, complementary])
    if first.get("metric") != "product2_conditional":
        raise ValueError("Primary metric must be the original product mix among own sales")
    load_sources()  # Check the original campaign before changing directory locations.

    individuals = figures / "individual p0s"
    relocate_with_legacy_link(figures / "p0_sweep", individuals)
    relocate_with_legacy_link(figures / "p0_0p5_common_grid", individuals / "p0_0p5_common_grid")
    load_sources()  # Archived fingerprints and paths must still work after relocation.

    copies = [
        (primary/"figures/01_product2_given_sale_by_p0.png", "01_primary_product2_given_sale_by_p0.png"),
        (primary/"figures/02_product2_given_sale_by_rho.png", "02_primary_product2_given_sale_by_rho.png"),
        (complementary/"figures/02_customer_product2_by_p0.png", "03_complementary_customer_product2_by_p0.png"),
        (complementary/"figures/01_customer_product2_by_rho.png", "04_complementary_customer_product2_by_rho.png"),
    ]
    outputs = []
    for source, name in copies:
        target = figures / name
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise ValueError("Compiled PNG differs from its source: " + name)
        outputs.append(target)

    pdf_sources = [primary/"figures/original_metric_comparison.pdf",
                   complementary/"figures/customer_product2_comparison.pdf"]
    pdf_output = figures / "primary_and_complementary_metrics.pdf"
    with tempfile.TemporaryDirectory(prefix="main-metrics-pdf-") as temporary:
        temp = Path(temporary)
        pages = []
        for index, source in enumerate(pdf_sources):
            prefix = temp / f"source_{index}_%d.pdf"
            subprocess.run(["pdfseparate", "-f", "1", "-l", "2", str(source), str(prefix)], check=True)
            page_order = [1, 2] if index == 0 else [2, 1]
            pages.extend(temp/f"source_{index}_{page}.pdf" for page in page_order)
        subprocess.run(["pdfunite", *map(str, pages), str(pdf_output)], check=True)
    info = subprocess.run(["pdfinfo", str(pdf_output)], capture_output=True, text=True, check=True).stdout
    if re.search(r"^Pages:\s+4\s*$", info, re.MULTILINE) is None:
        raise ValueError("The compilation must have exactly four pages")
    outputs.append(pdf_output)

    keys = ["p0", "rho", "n", "model"]
    data = pd.read_csv(primary/"data/product2_given_sale_summary.csv")
    customer = pd.read_csv(complementary/"data/customer_product2_summary.csv")
    paired = data[keys+["product2_conditional", "sale_rate", "outside_rate"]].merge(
        customer[keys+["customer_product2_calendar"]], on=keys, validate="one_to_one")
    if len(paired) != 54:
        raise ValueError("The compiled metrics must cover the same 54 model cases")
    seller_count = paired.model.map({"monopoly": 1, "competition": 2, "conditional_competition": 2})
    expected = seller_count*paired.sale_rate*paired.product2_conditional
    if (abs(expected-paired.customer_product2_calendar) > 1e-12).any():
        raise ValueError("Primary and complementary metrics do not reconcile")
    paired = paired.rename(columns={"product2_conditional": "primary_product2_given_own_sale",
                                    "customer_product2_calendar": "complementary_customer_product2_per_period"})
    paired.insert(3, "c2", .65)
    table_path = figures / "main_metrics_data.csv"
    paired.sort_values(keys).to_csv(table_path, index=False)
    outputs.append(table_path)

    (figures / "README.md").write_text("\n".join([
        "# Compilados: métrica principal y complementaria", "",
        "Calibración común: c2=0.65, grilla de 101 puntos, seis p0 y rho=0.6,0.7,0.8.",
        "Se reúnen los gráficos existentes y auditados; no se ejecutan simulaciones ni se resuelven nuevos equilibrios.", "",
        "| Métrica | Frente a p0 | Frente a rho |", "| --- | --- | --- |",
        "| **Principal:** producto 2 como porcentaje de las ventas propias | [Tres paneles](01_primary_product2_given_sale_by_p0.png) | [Seis paneles](02_primary_product2_given_sale_by_rho.png) |",
        "| **Complementaria:** producto 2 recibido por el customer por período, sumando A y B | [Tres paneles](03_complementary_customer_product2_by_p0.png) | [Seis paneles](04_complementary_customer_product2_by_rho.png) |", "",
        "La principal es `P(producto 2 | venta propia)`. La complementaria es `P(customer recibe producto 2 de algún seller)` por período calendario.",
        "En monopolio la complementaria cuenta al único seller; en competencia suma ambos. La opción externa no cuenta como producto 2.", "",
        "- [PDF con los cuatro compilados](primary_and_complementary_metrics.pdf): páginas 1-2 principal y páginas 3-4 complementaria.",
        "- [Tabla conjunta de las dos métricas](main_metrics_data.csv).",
        "- [Carpetas individuales por p0](individual%20p0s/).", "",
        "Los seis casos individuales del barrido están en `individual p0s/p0_0p1`, `p0_0p3`, `p0_0p5`, `p0_0p7`, `p0_0p8` y `p0_0p9`.",
        "La comparación histórica ampliada de p0=0.5 está en `individual p0s/p0_0p5_common_grid`.",
        "Los accesos `p0_sweep` y `p0_0p5_common_grid` conservan las rutas históricas necesarias para los certificados y enlaces originales.",
        "La campaña `p0_sweep_201` continúa pausada y conserva su ubicación.", "",
        "Los gráficos originales del customer, incluidos los de compras a p0, A y B, siguen en `../customer_product2_per_period/`.", "",
        "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.compile_main_metrics", "```", ""]))
    (individuals / "README.md").write_text("\n".join([
        "# Gráficos individuales por p0", "",
        "Los datos y gráficos individuales se conservan sin cambios.", "",
        *[f"- [p0 = {p0:g}](p0_{str(p0).replace('.', 'p')}/)." for p0 in P0S],
        "- [Comparación histórica ampliada de p0=0.5](p0_0p5_common_grid/).", "",
        "[Volver a los compilados principal y complementario](../README.md).", ""]))
    outputs += [figures/"README.md", individuals/"README.md"]
    verify_inputs(inputs)
    source = Path(__file__).resolve()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="complete",
        displayed_n=GRID, c2=.65, p0s=P0S, rhos=RHOS, rows=len(paired),
        primary="P(product 2 | own sale)",
        complementary="P(customer receives product 2 from any strategic seller in a calendar period)",
        pdf_page_order=[dict(page=i+1, metric=metric, horizontal=horizontal,
                             source_pdf=str(pdf_sources[i//2].relative_to(ROOT)), source_page=[1, 2, 2, 1][i])
                        for i, (metric, horizontal) in enumerate([
                            ("primary", "p0"), ("primary", "rho"),
                            ("complementary", "p0"), ("complementary", "rho")])],
        png_copies={name: str(source.relative_to(ROOT)) for source, name in copies},
        source_input_sha256=inputs,
        compilation_source_sha256={str(source.relative_to(ROOT)): digest(source)},
        generated_file_sha256={str(p.relative_to(figures)): digest(p) for p in outputs},
        individual_directories=str(individuals.relative_to(ROOT)),
        legacy_directory_aliases={"p0_sweep": "individual p0s", "p0_0p5_common_grid": "individual p0s/p0_0p5_common_grid"},
        source_modification="Individual directories relocated with legacy aliases; all original source and figure bytes preserved")
    (figures/"main_metrics_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(figures=str(figures), individuals=str(individuals), rows=len(paired), pdf_pages=4), indent=2))


if __name__ == "__main__":
    main()
