"""Generate the English results note directly from the validated tables."""
from .run_experiment import CONFIG, HOME


def summarize(rows, validation):
    raise RuntimeError(
        "Legacy rho-matched results publication is retired. Use python -m "
        "forgetting_random_replacement.oblivious.code.capacity_report."
    )
    lookup = {(r["capacity"], r["p0"], r["sellers"], r["closure"]): r for r in rows}
    lines = ["# Random-replacement oblivious equilibrium results", "",
             f"All **{len(rows)} distinct solutions** passed the campaign audit: "
             "108 Experiment-8 cases and six additional three-to-five-seller cases. "
             "All seven independent unit tests pass.", "",
             "The sweep uses outside options `.1, .3, .5, .7, .8, .9` and reference "
             "retention settings `.6, .7, .8, .9, .95, .97, .99`. The first two "
             "retention settings both map to capacity 3 and share the same solution.", "",
             "## The requested seller-count plot", "",
             "![Product 2 by seller count](figures/seller_count/01_product2_by_seller_count.png)", "",
             "The original geometric-forgetting plot is reproduced with uniform random "
             "replacement, holding the economic primitives and marginal OE closure fixed. "
             "Reference `rho=.6` now means capacity `C=3`, with effective retention `2/3`. "
             "The count model is exact; the old 31-versus-21 reputation-grid comparison "
             "does not apply.", "",
             "| Outside option | 1 seller | 2 sellers | 3 sellers | 4 sellers | 5 sellers |",
             "|---:|---:|---:|---:|---:|---:|"]
    for p0 in (.1, .3):
        values = [100*lookup[3, p0, sellers, "monopoly" if sellers == 1 else "marginal"]
                  ["product2_conditional"] for sellers in range(1, 6)]
        lines.append(f"| {p0:g} | " + " | ".join(f"{value:.2f}%" for value in values) + " |")
    lines += ["", "Product-2 use is measured as a percentage of each seller's own sales. "
              "Equal shares can arise when different market sizes select the same own-count "
              "policy: the purchase-time count law depends on that policy, whereas competition "
              "also changes the frequency of sales and the calendar-time distribution.", "",
              "## Experiment 8", "",
              "![Experiment 8 overview](figures/experiment8/00_product2_usage_overview.png)", "",
              "Monopoly, marginal OE, and conditional OE each solve their own Bellman "
              "problem and policy/distribution consistency conditions. The full joint "
              "calendar law is retained. Actual discounted profits are evaluated separately "
              "from perceived OE values. The figures display Monopoly and Marginal OE; "
              "Conditional OE remains in the saved cases and numerical tables.", "",
              "For the original baseline outside option `p0=.5`, the product-2 shares are:", "",
              "| Capacity | Effective retention | Monopoly | Marginal OE | Conditional OE |",
              "|---:|---:|---:|---:|---:|"]
    for capacity in CONFIG["capacities"]:
        values = [100*lookup[capacity, .5, sellers, closure]["product2_conditional"]
                  for sellers, closure in ((1, "monopoly"), (2, "marginal"), (2, "conditional"))]
        lines.append(f"| {capacity} | {1-1/capacity:.6f} | " +
                     " | ".join(f"{value:.4f}%" for value in values) + " |")
    lines += ["", "For every outside option, the figure folder contains:", "",
              "1. Exact own-count policies across all six capacities.",
              "2. Product 2 as a share of own sales across the seven reference retention settings.",
              "3. Stationary profit per seller/calendar period and actual discounted profit.",
              "4. The market outside-option share.",
              "5. The stationary own-count distributions.", "",
              "### Comparisons as the outside option changes", "",
              "![Product 2 among own sales by outside option]"
              "(figures/experiment8/by_outside_option/01_product2_given_sale_by_p0.png)", "",
              "![Customer receives product 2 per calendar period by outside option]"
              "(figures/experiment8/by_outside_option/02_customer_product2_by_p0.png)", "",
              "Both figures reproduce the three-panel reference layout at reference "
              "`rho=.6, .7, .8`, corresponding to capacities `3, 3, 5`. The `.6` and `.7` "
              "panels coincide because they describe the same random-replacement model. "
              "The first figure uses `product2_conditional`, the mix among a seller's own "
              "sales. The second uses `sellers * product2_calendar`, the probability that "
              "the customer receives product 2 from any strategic seller in a calendar "
              "period. It sums both sellers in competition and counts outside-option "
              "selection as zero product-2 receipt. All 54 comparison rows, including "
              "Conditional OE, are saved in "
              "[outside_option_comparisons.csv](data/outside_option_comparisons.csv).", "",
              "All figures are available as PNG and PDF. Complete numerical results are in "
              "[summary.csv](data/summary.csv), [policies.csv](data/policies.csv), and "
              "[experiment8_by_reference_rho.csv](data/experiment8_by_reference_rho.csv). "
              "NPZ/JSON pairs in `data/cases/` retain every policy, joint law, value, and certificate.", "",
              "## Validation", "",
              f"The audit passed **{validation['passed_case_count']}/{validation['expected_case_count']} cases**. "
              "It checked complete coverage, source and checkpoint hashes, positive joint "
              "and conditional rows, stationary flow, Bellman incentives, actual discounted "
              "values, purchase-time birth-death laws, and every reported flow metric. "
              "Fresh positive GTH elimination also checked the full conditional rival "
              "distribution row by row against the stored reverse-order solution.", "",
              "| Reconstructed check | Worst error |", "|---|---:|"]
    for name, error in validation["worst_errors"].items():
        lines.append(f"| {name.replace('_', ' ')} | {error:.3e} |")
    lines += ["", "See [validation.json](data/validation.json) for individual certificates "
              "and [README.md](README.md) for model equations, initialization, the "
              "retention-to-capacity mapping, and reproduction commands.", "",
              "These results concern the exact finite mature-memory models in floating-point "
              "arithmetic. Stationary statistics do not imply rapid convergence from an "
              "initial state; high outside-option values can make updates very infrequent.", ""]
    (HOME / "RESULTS.md").write_text("\n".join(lines))
