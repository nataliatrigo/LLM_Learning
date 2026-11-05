"""Experiment locations and read-only resolution of historical saved paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FOLDERS = {
    "oe_distribution": "01_oe_distribution",
    "oe_p0": "02_oe_p0",
    "oe_following_monopoly": "03_oe_following_monopoly",
    "oe_conditional_distribution": "04_oe_conditional_distribution",
    "oe_p0_temporal": "05_oe_p0_temporal",
    "equilibrium_comparison": "06_equilibrium_comparison",
    "full_comparison": "07_full_comparison",
    "oe_distribution_fixed_p0": "08_oe_distribution_fixed_p0",
    "oe_three_sellers_fixed_p0": "09_oe_three_sellers_fixed_p0",
    "fluid_oblivious": "10_fluid_oblivious",
    "fluid_oblivious_initial_conditions": "11_fluid_oblivious_initial_conditions",
    "fluid_average_profit": "12_fluid_average_profit",
    "fluid_asymmetric": "13_fluid_asymmetric",
}
LEGACY_FOLDERS = dict(zip(
    ("oblivious_real_marginal", "endogenous_p0", "following_monopoly",
     "conditional_oe", "temporal_p0", "equilibrium_comparison", "Full comparison"),
    EXPERIMENT_FOLDERS.values(),
))


def resolve_saved_path(path, project_root=None):
    """Locate a saved repository path without rewriting its provenance metadata.

    Absolute paths outside this repository are preserved. Paths saved under a
    previous checkout named LLM_Learning can also be read from this checkout.
    This only resolves locations; source and checkpoint hash checks still apply.
    """
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            matches = [i for i, part in enumerate(candidate.parts) if part == PROJECT_ROOT.name]
            if not matches:
                return candidate
            relative = Path(*candidate.parts[matches[-1] + 1:])
            if not relative.parts or relative.parts[0] not in {*LEGACY_FOLDERS, "oblivious"}:
                return candidate
    else:
        relative = candidate
    if relative.parts and relative.parts[0] in LEGACY_FOLDERS:
        relative = Path("oblivious", LEGACY_FOLDERS[relative.parts[0]], *relative.parts[1:])
    return root / relative
