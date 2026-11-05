"""Audit saved fixed-n sections without fitting or assuming threshold bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def three_point_witness(lower, upper=None, margin=0.0):
    """Find i<j<k with lower[i],lower[k]>margin and upper[j]<-margin.

    Passing the same vector as both bounds tests nominal signs. Zeros and
    uncertain middle cells do not count as strict product-1 separation.
    """
    low = np.asarray(lower, dtype=float)
    high = low if upper is None else np.asarray(upper, dtype=float)
    if low.ndim != 1 or high.shape != low.shape or margin < 0:
        raise ValueError("Expected matching one-dimensional bounds and margin >= 0")
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Bounds must be finite")
    if np.any(low > high + 1e-12):
        raise ValueError("Lower advantage cannot exceed upper advantage")
    positives = np.flatnonzero(low > margin)
    if len(positives) < 2:
        return None
    middle = np.flatnonzero(high < -margin)
    middle = middle[(middle > positives[0]) & (middle < positives[-1])]
    if not len(middle):
        return None
    j = int(middle[np.argmin(high[middle])])
    left, right = positives[positives < j], positives[positives > j]
    i, k = int(left[np.argmax(low[left])]), int(right[np.argmax(low[right])])
    return {"indices": [i, j, k], "lower": low[[i, j, k]].tolist(),
            "upper": high[[i, j, k]].tolist(),
            "strength": float(min(low[i], -high[j], low[k]))}


def audit_file(path: Path, strict_margin=1e-5):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        offsets = saved["row_offsets"]
        advantage = saved["advantages"]
        low, high = saved["advantage_lower"], saved["advantage_upper"]
    nominal = strict = max_components = 0
    first = None
    for level, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row = advantage[start:end]
        mask = row > 0
        max_components = max(max_components, int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]])))
        nominal += three_point_witness(row) is not None
        witness = three_point_witness(low[start:end], high[start:end], strict_margin)
        strict += witness is not None
        if witness is not None and first is None:
            first = {"n": level * metadata["h"], "S": [j * metadata["h"] for j in witness["indices"]], **witness}
    return {"file": str(path), "p0": metadata["parameters"]["p0"], "h": metadata["h"],
            "report_n": metadata["report_n"], "levels": len(offsets)-1,
            "nominal_sign_margin": 0, "strict_gap_margin": strict_margin,
            "max_nominal_components": max_components,
            "nominal_disconnected_levels": nominal, "strict_violation_levels": strict,
            "first_witness": first,
            "limitation": "Floating-point continuation brackets cover spatial truncation of this grid, not the continuous HJB mesh error."}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "outputs/p0_comparison")
    parser.add_argument("--output", type=Path, default=root / "outputs/interval_search/baseline_audit.json")
    args = parser.parse_args()
    files = sorted(args.baseline_dir.glob("p0_*/data/*.npz"))
    if not files:
        raise ValueError("No saved triangular solutions found")
    results = [audit_file(path) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved_solutions": len(results),
                      "fixed_n_levels": sum(r["levels"] for r in results),
                      "nominal_witness_levels": sum(r["nominal_disconnected_levels"] for r in results),
                      "strict_witness_levels": sum(r["strict_violation_levels"] for r in results)}, indent=2))


if __name__ == "__main__":
    main()
