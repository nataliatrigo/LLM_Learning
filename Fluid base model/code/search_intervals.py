"""Reproducible parameter search for strict product-2 / product-1 / product-2.

Every horizontal section holds n=S+F fixed.  A witness must contain three
ordered S values whose action advantages have signs +,-,+.  Boundary brackets
are exact for this finite-difference grid; they do not certify the continuous
fluid HJB.  Run finer grids independently before interpreting a witness.

The normalization R-c1=1 and c2-c1=cost loses no positive-margin scale.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import betaincc


def _floats(text):
    return [float(item) for item in text.split(",")]


def search_group(configurations, p0, delta, h, report_n, boundary_tolerance,
                 gap_tolerance=1e-5, nominal_sign_tolerance=0.0):
    """Batch all (p1,p2,cost) at fixed demand, discount, and grid.

    Keep only consecutive diagonals. The outer upper continuation is 1/delta,
    and the lower continuation is zero.  The reported gap bounds use both.
    """
    cfg = np.asarray(configurations, dtype=float)
    p1, p2, cost = cfg.T[:, :, None]
    count = len(cfg)
    report = int(round(report_n / h))
    buffer = max(1, math.ceil(math.log(1 / delta / boundary_tolerance)
                              / math.log1p(delta * h)))
    outer = report + buffer
    low = np.zeros((count, outer + 1))
    high = np.full((count, outer + 1), 1 / delta)
    witnesses = []
    nominal_witnesses = []
    max_boundary_width = 0.0
    rows_checked = 0
    product2_configs = np.zeros(count, dtype=bool)

    def find_witnesses(positive, negative, advantage, level, strict):
        # Positive witnesses on both sides of a strictly negative middle cell.
        left = np.maximum.accumulate(positive, axis=1)
        right = np.maximum.accumulate(positive[:, ::-1], axis=1)[:, ::-1]
        eligible = negative & left & right
        found_configs = np.flatnonzero(np.any(eligible, axis=1))
        target = witnesses if strict else nominal_witnesses
        for index in found_configs:
            middle_options = np.flatnonzero(eligible[index])
            middle = int(middle_options[np.argmin(advantage[index, middle_options])])
            before = np.flatnonzero(positive[index, :middle])
            after = np.flatnonzero(positive[index, middle+1:]) + middle+1
            first = int(before[np.argmax(advantage[index, before])])
            third = int(after[np.argmax(advantage[index, after])])
            indices = [first, middle, third]
            target.append({
                "configuration_index": int(index), "p0": float(p0),
                "delta": float(delta), "p1": float(cfg[index, 0]),
                "p2": float(cfg[index, 1]), "cost": float(cfg[index, 2]),
                "h": float(h), "n": float(level*h),
                "S": [float(j*h) for j in indices],
                "F": [float((level-j)*h) for j in indices],
                "advantages": advantage[index, indices].tolist(),
                "advantage_lower": alo[index, indices].tolist(),
                "advantage_upper": ahi[index, indices].tolist(),
                "strict_boundary_brackets": strict,
            })

    started = time.monotonic()
    for level in range(outer - 1, -1, -1):
        indices = np.arange(level+1, dtype=float)
        d = betaincc(indices*h + 1, (level-indices)*h + 1, p0)
        d = np.clip(d, 0, 1)
        if level <= report:
            advantage = -cost + (p2-p1)*(low[:, 1:] - low[:, :-1])/h
            alo = -cost + (p2-p1)*(low[:, 1:] - high[:, :-1])/h
            ahi = -cost + (p2-p1)*(high[:, 1:] - low[:, :-1])/h
            # Zero demand yields indifference, not a product-2 component.
            active = d[None, :] > 0
            pos = (alo > gap_tolerance) & active
            neg = (ahi < -gap_tolerance) & active
            product2_configs |= np.any(pos, axis=1)
            if level >= 2:
                find_witnesses(pos, neg, advantage, level, True)
                nominal_pos = (advantage > nominal_sign_tolerance) & active
                nominal_neg = (advantage < -nominal_sign_tolerance) & active
                # Diagnostics retain nominal candidates even if brackets fail.
                find_witnesses(nominal_pos, nominal_neg, advantage, level, False)
                rows_checked += count
        factor = d / (delta*h+d)
        low_diff = low[:, 1:] - low[:, :-1]
        high_diff = high[:, 1:] - high[:, :-1]
        low_base = low[:, :-1]
        high_base = high[:, :-1]
        low = factor * (low_base + np.maximum(h + p1*low_diff,
                                             h*(1-cost) + p2*low_diff))
        high = factor * (high_base + np.maximum(h + p1*high_diff,
                                               h*(1-cost) + p2*high_diff))
        if level <= report:
            max_boundary_width = max(max_boundary_width,
                                     float(np.max(high-low)))
    result = {
        "p0": float(p0), "delta": float(delta), "h": float(h),
        "report_n": float(report*h), "cutoff_n": float(outer*h),
        "configuration_count": count,
        "configurations": [{"p1": float(a), "p2": float(b), "cost": float(c)}
                           for a,b,c in cfg],
        "gap_tolerance": float(gap_tolerance),
        "nominal_sign_tolerance": float(nominal_sign_tolerance),
        "rows_checked": rows_checked,
        "product2_configuration_count": int(product2_configs.sum()),
        "max_actual_value_bracket_width": max_boundary_width,
        "geometric_value_boundary_bound": (1/delta)*math.exp(-buffer*math.log1p(delta*h)),
        "strict_witness_count": len(witnesses),
        "nominal_witness_count": len(nominal_witnesses),
        "strict_witnesses": witnesses,
        "nominal_witnesses": nominal_witnesses,
        "elapsed_seconds": time.monotonic()-started,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=_floats, default=_floats(".05,.1,.3,.5,.7,.9,.95"))
    parser.add_argument("--delta", type=_floats, default=_floats(".03,.1,.3,1"))
    parser.add_argument("--p", type=_floats, default=_floats("0,.1,.2,.35,.5,.65,.8,.9,1"))
    parser.add_argument("--cost", type=_floats, default=_floats(".01,.03,.06,.1,.2,.4,.6,.8,.95"))
    parser.add_argument("--random-configurations", type=int, default=0,
                        help="If >0, replace the probability/cost grid by a seeded random sample")
    parser.add_argument("--seed", type=int, default=261020)
    parser.add_argument("--h", type=float, default=1)
    parser.add_argument("--report-n", type=float, default=80)
    parser.add_argument("--boundary-tolerance", type=float, default=1e-5)
    parser.add_argument("--gap-tolerance", type=float, default=1e-4)
    parser.add_argument("--nominal-sign-tolerance", type=float, default=0.0,
                        help="Raw nominal signs by default; strict boundary signs retain --gap-tolerance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.random_configurations:
        rng = np.random.default_rng(args.seed)
        probabilities = np.sort(rng.uniform(0,1,size=(args.random_configurations,2)),axis=1)
        costs = np.exp(rng.uniform(math.log(.001), math.log(.999),
                                  size=args.random_configurations))
        configurations = np.column_stack([probabilities,costs]).tolist()
    else:
        configurations = [(p1,p2,cost) for p1,p2 in itertools.combinations(args.p,2)
                          for cost in args.cost]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {key: str(value) if isinstance(value,Path) else value
                for key,value in vars(args).items()}
    manifest.update({"normalization":"revenue=1,c1=0,c2=cost",
                     "meaning":"fixed n sections; strict positive-negative-positive advantage",
                     "limitation":"brackets certify spatial truncation of the grid only, not discretization"})
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summaries=[]
    with (args.output/"progress.jsonl").open("w") as log:
        for number,(p0,delta) in enumerate(itertools.product(args.p0,args.delta)):
            result = search_group(configurations,p0,delta,args.h,args.report_n,
                                  args.boundary_tolerance,args.gap_tolerance,
                                  args.nominal_sign_tolerance)
            filename=f"group_{number:03d}_p0_{p0:g}_delta_{delta:g}.json"
            (args.output/filename).write_text(json.dumps(result,indent=2)+"\n")
            summary={key:value for key,value in result.items()
                     if key not in ("configurations","strict_witnesses","nominal_witnesses")}
            summary["detail_file"]=filename
            summaries.append(summary)
            log.write(json.dumps(summary)+"\n")
            log.flush()
            print(json.dumps(summary),flush=True)
    total={"configuration_solves":sum(s["configuration_count"] for s in summaries),
           "fixed_n_rows_checked":sum(s["rows_checked"] for s in summaries),
           "strict_witness_count":sum(s["strict_witness_count"] for s in summaries),
           "nominal_witness_count":sum(s["nominal_witness_count"] for s in summaries),
           "groups":summaries}
    (args.output/"summary.json").write_text(json.dumps(total,indent=2)+"\n")


if __name__ == "__main__":
    main()
