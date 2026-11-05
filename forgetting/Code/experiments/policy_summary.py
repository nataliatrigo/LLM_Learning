"""Summarize the numerical policy cutoff, thresholds, and late customer-1 states."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from forgetting.Code.layers import solve_observation_layers
from forgetting.Code.model import Primitives, solve_boundary, reachable_maximum
from forgetting.Code.experiments.states import policy_intervals


def continue_customer(meta, prefix, advantage, coordinate, terminal, *, interactions, burn_in):
    """Continue the exact customer-1 random stream; retain only late extrema.

    The interpolation is the scalar version of simulation._policy_and_demand.
    The original RNG width matters even though only customer 1 is retained.
    """
    rho = meta['rho']
    primitives = Primitives.from_mapping(meta['seller'])
    width = meta.get('original_paths', meta['shape'][1])
    uniforms = np.random.default_rng(meta['seed']).random((interactions, width))[:, 0]
    length = len(prefix)
    if not length <= burn_in < interactions or burn_in < terminal:
        raise ValueError('Require saved prefix <= burn-in < interactions and burn-in >= K')
    expected = uniforms[:length] < np.where(prefix['product2'], primitives.p2, primitives.p1)
    np.testing.assert_array_equal(expected, prefix['success'])
    x = rho * prefix['x_before'][-1] + (1-rho) * prefix['success'][-1]
    masses = 1 - np.power(rho, np.arange(interactions))
    tolerance = meta['policy_solver']['action_tolerance']
    nodes = len(coordinate)
    minimum, maximum = float('inf'), -float('inf')
    min_at = max_at = product2_count = 0
    for k in range(length, interactions):
        position = min(max(x/masses[k], 0.), 1.) * (nodes-1)
        left = min(int(position), nodes-2)
        weight = position-left
        row = advantage[min(k, terminal)] if advantage.ndim == 2 else advantage
        product2 = (1-weight)*row[left] + weight*row[left+1] > tolerance
        if k >= burn_in:
            if x < minimum:
                minimum, min_at = x, k+1
            if x > maximum:
                maximum, max_at = x, k+1
            product2_count += int(product2)
        success = uniforms[k] < (primitives.p2 if product2 else primitives.p1)
        x = rho*x + (1-rho)*success
    return dict(S_min=minimum/(1-rho), S_max=maximum/(1-rho),
                min_at_interaction=min_at, max_at_interaction=max_at,
                late_product2_count=product2_count)


def advantage_summary(rho, coordinate, advantage):
    """Raw Q2-Q1 maximum on the reachable set, without subtracting action tolerance."""
    maximum, argmax, radius = reachable_maximum(
        SimpleNamespace(rho=rho, coordinate=coordinate, advantage=advantage),
        reachable_depth=16)
    return dict(A_rho=maximum, A_rho_argmax_S=argmax/(1-rho),
                A_rho_reachable_depth=16, A_rho_cover_radius_x=radius)


def summarize_pair(manifest, *, interactions, burn_in):
    meta = json.loads(manifest.read_text())
    solver = meta['policy_solver']
    rho = meta['rho']
    terminal = math.ceil(math.log(solver['terminal_gap']) / math.log(rho))
    records = np.load(manifest.parent / 'observations.npy', mmap_mode='r')[:, 0]
    primitives = Primitives.from_mapping(meta['seller'])
    if terminal > len(records):
        _, policy = solve_observation_layers(primitives, rho, **solver, return_policy=True)
        advantage = policy['advantage_table']
        coordinate = policy['coordinate']
        mature = advantage[-1]
    else:
        solution = solve_boundary(primitives, rho,
            central_nodes=solver['nodes'], outer_nodes=2, half_widths=1e6,
            tolerance=solver['tolerance'], max_iterations=solver['max_iterations'])
        advantage = mature = solution.advantage
        coordinate = solution.coordinate
    intervals = policy_intervals(SimpleNamespace(coordinate=coordinate, advantage=mature),
                                 solver['action_tolerance'])
    regions = [(lo/(1-rho), hi/(1-rho)) for lo, hi, action in intervals if action == 'product2']
    switches = []
    for lo, hi, action in intervals:
        if action != 'product2':
            continue
        if lo > 0:
            switches.append(f'{lo/(1-rho):.6f} (1→2)')
        if hi < 1:
            switches.append(f'{hi/(1-rho):.6f} (2→1)')
    extrema = continue_customer(meta, records, advantage, coordinate, terminal,
                               interactions=interactions, burn_in=burn_in)
    return dict(**primitives.as_dict(), rho=rho, K=terminal, first_limit_interaction=terminal+1,
        thresholds_S='; '.join(switches) or 'None',
        product2_intervals_S=json.dumps(regions), **extrema,
        **advantage_summary(rho, coordinate, mature),
        burn_in=burn_in, end_interaction=interactions, trajectory=1,
        terminal_gap=solver['terminal_gap'], action_tolerance=solver['action_tolerance'],
        seed=str(meta['seed']))


def write_table(frame, directory):
    frame = frame.sort_values(['p0', 'c2', 'rho']).drop(columns=['first_limit_interaction'], errors='ignore')
    def region_text(row, *, reachable):
        regions = json.loads(row.product2_intervals_S)
        if not regions or (reachable and row.A_rho <= row.action_tolerance):
            return 'None'
        parts = []
        for lower, upper in regions:
            left = '≤' if lower == 0 else '<'
            right = '≤' if upper == 1/(1-row.rho) else '<'
            parts.append(f'{lower:.6f} {left} S {right} {upper:.6f}')
        region = ' or '.join(parts)
        if reachable and row.rho < .5:
            return f'({region}) ∩ C_rho^S'
        return region

    frame['product2_full_boundary_region_S'] = [
        region_text(row, reachable=False) for row in frame.itertuples()]
    frame['product2_reachable'] = frame.A_rho > frame.action_tolerance
    frame['product2_region_S'] = [
        region_text(row, reachable=True) for row in frame.itertuples()]
    frame.to_csv(directory / 'policy_summary.csv', index=False)
    lines = [
        '# Policy thresholds and observed long-run states', '',
        'Original state S; customer 1. Extrema use pre-choice states from purchases '
        f'{int(frame.burn_in.iloc[0])+1:,}–{int(frame.end_interaction.iloc[0]):,} with A.', '',
        'K = ceil(log(terminal_gap)/log(rho)) counts completed purchases. The code '
        'uses its limiting-policy approximation starting at purchase K+1. '
        'This is a numerical cutoff, not an exact stabilization time.', '',
        'The product-2 column shows the positive-advantage boundary region '
        'intersected with the limiting reachable set, using the simulation action '
        'tolerance (1e-6). For rho < 0.5, C_rho^S = {S : (1-rho)S belongs to C_rho} '
        'is the Cantor set expressed in original S units; the intersection notation '
        'excludes its gaps rather than treating the displayed bounds as a full '
        'interval. For rho >= 0.5, the entire boundary is reachable. None means '
        'no product-2 region was detected on the limiting reachable set; it does '
        'not rule out transient product-2 choices. Bounds are rounded to six '
        'decimals; ≤ includes a domain endpoint.', '',
        'The c2 column distinguishes calibrations. The CSV records all seller '
        'primitives and the source manifest for each row. Additional histories '
        'are stored separately under policy_summary_cases.', '',
        'A_rho estimates max(Q2-Q1) over the limiting reachable set, without '
        'subtracting the action tolerance. It uses the same boundary grid as the '
        'simulation. For rho < 0.5 the Cantor set is sampled at depth-16 cylinder '
        'endpoints; for rho >= 0.5 the full interval is used. These are numerical '
        'estimates, not certified signs of the exact maximum. The CSV records '
        'the maximizer and the normalized-state sampling radius. Reachable-region '
        'emptiness is evaluated with that same numerical maximum and the action '
        'tolerance. The CSV retains the unfiltered full-boundary intervals in '
        'product2_intervals_S and their text in product2_full_boundary_region_S. '
        'A full-boundary region wholly inside an unreachable gap is None in '
        'the table.', '',
        'The minimum and maximum are observed sample extrema, not theoretical '
        'support bounds or a proof of convergence. Seeds and the customer-1 '
        'prefix match the existing plots for the original cases. Additional cases '
        'use newly simulated prefixes with the same seed and customer-1 draw '
        'ordering as the p0 = 0.9 reference histories. terminal_gap = 1e-5.', '',
        '| p0 | c2 | rho | K | Product 2 region (reachable boundary) | $\\mathcal A_\\rho$ | Min S | Max S |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in frame.itertuples():
        region = row.product2_region_S.replace('C_rho^S', '$C_\\rho^S$')
        lines.append(f'| {row.p0:g} | {row.c2:g} | {row.rho:g} | {row.K} '
                     f'| {region} | {row.A_rho:.6g} | {row.S_min:.6g} | {row.S_max:.6g} |')
    (directory / 'policy_summary.md').write_text('\n'.join(lines)+'\n')


def run(root, *, interactions=200000, burn_in=100000):
    directory = root / 'tables'
    directory.mkdir(exist_ok=True)
    cache_path = directory / 'policy_summary_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = []
    # Keep comparison histories separate from the plot datasets, whose names
    # identify only p0 and rho and therefore cannot distinguish different costs.
    manifests = sorted((root / 'data').glob('*/records_manifest.json'))
    manifests += sorted((root / 'policy_summary_cases').glob('*/records_manifest.json'))
    if not manifests:
        raise ValueError(f'No saved experiments under {root}')
    for index, manifest in enumerate(manifests, 1):
        signature = hashlib.sha256(manifest.read_bytes() +
            (manifest.parent/'observations.npy').read_bytes() +
            f'v1:{interactions}:{burn_in}'.encode()).hexdigest()
        print(f'[{index}/{len(manifests)}] {manifest.parent.name}', flush=True)
        if signature not in cache:
            cache[signature] = summarize_pair(manifest, interactions=interactions, burn_in=burn_in)
            cache_path.write_text(json.dumps(cache, indent=2)+'\n')
        elif 'A_rho' not in cache[signature]:
            # Upgrade existing tables without repeating any simulated trajectories.
            meta = json.loads(manifest.read_text())
            solver = meta['policy_solver']
            solution = solve_boundary(Primitives.from_mapping(meta['seller']), meta['rho'],
                central_nodes=solver['nodes'], outer_nodes=2, half_widths=1e6,
                tolerance=solver['tolerance'], max_iterations=solver['max_iterations'])
            cache[signature].update(advantage_summary(meta['rho'], solution.coordinate, solution.advantage))
            cache_path.write_text(json.dumps(cache, indent=2)+'\n')
        # Older cached rows predate the explicit calibration columns. Preserve
        # their numerical results and recorded trajectories when upgrading them.
        meta = json.loads(manifest.read_text())
        primitives = Primitives.from_mapping(meta['seller']).as_dict()
        if any(cache[signature].get(key) != value for key, value in primitives.items()):
            cache[signature].update(primitives)
            cache_path.write_text(json.dumps(cache, indent=2)+'\n')
        rows.append({**cache[signature], 'source_manifest': str(manifest.relative_to(root))})
        write_table(pd.DataFrame(rows), directory)
    print(f'Tables: {directory}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/first_2000'))
    parser.add_argument('--interactions', type=int, default=200000)
    parser.add_argument('--burn-in', type=int, default=100000)
    args = parser.parse_args(argv)
    run(args.root, interactions=args.interactions, burn_in=args.burn_in)


if __name__ == '__main__':
    main()
