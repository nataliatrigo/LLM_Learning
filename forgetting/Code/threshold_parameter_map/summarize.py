"""Summarize the saved numerical exploration without repeating the solves."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map'
DATA = ROOT / 'data'


def main():
    frame = pd.read_csv(DATA / 'parameter_map.csv')
    frontier = pd.read_csv(DATA / 'fixed_policy_frontier.csv')
    baseline = pd.read_csv(DATA / 'baseline_summary.csv')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert len(frame) == 28200
    assert not frame.duplicated(['rho', 'delta_p', 'delta_c']).any()
    assert set(frame.rho) == {.6, .7, .8, .9, .95, .99}
    assert np.all(frame.delta_c < .95) and np.all(frame.delta_p < .65)
    counts = frame.groupby(['rho', 'regime']).size().unstack(fill_value=0)
    counts.to_csv(DATA / 'regime_counts.csv')
    checked = frame.merge(frontier[['rho', 'K_rho']], on='rho', validate='many_to_one')
    checked['fixed_policy_existence_margin'] = checked.K_rho * checked.delta_p - checked.delta_c
    # This is a numerical cross-check, not a continuum error certificate.
    away = checked.fixed_policy_existence_margin.abs() > 1e-4
    mismatch = away & ((checked.fixed_policy_existence_margin > 0) != (checked.a_max > 0))
    checked.loc[mismatch].to_csv(DATA / 'existence_check_mismatches.csv', index=False)
    if mismatch.any():
        raise AssertionError('Investigate existence-boundary disagreement away from the numerical guard band.')
    box = frame[np.isclose(frame.rho, .95)
                & frame.delta_p.between(.30, .60)
                & frame.delta_c.between(.20, .80)]
    metrics = dict(
        parameter_points=len(frame),
        existence_crosscheck_pairs=int(away.sum()),
        existence_crosscheck_mismatches=int(mismatch.sum()),
        disconnected_detected=int((frame.detected_regime == 'multiple').sum()),
        unresolved=int((frame.regime == 'uncertain').sum()),
        candidate_rectangle_points=len(box),
        candidate_rectangle_all_two=bool((box.regime == 'two').all()),
        candidate_rectangle_minimum_peak_advantage=float(box.a_max.min()),
        candidate_rectangle_largest_endpoint_advantage=float(max(box.a0.max(), box.a1.max())),
    )
    (DATA / 'research_summary.json').write_text(json.dumps(metrics, indent=2) + '\n')

    lines = [
        '# Exploring the two-threshold policy through cost and quality differences', '',
        'We solve the exact discrete Beta-demand Bellman model on numerical state grids. '
        'This exploration concerns the limiting boundary x_s+x_f=1, not all transient states.', '',
        '## What was held fixed', '',
        'R=1, gamma=0.98, c1=0.05, p1=0.35, p0=0.50. We set '
        'c2=c1+Delta c and p2=p1+Delta p. Fixing these anchors matters: the differences '
        'alone do not specify the model.', '',
        'The scan uses Delta p=0.15,0.16,...,0.64 and Delta c=0.01,0.02,...,0.94, '
        'at rho=0.60,0.70,0.80,0.90,0.95,0.99: 28,200 parameter pairs. '
        'All satisfy p0<=p2, p2<1, and c2<R. The baseline differences are (Delta p,Delta c)=(0.45,0.60).', '',
        '## Figures', '',
        '- [Policy patterns across both differences](figures/01_policy_phase_maps.png) '
        '([PDF](figures/01_policy_phase_maps.pdf)). Each tile is a sampled parameter pair; '
        'the product sequence is read as state x increases.',
        '- [Locations of the two thresholds at rho=0.95](figures/02_threshold_locations.png) '
        '([PDF](figures/02_threshold_locations.pdf)). Only two-threshold points are colored.',
        '- [Slices through the base differences](figures/03_threshold_slices.png) '
        '([PDF](figures/03_threshold_slices.pdf)). Shading is the product-2 state interval.',
        '- [Baseline advantage across rho](figures/04_baseline_advantage_curves.png) '
        '([PDF](figures/04_baseline_advantage_curves.pdf)).', '',
        '## Main findings', '',
        'Substantial cost and success differences alone do not select the two-threshold pattern. '
        'At rho=0.90, the baseline differences produce a middle interval, but increasing both '
        'proportionally to (Delta p,Delta c)=(0.60,0.80) makes product 2 optimal at x=0, '
        'leaving only an upper threshold. Both pairs have Delta c/Delta p=4/3.', '',
        '| rho | Delta p | Delta c | Product-2 interval, approximate |',
        '|---:|---:|---:|---|',
        '| 0.90 | 0.45 | 0.60 | (0.004631, 0.617361) |',
        '| 0.90 | 0.60 | 0.80 | [0, 0.598516) |',
        '| 0.95 | 0.15 | 0.20 | (0.239681, 0.669306) |',
        '| 0.95 | 0.45 | 0.60 | (0.207835, 0.598284) |',
        '| 0.95 | 0.60 | 0.80 | (0.197868, 0.587995) |', '',
        'These five illustrative cases were independently checked using the original Python '
        'solver at 8,001 and 16,001 state nodes; threshold movements were below 4e-7. '
        'Their diagnostics are in independent_checks.json.', '',
    ]
    if metrics['candidate_rectangle_all_two']:
        lines += [
            'A concrete candidate region for analytical study is rho=0.95, '
            '0.30<=Delta p<=0.60 and 0.20<=Delta c<=0.80, with the fixed anchors above. '
            f'All {len(box):,} sampled points in this rectangle exhibit two thresholds. '
            f'The largest sampled endpoint advantage is {metrics["candidate_rectangle_largest_endpoint_advantage"]:.6f}; '
            f'the smallest sampled maximum advantage is {metrics["candidate_rectangle_minimum_peak_advantage"]:.6f}. '
            'These are sampled margins, not bounds holding throughout the continuous parameter rectangle.', '',
        ]
    lines += [
        f'No disconnected product-2 region was detected at {len(frame)-metrics["disconnected_detected"]:,} '
        f'of the {len(frame):,} sampled pairs; disconnected detections: {metrics["disconnected_detected"]}. '
        f'Unresolved points near numerical transitions: {metrics["unresolved"]}. '
        'Absence of detections is not a connectedness theorem.', '',
        '## The analytical role of the fixed-policy value', '',
        'With c1,p1,p0,rho fixed, U_rho^1 does not depend on Delta c or Delta p. '
        'The proven positivity equivalence therefore gives', '',
        r'$$\mathcal A_\rho>0\quad\Longleftrightarrow\quad '
        r'\Delta c<\Delta p\,\max_{x\in[0,1]}G_\rho^1(x).$$', '',
        'Consequently the boundary between existence and absence of a strict product-2 region '
        'is a straight line in these parameter coordinates. The dashed lines use numerically '
        'computed slopes from U_rho^1, checked at 8,001 and 16,001 nodes. The exact linear '
        'form follows analytically; the displayed slopes are numerical.', '',
        f'The predicted existence classification agreed with the optimal-policy scan at all '
        f'{int(away.sum()):,} checked pairs outside a 1e-4 comparison-margin guard band.', '',
        'This criterion does not distinguish an interior interval from one touching x=0 or x=1. '
        'For two thresholds, one also needs negative endpoint advantages and connectedness '
        'of the positive region. The proposed parameter box is a numerical target for proving '
        'those properties, not an already proved sufficient condition.', '',
        '## Numerical checks and limits', '',
        'Every parameter pair was solved at 2,001 and 4,001 state nodes. Classification evaluates '
        'all transformed interpolation knots of the continuation advantage, so it records '
        'every component of the piecewise-linear approximation. Sensitive points are refined '
        'to 8,001 nodes. This does not supply a certified error bound for the original continuous-state problem.', '',
        f'Maximum final grid Bellman residual: {manifest["max_bellman_residual"]:.3g}. '
        f'Extra 8,001-node cases: {manifest["extra_8001_points"]}. '
        f'Final maximum threshold movement on the refinement check: {manifest["max_final_root_movement"]:.3g}.', '',
        'An accelerated Gauss-Seidel solver solves the same interpolated Bellman equation and '
        'was checked against forgetting.Code.model.solve_boundary at eleven representative parameter pairs. '
        'The scan uses fixed-policy shortcuts only when their grid Bellman inequalities hold. '
        'No production model files were changed.', '',
        '## Reproduce', '',
        'From the repository root:', '',
        '```bash',
        '.venv/bin/python -B forgetting/Code/threshold_parameter_map/sweep.py',
        '.venv/bin/python -B forgetting/Code/threshold_parameter_map/plot_maps.py',
        '.venv/bin/python -B forgetting/Code/threshold_parameter_map/summarize.py',
        '```', '',
        'The sweep requires g++ to compile a temporary solver library, along with the project Python dependencies. '
        'The separate fixed-policy frontier and independent checks are saved in the data directory '
        'and independent_checks.json, respectively.', '',
        'Full results: data/parameter_map.csv. Grid and solver diagnostics: data/manifest.json. '
        'Category counts: data/regime_counts.csv. The state thresholds are in normalized x units.', '',
    ]
    (ROOT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(metrics, indent=2))
    print(counts.to_string())


if __name__ == '__main__':
    main()
