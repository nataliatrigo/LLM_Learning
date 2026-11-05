"""Create the explanatory report for the full-difference exploration."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/full_delta_map'
DATA = ROOT / 'data'


def main():
    raw = pd.read_csv(DATA / 'parameter_map.csv')
    agg = pd.read_csv(DATA / 'aggregated_map.csv')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert len(raw) == 54756 and len(agg) == 6084
    assert (agg.n_total == 9).all()
    assert ((raw.p1 > 0) & (raw.p1 < raw.p2) & (raw.p2 < 1)).all()
    assert ((raw.c1 > 0) & (raw.c1 < raw.c2) & (raw.c2 < 1)).all()
    assert np.allclose(raw.p2 - raw.p1, raw.delta_p)
    assert np.allclose(raw.c2 - raw.c1, raw.delta_c)
    counts = agg.groupby(['rho', 'status']).size().unstack(fill_value=0)
    counts.to_csv(DATA / 'aggregate_counts.csv')
    base = agg[np.isclose(agg.delta_p, .45) & np.isclose(agg.delta_c, .60)]
    base.to_csv(DATA / 'base_differences_summary.csv', index=False)
    lines = [
        '# Both differences span the unit interval; baseline levels also vary', '',
        'This corrects the earlier fixed-p1, fixed-c1 slice. Fixing p1=0.35 '
        'restricted Delta p<0.65, and fixing c1=0.05 restricted Delta c<0.95. '
        'Those limits were choices of that experiment, not limits inherent in the model.', '',
        'For any differences in (0,1), the feasible remaining levels satisfy', '',
        r'$$0<p_1<1-\Delta p,\qquad 0<c_1<1-\Delta c,$$', '',
        r'with $p_2=p_1+\Delta p$ and $c_2=c_1+\Delta c$.', '',
        'The continuation value depends on the individual probabilities and costs. '
        'Therefore identical differences need not imply identical optimal policies. '
        'A two-dimensional difference plot must either specify those remaining levels '
        'or explicitly summarize their variation.', '',
        '## Corrected exploration', '',
        'Fixed: R=1, gamma=0.98, p0=0.50. Compared rho=0.70,0.90,0.95,0.99. '
        'Both differences take 39 equally spaced values 0.025,0.050,...,0.975. '
        'The endpoints 0 and 1 are outside the open domain and are not solved.', '',
        'At every pair of differences, p1 takes 10%, 50%, and 90% of its feasible '
        'interval (0,1-Delta p); independently c1 takes 10%, 50%, and 90% of '
        'its feasible interval (0,1-Delta c). Thus nine feasible probability/cost '
        'combinations are solved per displayed cell: 54,756 cases overall.', '',
        'The optional p0<=p2 restriction is not imposed in this corrected exploration. '
        'All cases satisfy the model constraints 0<p1<p2<1 and 0<c1<c2<R=1.', '',
        '## Reading the plots', '',
        '- [Count map](figures/01_two_threshold_counts.png) '
        '([PDF](figures/01_two_threshold_counts.pdf)): the number of the nine tested '
        'combinations giving the strict interior product-2 interval. This is a count, '
        'not a probability. A count of nine does not assert the result for every '
        'unexamined feasible baseline level.',
        '- [Separate choices at rho=0.95](figures/02_baseline_fraction_maps.png) '
        '([PDF](figures/02_baseline_fraction_maps.pdf)): each panel identifies '
        'the chosen fraction of the feasible p1 and c1 intervals. These levels vary '
        'with the differences, so these are not panels holding p1 or c1 constant.', '',
        'Both figures concern the limiting boundary x_s+x_f=1. The sequence '
        '1 -> 2 -> 1 describes the optimal products as normalized state x increases.', '',
        '## Identical differences can have different policies', '',
        'Take rho=0.95, Delta p=0.45, Delta c=0.60, p1=0.055, and p2=0.505. '
        'With c1=0.04,c2=0.64, the computed product-2 region is approximately '
        '(0.291874,0.540953). With c1=0.20,c2=0.80, product 1 is strictly optimal '
        'throughout the boundary (maximum advantage approximately -0.019709). '
        'These examples also satisfy p0<=p2. They were independently checked '
        'at 8,001 and 16,001 state nodes; details are in same_differences_examples.json.', '',
        '| rho | Two-threshold choices at the base differences |',
        '|---:|---:|',
    ]
    for row in base.itertuples():
        lines.append(f'| {row.rho:.2f} | {row.n_two} of {row.n_total} |')
    lines += [
        '', 'These rows refer to the base differences, not the full original calibration: '
        'the nine feasible baseline choices generally do not include p1=0.35,c1=0.05.', '',
        '## Numerical checks', '',
        'All 54,756 cases were compared at 2,001 and 4,001 state nodes. '
        'The continuation advantage is evaluated at every transformed interpolation knot. '
        'Sensitive cases are further refined to 8,001 nodes. The accelerated solver '
        'was independently checked against forgetting.Code.model.solve_boundary.', '',
        f'Additional 8,001-node cases: {manifest["extra_8001_points"]}. '
        f'Unresolved final cases: {manifest["uncertain_points"]}. '
        f'Maximum final threshold movement: {manifest["max_final_root_movement"]:.3g}. '
        f'Maximum final grid Bellman residual: {manifest["max_bellman_residual"]:.3g}.', '',
        'These are finite numerical comparisons, not an analytical threshold proof '
        'or a certified discretization-error bound. The nine baseline choices are '
        'representative samples rather than an exhaustive search.', '',
        '## Reproduce', '', 'From the repository root:', '', '```bash',
        '.venv/bin/python -B forgetting/Code/full_delta_map/sweep.py',
        '.venv/bin/python -B forgetting/Code/full_delta_map/plot_maps.py',
        '.venv/bin/python -B forgetting/Code/full_delta_map/summarize.py',
        '```', '',
        'The sweep reuses the temporary C++ solver from the prior threshold_parameter_map '
        'experiment, requires g++, and leaves the production model unchanged. '
        'Full scenarios are in data/parameter_map.csv; counts are in data/aggregated_map.csv; '
        'solver and sampling details are in data/manifest.json.', '',
    ]
    (ROOT / 'README.md').write_text('\n'.join(lines))
    print(counts.to_string())
    print(base[['rho', 'n_two', 'n_uncertain', 'patterns']].to_string(index=False))


if __name__ == '__main__':
    main()
