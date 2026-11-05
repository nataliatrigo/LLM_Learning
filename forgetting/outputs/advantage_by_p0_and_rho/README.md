# Product-2 advantage across p0 and rho

The two combined figures group the same results by p0 and by rho. All combined panels share the same x and y scales. Each p0 folder contains an eight-panel figure, an overlay, and CSV curve data. All figures are available as PNG and vector PDF.

The horizontal axis follows the long-run boundary z=(x,1-x), with x=(1-rho)*S. Positive advantage favors product 2. For rho<0.5, only a Cantor subset of the plotted boundary is limiting-reachable; individual panels mark its endpoints. summary.csv reports numerical reachable-set maxima and full-boundary zero crossings.

Calibration and convergence checks are recorded in manifest.json. The optimal value is recomputed for every parameter pair.

Reproduce from the repository root:

```bash
uv run python -B forgetting/Code/visualizations/plot_advantage_by_p0_rho.py
```
