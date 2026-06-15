# LLM Learning

This repo contains a single-file Python simulation of a dynamic reputation
model with two competing LLM providers.

## Run

```bash
uv run python main.py
```

The default run uses `T = 300` periods and `n_rep = 500` Monte Carlo
replications. It saves plots and the summary table in `outputs/`.

The learning-check plots compare posterior beliefs against the true delivered
utility and the true probability that the realized signal is above zero.

For a quick smoke test:

```bash
uv run python main.py --T 30 --n-rep 20 --outputs-dir /tmp/llm_learning_smoke
```

## Outputs

- `outputs/summary_table.csv`
- `outputs/product2_usage_heuristic.png`
- `outputs/product2_usage_onestep.png`
- `outputs/market_share_A_heuristic.png`
- `outputs/market_share_A_onestep.png`
- `outputs/posterior_means_heuristic.png`
- `outputs/posterior_means_onestep.png`
- `outputs/success_probability_heuristic.png`
- `outputs/success_probability_onestep.png`
- `outputs/user_welfare_heuristic.png`
- `outputs/user_welfare_onestep.png`
- `outputs/cumulative_profits_heuristic.png`
- `outputs/cumulative_profits_onestep.png`
