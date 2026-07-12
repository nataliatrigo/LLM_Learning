# BinaryBelief-Logit

1-D "hidden product choice" reputation model with **logit demand**
microfounded by heterogeneous outside options, and **stochastic per-period
engagement**: each period one user engages w.p. `D(pi)`; if she does not,
the public belief is frozen and nothing is earned or paid.

- State: log-odds `ell = logit(pi)`; Bayes updates are additive
  (`+dS` on success, `+dF` on failure), only when a transaction occurs.
- Demand: `D(pi) = sigmoid(beta * (p1 + pi*dp - mu))` — the fraction of
  Logistic(mu, 1/beta) outside options below the posterior mean quality.
- Bellman: `V = gamma*(1-D)*V + D*max_x{ R - c_x + gamma*E_x[V'] }`,
  iterated in the fixed-point form `V = D/(1 - gamma*(1-D)) * M`.
- Policy: invest (`x*=2`) iff `g(ell) = gamma*[V(ell+dS)-V(ell+dF)] >= dc/dp`
  (constant threshold).

## Layout

- `config.py` — all parameters (model, grid, tolerances, seeds).
- `src/model.py` — solver (value iteration), band extraction, simulator.
- `run_all.py` — entry point; runs experiments E0-E6.
- `outputs/plots/`, `outputs/tables/`, `outputs/SUMMARY.md` — generated.

## Usage

```bash
python run_all.py                      # everything at defaults
python run_all.py --mu 0.6 --beta 16   # override any Config field
python run_all.py --only E1 E2         # subset of experiments
```

Deterministic given `Config.seed`; matplotlib only; ~3 s for a full run.

## Experiments

- **E0** demand shapes vs beta (microfoundation sanity check)
- **E1** baseline solve: V, g vs dc/dp, policy, band
- **E2** sweep in mu — band location vs the greedy reference pibar(mu)
- **E3** sweep in beta — greedy limit and critical beta where the band dies
- **E4** sweep in dc — band shrinks in cost; critical dc
- **E5** 200 simulated paths, true frozen-belief mechanism, escape times
- **E6** Thompson demand `D(pi)=pi` comparison
