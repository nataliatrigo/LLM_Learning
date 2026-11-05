"""Actual-game stationary and Monte Carlo quality diagnostics."""

import numpy as np

from oblivious.oe_p0.code.diagnostics import demand_diagnostics
from oblivious.equilibrium_comparison.code.analysis import interval
from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
from oblivious.equilibrium_comparison.code.simulation import simulate_pair


def stationary_predictions(experiment, candidate):
    game, p = experiment.game, experiment.params
    nu, alpha = candidate["nu"], candidate["policy"]
    a, b = alpha[:, None], alpha[None, :]
    use = float(np.sum(nu*(game.D*a + game.D.T*b)))
    profit = .5*(p.R-p.c1-(p.c2-p.c1)*use)
    success = p.p1+(p.p2-p.p1)*use
    quality = experiment.quality_statistics(alpha, nu)
    if experiment.weighting == "purchase" and abs(success-quality["purchase_quality_B"]) > 1e-9:
        raise ArithmeticError("Symmetric customer-success and rival-sales quality disagree")
    return dict(product2_market=use, seller_profit=profit, success_rate=success, **quality)


def run_diagnostics(experiment, candidate, config, *, progress=None):
    pair = dict(grid=experiment.grid, params=experiment.params,
                profiles={"quality": dict(grid=experiment.grid, policy=candidate["policy"])})
    simulation = simulate_pair(
        pair, periods=config["periods"], burn_in=config["burn_in"],
        replications=config["replications"], seed=config["seed"], mode="grid",
        short_periods=0, initial=(.5, .5), assignments={"quality": ("quality", "quality")},
        progress=progress)
    raw = simulation["profiles"]["quality"]
    start, length = config["burn_in"], config["periods"]
    summarized = summarize_profile(experiment.grid, raw, start=start, length=length)
    take = slice(start, start+length)
    selected = raw["selected"][:, take]
    states = raw["before"][:, take, 1]
    indices = np.searchsorted(experiment.grid, states)
    q = experiment.params.p1+(experiment.params.p2-experiment.params.p1)*candidate["policy"][indices]
    sales = selected == 1
    counts = sales.sum(axis=1)
    purchase = np.divide((q*sales).sum(axis=1), counts,
                         out=np.full(len(counts), np.nan), where=counts > 0)
    observed = np.divide((raw["success"][:, take]*sales).sum(axis=1), counts,
                         out=np.full(len(counts), np.nan), where=counts > 0)
    for key, values in (("purchase_quality_B", purchase),
                        ("calendar_quality_B", q.mean(axis=1)),
                        ("realized_success_B", observed)):
        summarized["summary"][key] = interval(values)
        for row, value in zip(summarized["replications"], values):
            row[key] = float(value)
    demand = demand_diagnostics(experiment.game, candidate)
    nu = candidate["nu"]
    demand["arrays"]["joint_minus_independent"] = nu-np.outer(nu.sum(axis=1), nu.sum(axis=0))
    return dict(
        rho=experiment.params.rho, n=experiment.n, p0=candidate["p0"],
        candidate_verified=candidate["candidate_verified"],
        predicted=stationary_predictions(experiment, candidate), demand=demand,
        simulation=dict(**summarized, config=dict(config), metadata=simulation["metadata"],
                        interpretation="Actual two-seller TS on the saved grid; common initial (0.5,0.5). Monte Carlo intervals do not include initialization bias. Only rival sales enter purchase quality."))
