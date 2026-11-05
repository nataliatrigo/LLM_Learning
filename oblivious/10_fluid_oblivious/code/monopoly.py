"""Fluid monopoly benchmark for the fluid competition experiment.

The monopoly has one optimal own-state policy for each (p0,rho). That policy
can have more than one optimal stationary state; all verified states are
retained, while its realized path from the common initial state is stored only
once. No stochastic stationary distribution enters this benchmark.
"""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from forgetting.Code.model import Primitives, boundary_demand
from forgetting.Code.fluid_convergence.numerics import solve_fluid
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.model import _policy_description
from oblivious.fluid_oblivious.code.trajectories import simulate_pair


def _primitives(params, p0):
    return Primitives(p0=p0, p1=params.p1, p2=params.p2, c1=params.c1,
                      c2=params.c2, revenue=params.R, gamma=params.gamma)


def solve_monopoly(params, p0, n=4001):
    """Reuse the established mature-boundary fluid monopoly HJB directly."""
    response = solve_fluid(_primitives(params, p0), params.rho, n)
    policy = _policy_description(params, response)
    response.update(policy)
    return response


def stationary_states(params, p0, response):
    """Enumerate globally optimal stationary states of the optimized policy.

    In addition to pure endpoints p1,p2, all policy roots inside [p1,p2] are
    tested as holding mixtures. A repelling root alone never suffices: the
    held payoff must match the full optimized value in conditional units.
    """
    dp, dc = params.p2-params.p1, params.c2-params.c1
    lam, delta = -math.log(params.rho), -math.log(params.gamma)
    x = np.asarray(response["x"])
    h = float(np.max(np.diff(x)))
    k = dc/(lam*dp)
    states, rejected = [], []
    candidates = [(params.p1, 0., "pure_product1"),
                  (params.p2, 1., "pure_product2")]
    candidates.extend((root, (root-params.p1)/dp, "mixed")
                      for root in response["roots"] if params.p1 < root < params.p2)
    policy = _policy_description(params, response)
    for z, alpha, kind in candidates:
        demand = float(boundary_demand(z, params.rho, p0))
        reward = params.R-params.c1-dc*alpha
        hold = demand*reward/delta
        optimum = float(np.interp(z, x, response["value"]))
        gap = float(np.interp(z, x, response["gap"]))
        defect = delta*(optimum-hold)/max(demand,np.finfo(float).tiny)
        value_tol = max(2e-6,50*h*h*(1+abs(k))*(1+abs(optimum)))
        conditional_tol = max(2e-5,10*h*h*(1+dc+lam*abs(k)))
        action_valid = (gap <= 1e-8 if kind == "pure_product1" else
                        gap >= -1e-8 if kind == "pure_product2" else
                        abs(gap) <= max(2e-5,8*h*(1+abs(k))*lam))
        root_kind = None
        if kind == "mixed":
            index = int(np.argmin(abs(np.asarray(policy["roots"])-z)))
            root_kind = policy["root_kinds"][index]
        valid = bool(action_valid and abs(optimum-hold) <= value_tol
                     and abs(defect) <= conditional_tol and response["residual"] <= 1e-8)
        record = dict(x_star=float(z),alpha=float(alpha),kind=kind,verified=valid,
                      diagnostics=dict(holding_value=hold,optimized_value=optimum,
                        holding_value_gap=optimum-hold,conditional_holding_defect=defect,
                        hamiltonian_gap=gap,holding_value_tolerance=value_tol,
                        conditional_tolerance=conditional_tol,hjb_residual=response["residual"],
                        root_kind=root_kind,action_valid=bool(action_valid)))
        if valid:
            if not any(abs(z-old["x_star"]) < 1e-8 for old in states):
                states.append(record)
        else:
            rejected.append(record)
    states.sort(key=lambda row:row["x_star"])
    return states,rejected


def simulate_monopoly(params,p0,policy,*,times=None,initial=.5,bin_width=25,
                      rtol=1e-9,atol=1e-11,max_step=10.):
    """Scalar monopoly path using the shared event and counter integrator.

    The unused B coordinate has zero demand, sales, profit, and drift. This
    exact embedding avoids a second implementation of switching integration.
    Consumer product-2 rate equals the single seller's rate, without a factor2.
    """
    def selection(a,_b):
        own = float(boundary_demand(a,params.rho,p0))
        return own,0.,1-own
    result = simulate_pair(params,p0,policy,times=times,initial=(initial,initial),
                           bin_width=bin_width,rtol=rtol,atol=atol,max_step=max_step,
                           demand_function=selection,
                           response_b=dict(roots=[],products=[1]))
    result["diagnostics"]["selection"] = "Monopoly Thompson demand D_M(x_A); B is a zero-demand dummy used only by the shared integrator"
    result["diagnostics"]["model"] = "fluid_monopoly"
    result["diagnostics"]["initial_reputations"] = [float(initial)]
    result["diagnostics"]["seller_count"] = 1
    return result


def _refinement(params,p0,coarse,fine,coarse_states,fine_states,limits):
    rows=[]
    if fine_states and coarse_states:
        distances=np.asarray([[abs(a["x_star"]-b["x_star"]) for b in coarse_states] for a in fine_states])
        row_ids,col_ids=linear_sum_assignment(distances)
        matching=dict(zip(row_ids.tolist(),col_ids.tolist()))
    else:
        matching={}
    common=np.linspace(0,1,8001)
    disagreement=float(np.trapezoid((
        (np.interp(common,fine["x"],fine["gap"])>0) !=
        (np.interp(common,coarse["x"],coarse["gap"])>0)
    ).astype(float),common))
    for i,state in enumerate(fine_states):
        row=dict(p0=p0,rho=params.rho,coarse_n=len(coarse["x"]),fine_n=len(fine["x"]),
                 equilibrium_id=i,coarse_count=len(coarse_states),fine_count=len(fine_states),
                 policy_disagreement_measure=disagreement,passed=False)
        if i in matching:
            previous=coarse_states[matching[i]]
            share=float(boundary_demand(state["x_star"],params.rho,p0))
            old_share=float(boundary_demand(previous["x_star"],params.rho,p0))
            row.update(reputation_change=abs(state["x_star"]-previous["x_star"]),
                       alpha_change=abs(state["alpha"]-previous["alpha"]),
                       market_share_change=abs(share-old_share))
            row["passed"]=bool(len(coarse_states)==len(fine_states)
                and row["reputation_change"]<=limits["equilibrium_reputation_absolute"]
                and row["alpha_change"]<=limits["equilibrium_fraction_absolute"]
                and row["market_share_change"]<=limits["market_share_absolute"]
                and disagreement<=limits["policy_disagreement_measure"])
        rows.append(row)
    return rows


def build_monopoly_benchmark(config):
    """Return summary, policies, common-start paths, bins, and diagnostics."""
    summary,policies,paths,bins,case_records,refinements=[],[],[],[],[],[]
    settings=config["trajectories"]
    initial=float(settings["initial"][0])
    times=np.linspace(0,settings["horizon"],settings["samples"])
    for p0 in config["p0s"]:
        for rho in config["rhos"]:
            params=Parameters(**config["parameters"],rho=rho)
            grids=[]
            for n in config["grids"]:
                response=solve_monopoly(params,p0,n)
                states,rejected=stationary_states(params,p0,response)
                grids.append((response,states,rejected))
            local_refinements=[]
            for (coarse,cs,_),(fine,fs,_) in zip(grids,grids[1:]):
                local_refinements.extend(_refinement(params,p0,coarse,fine,cs,fs,config["refinement"]))
            refinements.extend(local_refinements)
            response,states,rejected=grids[-1]
            policy=_policy_description(params,response)
            # Add any globally validated repelling tie as an explicit holding
            # choice. Attractive roots are already in the common description.
            for state in states:
                if state["kind"]=="mixed" and not any(abs(point["x"]-state["x_star"])<1e-10 for point in policy["holding_points"]):
                    policy["holding_points"].append(dict(x=state["x_star"],alpha=state["alpha"]))
            trajectory=simulate_monopoly(params,p0,policy,times=times,initial=initial,
                                         bin_width=settings["bin_width"],rtol=settings["rtol"],
                                         atol=settings["atol"],max_step=settings["max_step"])
            for key,destination in (("by_time",paths),("binned",bins)):
                frame=trajectory[key].copy()
                frame["p0"],frame["rho"]=p0,rho
                frame["model"]="fluid_monopoly"
                frame["equilibrium_id"]=-1
                frame["trajectory_id"]=0
                destination.append(frame)
            final_state=float(trajectory["by_time"].x_a.iloc[-1])
            base=dict(p0=p0,rho=rho,n=len(response["x"]),model="fluid_monopoly",seller_count=1)
            stationarity=[]
            if not states:
                summary.append(dict(base,equilibrium_id=-1,verified=False,status="no_stationary_state_found",equilibrium_count=0))
            for i,state in enumerate(states):
                checks=[row for row in local_refinements if row["fine_n"]==len(response["x"]) and row["equilibrium_id"]==i]
                refined=len(checks)==1 and bool(checks[0]["passed"])
                verified=bool(state["verified"] and refined)
                z,alpha=state["x_star"],state["alpha"]
                share=float(boundary_demand(z,rho,p0))
                reward=params.R-params.c1-(params.c2-params.c1)*alpha
                diag=state["diagnostics"]
                row=dict(base,equilibrium_id=i,equilibrium_count=len(states),verified=verified,
                    core_verified=state["verified"],refined=refined,
                    status="verified" if verified else "refinement_failed",
                    x_star=z,alpha=alpha,equilibrium_kind=state["kind"],stationary_kind=state["kind"],
                    product2_conditional=alpha,product2_calendar=share*alpha,
                    customer_product2_calendar=share*alpha,sale_rate=share,outside_rate=1-share,
                    profit_calendar=share*reward,value_stationary=share*reward/(-math.log(params.gamma)),
                    perceived_value_at_equilibrium=diag["optimized_value"],hjb_residual=response["residual"],
                    value_initial=trajectory["diagnostics"]["value_initial_a"],
                    trajectory_terminal_x=final_state,trajectory_distance_to_equilibrium=abs(final_state-z),
                    common_start_attractor=bool(abs(final_state-z)<=config["refinement"]["equilibrium_reputation_absolute"]))
                row.update({"diagnostic_"+key:value for key,value in diag.items()})
                summary.append(row)
                for x,action in zip(response["x"],response["policy"]):
                    policies.append(dict(base,equilibrium_id=i,verified=verified,reputation=float(x),
                                         product2_probability=float(action),holding=False))
                for point in policy["holding_points"]:
                    policies.append(dict(base,equilibrium_id=i,verified=verified,reputation=point["x"],
                                         product2_probability=point["alpha"],holding=True))
                stationary=simulate_monopoly(params,p0,policy,times=[0.,1.,10.],initial=z,bin_width=5)
                err=float(np.max(abs(stationary["by_time"].x_a-z)))
                alpha_err=float(np.max(abs(stationary["by_time"].alpha_a-alpha)))
                if err>1e-9 or alpha_err>1e-8:
                    raise AssertionError("Monopoly stationary policy does not hold its reported state")
                stationarity.append(dict(equilibrium_id=i,reputation_error=err,alpha_error=alpha_err,passed=True))
            case_records.append(dict(p0=p0,rho=rho,policy=policy,
                grids=[dict(n=len(rr["x"]),hjb_residual=rr["residual"],roots=rr["roots"],
                            stationary_states=ss,rejected_candidates=rej) for rr,ss,rej in grids],
                stationary_policy_checks=stationarity,trajectory=trajectory["diagnostics"],
                trajectory_events=trajectory["events"]))
    source=Path(__file__)
    diagnostics=dict(model="fluid_monopoly",case_count=len(case_records),
        stationary_state_count=len([row for row in summary if row["verified"]]),
        common_start_path_count=len(paths),initial_reputation=initial,
        calibration="lambda=-log(rho), delta=-log(gamma), same original Beta posterior and outside option",
        source_sha256={str(source):sha256(source.read_bytes()).hexdigest()},
        stationary_vs_path="Every optimal stationary state is retained; there is one common-start monopoly policy/path per parameter case.",
        cases=case_records,grid_refinement=refinements)
    return (pd.DataFrame(summary),pd.DataFrame(policies),pd.concat(paths,ignore_index=True),
            pd.concat(bins,ignore_index=True),diagnostics)
