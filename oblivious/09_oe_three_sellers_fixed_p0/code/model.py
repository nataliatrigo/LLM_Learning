"""Symmetric own-reputation OE with one, two, or three strategic sellers.

Marginal beliefs preserve the joint distribution of *all* rivals. Conditional
beliefs additionally condition that distribution on the focal reputation. The
permutation quotient is exact for symmetric policies, not an independence
approximation. Search derivatives differentiate the actual finite joint chain.
"""

from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import root
from scipy.special import expit
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from .kernel import FixedOutsideKernel
from .stationary import jump_system, symmetric_lumping, checked_symmetric_stationary, checked_stationary


def regret(alpha, advantage):
    return float(np.max(np.maximum(advantage, 0) - alpha * advantage))


class Experiment:
    def __init__(self, n=21, sellers=3, closure="marginal", p0=.5,
                 params=Parameters(rho=.6), order=256, tolerance=1e-7):
        if int(n) != n or n < 3 or n % 2 != 1:
            raise ValueError("Use an odd grid size >=3 so initial reputation .5 is a node.")
        if sellers not in (1, 2, 3) or closure not in ("monopoly", "marginal", "conditional"):
            raise ValueError("Require 1..3 sellers and a supported belief closure.")
        if (sellers == 1) != (closure == "monopoly") or not 0 <= p0 < 1:
            raise ValueError("Monopoly means one seller; the stationary experiment requires p0<1.")
        self.n, self.sellers, self.closure = int(n), int(sellers), closure
        self.params, self.p0, self.order, self.tolerance = params, float(p0), int(order), tolerance
        self.grid = np.linspace(0, 1, self.n)
        self.game = FixedOutsideKernel(self.grid, params, p0, sellers, order)
        self.S, self.F = self.game.success, self.game.failure
        self.calls, self.last_alpha, self.last_result = 0, None, None
        if sellers > 1:
            jump, rates = self.game.jump(np.zeros(self.n))
            lump = symmetric_lumping(jump, rates, self.game.shape)
            self.representatives, self.inverse, self.counts = (
                lump['representatives'], lump['inverse'], lump['counts'])
            size, reduced_size = self.game.size, len(self.representatives)
            self.membership = csr_matrix((np.ones(size), (np.arange(size), self.inverse)),
                                         shape=(size, reduced_size))
            coordinates = np.indices(self.game.shape).reshape(sellers, -1)
            shares = 1 / self.counts[self.inverse]
            self.own_marginal = csr_matrix((shares, (coordinates[0], self.inverse)),
                                           shape=(self.n, reduced_size))
            self.sale_marginal = csr_matrix((shares * self.game.demand[0].ravel(),
                                            (coordinates[0], self.inverse)),
                                           shape=(self.n, reduced_size))
            rival_ids = np.ravel_multi_index(coordinates[1:], self.game.shape[1:])
            rival_marginal = csr_matrix((shares, (rival_ids, self.inverse)),
                                       shape=(self.n ** (sellers-1), reduced_size))
            self.marginal_demand = np.asarray(
                rival_marginal.T @ self.game.demand[0].reshape(self.n, -1).T).T
            self.derivatives = self._transition_derivatives()
            self.rate_derivatives = np.column_stack(
                [np.asarray(delta.sum(axis=1)).ravel() for delta in self.derivatives])

    def configuration(self):
        return dict(n=self.n, sellers=self.sellers, closure=self.closure, p0=self.p0,
                    parameters=asdict(self.params), order=self.order, tolerance=self.tolerance,
                    strategy="Symmetric mixed own-reputation policy; action after selection",
                    marginal_belief="Joint calendar-time distribution of all rivals, independent of focal state",
                    conditional_belief="Actual joint rival distribution conditional on own calendar-time state",
                    outside_event="No seller sells; all reputations freeze",
                    forgetting_clock="Own observations only", initial_reputations=.5,
                    scope="Finite-grid oblivious equilibrium; not full-state MPE or uniqueness proof")

    def _transition_derivatives(self):
        """Exact d(off-diagonal quotient flows)/d alpha[k]."""
        game, dp = self.game, self.params.p2-self.params.p1
        derivatives = []
        for k in range(self.n):
            rr, cc, ww = [], [], []
            for seller in range(self.sellers):
                own = game._own_indices[seller]
                rows = game._rows[own == k]
                selection = game.demand[seller].ravel()[rows]
                for operator, sign in ((self.S, 1), (self.F, -1)):
                    start, end = operator.indptr[k:k+2]
                    for dest, weight in zip(operator.indices[start:end], operator.data[start:end]):
                        columns = rows+(dest-k)*game._strides[seller]
                        rr.append(rows); cc.append(columns)
                        ww.append(sign*dp*weight*selection)
            full = csr_matrix((np.concatenate(ww), (np.concatenate(rr), np.concatenate(cc))),
                              shape=(game.size, game.size))
            reduced = (full[self.representatives] @ self.membership).tocsr()
            reduced.setdiag(0); reduced.eliminate_zeros()
            derivatives.append(reduced)
        return derivatives

    def belief_demand(self, joint):
        joint = np.asarray(joint).reshape(self.game.shape)
        own = joint.sum(axis=tuple(range(1, self.sellers)))
        rivals = joint.sum(axis=0)
        marginal = self.game.demand[0].reshape(self.n, -1) @ rivals.ravel()
        if self.closure == "conditional":
            selected = (joint*self.game.demand[0]).sum(axis=tuple(range(1, self.sellers)))
            positive = own > 0
            marginal[positive] = selected[positive]/own[positive]
        return marginal

    def calculate(self, alpha):
        """Exploratory sparse evaluation with analytic equilibrium Jacobian."""
        alpha = np.clip(np.asarray(alpha, dtype=float), 0, 1)
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        self.calls += 1
        full = self.game.transition(alpha)
        reduced = (full[self.representatives] @ self.membership).tocsr()
        jump, rates = jump_system(reduced)
        size = len(rates)
        anchor = size//2
        linear = (jump.T-eye(size, format="csr")).tolil()
        linear[anchor, :] = np.ones(size)
        lu = splu(linear.tocsc())
        rhs = np.zeros(size); rhs[anchor] = 1
        w = lu.solve(rhs)
        calendar = w/rates
        calendar /= calendar.sum()
        if np.any(~np.isfinite(calendar)) or np.any(calendar <= 0):
            # A signed search law is never used to condition on a rare state.
            checked = checked_stationary(jump, rates)
            calendar = checked['law']
            w = calendar*rates; w /= w.sum()
        mu = np.asarray(self.own_marginal @ calendar).ravel()
        demand = self.marginal_demand @ calendar
        if self.closure == 'conditional':
            demand = np.asarray(self.sale_marginal @ calendar).ravel()/mu
        if np.any(demand < -1e-12) or np.any(demand > 1+1e-12):
            raise ArithmeticError("Invalid exploratory belief demand")
        demand = np.clip(demand, 0, 1)
        evaluation = evaluate_policy(self.grid, demand, alpha, self.params)
        value, advantage = evaluation['value'], evaluation['advantage']
        derivative_rhs = np.empty((size, self.n))
        relative_dr = self.rate_derivatives/rates[:, None]
        for k, delta in enumerate(self.derivatives):
            derivative_rhs[:, k] = -(delta.T@(w/rates)-jump.T@(w*relative_dr[:, k]))
        derivative_rhs[anchor, :] = 0
        dw = lu.solve(derivative_rhs)
        unnormalized = w/rates
        dc_raw = dw/rates[:, None]-unnormalized[:, None]*relative_dr
        dc = dc_raw/unnormalized.sum()-calendar[:, None]*(dc_raw.sum(axis=0)/unnormalized.sum())
        dd = self.marginal_demand @ dc
        if self.closure == 'conditional':
            dd = (self.sale_marginal@dc-demand[:, None]*(self.own_marginal@dc))/mu[:, None]
        p = self.params
        selected = diags(p.p1+(p.p2-p.p1)*alpha)@self.S + diags(1-p.p1-(p.p2-p.p1)*alpha)@self.F
        own = diags(1-demand)+diags(demand)@selected
        own_lu = splu((eye(self.n, format='csr')-p.gamma*own).tocsc())
        flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected@value-value)
        dv = own_lu.solve(flow[:, None]*dd+np.diag(demand*advantage))
        da = p.gamma*(p.p2-p.p1)*((self.S-self.F)@dv)
        result = dict(policy=alpha, advantage=advantage, jacobian=da, demand=demand,
                      value=value, reduced_law=calendar, regret=regret(alpha, advantage))
        self.last_alpha, self.last_result = alpha.copy(), result
        return result

    def certify(self, alpha):
        """Fresh independent joint law, rowwise beliefs, DP and actual payoffs."""
        started = time.monotonic()
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError("Invalid candidate policy")
        jump, rates = self.game.jump(alpha)
        if self.sellers == 1:
            stationary = checked_stationary(jump, rates)
            joint = stationary['law'].reshape(self.game.shape)
            demand = self.game.demand[0].copy()
        else:
            stationary = checked_symmetric_stationary(jump, rates, self.game.shape)
            joint = stationary['law'].reshape(self.game.shape)
            demand = self.belief_demand(joint)
        evaluated = evaluate_policy(self.grid, demand, alpha, self.params)
        best = solve_dp(self.grid, demand, self.params)
        conditional_regret = regret(alpha, evaluated['advantage'])
        value_difference = float(np.max(abs(best['value']-evaluated['value'])))
        calendar = self.game.transition(alpha)
        p = self.params
        reward = self.game.demand[0]*(p.R-p.c1-(p.c2-p.c1)*alpha.reshape((self.n,)+(1,)*(self.sellers-1)))
        actual_value = np.zeros(self.game.size)
        flat_reward = reward.ravel()
        for value_iterations in range(1, 10001):
            updated = flat_reward+p.gamma*(calendar@actual_value)
            value_residual = float(np.max(abs(updated-actual_value)))
            actual_value = updated
            if value_residual <= 1e-11:
                break
        else:
            raise ArithmeticError("Actual discounted payoff evaluation exceeded its budget")
        actual_value = actual_value.reshape(self.game.shape)
        event = purchase_chain_statistics(self.grid, alpha, p)
        sales = (joint*self.game.demand[0]).reshape(self.n, -1).sum(axis=1)
        sale_rate = float(sales.sum())
        sale_law = sales/sale_rate
        sale_error = float(np.sum(abs(sale_law-event['sale_stationary'])))
        use = float(sale_law@alpha)
        outside = float(np.sum(joint*self.game.outside))
        market_error = abs(outside+self.sellers*sale_rate-1)
        sale_symmetry = max(abs(float(np.sum(joint*self.game.demand[i]))-sale_rate)
                            for i in range(self.sellers))
        diagnostics = dict(conditional_regret=conditional_regret,
                           bellman_residual=evaluated['residual'], optimal_value_difference=value_difference,
                           actual_value_residual=value_residual, actual_value_error_bound=value_residual/(1-p.gamma),
                           actual_value_iterations=value_iterations, sale_law_l1=sale_error,
                           market_partition_error=market_error, seller_sale_rate_symmetry=sale_symmetry,
                           stationary=stationary['summary'], sale_time=event,
                           certificate_seconds=time.monotonic()-started)
        # Positive rare rows must have consistent relative conditional beliefs.
        if self.sellers > 1 and self.closure == 'conditional':
            # Independent checked-stationary candidates are compared globally;
            # perform another reversed-coordinate quotient solve and compare
            # every positive conditional row, not only total joint L1.
            permutation = np.arange(self.game.size)[::-1]
            alternate = checked_symmetric_stationary(jump[permutation, :][:, permutation],
                                                      rates[permutation], self.game.shape)
            independent = alternate['law'][permutation].reshape(self.game.shape)
            own = joint.reshape(self.n, -1).sum(axis=1)
            own_alt = independent.reshape(self.n, -1).sum(axis=1)
            if np.any(own <= 0) or np.any(own_alt <= 0):
                raise ArithmeticError("Positive joint law lost an own-state conditional row")
            row_l1 = np.sum(abs(joint.reshape(self.n, -1)/own[:, None]
                                -independent.reshape(self.n, -1)/own_alt[:, None]), axis=1)
            diagnostics['conditional_row_stability_l1'] = float(row_l1.max())
            if row_l1.max() > self.tolerance:
                raise ArithmeticError("Conditional laws are unstable on a positive-mass row")
        verified = bool(conditional_regret <= self.tolerance and evaluated['residual'] <= 1e-10
                        and value_difference <= 1e-5 and sale_error <= 1e-7
                        and market_error <= 1e-10 and sale_symmetry <= 1e-10)
        summary = dict(p0=self.p0, rho=p.rho, n=self.n, sellers=self.sellers, closure=self.closure,
                       verified=verified, product2_conditional=use, sale_rate=sale_rate,
                       outside_rate=outside, product2_calendar=sale_rate*use,
                       profit_calendar=sale_rate*(p.R-p.c1-(p.c2-p.c1)*use),
                       value_initial=float(actual_value[(self.n//2,)*self.sellers]),
                       perceived_value_initial=float(evaluated['value'][self.n//2]))
        return dict(verified=verified, summary=summary, policy=alpha, grid=self.grid,
                    joint=joint, value=evaluated['value'], actual_value=actual_value,
                    demand=demand, advantage=evaluated['advantage'], diagnostics=diagnostics)

    def solve(self, initial_policy=None, max_evaluations=500, progress=None):
        if self.sellers == 1:
            policy = solve_dp(self.grid, self.game.demand[0], self.params)['policy']
            return self.certify(policy)
        initial = np.full(self.n, .5) if initial_policy is None else np.clip(np.asarray(initial_policy), 0, 1)
        if initial.shape != (self.n,):
            raise ValueError("Initial policy must match the grid")
        alpha, history, seen = initial.copy(), [], set()
        best = None
        # Pure improvements cheaply solve the common threshold cases.
        for _ in range(10):
            data = self.calculate(alpha)
            history.append(dict(stage='pure improvement', regret=data['regret'], calls=self.calls))
            if progress:
                progress(history[-1])
            if best is None or data['regret'] < best['regret']:
                best = dict(data)
            if data['regret'] <= self.tolerance:
                result = self.certify(alpha)
                if result['verified']:
                    result.update(history=history, search_calls=self.calls)
                    return result
            key = (data['advantage'] >= 0).tobytes()
            if key in seen:
                break
            seen.add(key)
            alpha = (data['advantage'] >= 0).astype(float)
        # A natural-map equation permits genuine interior policies on cycles.
        schedules = [[0.], [.1, .03, .01, .003, .001, .0003, .0001, 0.]]
        for schedule in schedules:
            alpha = best['policy'].copy()
            for temperature in schedule:
                if self.calls >= max_evaluations:
                    raise RuntimeError(f"Equilibrium search budget exhausted: best regret={best['regret']:g}")
                def equations(raw):
                    nonlocal best
                    data = self.calculate(raw)
                    if data['regret'] < best['regret']:
                        best = dict(data)
                    target = expit(data['advantage']/temperature) if temperature else np.clip(raw+data['advantage'], 0, 1)
                    return raw-target
                def jacobian(raw):
                    data = self.calculate(raw)
                    da = data['jacobian']*((raw >= 0)&(raw <= 1))[None, :]
                    if temperature:
                        target = expit(data['advantage']/temperature)
                        return np.eye(self.n)-(target*(1-target)/temperature)[:, None]*da
                    inside = (raw+data['advantage'] > 0)&(raw+data['advantage'] < 1)
                    return np.eye(self.n)-inside[:, None]*(np.eye(self.n)+da)
                solved = root(equations, alpha, jac=jacobian, method='hybr',
                              options=dict(xtol=1e-10, maxfev=min(120, max_evaluations-self.calls), factor=.3))
                alpha = np.clip(solved.x, 0, 1)
                alpha[alpha < 1e-9] = 0; alpha[alpha > 1-1e-9] = 1
                data = self.calculate(alpha)
                history.append(dict(stage='natural-map continuation', temperature=temperature,
                                    regret=data['regret'], calls=self.calls, success=bool(solved.success)))
                if progress:
                    progress(history[-1])
                if data['regret'] <= self.tolerance:
                    result = self.certify(alpha)
                    if result['verified']:
                        result.update(history=history, search_calls=self.calls)
                        return result
        raise RuntimeError(f"No certified equilibrium found; best regret={best['regret']:g}")
