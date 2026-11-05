"""Exact mature-count oblivious equilibria with uniform random replacement.

Strategies depend on the seller's own retained success count. The actual
calendar law retains every rival correlation through the exact permutation
quotient. Marginal and conditional closures solve separate fixed points.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import product
import math
import time

import numpy as np
from scipy.optimize import root
from scipy.special import expit, logsumexp
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from forgetting.Code.model import Primitives
from forgetting_random_replacement.code.model import solve_counts
from oblivious.oe_p0.code.positive_stationary import positive_invariant
from .kernel import CountKernel


def regret(alpha, advantage):
    """Largest conditional gain from changing the chosen product."""
    return float(np.max(np.maximum(advantage, 0) - alpha * advantage))


def evaluate_policy(demand, alpha, primitives, success, failure):
    """Calendar Bellman evaluation including waiting and replacement risk."""
    p = primitives
    demand, alpha = np.asarray(demand), np.asarray(alpha)
    probability = p.p1 + p.delta_p * alpha
    selected = diags(probability) @ success + diags(1-probability) @ failure
    own = diags(1-demand) + diags(demand) @ selected
    reward = demand * (p.revenue-p.c1-p.delta_c*alpha)
    linear = eye(len(alpha), format='csc') - p.gamma*own
    value = splu(linear.tocsc()).solve(reward)
    advantage = -p.delta_c + p.gamma*p.delta_p*((success-failure)@value)
    residual = float(np.max(abs(reward+p.gamma*(own@value)-value)))
    return dict(value=value, advantage=advantage, residual=residual,
                own_kernel=own, own_linear=linear, selected_kernel=selected)


def solve_dp(demand, primitives, success, failure, tolerance=1e-11):
    """Howard policy improvement on every exact mature count state."""
    policy = np.zeros(len(demand))
    for iteration in range(1000):
        data = evaluate_policy(demand, policy, primitives, success, failure)
        improved = (data['advantage'] > 0).astype(float)
        if regret(policy, data['advantage']) <= tolerance:
            return dict(data, policy=policy, iterations=iteration+1)
        policy = improved
    raise ArithmeticError('Exact count DP policy improvement exceeded its budget.')


def _calendar(jump_law, rates):
    log_mass = np.log(jump_law)-np.log(rates)
    mass = np.exp(log_mass-logsumexp(log_mass))
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ArithmeticError('Positive count calendar law lost representable mass.')
    return mass


def checked_count_stationary(jump, rates, *, row_operators=None, tolerance=1e-11,
                             stability_tolerance=1e-7):
    """Subtraction-free sparse GTH in both state elimination orders.

    Compare calendar and jump distributions, and optionally every positive
    conditional row. A small absolute residual alone does not certify a law.
    """
    started = time.monotonic()
    size = len(rates)
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ArithmeticError('Count invariant requires positive finite exit rates.')
    row_error = float(np.max(abs(np.asarray(jump.sum(axis=1)).ravel()-1)))
    if row_error > 1e-12:
        raise ArithmeticError('Count jump matrix is not row stochastic.')
    jump_laws, laws = [], []
    for permutation in (None, np.arange(size)[::-1]):
        w = positive_invariant(jump, np.ones(size), permutation)
        if np.any(w <= 0):
            raise ArithmeticError('Positive count GTH lost a positive jump coordinate.')
        jump_laws.append(w)
        laws.append(_calendar(w, rates))
    law, w = laws[-1], jump_laws[-1]
    stability = max(float(np.sum(abs(laws[0]-law))),
                    float(np.sum(abs(jump_laws[0]-w))))
    conditional_error = 0.
    if row_operators is not None:
        own_operator, selected_operator = row_operators
        means = [np.asarray(own_operator@nu).ravel() for nu in laws]
        selected = [np.asarray(selected_operator@nu).ravel() for nu in laws]
        if any(np.any(mu <= 0) for mu in means):
            raise ArithmeticError('Conditional count law lost an own-state row.')
        conditional_error = float(np.max(abs(selected[0]/means[0]-selected[1]/means[1])))
    jump_residual = float(np.sum(abs(jump.T@w-w)))
    flow = law*rates
    calendar_residual = float(np.sum(abs(jump.T@flow-flow)))
    if (stability > stability_tolerance or conditional_error > stability_tolerance
            or max(jump_residual, calendar_residual) > tolerance):
        raise ArithmeticError('Independent count invariant audit failed: '
                              f'L1={stability:g}, conditional={conditional_error:g}, '
                              f'jump flow={jump_residual:g}.')
    return dict(law=law, jump_law=w, summary=dict(
        verified=True, methods=['Subtraction-free sparse positive GTH',
                                 'Subtraction-free sparse positive GTH, reversed elimination order'],
        probability_sum=float(law.sum()), minimum_probability=float(law.min()),
        minimum_exit_rate=float(rates.min()), maximum_exit_rate=float(rates.max()),
        stability_l1=stability, conditional_demand_stability=conditional_error,
        rate_weighted_residual=jump_residual, residual=calendar_residual,
        row_sum_error=row_error, solved_states=size,
        elapsed_seconds=time.monotonic()-started,
        scope='Independent finite-chain numerical stability; no mixing-time or uniqueness proof'))


class _LazyJacobian(dict):
    def __init__(self, values, callback):
        super().__init__(values)
        self.callback = callback

    def __missing__(self, key):
        if key != 'jacobian':
            raise KeyError(key)
        value = self.callback()
        self[key] = value
        self.callback = None
        return value


class CountExperiment:
    def __init__(self, capacity, sellers=2, closure='marginal', p0=.5,
                 primitives=None, order=128, tolerance=1e-7):
        if closure not in ('monopoly', 'marginal', 'conditional'):
            raise ValueError('Unsupported oblivious belief closure.')
        if (sellers == 1) != (closure == 'monopoly'):
            raise ValueError('One seller requires the monopoly closure.')
        self.capacity, self.sellers, self.closure = int(capacity), int(sellers), closure
        self.p0, self.tolerance = float(p0), float(tolerance)
        self.primitives = Primitives(p0=p0) if primitives is None else primitives
        if abs(self.primitives.p0-p0) > 1e-15:
            raise ValueError('Primitives and experiment must use the same outside option.')
        self.game = CountKernel(capacity, sellers, p0, order=order)
        self.kernel = self.game
        self.n, self.grid = self.capacity+1, np.arange(self.capacity+1)/self.capacity
        self.S, self.F = self.game.success, self.game.failure
        self.calls, self.last_alpha, self.last_result = 0, None, None
        self.derivatives = None

    def configuration(self):
        return dict(capacity=self.capacity, effective_rho=1-1/self.capacity,
                    sellers=self.sellers, closure=self.closure, p0=self.p0,
                    primitives=asdict(self.primitives), count_states=self.n,
                    joint_states=self.game.size, tolerance=self.tolerance,
                    state='Mature memory: retained successes S=0,...,capacity',
                    replacement='Uniform old slot independent of new outcome; only selected seller updates',
                    posterior='Beta(1+S,1+capacity-S)',
                    joint_representation='Exact seller-permutation quotient; correlated rival law preserved',
                    initial_state='Independent equal mixture of floor(capacity/2),ceil(capacity/2) for odd capacity; exact midpoint for even capacity',
                    actual_value='Mean seller discounted payoff; equals each seller at exchangeable initialization',
                    scope='Finite exact-count symmetric oblivious equilibrium; no full-state MPE or uniqueness claim')

    def belief_demand(self, law):
        game = self.game
        mu = np.asarray(game.own_marginal@law).ravel()
        if self.sellers == 1:
            demand = np.asarray(game.selection[:, 0]).copy()
        elif self.closure == 'conditional':
            if np.any(mu <= 0):
                raise ArithmeticError('Every exact count conditional row must remain positive.')
            demand = np.asarray(game.sale_marginal@law).ravel()/mu
        else:
            demand = np.asarray(game.marginal_demand@law).ravel()
        if np.any(~np.isfinite(demand)) or np.any(demand <= 0) or np.any(demand > 1+1e-12):
            raise ArithmeticError('Invalid own-state count belief demand.')
        return np.clip(demand, 0, 1), mu

    def _jump(self, alpha):
        p = self.primitives
        return self.game.jump(alpha, p1=p.p1, p2=p.p2)

    def calculate(self, alpha):
        alpha = np.clip(np.asarray(alpha, dtype=float), 0, 1)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError('Count policy must match all retained success counts.')
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        self.calls += 1
        jump, rates = self._jump(alpha)
        w = positive_invariant(jump, np.ones(len(rates)))
        if np.any(w <= 0):
            raise ArithmeticError('Exploratory positive count law lost mass.')
        law = _calendar(w, rates)
        demand, mu = self.belief_demand(law)
        p = self.primitives
        evaluated = evaluate_policy(demand, alpha, p, self.S, self.F)
        value, advantage = evaluated['value'], evaluated['advantage']

        def jacobian():
            if self.derivatives is None:
                self.derivatives = [self.game.transition_derivative(k, p1=p.p1, p2=p.p2, off_diagonal_only=True)
                                    for k in range(self.n)]
                self.rate_derivatives = np.column_stack(
                    [np.asarray(delta.sum(axis=1)).ravel() for delta in self.derivatives])
            size = len(rates)
            anchor = int(np.argmax(w))
            keep = np.arange(size) != anchor
            linear = (eye(size, format='csr')-jump.T)[keep, :][:, keep].tocsc()
            factor = splu(linear, permc_spec='MMD_AT_PLUS_A')
            # Scaled w/r avoids overflow from very small calendar exit rates.
            log_q = np.log(w)-np.log(rates)
            q = np.exp(log_q-log_q.max())
            q_sum = q.sum()
            relative_dr = self.rate_derivatives/rates[:, None]
            rhs = np.column_stack([delta.T@q-jump.T@(q*self.rate_derivatives[:, k])
                                   for k, delta in enumerate(self.derivatives)])
            dw = np.zeros((size, self.n))
            dw[keep] = factor.solve(rhs[keep])
            raw = dw/rates[:, None]-q[:, None]*relative_dr
            dc = raw/q_sum-law[:, None]*(raw.sum(axis=0)/q_sum)
            dd = self.game.marginal_demand@dc
            if self.closure == 'conditional':
                dd = (self.game.sale_marginal@dc-demand[:, None]*(self.game.own_marginal@dc))/mu[:, None]
            selected = evaluated['selected_kernel']
            flow = p.revenue-p.c1-p.delta_c*alpha+p.gamma*(selected@value-value)
            dv = splu(evaluated['own_linear'].tocsc()).solve(
                flow[:, None]*dd+np.diag(demand*advantage))
            derivative = p.gamma*p.delta_p*((self.S-self.F)@dv)
            if np.any(~np.isfinite(derivative)):
                raise ArithmeticError('Nonfinite exact-count equilibrium Jacobian.')
            return derivative

        result = _LazyJacobian(dict(policy=alpha.copy(), value=value, advantage=advantage,
            demand=demand, joint=law, regret=regret(alpha, advantage)), jacobian)
        self.last_alpha, self.last_result = alpha.copy(), result
        return result

    def _initial(self, values):
        low, high = self.capacity//2, (self.capacity+1)//2
        if np.asarray(values).shape == (self.n,):
            return float((values[low]+values[high])/2)
        lookup = {tuple(state): k for k, state in enumerate(self.game.states)}
        initial = 0.
        for choices in product((low, high), repeat=self.sellers):
            initial += float(values[lookup[tuple(sorted(choices))]])/(2**self.sellers)
        return initial

    def certify(self, alpha):
        started = time.monotonic()
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)) or np.any((alpha < 0)|(alpha > 1)):
            raise ValueError('Invalid exact-count candidate policy.')
        jump, rates = self._jump(alpha)
        stationary = checked_count_stationary(jump, rates,
            row_operators=(self.game.own_marginal, self.game.sale_marginal))
        law = stationary['law']
        demand, mu = self.belief_demand(law)
        p = self.primitives
        evaluated = evaluate_policy(demand, alpha, p, self.S, self.F)
        best = solve_dp(demand, p, self.S, self.F)
        conditional_regret = regret(alpha, evaluated['advantage'])
        value_difference = float(np.max(abs(best['value']-evaluated['value'])))
        calendar = self.game.transition(alpha, p1=p.p1, p2=p.p2)
        states = np.asarray(self.game.states)
        mean_reward = np.sum(self.game.selection*(p.revenue-p.c1-p.delta_c*alpha[states]), axis=1)/self.sellers
        actual_linear = eye(self.game.size, format='csc')-p.gamma*calendar
        actual_value = splu(actual_linear.tocsc()).solve(mean_reward)
        actual_residual = float(np.max(abs(mean_reward+p.gamma*(calendar@actual_value)-actual_value)))
        sales = np.asarray(self.game.sale_marginal@law).ravel()
        sale_rate = float(sales.sum())
        sale_law = sales/sale_rate
        probability = p.p1+p.delta_p*alpha
        log_eta = np.zeros(self.n)
        for s in range(self.capacity):
            log_eta[s+1] = log_eta[s]+math.log(probability[s])+math.log(self.capacity-s)-math.log(1-probability[s+1])-math.log(s+1)
        eta = np.exp(log_eta-logsumexp(log_eta))
        sale_error = float(np.sum(abs(sale_law-eta)))
        use = float(sale_law@alpha)
        outside = float(law@self.game.outside)
        market_error = abs(outside+self.sellers*sale_rate-1)
        diagnostics = dict(conditional_regret=conditional_regret,
            bellman_residual=evaluated['residual'], optimal_value_difference=value_difference,
            actual_value_residual=actual_residual,
            actual_value_error_bound=actual_residual/(1-p.gamma),
            sale_law_l1=sale_error, market_partition_error=market_error,
            stationary=stationary['summary'], demand=self.game.demand_diagnostics,
            certificate_seconds=time.monotonic()-started)
        if self.sellers == 1:
            independent = solve_counts(p, self.capacity)
            diagnostics['independent_monopoly_value_difference'] = float(np.max(abs(independent.values[-1]-evaluated['value'])))
            if diagnostics['independent_monopoly_value_difference'] > 1e-7:
                raise ArithmeticError('Independent monopoly count Bellman values disagree.')
        verified = bool(conditional_regret <= self.tolerance and evaluated['residual'] <= 1e-10
            and value_difference <= 1e-5 and actual_residual <= 1e-10
            and sale_error <= 1e-7 and market_error <= 1e-10)
        summary = dict(capacity=self.capacity, effective_rho=1-1/self.capacity,
            p0=self.p0, sellers=self.sellers, closure=self.closure, verified=verified,
            product2_conditional=use, sale_rate=sale_rate, outside_rate=outside,
            product2_calendar=sale_rate*use,
            profit_calendar=sale_rate*(p.revenue-p.c1-p.delta_c*use),
            value_initial=self._initial(actual_value),
            perceived_value_initial=self._initial(evaluated['value']),
            count_states=self.n, joint_states=self.game.size)
        return dict(verified=verified, summary=summary, policy=alpha, grid=self.grid,
            states=states, joint=law, value=evaluated['value'], actual_value=actual_value,
            demand=demand, advantage=evaluated['advantage'], sale_stationary=eta,
            diagnostics=diagnostics, configuration=self.configuration())

    def solve(self, initial_policy=None, max_evaluations=500, progress=None):
        if self.sellers == 1:
            policy = solve_dp(self.game.selection[:, 0], self.primitives, self.S, self.F)['policy']
            return self.certify(policy)
        alpha = np.full(self.n, .5) if initial_policy is None else np.clip(np.asarray(initial_policy), 0, 1)
        if alpha.shape != (self.n,):
            raise ValueError('Initial count policy has the wrong shape.')
        history, seen, best = [], set(), None
        for _ in range(12):
            data = self.calculate(alpha)
            history.append(dict(stage='pure improvement', regret=data['regret'], calls=self.calls))
            if progress:
                progress(history[-1])
            if best is None or data['regret'] < best['regret']:
                best = dict(data)
            if data['regret'] <= self.tolerance:
                result = self.certify(alpha)
                if result['verified']:
                    return dict(result, history=history, search_calls=self.calls)
            signature = (data['advantage'] >= 0).tobytes()
            if signature in seen:
                break
            seen.add(signature)
            alpha = (data['advantage'] >= 0).astype(float)
        for schedule in ([0.], [.1, .03, .01, .003, .001, .0003, .0001, 0.]):
            alpha = best['policy'].copy()
            for temperature in schedule:
                if self.calls >= max_evaluations:
                    raise RuntimeError(f'Count equilibrium budget exhausted; best regret={best["regret"]:g}')
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
                    options=dict(xtol=1e-10, maxfev=min(150, max_evaluations-self.calls), factor=.3))
                alpha = np.clip(solved.x, 0, 1)
                alpha[alpha < 1e-9] = 0
                alpha[alpha > 1-1e-9] = 1
                data = self.calculate(alpha)
                history.append(dict(stage='natural-map continuation', temperature=temperature,
                    regret=data['regret'], calls=self.calls, success=bool(solved.success)))
                if progress:
                    progress(history[-1])
                if data['regret'] <= self.tolerance:
                    result = self.certify(alpha)
                    if result['verified']:
                        return dict(result, history=history, search_calls=self.calls)
        raise RuntimeError(f'No certified count equilibrium; best regret={best["regret"]:g}')


__all__ = ['CountExperiment', 'evaluate_policy', 'solve_dp', 'regret', 'checked_count_stationary']
