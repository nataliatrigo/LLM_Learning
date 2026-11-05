"""Scalable exact-count oblivious equilibria on large permutation quotients.

The economic equations and count-state representation are unchanged. Large
joint chains use normalized matrix-free GMRES instead of sparse elimination,
and discounted payoff evaluation uses the calendar Bellman contraction.
Independent coordinate-order and positive two-start stationary checks remain
mandatory; neither a small residual alone nor clipped probabilities certify a
solution.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import logsumexp
from scipy.sparse import eye
from scipy.sparse.linalg import LinearOperator, splu, spilu, gmres

from oblivious.oe_three_sellers_fixed_p0.code.five_seller_model import (
    IterativeFactor, CERTIFICATION, ITERATIVE_SETTINGS)
from oblivious.oe_three_sellers_fixed_p0.code.four_seller_iterative import SETTINGS
from oblivious.oe_three_sellers_fixed_p0.code.stationary import (
    _validated_system, _flow_checks, _jump_mass, _positive_iterates)
from .model import CountExperiment, _LazyJacobian, evaluate_policy, solve_dp, regret, _calendar


DIRECT_STATE_LIMIT = 2000
SCALABLE_SETTINGS = dict(ITERATIVE_SETTINGS,
    method='Matrix-free exact normalized GMRES with bounded-fill shifted ILU/Sherman-Morrison preconditioning',
    preconditioning='Shifted sparse M-matrix inverse and exact rank-one Sherman-Morrison correction; shift changes only the preconditioner',
    ilu_shifts=[1e-5, 1e-4, 1e-3], ilu_drop_tolerance=1e-4, ilu_fill_factor=5.,
    positive_initial_iterations=100, positive_search_warm_iterations=10,
    positivity_recovery='Exact lazy Markov propagation of an algebraic candidate until strictly positive; no beliefs before positivity, no clipping or flooring',
    derivative_scaling='Unit-norm right-hand side, then exact rescaling and unchanged pinned backward-error check')


class _WarmFactor(IterativeFactor):
    """Same normalized equations and GMRES tolerances, with a positive warm start.

    A previous calendar law can be rate-weighted into a positive initial jump
    law. Algebraic GMRES iterates are propagated by the exact lazy Markov
    operator until strictly positive; no coordinate is clipped or floored. Final
    independent certification always starts afresh and does not use this
    exploratory warm-start shortcut.
    """

    def __init__(self, jump, reference, initial=None):
        self.transpose = jump.T.tocsr()
        self.size = jump.shape[0]
        self.source = np.full(self.size, .99/self.size)
        self.source[reference] += .01
        self.operator = LinearOperator((self.size, self.size),
            matvec=lambda value: value-self.transpose@value+self.source*value.sum(), dtype=float)
        self.record = dict(size=self.size, matrix_nnz=jump.nnz,
            normalization_reference=int(reference), solves=0, total_inner_iterations=0,
            maximum_relative_residual=0., maximum_absolute_residual=0.,
            maximum_backward_error=0., solve_seconds=0.)
        for epsilon in (1e-5, 1e-4, 1e-3):
            try:
                shifted = eye(self.size, format='csc')-self.transpose+epsilon*eye(self.size, format='csc')
                self.preconditioner_factor = spilu(shifted.tocsc(), drop_tol=1e-4,
                                                   fill_factor=5., permc_spec='MMD_AT_PLUS_A')
                break
            except RuntimeError:
                if epsilon == 1e-3:
                    raise
        normalized_source = self.preconditioner_factor.solve(self.source)
        denominator = 1+normalized_source.sum()

        def precondition(value):
            solved = self.preconditioner_factor.solve(value)
            return solved-normalized_source*(solved.sum()/denominator)

        self.preconditioner = LinearOperator((self.size, self.size), matvec=precondition, dtype=float)
        self.record.update(ilu_shift=epsilon, ilu_drop_tolerance=1e-4, ilu_fill_factor=5.,
                           ilu_nnz=self.preconditioner_factor.nnz)
        candidate = self.source.copy() if initial is None else np.asarray(initial, dtype=float).copy()
        if candidate.shape != (self.size,) or np.any(~np.isfinite(candidate)) or np.any(candidate <= 0):
            raise ArithmeticError('Large-count iterative start must have strictly positive finite mass.')
        candidate /= candidate.sum()
        warmup = 100 if initial is None else 10
        for _ in range(warmup):
            candidate = .5*(candidate+self.transpose@candidate)
        self.law = self._normalized_solve(self.source, candidate)
        if np.any(~np.isfinite(self.law)) or np.any(self.law <= 0):
            raise ArithmeticError('Large-count normalized GMRES lost a positive coordinate.')
        self.law /= self.law.sum()
        self.anchor = int(np.argmax(self.law))
        self.keep = np.arange(self.size) != self.anchor

    def _normalized_solve(self, rhs, initial=None):
        rhs = np.asarray(rhs, dtype=float)
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0:
            return np.zeros_like(rhs)
        started, iterations = time.monotonic(), 0

        def callback(_):
            nonlocal iterations
            iterations += 1
            if time.monotonic()-started > 240:
                raise ArithmeticError('Large-count preconditioned GMRES exceeded its explicit time budget.')

        solution, info = gmres(self.operator, rhs, x0=initial, M=self.preconditioner,
            rtol=SETTINGS['gmres_relative_tolerance'], atol=SETTINGS['gmres_absolute_tolerance'],
            restart=SETTINGS['gmres_restart'], maxiter=SETTINGS['gmres_max_restart_cycles'],
            callback=callback, callback_type='pr_norm')
        target = max(SETTINGS['gmres_absolute_tolerance'], SETTINGS['gmres_relative_tolerance']*rhs_norm)
        signed_entries = int(np.sum(solution <= 0)) if initial is not None else 0
        polishing = 0
        if signed_entries:
            for polishing in range(1, CERTIFICATION['positive_max_iterations']+1):
                solution = .5*(solution+self.transpose@solution)
                if polishing % 25 == 0 and np.all(solution > 0):
                    break
                if time.monotonic()-started > CERTIFICATION['positive_time_limit']:
                    raise ArithmeticError('Large-count Markov sign recovery exceeded its explicit budget.')
            if np.any(solution <= 0):
                raise ArithmeticError('Large-count Markov propagation did not restore strict positivity.')
        residual = float(np.linalg.norm(rhs-self.operator@solution))
        if (info != 0 or np.any(~np.isfinite(solution))
                or residual > SETTINGS['true_residual_safety_factor']*target):
            raise ArithmeticError('Large-count normalized GMRES failed its true residual check: '
                                  f'info={info}, residual={residual:g}, target={target:g}.')
        self.record['solves'] += 1
        self.record['total_inner_iterations'] += iterations
        self.record['maximum_relative_residual'] = max(self.record['maximum_relative_residual'], residual/rhs_norm)
        self.record['maximum_absolute_residual'] = max(self.record['maximum_absolute_residual'], residual)
        self.record['solve_seconds'] += time.monotonic()-started
        if initial is not None:
            self.record.update(rejected_signed_tail_entries=signed_entries,
                               exact_markov_positivity_iterations=polishing,
                               positivity_method='Exact lazy Markov propagation; no clipping or flooring')
        return solution

    def solve(self, rhs):
        """Normalize derivative right-hand sides before the pinned solve.

        The unchanged absolute Krylov tolerance must not dominate tiny
        derivative flows. Rescaling the exact linear equation retains the
        same pinned backward-error check and improves relative accuracy.
        """
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(rhs[:, k]) for k in range(rhs.shape[1])])
        scale = float(np.linalg.norm(rhs))
        if scale == 0:
            return np.zeros_like(rhs)
        return super().solve(rhs/scale)*scale


def checked_large_stationary(jump, rates):
    """Independent ordered solves and unchanged positive two-start checks."""
    matrix, rates, row_error = _validated_system(jump, rates)
    started, size = time.monotonic(), len(rates)
    laws, jump_laws, records = [], [], []
    for reverse in (False, True):
        permutation = np.arange(size)[::-1] if reverse else np.arange(size)
        factor = _WarmFactor(matrix[permutation, :][:, permutation], 0 if reverse else size//2)
        w = np.empty(size)
        w[permutation] = factor.law
        jump_laws.append(w)
        laws.append(_calendar(w, rates))
        records.append(dict(factor.record))
        del factor
    law, jump_law = laws[-1], jump_laws[-1]
    calendar_difference = float(np.sum(abs(laws[0]-law)))
    jump_difference = float(np.sum(abs(jump_laws[0]-jump_law)))
    if max(calendar_difference, jump_difference) > CERTIFICATION['stability_tolerance']:
        raise ArithmeticError('Independent large-count stationary solves disagree.')
    positive = _positive_iterates(matrix, rates, law,
        tolerance=CERTIFICATION['stability_tolerance'],
        max_iterations=CERTIFICATION['positive_max_iterations'],
        time_limit=CERTIFICATION['positive_time_limit'])
    jump_residual, calendar_residual, jump_law = _flow_checks(matrix, rates, law)
    if max(jump_residual, calendar_residual) > CERTIFICATION['invariant_tolerance']:
        raise ArithmeticError('Large-count independent invariant flow check failed.')
    return dict(law=law, jump_law=jump_law, summary=dict(
        verified=True, probability_sum=float(law.sum()), minimum_probability=float(law.min()),
        minimum_exit_rate=float(rates.min()), maximum_exit_rate=float(rates.max()),
        residual=calendar_residual, rate_weighted_residual=jump_residual, row_sum_error=row_error,
        stability_l1=max(calendar_difference, jump_difference, positive['discrepancy']),
        independent_calendar_l1=calendar_difference, independent_jump_l1=jump_difference,
        methods=['ILU-preconditioned exact normalized GMRES, original order and middle reference',
                 'ILU-preconditioned exact normalized GMRES, reversed order and edge reference',
                 'Positive lazy Markov iteration from uniform and concentrated starts'],
        tolerance=CERTIFICATION['invariant_tolerance'],
        stability_tolerance=CERTIFICATION['stability_tolerance'],
        linear_solves=records, positive_iteration=positive, solved_states=size,
        iterative_settings=dict(SCALABLE_SETTINGS),
        communicating_components=1, elapsed_seconds=time.monotonic()-started,
        scope='Exact count quotient, independent numerical stability; no mixing-time or uniqueness guarantee'))


class LargeCountExperiment(CountExperiment):
    """Exact count model with a scalable solver for three to five sellers."""

    def __init__(self, capacity, sellers=5, closure='marginal', p0=.1,
                 primitives=None, order=128, tolerance=1e-7, direct_state_limit=DIRECT_STATE_LIMIT):
        super().__init__(capacity, sellers, closure, p0, primitives, order, tolerance)
        if self.sellers < 3 or self.closure != 'marginal':
            raise ValueError('The large-count adapter supports three to five sellers and marginal OE.')
        self.direct_state_limit = int(direct_state_limit)
        self._search_law = None
        self._progress = None

    def configuration(self):
        configuration = super().configuration()
        configuration.update(large_joint_solver=dict(SCALABLE_SETTINGS),
                             independent_stationary_checks=dict(CERTIFICATION),
                             direct_state_limit=self.direct_state_limit,
                             actual_value_method='Positive calendar Bellman contraction, unchanged discount')
        return configuration

    def calculate(self, alpha):
        if self.game.size <= self.direct_state_limit:
            return super().calculate(alpha)
        alpha = np.clip(np.asarray(alpha, dtype=float), 0, 1)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError('Count policy must match every retained-success state.')
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        started = time.monotonic()
        self.calls += 1
        jump, rates = self._jump(alpha)
        initial = None
        if self._search_law is not None:
            log_mass = np.log(self._search_law)+np.log(rates)
            initial = np.exp(log_mass-logsumexp(log_mass))
        factor = _WarmFactor(jump, len(rates)//2, initial)
        w = factor.law
        law = _calendar(w, rates)
        self._search_law = law.copy()
        demand, mu = self.belief_demand(law)
        p = self.primitives
        evaluated = evaluate_policy(demand, alpha, p, self.S, self.F)
        value, advantage = evaluated['value'], evaluated['advantage']

        def jacobian():
            # Stream one exact policy derivative at a time, avoiding several
            # size-by-count dense work arrays on the largest quotient.
            log_q = np.log(w)-np.log(rates)
            q = np.exp(log_q-log_q.max())
            q_sum = q.sum()
            dd = np.empty((self.n, self.n))
            for k in range(self.n):
                delta = self.game.transition_derivative(k, p1=p.p1, p2=p.p2,
                                                       off_diagonal_only=True)
                dr = np.asarray(delta.sum(axis=1)).ravel()
                rhs = delta.T@q-jump.T@(q*dr)
                dw = np.zeros(len(rates))
                dw[factor.keep] = factor.solve(rhs[factor.keep])
                raw = dw/rates-q*(dr/rates)
                dc = raw/q_sum-law*(raw.sum()/q_sum)
                dd[:, k] = self.game.marginal_demand@dc
            flow = p.revenue-p.c1-p.delta_c*alpha+p.gamma*(evaluated['selected_kernel']@value-value)
            dv = splu(evaluated['own_linear'].tocsc()).solve(
                flow[:, None]*dd+np.diag(demand*advantage))
            derivative = p.gamma*p.delta_p*((self.S-self.F)@dv)
            if np.any(~np.isfinite(derivative)):
                raise ArithmeticError('Large-count equilibrium Jacobian is nonfinite.')
            return derivative

        result = _LazyJacobian(dict(policy=alpha.copy(), value=value, advantage=advantage,
            demand=demand, joint=law, regret=regret(alpha, advantage)), jacobian)
        self.last_alpha, self.last_result = alpha.copy(), result
        if self._progress:
            self._progress(dict(stage='Large-count exploratory evaluation', calls=self.calls,
                regret=result['regret'], elapsed_seconds=time.monotonic()-started,
                stationary_solver=dict(factor.record)))
        return result

    def certify(self, alpha):
        if self.game.size <= self.direct_state_limit:
            return super().certify(alpha)
        started = time.monotonic()
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)) or np.any((alpha < 0)|(alpha > 1)):
            raise ValueError('Candidate policy must be a probability at every count state.')
        if self._progress:
            self._progress(dict(stage='Large-count independent certificate started'))
        jump, rates = self._jump(alpha)
        stationary = checked_large_stationary(jump, rates)
        law = stationary['law']
        demand, mu = self.belief_demand(law)
        p = self.primitives
        evaluated = evaluate_policy(demand, alpha, p, self.S, self.F)
        best = solve_dp(demand, p, self.S, self.F)
        conditional_regret = regret(alpha, evaluated['advantage'])
        value_difference = float(np.max(abs(best['value']-evaluated['value'])))
        states = self.game.states
        mean_reward = np.mean(self.game.selection*(p.revenue-p.c1-p.delta_c*alpha[states]), axis=1)
        off = jump.multiply(rates[:, None]).tocsr()
        actual_value = np.zeros(self.game.size)
        for iteration in range(1, 10001):
            updated = mean_reward+p.gamma*(actual_value+off@actual_value-rates*actual_value)
            step_residual = float(np.max(abs(updated-actual_value)))
            actual_value = updated
            if step_residual <= 1e-11:
                break
        else:
            raise ArithmeticError('Large-count discounted payoff iteration exceeded its budget.')
        actual_residual = float(np.max(abs(mean_reward+p.gamma*(actual_value+off@actual_value-rates*actual_value)-actual_value)))
        sales = np.asarray(self.game.sale_marginal@law).ravel()
        sale_rate = float(sales.sum())
        if sale_rate <= 0:
            raise ArithmeticError('Large-count stationary own-sale rate is not positive.')
        probability = p.p1+p.delta_p*alpha
        log_eta = np.zeros(self.n)
        for s in range(self.capacity):
            log_eta[s+1] = (log_eta[s]+math.log(probability[s])+math.log(self.capacity-s)
                           -math.log1p(-probability[s+1])-math.log(s+1))
        eta = np.exp(log_eta-logsumexp(log_eta))
        sale_error = float(np.sum(abs(sales/sale_rate-eta)))
        use = float((sales/sale_rate)@alpha)
        outside = float(law@self.game.outside)
        market_error = abs(outside+self.sellers*sale_rate-1)
        reward_error = abs(float(law@mean_reward)-sale_rate*(p.revenue-p.c1-p.delta_c*use))
        diagnostics = dict(conditional_regret=conditional_regret,
            bellman_residual=evaluated['residual'], optimal_value_difference=value_difference,
            actual_value_residual=actual_residual, actual_value_error_bound=actual_residual/(1-p.gamma),
            actual_value_iterations=iteration, sale_law_l1=sale_error,
            market_partition_error=market_error, stationary_profit_accounting_error=reward_error,
            stationary=stationary['summary'], demand=self.game.demand_diagnostics,
            certificate_seconds=time.monotonic()-started)
        verified = bool(conditional_regret <= self.tolerance and evaluated['residual'] <= 1e-10
            and value_difference <= 1e-5 and actual_residual <= 1e-10
            and sale_error <= 1e-7 and market_error <= 1e-10 and reward_error <= 1e-10)
        summary = dict(capacity=self.capacity, effective_rho=1-1/self.capacity,
            p0=self.p0, sellers=self.sellers, closure=self.closure, verified=verified,
            product2_conditional=use, sale_rate=sale_rate, outside_rate=outside,
            product2_calendar=sale_rate*use,
            profit_calendar=sale_rate*(p.revenue-p.c1-p.delta_c*use),
            value_initial=self._initial(actual_value),
            perceived_value_initial=self._initial(evaluated['value']),
            count_states=self.n, joint_states=self.game.size)
        if self._progress:
            self._progress(dict(stage='Large-count independent certificate finished', verified=verified,
                regret=conditional_regret, elapsed_seconds=time.monotonic()-started))
        return dict(verified=verified, summary=summary, policy=alpha.copy(), grid=self.grid,
            states=states, joint=law, value=evaluated['value'], actual_value=actual_value,
            demand=demand, advantage=evaluated['advantage'], sale_stationary=eta,
            diagnostics=diagnostics, configuration=self.configuration())

    def solve(self, initial_policy=None, max_evaluations=500, progress=None):
        self._progress = progress
        try:
            return super().solve(initial_policy, max_evaluations, progress)
        finally:
            self._progress = None


__all__ = ['LargeCountExperiment', 'DIRECT_STATE_LIMIT']
