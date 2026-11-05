"""Five-seller marginal OE on the exact seller-permutation quotient.

No independent-rival or representative-rival approximation is made.  The
calendar distribution is a law on sorted five-reputation tuples, and marginal
belief demand averages over the complete, correlated four-rival law.

The quotient discounted value is the *mean across the five sellers*.  At the
exchangeable all-.5 initial state it equals each seller's actual value; at a
general sorted state it must not be interpreted as a particular label's value.
"""

from dataclasses import asdict
import time

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from .five_seller_kernel import FiveSellerKernel
from .four_seller_iterative import IterativeFactor as FrozenIterativeFactor, SETTINGS as FROZEN_ITERATIVE_SETTINGS
from .model import Experiment as FrozenExperiment, regret
from .search_only import _LazyJacobian
from .stationary import (
    _calendar, _flow_checks, _jump_mass,
    _positive_iterates, _validated_system, jump_system,
)


CERTIFICATION = dict(
    invariant_tolerance=1e-11, stability_tolerance=1e-7,
    positive_max_iterations=1_000_000, positive_time_limit=240,
    failure_policy="Explicit failure; no tolerance relaxation or residual-only acceptance",
)
ITERATIVE_SETTINGS = dict(
    FROZEN_ITERATIVE_SETTINGS,
    method='Matrix-free normalized GMRES; positive lazy warmup for signed rare tails; no large LU',
    positivity_recovery='Continue the positive initial lazy Markov iterate until the same '
                        'normalized-system true residual target, then rerun unchanged GMRES; '
                        'no clipping or flooring',
    positivity_max_iterations=1_000_000, positivity_time_limit=240.,
)
CERTIFICATION['sparse_linear_solver'] = dict(ITERATIVE_SETTINGS)


class IterativeFactor(FrozenIterativeFactor):
    """Unchanged normalized equations, with an explicit positive-law recovery.

    A rare coordinate can be smaller than the absolute GMRES tolerance. If
    GMRES gives that coordinate a nonpositive sign, reject its vector and
    continue the already positive lazy Markov iterate and retry unchanged
    GMRES. This warmup must meet the same true normalized-system residual target,
    and the final law must still pass both independent coordinate solves and
    two positive starts.
    Derivative right-hand sides use the frozen fundamental solve unchanged.
    """

    def _normalized_solve(self, rhs, initial=None):
        solution = super()._normalized_solve(rhs, initial)
        if initial is None or np.all(solution > 0):
            return solution
        candidate = np.asarray(initial, dtype=float).copy()
        if np.any(~np.isfinite(candidate)) or np.any(candidate <= 0):
            raise ArithmeticError('Five-seller positive recovery has no strictly positive initial law.')
        candidate /= candidate.sum()
        target = max(FROZEN_ITERATIVE_SETTINGS['gmres_absolute_tolerance'],
                     FROZEN_ITERATIVE_SETTINGS['gmres_relative_tolerance']*np.linalg.norm(rhs))
        allowed = FROZEN_ITERATIVE_SETTINGS['true_residual_safety_factor']*target
        started = time.monotonic()
        for iteration in range(1, ITERATIVE_SETTINGS['positivity_max_iterations']+1):
            candidate = .5*(candidate+self.transpose@candidate)
            candidate /= candidate.sum()
            if np.any(~np.isfinite(candidate)) or np.any(candidate <= 0):
                raise ArithmeticError('Five-seller positive recovery lost a positive coordinate.')
            if iteration % 10 == 0:
                residual = float(np.linalg.norm(rhs-self.operator@candidate))
                if residual <= .5*target:
                    recovered = super()._normalized_solve(rhs, candidate)
                    if np.any(recovered <= 0):
                        if time.monotonic()-started >= ITERATIVE_SETTINGS['positivity_time_limit']:
                            break
                        continue
                    final_residual = float(np.linalg.norm(rhs-self.operator@recovered))
                    self.record['positive_recovery'] = dict(
                        iterations=iteration, seconds=time.monotonic()-started,
                        true_residual=final_residual, true_residual_target=allowed,
                        minimum_probability=float(recovered.min()),
                        rejected_gmres_nonpositive_entries=int(np.sum(solution <= 0)),
                        method='Positive lazy Markov warmup, then unchanged normalized GMRES; '
                               'no clipping or flooring')
                    return recovered
                if time.monotonic()-started >= ITERATIVE_SETTINGS['positivity_time_limit']:
                    break
        raise ArithmeticError('Five-seller positive recovery exceeded its explicit budget.')


def _pinned_jump(jump, anchor):
    """Normalized invariant and pinned derivative gauge, without any large LU.

    The frozen N=4 iterative factor is dimension independent. Reusing its
    equations directly leaves every frozen N=4 global and source unchanged.
    The retained reference allows a released lazy Jacobian to recreate the
    same factor without retaining a second large sparse system.
    """
    factor = IterativeFactor(jump, anchor)
    return factor.law, factor, int(anchor), factor.keep


def checked_quotient_stationary(jump, rates):
    """Two coordinate orders/references plus positive iteration from two starts.

    This is the unchanged large-chain audit criterion used by experiment 09,
    with normalized matrix-free GMRES instead of a large sparse factorization.
    A stationary residual alone never certifies a candidate.
    """
    matrix, rates, row_error = _validated_system(jump, rates)
    size = len(rates)
    started = time.monotonic()
    candidates = []
    linear_solves = []
    for reverse in (False, True):
        permutation = np.arange(size)[::-1] if reverse else np.arange(size)
        anchor = 0 if reverse else size // 2
        ordered, factor, _, _ = _pinned_jump(matrix[permutation, :][:, permutation], anchor)
        linear_solves.append(dict(factor.record))
        del factor
        mass = np.empty(size)
        mass[permutation] = ordered
        candidates.append(_calendar(mass, rates))
    law = candidates[-1]
    calendar_discrepancy = max(float(np.sum(abs(candidate-law))) for candidate in candidates)
    target_jump = _jump_mass(law, rates)
    jump_discrepancy = max(float(np.sum(abs(_jump_mass(candidate, rates)-target_jump)))
                           for candidate in candidates)
    discrepancy = max(calendar_discrepancy, jump_discrepancy)
    if discrepancy > CERTIFICATION['stability_tolerance']:
        raise ArithmeticError(f"Five-seller independent GMRES laws disagree: L1={discrepancy:g}")
    positive = _positive_iterates(
        matrix, rates, law, tolerance=CERTIFICATION['stability_tolerance'],
        max_iterations=CERTIFICATION['positive_max_iterations'],
        time_limit=CERTIFICATION['positive_time_limit'])
    discrepancy = max(discrepancy, positive['discrepancy'])
    jump_residual, calendar_residual, jump_law = _flow_checks(matrix, rates, law)
    if max(jump_residual, calendar_residual) > CERTIFICATION['invariant_tolerance']:
        raise ArithmeticError("Five-seller invariant flow verification failed: "
                              f"jump={jump_residual:g}, calendar={calendar_residual:g}")
    log_range = float(np.log(rates.max())-np.log(rates.min()))
    dynamic_range = float(np.exp(log_range)) if log_range < np.log(np.finfo(float).max) else 'overflow'
    summary = dict(
        verified=True, probability_sum=float(law.sum()), minimum_probability=float(law.min()),
        minimum_exit_rate=float(rates.min()), maximum_exit_rate=float(rates.max()),
        exit_rate_dynamic_range=dynamic_range, residual=calendar_residual,
        rate_weighted_residual=jump_residual, row_sum_error=row_error,
        stability_l1=discrepancy, independent_calendar_l1=calendar_discrepancy,
        independent_jump_l1=jump_discrepancy,
        tolerance=CERTIFICATION['invariant_tolerance'],
        stability_tolerance=CERTIFICATION['stability_tolerance'],
        methods=['Normalized matrix-free GMRES jump solve, original order and middle reference; '
                 'positive lazy recovery if needed',
                 'Normalized matrix-free GMRES jump solve, reversed order and edge reference; '
                 'positive lazy recovery if needed',
                 'positive lazy jump iteration, uniform and concentrated starts'],
        iterative_settings=dict(ITERATIVE_SETTINGS), linear_solves=linear_solves,
        preliminary_failures=[], positive_iteration=positive, solved_states=size,
        communicating_components=1, elapsed_seconds=time.monotonic()-started,
        scope='Exact finite seller-permutation quotient; independent numerical stability, '
              'not a mixing-time or continuous-state convergence guarantee.')
    return dict(law=law, jump_law=jump_law, summary=summary)


class FiveSellerExperiment(FrozenExperiment):
    """Same own-reputation marginal OE, extended to five strategic sellers."""

    def __init__(self, n=21, sellers=5, closure='marginal', p0=.1,
                 params=Parameters(rho=.6), order=256, tolerance=1e-7):
        if int(n) != n or n < 3 or n % 2 != 1:
            raise ValueError('Use an odd grid size >=3 so initial reputation .5 is a node.')
        if sellers != 5 or closure != 'marginal':
            raise ValueError('This new solver supports exactly five sellers and marginal beliefs.')
        if not np.isfinite(p0) or not 0 <= p0 < 1 or not 0 < tolerance <= 1e-7:
            raise ValueError('Require p0 in [0,1) and the original or stricter regret tolerance.')
        self.n, self.sellers, self.closure = int(n), 5, closure
        self.params, self.p0, self.order, self.tolerance = params, float(p0), int(order), float(tolerance)
        self.grid = np.linspace(0, 1, self.n)
        self.game = FiveSellerKernel(self.grid, params, self.p0, self.order)
        self.kernel = self.game
        self.S, self.F = self.game.success, self.game.failure
        self.representatives = self.game.representatives
        self.own_marginal = self.game.own_marginal
        self.sale_marginal = self.game.sale_marginal
        self.marginal_demand = self.game.marginal_demand
        self.derivatives, self.rate_derivatives = None, None
        self.calls, self.last_alpha, self.last_result = 0, None, None
        self._progress = None

    def configuration(self):
        return dict(n=self.n, sellers=5, closure='marginal', p0=self.p0,
                    parameters=asdict(self.params), order=self.order, tolerance=self.tolerance,
                    strategy='Symmetric mixed own-reputation policy; action after selection',
                    marginal_belief='Joint calendar-time distribution of all four rivals, '
                                    'independent of focal state; rival correlations preserved',
                    outside_event='No seller sells; all reputations freeze',
                    forgetting_clock='Own observations only', initial_reputations=.5,
                    joint_representation='Exact seller-permutation quotient, sorted reputation index tuples',
                    actual_value_scope='Mean across five seller payoffs; equals each seller value '
                                       'at the exchangeable all-.5 initial state',
                    certification=dict(CERTIFICATION),
                    scope='Finite-grid marginal oblivious equilibrium; not full-state MPE '
                          'or uniqueness/continuous-state convergence proof')

    def _report(self, message):
        if self._progress is not None:
            self._progress(message)

    def _ensure_derivatives(self):
        if self.derivatives is None:
            self.derivatives = []
            for k in range(self.n):
                derivative = self.game.transition_derivative(k)
                off = derivative.copy().tocsr()
                off.setdiag(0)
                off.eliminate_zeros()
                self.derivatives.append(off)
            self.rate_derivatives = np.column_stack(
                [np.asarray(delta.sum(axis=1)).ravel() for delta in self.derivatives])

    def calculate(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError('Search policy must be a finite vector matching the grid.')
        alpha = np.clip(alpha, 0, 1)
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        if isinstance(self.last_result, _LazyJacobian):
            self.last_result.release_factorization()
        started = time.monotonic()
        self.calls += 1
        calendar_kernel = self.game.transition(alpha)
        jump, rates = jump_system(calendar_kernel)
        size, anchor = len(rates), len(rates)//2
        w, lu, normalization_reference, keep = _pinned_jump(jump, anchor)
        log_q = np.log(w)-np.log(rates)
        q = np.exp(log_q-log_q.max())
        q_sum = q.sum()
        calendar = q/q_sum
        if np.any(~np.isfinite(calendar)) or np.any(calendar <= 0):
            raise ArithmeticError('Five-seller exploratory calendar law lost positive mass.')
        demand = np.asarray(self.marginal_demand@calendar).ravel()
        if np.any(~np.isfinite(demand)) or np.any(demand < -1e-12) or np.any(demand > 1+1e-12):
            raise ArithmeticError('Invalid five-seller exploratory belief demand.')
        demand = np.clip(demand, 0, 1)
        evaluated = evaluate_policy(self.grid, demand, alpha, self.params)
        value, advantage = evaluated['value'], evaluated['advantage']
        factorization = dict(lu=lu)

        def jacobian():
            self._ensure_derivatives()
            derivative_rhs = np.empty((size, self.n))
            for k, delta in enumerate(self.derivatives):
                derivative_rhs[:, k] = delta.T@q-jump.T@(q*self.rate_derivatives[:, k])
            dw = np.zeros((size, self.n))
            factor = factorization['lu']
            if factor is None:
                factor = IterativeFactor(jump, normalization_reference)
                if not np.array_equal(factor.keep, keep):
                    raise ArithmeticError('Recreated five-seller derivative gauge changed.')
            dw[keep] = factor.solve(derivative_rhs[keep])
            dc_raw = dw/rates[:, None]-q[:, None]*(self.rate_derivatives/rates[:, None])
            dc = dc_raw/q_sum-calendar[:, None]*(dc_raw.sum(axis=0)/q_sum)
            dd = self.marginal_demand@dc
            p = self.params
            success = p.p1+(p.p2-p.p1)*alpha
            selected = diags(success)@self.S+diags(1-success)@self.F
            own = diags(1-demand)+diags(demand)@selected
            own_factor = splu((eye(self.n, format='csr')-p.gamma*own).tocsc())
            flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected@value-value)
            dv = own_factor.solve(flow[:, None]*dd+np.diag(demand*advantage))
            result = p.gamma*(p.p2-p.p1)*((self.S-self.F)@dv)
            if np.any(~np.isfinite(result)):
                raise ArithmeticError('Five-seller exploratory Jacobian is nonfinite.')
            return result

        result = _LazyJacobian(dict(policy=alpha.copy(), advantage=advantage, demand=demand,
                                    value=value, reduced_law=calendar,
                                    regret=regret(alpha, advantage)), jacobian,
                               lambda: factorization.update(lu=None))
        self.last_alpha, self.last_result = alpha.copy(), result
        self._report(dict(stage='five-seller exploratory evaluation', calls=self.calls,
                          regret=result['regret'], elapsed_seconds=time.monotonic()-started))
        return result

    def certify(self, alpha):
        started = time.monotonic()
        alpha = np.asarray(alpha, dtype=float)
        if (alpha.shape != (self.n,) or np.any(~np.isfinite(alpha))
                or np.any(alpha < 0) or np.any(alpha > 1)):
            raise ValueError('Invalid candidate policy')
        # Discard any exploratory factor before the independent audit.
        if isinstance(self.last_result, _LazyJacobian):
            self.last_result.release_factorization()
        self._report(dict(stage='five-seller independent certificate started'))
        kernel = self.game.transition(alpha)
        jump, rates = jump_system(kernel)
        stationary = checked_quotient_stationary(jump, rates)
        law = stationary['law']
        demand = np.asarray(self.marginal_demand@law).ravel()
        evaluated = evaluate_policy(self.grid, demand, alpha, self.params)
        best = solve_dp(self.grid, demand, self.params)
        conditional_regret = regret(alpha, evaluated['advantage'])
        value_difference = float(np.max(abs(best['value']-evaluated['value'])))
        p = self.params
        margins = p.R-p.c1-(p.c2-p.c1)*alpha[self.representatives]
        mean_reward = np.mean(self.game.demand*margins, axis=1)
        actual_value = np.zeros(self.game.size)
        for value_iterations in range(1, 10001):
            updated = mean_reward+p.gamma*(kernel@actual_value)
            value_residual = float(np.max(abs(updated-actual_value)))
            actual_value = updated
            if value_residual <= 1e-11:
                break
        else:
            raise ArithmeticError('Five-seller actual discounted payoff evaluation exceeded its budget')
        event = purchase_chain_statistics(self.grid, alpha, p)
        sales = np.asarray(self.sale_marginal@law).ravel()
        sale_rate = float(sales.sum())
        if sale_rate <= 0:
            raise ArithmeticError('Five-seller stationary own-sale rate is not positive')
        sale_law = sales/sale_rate
        sale_error = float(np.sum(abs(sale_law-event['sale_stationary'])))
        use = float(sale_law@alpha)
        outside = float(law@self.game.outside)
        market_error = abs(outside+5*sale_rate-1)
        reward_error = abs(float(law@mean_reward)-sale_rate*(p.R-p.c1-(p.c2-p.c1)*use))
        initial = int(self.game.orbit_index(np.full((1, 5), self.n//2, dtype=int))[0])
        stationary_summary = dict(stationary['summary'], symmetric_lumping=True,
            full_states=self.n**5, minimum_full_probability=float(np.min(law/self.game.orbit_sizes)),
            full_l1_residual_equals_quotient_l1_by_exchangeability=True,
            representation='Sorted five-reputation tuples, exact permutation quotient')
        diagnostics = dict(
            conditional_regret=conditional_regret, bellman_residual=evaluated['residual'],
            optimal_value_difference=value_difference, actual_value_residual=value_residual,
            actual_value_error_bound=value_residual/(1-p.gamma), actual_value_iterations=value_iterations,
            actual_value_scope='Mean of five seller payoffs; per-seller value at all-.5 initial state',
            sale_law_l1=sale_error, market_partition_error=market_error,
            seller_sale_rate_symmetry=0., seller_symmetry_method='Exact exchangeability of labels within each orbit',
            stationary_profit_accounting_error=reward_error,
            stationary=stationary_summary, sale_time=event,
            certificate_seconds=time.monotonic()-started)
        verified = bool(conditional_regret <= self.tolerance and evaluated['residual'] <= 1e-10
                        and value_difference <= 1e-5 and sale_error <= 1e-7
                        and market_error <= 1e-10 and reward_error <= 1e-10)
        summary = dict(p0=self.p0, rho=p.rho, n=self.n, sellers=5, closure='marginal',
            verified=verified, product2_conditional=use, sale_rate=sale_rate, outside_rate=outside,
            product2_calendar=sale_rate*use,
            profit_calendar=sale_rate*(p.R-p.c1-(p.c2-p.c1)*use),
            value_initial=float(actual_value[initial]),
            perceived_value_initial=float(evaluated['value'][self.n//2]))
        self._report(dict(stage='five-seller independent certificate finished', verified=verified,
                          regret=conditional_regret, elapsed_seconds=time.monotonic()-started))
        return dict(verified=verified, summary=summary, policy=alpha.copy(), grid=self.grid,
                    joint=law, representatives=self.representatives,
                    orbit_sizes=self.game.orbit_sizes, value=evaluated['value'],
                    actual_value=actual_value, demand=demand, advantage=evaluated['advantage'],
                    diagnostics=diagnostics)

    def solve(self, initial_policy=None, max_evaluations=500, progress=None):
        self._progress = progress
        try:
            return super().solve(initial_policy, max_evaluations=max_evaluations, progress=progress)
        finally:
            self._progress = None


__all__ = ['FiveSellerExperiment', 'checked_quotient_stationary', 'CERTIFICATION']
