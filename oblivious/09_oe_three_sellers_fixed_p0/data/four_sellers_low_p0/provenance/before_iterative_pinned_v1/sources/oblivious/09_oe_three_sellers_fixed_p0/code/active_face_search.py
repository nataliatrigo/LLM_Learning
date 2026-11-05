"""Optional active-face proposals with exact selected-column derivatives.

This is not a production entrypoint.  It changes only exploratory equilibrium
search: the finite kernel, marginal beliefs, linear equations and inherited
full equilibrium certificate are unchanged.  A caller that later uses this
helper must install and fingerprint the campaign's hybrid stationary checker.
Every returned solution passes ``self.certify`` on the complete state grid.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.policy import evaluate_policy
from .model import regret
from .search_only import FastExperiment, _LazyJacobian
from .stationary import checked_stationary, jump_system
from .run_experiment import ROOT


def source_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()}


class ActiveFaceFailure(RuntimeError):
    """An explicit search failure; no uncertified proposal is returned."""

    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


class _ExpandFace(Exception):
    def __init__(self, policy, indices):
        self.policy, self.indices = policy.copy(), [int(k) for k in indices]


def _columns(indices, n):
    values = list(indices)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
           or not 0 <= k < n for k in values):
        raise ValueError('Active columns must be integer grid indices')
    if len(values) != len(set(values)):
        raise ValueError('Active columns must be distinct')
    return np.asarray(sorted(values), dtype=int)


def incentive_violations(policy, advantage):
    """All-node own-sale regret, including nodes fixed outside the face."""
    return np.maximum(advantage, 0) - policy*advantage


class ActiveFaceExperiment(FastExperiment):
    """A frozen experiment with a separate optional active-face search API."""

    def calculate_active(self, alpha, columns):
        """Return full advantages and an exact n-by-k analytic Jacobian.

        The pinned stationary system and chain rule are the frozen search's
        equations.  Only the requested k transition-derivative RHS are formed
        and solved; no full n-column Jacobian is materialized underneath.
        """
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError('Policy must be finite and match the grid')
        alpha = np.clip(alpha, 0, 1)
        columns = _columns(columns, self.n)
        if self.sellers == 1:
            raise ValueError('Monopoly uses the inherited direct DP')
        previous = getattr(self, '_active_context', None)
        if previous is not None and np.array_equal(previous['alpha'], alpha):
            context = previous
        else:
            if previous is not None:
                previous['factor']['lu'] = None
            self.calls += 1
            full = self.game.transition(alpha)
            reduced = (full[self.representatives] @ self.membership).tocsr()
            jump, rates = jump_system(reduced)
            size, anchor = len(rates), len(rates)//2
            keep = np.arange(size) != anchor
            transpose = jump.T.tocsr()
            linear = (eye(size, format='csr')-transpose)[keep, :][:, keep].tocsc()
            lu = splu(linear, permc_spec='MMD_AT_PLUS_A')
            w = np.empty(size)
            w[anchor] = 1.
            w[keep] = lu.solve(np.asarray(transpose[keep, anchor].toarray()).ravel())
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                checked = checked_stationary(jump, rates)
                log_w = np.log(checked['law'])+np.log(rates)
                w = np.exp(log_w-log_w.max())
            w /= w.sum()
            if np.any(w <= 0) or np.any(~np.isfinite(w)):
                raise ArithmeticError('Pinned active-face invariant lost positive mass')
            log_q = np.log(w)-np.log(rates)
            q = np.exp(log_q-log_q.max())
            q_sum = q.sum()
            calendar = q/q_sum
            if np.any(calendar <= 0) or np.any(~np.isfinite(calendar)):
                raise ArithmeticError('Pinned active-face calendar law lost positive mass')
            mu = np.asarray(self.own_marginal @ calendar).ravel()
            demand = self.marginal_demand @ calendar
            if self.closure == 'conditional':
                demand = np.asarray(self.sale_marginal @ calendar).ravel()/mu
            if np.any(~np.isfinite(demand)) or np.any(demand < -1e-12) or np.any(demand > 1+1e-12):
                raise ArithmeticError('Invalid active-face belief demand')
            demand = np.clip(demand, 0, 1)
            evaluation = evaluate_policy(self.grid, demand, alpha, self.params)
            context = dict(alpha=alpha.copy(), jump=jump, rates=rates, linear=linear,
                keep=keep, q=q, q_sum=q_sum, calendar=calendar, mu=mu, demand=demand,
                value=evaluation['value'], advantage=evaluation['advantage'],
                factor={'lu': lu}, results={})
            self._active_context = context
        key = tuple(columns)
        if key in context['results']:
            return context['results'][key]

        def jacobian():
            k = len(columns)
            if k == 0:
                return np.empty((self.n, 0))
            rates, jump, q = context['rates'], context['jump'], context['q']
            derivative_rhs = np.empty((len(rates), k))
            for j, column in enumerate(columns):
                derivative_rhs[:, j] = (self.derivatives[column].T @ q
                    - jump.T @ (q*self.rate_derivatives[:, column]))
            factor = context['factor']['lu']
            if factor is None:
                factor = splu(context['linear'], permc_spec='MMD_AT_PLUS_A')
            dw_scaled = np.zeros((len(rates), k))
            dw_scaled[context['keep']] = factor.solve(derivative_rhs[context['keep']])
            relative_dr = self.rate_derivatives[:, columns]/rates[:, None]
            dc_raw = dw_scaled/rates[:, None]-q[:, None]*relative_dr
            dc = (dc_raw/context['q_sum']
                  - context['calendar'][:, None]*(dc_raw.sum(axis=0)/context['q_sum']))
            demand, alpha = context['demand'], context['alpha']
            dd = self.marginal_demand @ dc
            if self.closure == 'conditional':
                dd = (self.sale_marginal @ dc-demand[:, None]*(self.own_marginal @ dc))/context['mu'][:, None]
            p = self.params
            probability = p.p1+(p.p2-p.p1)*alpha
            selected = diags(probability) @ self.S+diags(1-probability) @ self.F
            own = diags(1-demand)+diags(demand) @ selected
            own_lu = splu((eye(self.n, format='csr')-p.gamma*own).tocsc())
            value, advantage = context['value'], context['advantage']
            flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected @ value-value)
            direct = np.zeros((self.n, k))
            direct[columns, np.arange(k)] = demand[columns]*advantage[columns]
            dv = own_lu.solve(flow[:, None]*dd+direct)
            result = p.gamma*(p.p2-p.p1)*((self.S-self.F) @ dv)
            if np.any(~np.isfinite(result)):
                raise ArithmeticError('Active-face analytic Jacobian is not finite')
            return result

        result = _LazyJacobian(dict(policy=context['alpha'].copy(),
            advantage=context['advantage'], value=context['value'], demand=context['demand'],
            reduced_law=context['calendar'], regret=regret(context['alpha'], context['advantage']),
            jacobian_columns=columns.copy()), jacobian,
            lambda: context['factor'].update(lu=None))
        context['results'][key] = result
        return result

    def solve_active_face(self, initial_policy, active_columns, *, max_evaluations=100,
                          max_expansions=None, progress=None):
        """Search bounded faces, expand on any outside violation, fully certify.

        The face is a proposal device, not a restriction on the equilibrium:
        every outside incentive is checked at every evaluation.  A violation
        adds that node and restarts the bounded root.  Numerical failures and
        exhausted budgets raise ``ActiveFaceFailure`` with their history.
        """
        alpha = np.asarray(initial_policy, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError('Initial policy must be finite and match the grid')
        alpha = np.clip(alpha, 0, 1)
        active = _columns(active_columns, self.n)
        if max_evaluations < 1 or (max_expansions is not None and max_expansions < 0):
            raise ValueError('Use positive evaluation and nonnegative expansion budgets')
        started_calls, expansions, history = self.calls, 0, []

        def note(event):
            history.append(event)
            if progress:
                progress(event)

        try:
            while True:
                fixed = alpha.copy()

                def evaluate(raw):
                    candidate = fixed.copy()
                    candidate[active] = raw
                    context = getattr(self, '_active_context', None)
                    cached = context is not None and np.array_equal(context['alpha'], candidate)
                    if not cached and self.calls-started_calls >= max_evaluations:
                        raise ActiveFaceFailure('Active-face evaluation budget exhausted', history)
                    before_calls = self.calls
                    data = self.calculate_active(candidate, active)
                    if self.calls > before_calls:
                        note(dict(stage='active-face evaluation', calls=self.calls-started_calls,
                                  active=active.tolist(), alpha_active=data['policy'][active].tolist(),
                                  full_regret=float(data['regret'])))
                    violations = incentive_violations(data['policy'], data['advantage'])
                    outside = np.ones(self.n, dtype=bool)
                    outside[active] = False
                    missing = np.flatnonzero(outside & (violations > self.tolerance))
                    if len(missing):
                        raise _ExpandFace(data['policy'], missing)
                    return data

                try:
                    data = evaluate(alpha[active])
                    if data['regret'] <= self.tolerance:
                        result = self.certify(data['policy'])
                        if not result['verified']:
                            raise ActiveFaceFailure('Full inherited certificate rejected the proposal', history)
                        note(dict(stage='full inherited certificate', active=active.tolist(),
                                  regret=data['regret'], calls=self.calls-started_calls))
                        result.update(history=history, search_calls=self.calls-started_calls)
                        return result

                    def equations(raw):
                        current = evaluate(raw)
                        return raw-np.clip(raw+current['advantage'][active], 0, 1)

                    def jacobian(raw):
                        current = evaluate(raw)
                        da = current['jacobian'][active, :]
                        shifted = raw+current['advantage'][active]
                        inside = (shifted > 0) & (shifted < 1)
                        identity = np.eye(len(active))
                        return identity-inside[:, None]*(identity+da)

                    solved = least_squares(equations, alpha[active], jac=jacobian,
                        bounds=(0, 1), method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-15,
                        max_nfev=max(1, max_evaluations-(self.calls-started_calls)))
                    data = evaluate(solved.x)
                    if data['regret'] > self.tolerance:
                        # Natural-map residuals can be locally flat when an
                        # action strictly favors a boundary. Try that exact
                        # best-response endpoint once; all outside incentives
                        # and the complete final certificate still apply.
                        proposal = data['policy'].copy()
                        violations = incentive_violations(proposal, data['advantage'])
                        improve = active[violations[active] > self.tolerance]
                        proposal[improve] = (data['advantage'][improve] >= 0).astype(float)
                        if not np.array_equal(proposal, data['policy']):
                            data = evaluate(proposal[active])
                            note(dict(stage='active-face pure boundary proposal',
                                      active=active.tolist(), regret=data['regret'],
                                      calls=self.calls-started_calls))
                    note(dict(stage='bounded active-face root', active=active.tolist(),
                              regret=data['regret'], calls=self.calls-started_calls,
                              success=bool(solved.success)))
                    alpha = data['policy'].copy()
                    if data['regret'] > self.tolerance:
                        raise ActiveFaceFailure('Bounded face root left an incentive violation', history)
                except _ExpandFace as expand:
                    if max_expansions is not None and expansions >= max_expansions:
                        raise ActiveFaceFailure('Active-face expansion budget exhausted', history)
                    active = np.union1d(active, expand.indices)
                    alpha = expand.policy.copy()
                    expansions += 1
                    note(dict(stage='expand active face', added=expand.indices,
                              active=active.tolist(), calls=self.calls-started_calls))
        except ActiveFaceFailure:
            raise
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise ActiveFaceFailure('Active-face search failed: '+str(exc), history) from exc


__all__ = ['ActiveFaceExperiment', 'ActiveFaceFailure', 'incentive_violations', 'source_reference']
