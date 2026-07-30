"""Belief-to-demand rules used by the modular dynamic programs.

Discrete rules expose states as integer success/failure counts ``(S, F)``.
Forgetting rules expose the bounded normalized effective-count state ``(x,y)``
defined by ``x=(1-rho)(a-1)`` and ``y=(1-rho)(b-1)``.  This module contains no
Bellman logic; solvers can therefore share it without duplicating the learning
rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincc

from .primitives import SellerPrimitives


DiscreteState = tuple[int, int]
NormalizedState = tuple[float, float]
FloatArray = NDArray[np.float64]


def _positive_parameter(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _unit_interval_parameter(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _rho_parameter(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    return value


def _discrete_state(state: Sequence[int]) -> DiscreteState:
    if len(state) != 2:
        raise ValueError("a discrete state must be a pair (S, F)")
    successes, failures = state
    if isinstance(successes, (bool, np.bool_)) or isinstance(failures, (bool, np.bool_)):
        raise TypeError("S and F must be nonnegative integers")
    if int(successes) != successes or int(failures) != failures:
        raise TypeError("S and F must be nonnegative integers")
    successes, failures = int(successes), int(failures)
    if successes < 0 or failures < 0:
        raise ValueError("S and F must be nonnegative")
    return successes, failures


def _count_arrays(
    successes: ArrayLike, failures: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    successes_array, failures_array = np.broadcast_arrays(
        np.asarray(successes, dtype=float), np.asarray(failures, dtype=float)
    )
    if (
        np.any(~np.isfinite(successes_array))
        or np.any(~np.isfinite(failures_array))
        or np.any(successes_array < 0.0)
        or np.any(failures_array < 0.0)
    ):
        raise ValueError("S and F arrays must contain finite nonnegative values")
    return successes_array, failures_array


def _normalized_state(state: Sequence[float]) -> NormalizedState:
    if len(state) != 2:
        raise ValueError("a normalized state must be a pair (x, y)")
    x, y = float(state[0]), float(state[1])
    if not math.isfinite(x) or not math.isfinite(y) or x < 0.0 or y < 0.0:
        raise ValueError("x and y must be finite and nonnegative")
    # The transition maps preserve x+y <= 1.  Permit tiny floating-point
    # excursions so interpolators can evaluate a boundary successor safely.
    if x + y > 1.0 + 1e-12:
        raise ValueError("normalized states must satisfy x + y <= 1")
    return x, y


def _normalized_arrays(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    if (
        np.any(~np.isfinite(x_array))
        or np.any(~np.isfinite(y_array))
        or np.any(x_array < 0.0)
        or np.any(y_array < 0.0)
        or np.any(x_array + y_array > 1.0 + 1e-12)
    ):
        raise ValueError("normalized states require x >= 0, y >= 0, and x+y <= 1")
    return x_array, y_array


def _tail_probability(a: ArrayLike, b: ArrayLike, p0: float) -> FloatArray:
    probability = np.asarray(betaincc(a, b, p0), dtype=float)
    return np.clip(probability, 0.0, 1.0)


def _resolve_primitives(
    primitives: SellerPrimitives | None, p0: float | None
) -> SellerPrimitives:
    resolved = primitives or SellerPrimitives()
    if p0 is None:
        return resolved
    p0 = float(p0)
    if not math.isfinite(p0) or not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must lie in [0, 1]")
    if primitives is not None and not math.isclose(
        primitives.p0, p0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("p0 conflicts with primitives.p0")
    return replace(resolved, p0=p0)


class DiscreteBeliefDemand(ABC):
    """Common interface for integer success/failure learning rules."""

    name = "discrete belief demand"
    method = "discrete"
    parameter_name: str | None = None
    parameter_value: float | None = None
    has_exact_idle_self_loop = True

    def __init__(
        self,
        primitives: SellerPrimitives | None = None,
        *,
        p0: float | None = None,
    ) -> None:
        self.primitives = _resolve_primitives(primitives, p0)

    @property
    def p0(self) -> float:
        """Outside-option quality used by this demand rule."""

        return self.primitives.p0

    @abstractmethod
    def beta_parameters_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Parameters of the sampling distribution for count arrays."""

    def beta_parameters(self, state: Sequence[int]) -> tuple[float, float]:
        successes, failures = _discrete_state(state)
        a, b = self.beta_parameters_counts(successes, failures)
        return float(a), float(b)

    def demand_counts(self, successes: ArrayLike, failures: ArrayLike) -> FloatArray:
        a, b = self.beta_parameters_counts(successes, failures)
        return _tail_probability(a, b, self.primitives.p0)

    def demand_array(self, successes: ArrayLike, failures: ArrayLike) -> FloatArray:
        """Alias used by generic vectorized diagnostics."""

        return self.demand_counts(successes, failures)

    def demand(self, state: Sequence[int]) -> float:
        successes, failures = _discrete_state(state)
        return float(self.demand_counts(successes, failures))

    def demand_diagonal(self, n: int) -> FloatArray:
        if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 0:
            raise ValueError("n must be a nonnegative integer")
        successes = np.arange(int(n) + 1, dtype=float)
        return self.demand_counts(successes, int(n) - successes)

    def posterior_mean_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> FloatArray:
        """Mean of the Beta distribution used by this demand rule."""

        a, b = self.beta_parameters_counts(successes, failures)
        return np.asarray(a / (a + b), dtype=float)

    def effective_concentration_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> FloatArray:
        """Beta concentration ``a+b`` used to generate demand."""

        a, b = self.beta_parameters_counts(successes, failures)
        return np.asarray(a + b, dtype=float)

    def effective_sample_size_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> FloatArray:
        """Observation weight beyond this rule's initial sampling prior.

        This equals ``n`` for standard TS and epsilon-greedy, ``eta*n`` for
        scaled updates, and ``n/T`` for mean-preserving temperature.
        The separate effective-concentration field retains the prior weight.
        """

        concentration = self.effective_concentration_counts(successes, failures)
        initial_concentration = float(
            self.effective_concentration_counts(0.0, 0.0)
        )
        return np.asarray(concentration - initial_concentration, dtype=float)

    def success_state(self, state: Sequence[int]) -> DiscreteState:
        successes, failures = _discrete_state(state)
        return successes + 1, failures

    def failure_state(self, state: Sequence[int]) -> DiscreteState:
        successes, failures = _discrete_state(state)
        return successes, failures + 1

    def idle_state(self, state: Sequence[int]) -> DiscreteState:
        return _discrete_state(state)

    def state_coordinates(self, state: Sequence[int]) -> DiscreteState:
        return _discrete_state(state)

    def posterior_mean(self, state: Sequence[int]) -> float:
        successes, failures = _discrete_state(state)
        return float(self.posterior_mean_counts(successes, failures))

    def effective_concentration(self, state: Sequence[int]) -> float:
        successes, failures = _discrete_state(state)
        return float(self.effective_concentration_counts(successes, failures))

    def effective_sample_size(self, state: Sequence[int]) -> float:
        successes, failures = _discrete_state(state)
        return float(self.effective_sample_size_counts(successes, failures))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "state_type": "discrete_counts",
            "state_coordinates": ("S", "F"),
            "effective_sample_size_definition": (
                "sampling Beta concentration minus its concentration at S=F=0"
            ),
            "has_exact_idle_self_loop": self.has_exact_idle_self_loop,
            "primitives": self.primitives.as_dict(),
        }


class StandardThompsonSampling(DiscreteBeliefDemand):
    """Thompson sampling with the benchmark ``Beta(1,1)`` prior."""

    name = "Standard Thompson sampling"
    method = "standard_ts"

    def beta_parameters_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        successes_array, failures_array = _count_arrays(successes, failures)
        return 1.0 + successes_array, 1.0 + failures_array


class ScaledUpdates(DiscreteBeliefDemand):
    """Thompson sampling after multiplying each observation by ``eta``."""

    name = "Scaled updates"
    method = "scaled_updates"
    parameter_name = "eta"

    def __init__(
        self,
        eta: float,
        primitives: SellerPrimitives | None = None,
        *,
        p0: float | None = None,
    ) -> None:
        super().__init__(primitives, p0=p0)
        self.eta = _positive_parameter(eta, "eta")
        self.parameter_value = self.eta

    def beta_parameters_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        successes_array, failures_array = _count_arrays(successes, failures)
        return 1.0 + self.eta * successes_array, 1.0 + self.eta * failures_array


class MeanPreservingTemperature(DiscreteBeliefDemand):
    """Sampling temperature that preserves the standard posterior mean."""

    name = "Mean-preserving sampling temperature"
    method = "mean_preserving_temperature"
    parameter_name = "temperature"

    def __init__(
        self,
        temperature: float,
        primitives: SellerPrimitives | None = None,
        *,
        p0: float | None = None,
    ) -> None:
        super().__init__(primitives, p0=p0)
        self.temperature = _positive_parameter(temperature, "temperature")
        self.parameter_value = self.temperature

    def beta_parameters_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        successes_array, failures_array = _count_arrays(successes, failures)
        # a/T and b/T are algebraically equal to m*kappa/T and
        # (1-m)*kappa/T, while avoiding an unnecessary division by kappa.
        return (
            (1.0 + successes_array) / self.temperature,
            (1.0 + failures_array) / self.temperature,
        )


class EpsilonGreedy(DiscreteBeliefDemand):
    """Posterior-mean greedy choice with uniform random exploration."""

    name = "Epsilon-greedy"
    method = "epsilon_greedy"
    parameter_name = "epsilon"

    def __init__(
        self,
        epsilon: float,
        primitives: SellerPrimitives | None = None,
        tie_tolerance: float = 1e-12,
        *,
        p0: float | None = None,
    ) -> None:
        super().__init__(primitives, p0=p0)
        self.epsilon = _unit_interval_parameter(epsilon, "epsilon")
        self.parameter_value = self.epsilon
        self.tie_tolerance = float(tie_tolerance)
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and nonnegative")

    def beta_parameters_counts(
        self, successes: ArrayLike, failures: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        # These are the posterior parameters used to form the greedy mean;
        # epsilon-greedy does not draw from this Beta distribution.
        successes_array, failures_array = _count_arrays(successes, failures)
        return 1.0 + successes_array, 1.0 + failures_array

    def demand_counts(self, successes: ArrayLike, failures: ArrayLike) -> FloatArray:
        successes_array, failures_array = _count_arrays(successes, failures)
        posterior_mean = (successes_array + 1.0) / (
            successes_array + failures_array + 2.0
        )
        tie = np.isclose(
            posterior_mean,
            self.primitives.p0,
            rtol=0.0,
            atol=self.tie_tolerance,
        )
        greedy_demand = np.where(
            posterior_mean > self.primitives.p0,
            1.0 - self.epsilon / 2.0,
            self.epsilon / 2.0,
        )
        return np.asarray(np.where(tie, 0.5, greedy_demand), dtype=float)

    @property
    def metadata(self) -> dict[str, Any]:
        result = super().metadata
        result["tie_tolerance"] = self.tie_tolerance
        return result


class ForgettingThompsonSampling(ABC):
    """Common normalized-state interface for the two forgetting clocks."""

    name = "Forgetting Thompson sampling"
    method = "forgetting_ts"
    parameter_name = "rho"
    has_exact_idle_self_loop: bool

    def __init__(
        self,
        rho: float,
        primitives: SellerPrimitives | None = None,
        *,
        p0: float | None = None,
    ) -> None:
        self.rho = _rho_parameter(rho)
        self.parameter_value = self.rho
        self.primitives = _resolve_primitives(primitives, p0)

    @property
    def p0(self) -> float:
        """Outside-option quality used by this demand rule."""

        return self.primitives.p0

    def to_beta(self, state: Sequence[float]) -> tuple[float, float]:
        """Convert normalized effective counts ``(x,y)`` to ``(a,b)``."""

        x, y = _normalized_state(state)
        scale = 1.0 - self.rho
        return 1.0 + x / scale, 1.0 + y / scale

    def from_beta(self, state: Sequence[float]) -> NormalizedState:
        """Convert effective Beta parameters ``(a,b)`` to ``(x,y)``."""

        if len(state) != 2:
            raise ValueError("a Beta state must be a pair (a, b)")
        a, b = float(state[0]), float(state[1])
        if not math.isfinite(a) or not math.isfinite(b) or a < 1.0 or b < 1.0:
            raise ValueError("effective Beta parameters must be finite and at least one")
        normalized = ((1.0 - self.rho) * (a - 1.0), (1.0 - self.rho) * (b - 1.0))
        return _normalized_state(normalized)

    def beta_parameters_xy(
        self, x: ArrayLike, y: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        x_array, y_array = _normalized_arrays(x, y)
        scale = 1.0 - self.rho
        return 1.0 + x_array / scale, 1.0 + y_array / scale

    def demand_xy(self, x: ArrayLike, y: ArrayLike) -> FloatArray:
        a, b = self.beta_parameters_xy(x, y)
        return _tail_probability(a, b, self.primitives.p0)

    def demand_array(self, x: ArrayLike, y: ArrayLike) -> FloatArray:
        """Alias used by generic vectorized diagnostics."""

        return self.demand_xy(x, y)

    def demand(self, state: Sequence[float]) -> float:
        x, y = _normalized_state(state)
        return float(self.demand_xy(x, y))

    def success_state(self, state: Sequence[float]) -> NormalizedState:
        x, y = _normalized_state(state)
        innovation = 1.0 - self.rho
        return self.rho * x + innovation, self.rho * y

    def failure_state(self, state: Sequence[float]) -> NormalizedState:
        x, y = _normalized_state(state)
        innovation = 1.0 - self.rho
        return self.rho * x, self.rho * y + innovation

    @abstractmethod
    def idle_state(self, state: Sequence[float]) -> NormalizedState:
        """State following a calendar period in which seller B is selected."""

    def state_coordinates(self, state: Sequence[float]) -> NormalizedState:
        return _normalized_state(state)

    def posterior_mean(self, state: Sequence[float]) -> float:
        a, b = self.to_beta(state)
        return a / (a + b)

    def effective_concentration(self, state: Sequence[float]) -> float:
        a, b = self.to_beta(state)
        return a + b

    def effective_sample_size(self, state: Sequence[float]) -> float:
        """Effective number of observations, excluding the two prior counts."""

        x, y = _normalized_state(state)
        return (x + y) / (1.0 - self.rho)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "state_type": "normalized_effective_counts",
            "state_coordinates": ("x", "y"),
            "rho": self.rho,
            "has_exact_idle_self_loop": self.has_exact_idle_self_loop,
            "primitives": self.primitives.as_dict(),
        }


class ObservationForgettingTS(ForgettingThompsonSampling):
    """Forgetting applied only when seller A generates a new observation."""

    name = "Observation-time forgetting Thompson sampling"
    method = "observation_forgetting_ts"
    has_exact_idle_self_loop = True

    def idle_state(self, state: Sequence[float]) -> NormalizedState:
        return _normalized_state(state)


class CalendarForgettingTS(ForgettingThompsonSampling):
    """Forgetting applied in every calendar period, including idle periods."""

    name = "Calendar-time forgetting Thompson sampling"
    method = "calendar_forgetting_ts"
    has_exact_idle_self_loop = False

    def idle_state(self, state: Sequence[float]) -> NormalizedState:
        x, y = _normalized_state(state)
        return self.rho * x, self.rho * y


# A concise alias is useful in configurations while the descriptive class name
# remains the canonical public API.
StandardTS = StandardThompsonSampling


__all__ = [
    "CalendarForgettingTS",
    "DiscreteBeliefDemand",
    "EpsilonGreedy",
    "ForgettingThompsonSampling",
    "MeanPreservingTemperature",
    "ObservationForgettingTS",
    "ScaledUpdates",
    "StandardTS",
    "StandardThompsonSampling",
]
