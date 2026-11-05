"""Compare finite-horizon equilibrium policies at the SAME early states.

Example: solve H = 3, 5, 7 and compare every state with at most 2 observations.
The comparison states stay fixed as H grows. At state (1, 0, 0, 1), for
example, H=3 leaves 1 purchase and H=7 leaves 5 purchases. We ask whether
moving the ending date farther away changes today's equilibrium decision.

We compare BOTH sellers' policies, values, and product-2 gains. A policy can
stay unchanged while its underlying incentive changes substantially. Saved
gain values also help distinguish switches near indifference from larger
changes in incentives.

This is a numerical stability diagnostic on a finite set of states. Agreement
across finitely many horizons does not prove convergence to an infinite-game
equilibrium, nor does it describe behavior at states outside the checked set.
No simulated paths are used. The underlying solver still solves every state
before each H; only the comparison and retained results are limited to early
states. Start small with the readable solver.

Run the example:
    .venv/bin/python two_seller_horizon_comparison.py

Or call from another file or Python session:
    from two_seller_horizon_comparison import compare_horizons
    results = compare_horizons(horizons=[3, 5, 7], max_observations=2)
    results["comparisons"]
    results["snapshots"][7]["A"][(0, 0, 0, 0)]

The existing one-horizon example in two_seller_game.py remains separate.
"""

from math import isfinite

from competition.two_seller_game import (
    default_parameters,
    solve_game,
    states_with_total_observations,
    verify_equilibrium,
)


def compare_horizons(horizons, max_observations=2, parameters=None,
                     policy_tolerance=1e-8, value_tolerance=1e-6,
                     gain_tolerance=1e-6):
    """Solve increasing horizons and compare adjacent ones at fixed states.

    max_observations defines the comparison set: all four-count states with
    total counts from zero through this number, inclusive. It must be smaller
    than the smallest H, so every comparison state is a decision state.

    A comparison is 'numerically_stable' when all three maximum absolute
    changes (policy, value, gain) are within their specified tolerances.
    'consecutive_stable_comparisons' counts successive passing comparisons
    at the END of the sequence. It resets to zero whenever a comparison fails.
    This count is evidence to inspect, not an automatic convergence proof.

    Policies give probabilities of product 2, including 0.5 at numerical
    indifference. We compare these probabilities directly. Each change record
    identifies a seller and a state and includes the old and new incentives.
    """
    horizons = list(horizons)
    if len(horizons) < 2:
        raise ValueError("Supply at least two horizons to compare.")
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Every horizon must be a positive integer.")
    for index in range(1, len(horizons)):
        if horizons[index] <= horizons[index - 1]:
            raise ValueError("Horizons must be strictly increasing.")
    if (isinstance(max_observations, bool)
            or not isinstance(max_observations, int)
            or not 0 <= max_observations < horizons[0]):
        raise ValueError("max_observations must be an integer from 0 to min(H)-1.")
    for tolerance in (policy_tolerance, value_tolerance, gain_tolerance):
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("Comparison tolerances must be finite and nonnegative.")
    if parameters is None:
        parameters = default_parameters()
    parameters = parameters.copy()

    # STEP 1: Choose ONE set of exact states to use for every horizon.
    comparison_states = []
    for total in range(max_observations + 1):
        comparison_states.extend(states_with_total_observations(total))

    snapshots = {}
    comparisons = []
    previous_horizon = None
    consecutive_stable_comparisons = 0

    for horizon in horizons:
        # STEP 2: Solve and verify this finite game, keeping the same parameters
        # and the solver's same default numerical tie convention at every H.
        solution = solve_game(horizon=horizon, parameters=parameters)
        verification = verify_equilibrium(solution)
        if not verification["passed"]:
            raise ArithmeticError(f"The equilibrium check failed for H={horizon}.")

        # Retain only the fixed comparison states, rather than every solution.
        snapshot = {"A": {}, "B": {}}
        for seller in ("A", "B"):
            for state in comparison_states:
                snapshot[seller][state] = {
                    "policy": solution["policy"][seller][state],
                    "value": solution["value"][seller][state],
                    "gain": solution["gain"][seller][state],
                }
        snapshots[horizon] = snapshot
        del solution, verification

        # There is nothing to compare until we have solved two horizons.
        if previous_horizon is None:
            previous_horizon = horizon
            continue

        # STEP 3: Compare each seller at each identical state across horizons.
        previous_snapshot = snapshots[previous_horizon]
        maximum_policy_change = 0.0
        maximum_value_change = 0.0
        maximum_gain_change = 0.0
        policy_changes = []
        changed_states = set()  # Count each state once, even if both sellers change.

        for seller in ("A", "B"):
            for state in comparison_states:
                old = previous_snapshot[seller][state]
                new = snapshot[seller][state]
                policy_change = abs(new["policy"] - old["policy"])
                value_change = abs(new["value"] - old["value"])
                gain_change = abs(new["gain"] - old["gain"])

                maximum_policy_change = max(maximum_policy_change, policy_change)
                maximum_value_change = max(maximum_value_change, value_change)
                maximum_gain_change = max(maximum_gain_change, gain_change)

                if policy_change > policy_tolerance:
                    changed_states.add(state)
                    policy_changes.append({
                        "seller": seller,
                        "state": state,
                        "remaining_purchases_from": previous_horizon - sum(state),
                        "remaining_purchases_to": horizon - sum(state),
                        "policy_from": old["policy"],
                        "policy_to": new["policy"],
                        "gain_from": old["gain"],
                        "gain_to": new["gain"],
                        "value_from": old["value"],
                        "value_to": new["value"],
                    })

        numerically_stable = (
            maximum_policy_change <= policy_tolerance
            and maximum_value_change <= value_tolerance
            and maximum_gain_change <= gain_tolerance
        )
        if numerically_stable:
            consecutive_stable_comparisons += 1
        else:
            consecutive_stable_comparisons = 0

        comparisons.append({
            "H_from": previous_horizon,
            "H_to": horizon,
            "states_checked": len(comparison_states),
            "changed_states": len(changed_states),
            "changed_seller_state_pairs": len(policy_changes),
            "maximum_policy_change": maximum_policy_change,
            "maximum_value_change": maximum_value_change,
            "maximum_gain_change": maximum_gain_change,
            "numerically_stable": numerically_stable,
            "policy_changes": policy_changes,
        })
        previous_horizon = horizon

    return {
        "horizons": horizons,
        "parameters": parameters,
        "max_observations": max_observations,
        "comparison_states": comparison_states,
        "policy_tolerance": policy_tolerance,
        "value_tolerance": value_tolerance,
        "gain_tolerance": gain_tolerance,
        "snapshots": snapshots,
        "comparisons": comparisons,
        "consecutive_stable_comparisons": consecutive_stable_comparisons,
    }


if __name__ == "__main__":
    results = compare_horizons(horizons=[3, 5, 7], max_observations=2)

    print("Comparing the same states with 0, 1, or 2 observations.")
    for comparison in results["comparisons"]:
        print(f"\nH={comparison['H_from']} -> H={comparison['H_to']}")
        print("States where at least one seller's policy changed:",
              comparison["changed_states"], "of", comparison["states_checked"])
        print("Largest absolute value change:", comparison["maximum_value_change"])
        print("Largest absolute product-2 gain change:", comparison["maximum_gain_change"])
        print("Numerically stable under the chosen tolerances:",
              comparison["numerically_stable"])

        # Every switch is retained in policy_changes; display just the first five.
        for change in comparison["policy_changes"][:5]:
            print(f"  Seller {change['seller']}, state {change['state']}: "
                  f"P(product 2) {change['policy_from']} -> {change['policy_to']}")

    print("\nConsecutive stable comparisons at the end:",
          results["consecutive_stable_comparisons"])
    print("Finite-state numerical stability does not prove infinite-game convergence.")
