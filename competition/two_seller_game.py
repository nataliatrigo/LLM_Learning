"""Two strategic sellers: a readable finite-horizon game solver.

THE MODEL
---------
There is one customer purchase per period and two sellers, A and B.
Both sellers observe the complete state:

    state = (successes_A, failures_A, successes_B, failures_B)

The customer starts with independent Beta(1, 1) beliefs. At each purchase:
    1. She draws a quality estimate from each seller's current Beta belief.
    2. She selects the seller with the higher draw.
    3. Only that seller chooses product 1 or product 2, privately.
    4. The selected seller earns revenue minus the chosen product's cost.
    5. Success occurs with that product's actual success probability.
       Only the selected seller's success or failure count increases.

Product 2 costs more but succeeds more often. Revenue is received whether
the outcome is success or failure. The unselected seller earns zero now.
There is no forgetting and no outside option. Beliefs use observed outcomes;
the customer does not infer hidden product choices from sellers' strategies.
Both sellers discount profits every period, including when not selected.

WHAT WE SOLVE
-------------
Horizon H means H total purchases starting from (0, 0, 0, 0). At a state
with n observations, H - n purchases remain. Profits after H are zero.
Every transition adds one observation, so we can solve backward from H.
This is a finite-horizon equilibrium, not an infinite-horizon solution.

Notation in the returned dictionaries:
    value[seller][state]  = expected discounted profit BEFORE selection.
    policy[seller][state] = probability of product 2 IF selected.
    gain[seller][state]   = conditional payoff of product 2 minus product 1.

For a selected seller j and product i, the comparison is

    Q_j(i) = revenue - cost_i
             + discount * (p_i * V_j(after success)
                           + (1 - p_i) * V_j(after failure)).

Choose the product with the larger Q. Future values already incorporate
both sellers' future behavior. There is no alternating-policy-update loop.

HOW TO READ AND USE THIS FILE
----------------------------
Read the small functions in order, then solve_game, then best_response and
verify_equilibrium. The last two check the result without changing either
policy. Plain dictionaries and explicit loops favor readability over
speed. Start with H = 5: enumerating all states and integrating demand is
expensive at large H. No state approximation is made; integration and
arithmetic are numerical.

Example to type in a Python session (nothing runs automatically):

    from two_seller_game import default_parameters, solve_game, verify_equilibrium

    parameters = default_parameters()
    solution = solve_game(horizon=5, parameters=parameters)
    initial_state = (0, 0, 0, 0)
    solution["policy"]["A"][initial_state]   # 0 = product 1; 1 = product 2
    solution["value"]["A"][initial_state]
    solution["gain"]["A"][initial_state]

    verification = verify_equilibrium(solution)
    verification["passed"]  # True if both sellers pass at every decision state.
    verification["sellers"]["A"]["initial_improvement"]
    verification["sellers"]["A"]["maximum_improvement"]
    verification["sellers"]["A"]["state_of_maximum_improvement"]
    # The same results are available for B. Positive improvement means that
    # the seller can earn more by deviating from the calculated policy.
"""

from math import isfinite

from scipy.integrate import quad
from scipy.stats import beta


def default_parameters():
    """Return a fresh dictionary of the economic parameters; edit as needed."""
    return {
        "revenue": 1.0,
        "cost_1": 0.05,
        "cost_2": 0.65,
        "success_1": 0.35,
        "success_2": 0.80,
        "discount": 0.98,
    }


def states_with_total_observations(total):
    """List every nonnegative four-count state whose counts sum to total.

    Once the first three counts are chosen, the fourth is determined.
    We include unequal observation counts and all success/failure records.
    """
    states = []
    for successes_A in range(total + 1):
        for failures_A in range(total - successes_A + 1):
            remaining = total - successes_A - failures_A
            for successes_B in range(remaining + 1):
                failures_B = remaining - successes_B
                states.append((successes_A, failures_A, successes_B, failures_B))
    return states


def next_states(state):
    """The four possible states after exactly one purchase."""
    successes_A, failures_A, successes_B, failures_B = state
    return {
        "A_success": (successes_A + 1, failures_A, successes_B, failures_B),
        "A_failure": (successes_A, failures_A + 1, successes_B, failures_B),
        "B_success": (successes_A, failures_A, successes_B + 1, failures_B),
        "B_failure": (successes_A, failures_A, successes_B, failures_B + 1),
    }


def probability_customer_selects_A(state):
    """Compute P(A's Beta draw > B's Beta draw), without simulation.

    For a possible A draw x:
        density_A(x) * probability(B's draw < x)
    is the integrand. Integrating from 0 to 1 gives A's selection probability.
    SciPy's pdf is a density; its cdf is a cumulative probability.
    """
    successes_A, failures_A, successes_B, failures_B = state
    belief_A = beta(1 + successes_A, 1 + failures_A)
    belief_B = beta(1 + successes_B, 1 + failures_B)

    def integrand(x):
        return belief_A.pdf(x) * belief_B.cdf(x)

    probability, estimated_error = quad(
        integrand, 0.0, 1.0, epsabs=1e-11, epsrel=1e-11
    )
    if estimated_error > 1e-9 or not 0.0 <= probability <= 1.0:
        raise ArithmeticError("Demand integration was not sufficiently accurate.")
    return probability


def product_value(product, value_after_success, value_after_failure, parameters):
    """Expected profit from a product, conditional on this seller being selected.

    Both future values must belong to the seller whose choice we evaluate.
    Demand is absent here because selection has already occurred.
    """
    if product == 1:
        cost = parameters["cost_1"]
        success_probability = parameters["success_1"]
    elif product == 2:
        cost = parameters["cost_2"]
        success_probability = parameters["success_2"]
    else:
        raise ValueError("Product must be 1 or 2.")

    current_profit = parameters["revenue"] - cost
    expected_future_profit = (
        success_probability * value_after_success
        + (1 - success_probability) * value_after_failure
    )
    return current_profit + parameters["discount"] * expected_future_profit


def solve_game(horizon, parameters=None, tie_tolerance=1e-10):
    """Construct both sellers' finite-horizon policies by backward induction.

    At numerical indifference (absolute gain <= tie_tolerance), use each
    product with probability 1/2. Otherwise choose the better product.
    A numerical tie need not be an exact mathematical tie, so this rule can
    introduce a small optimality error. best_response measures that error.

    Returned dictionaries contain values at all states through H, and
    policies, gains and demand at all decision states (strictly before H).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")
    if parameters is None:
        parameters = default_parameters()
    parameters = parameters.copy()  # Keep the result separate from later edits.
    if not all(isfinite(number) for number in parameters.values()):
        raise ValueError("All parameters must be finite numbers.")
    if not 0 <= parameters["cost_1"] < parameters["cost_2"] < parameters["revenue"]:
        raise ValueError("Require 0 <= cost_1 < cost_2 < revenue.")
    if not 0 < parameters["success_1"] < parameters["success_2"] <= 1:
        raise ValueError("Require 0 < success_1 < success_2 <= 1.")
    if not 0 < parameters["discount"] < 1:
        raise ValueError("Discount must be strictly between 0 and 1.")
    if not isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative.")

    value = {"A": {}, "B": {}}
    policy = {"A": {}, "B": {}}
    gain = {"A": {}, "B": {}}
    demand_A = {}
    discount = parameters["discount"]

    # STEP 1: After H purchases, the game ends. Both future profits are zero.
    for state in states_with_total_observations(horizon):
        value["A"][state] = 0.0
        value["B"][state] = 0.0

    # STEP 2: Solve H-1 observations, then H-2, ..., finally zero observations.
    # Every successor has one MORE observation and has already been solved.
    for total in range(horizon - 1, -1, -1):
        for state in states_with_total_observations(total):
            following = next_states(state)
            demand_A[state] = probability_customer_selects_A(state)
            selection_probability = {"A": demand_A[state], "B": 1 - demand_A[state]}

            # STEP 2a: Find each seller's best product IF that seller is selected.
            # Calculate BOTH choices before evaluating either current value.
            for seller in ("A", "B"):
                success_state = following[seller + "_success"]
                failure_state = following[seller + "_failure"]
                value_after_success = value[seller][success_state]
                value_after_failure = value[seller][failure_state]

                payoff_1 = product_value(
                    1, value_after_success, value_after_failure, parameters
                )
                payoff_2 = product_value(
                    2, value_after_success, value_after_failure, parameters
                )
                gain[seller][state] = payoff_2 - payoff_1

                if gain[seller][state] > tie_tolerance:
                    policy[seller][state] = 1.0  # Always product 2.
                elif gain[seller][state] < -tie_tolerance:
                    policy[seller][state] = 0.0  # Always product 1.
                else:
                    policy[seller][state] = 0.5  # Numerical indifference.

            # STEP 2b: Given BOTH choices, evaluate profit BEFORE selection.
            for seller in ("A", "B"):
                probability_2 = policy[seller][state]
                expected_cost = (
                    (1 - probability_2) * parameters["cost_1"]
                    + probability_2 * parameters["cost_2"]
                )
                current_profit = selection_probability[seller] * (
                    parameters["revenue"] - expected_cost
                )

                expected_future_profit = 0.0
                for selected_seller in ("A", "B"):
                    probability_2 = policy[selected_seller][state]
                    success_probability = (
                        (1 - probability_2) * parameters["success_1"]
                        + probability_2 * parameters["success_2"]
                    )
                    success_state = following[selected_seller + "_success"]
                    failure_state = following[selected_seller + "_failure"]

                    # Even if B serves, A's future profit changes, and vice versa.
                    # Therefore these values use SELLER, not SELECTED_SELLER.
                    expected_future_profit += selection_probability[selected_seller] * (
                        success_probability * value[seller][success_state]
                        + (1 - success_probability) * value[seller][failure_state]
                    )

                value[seller][state] = current_profit + discount * expected_future_profit

    return {
        "horizon": horizon,
        "parameters": parameters,
        "tie_tolerance": tie_tolerance,
        "value": value,
        "policy": policy,
        "gain": gain,
        "demand_A": demand_A,
    }


def best_response(solution, seller):
    """Optimize one seller's entire policy against the opponent's fixed rule.

    This optional check does NOT change the solution or alternate updates.
    It builds new optimal values from zero, without reading saved values or
    the seller's prescribed policy. Only the opponent's policy is held fixed.
    The opponent still responds to the state, including after a deviation.

    Compare the returned values with solution['value'][seller]. Differences
    near zero at every state support finite-horizon equilibrium. This also
    checks randomized deviations: a mixture cannot beat the better product.
    """
    if seller not in ("A", "B"):
        raise ValueError("Seller must be 'A' or 'B'.")
    opponent = "B" if seller == "A" else "A"
    parameters = solution["parameters"]
    horizon = solution["horizon"]
    optimal_value = {}

    for state in states_with_total_observations(horizon):
        optimal_value[state] = 0.0

    for total in range(horizon - 1, -1, -1):
        for state in states_with_total_observations(total):
            following = next_states(state)
            probability_selected = solution["demand_A"][state]
            if seller == "B":
                probability_selected = 1 - probability_selected

            # If selected, optimize now AND use our own optimized future values.
            value_after_success = optimal_value[following[seller + "_success"]]
            value_after_failure = optimal_value[following[seller + "_failure"]]
            payoff_1 = product_value(
                1, value_after_success, value_after_failure, parameters
            )
            payoff_2 = product_value(
                2, value_after_success, value_after_failure, parameters
            )
            best_payoff_if_selected = max(payoff_1, payoff_2)

            # If the opponent is selected, we earn zero now, but play continues.
            opponent_probability_2 = solution["policy"][opponent][state]
            opponent_success_probability = (
                (1 - opponent_probability_2) * parameters["success_1"]
                + opponent_probability_2 * parameters["success_2"]
            )
            future_if_opponent_selected = (
                opponent_success_probability
                * optimal_value[following[opponent + "_success"]]
                + (1 - opponent_success_probability)
                * optimal_value[following[opponent + "_failure"]]
            )

            optimal_value[state] = (
                probability_selected * best_payoff_if_selected
                + (1 - probability_selected) * parameters["discount"]
                * future_if_opponent_selected
            )

    return optimal_value


def verify_equilibrium(solution, tolerance=1e-8):
    """Compare saved profits with best-response profits for BOTH sellers.

    Pass the unchanged result of solve_game. This function assumes its saved
    values correspond to its policies; it does not reevaluate edited policies.
    It calls best_response separately for A and B and checks every decision
    state, including states not visited along a particular path.

        improvement = best_response_profit - original_profit

    A positive improvement larger than tolerance means a profitable deviation.
    Small differences of either sign are allowed because of numerical error.
    A negative difference below -tolerance also fails: the best response should
    never earn less than the original policy, which is itself an available
    choice. Such a difference signals inconsistent values or numerical error.

    The returned report contains the initial improvement, largest improvement
    and its state, and the improvement at every decision state for each seller.
    'passed' is a finite-game numerical check, not an infinite-game guarantee.
    This function does not modify solution or run automatically in solve_game.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Verification tolerance must be finite and nonnegative.")

    report = {"passed": True, "tolerance": tolerance, "sellers": {}}
    initial_state = (0, 0, 0, 0)

    for seller in ("A", "B"):
        # Hold the opponent's policy fixed and optimize this seller's choices.
        best_possible_profit = best_response(solution, seller)
        improvement_by_state = {}
        seller_passed = True

        # Check all decision states, not only the initial state.
        for total in range(solution["horizon"]):
            for state in states_with_total_observations(total):
                original_profit = solution["value"][seller][state]
                improved_profit = best_possible_profit[state]
                improvement = improved_profit - original_profit
                improvement_by_state[state] = improvement

                if not isfinite(improvement) or abs(improvement) > tolerance:
                    seller_passed = False

        # Keep the signed differences: do not hide small negative roundoff.
        state_of_maximum = max(improvement_by_state, key=improvement_by_state.get)
        report["sellers"][seller] = {
            "passed": seller_passed,
            "states_checked": len(improvement_by_state),
            "initial_original_profit": solution["value"][seller][initial_state],
            "initial_best_response_profit": best_possible_profit[initial_state],
            "initial_improvement": improvement_by_state[initial_state],
            "maximum_improvement": improvement_by_state[state_of_maximum],
            "state_of_maximum_improvement": state_of_maximum,
            "minimum_improvement": min(improvement_by_state.values()),
            "improvement_by_state": improvement_by_state,
        }
        if not seller_passed:
            report["passed"] = False

    return report

if __name__ == "__main__":
    # Choose the total number of purchases.
    H = 5

    # Calculate the equilibrium policies and values.
    solution = solve_game(horizon=H)

    # Verify both sellers' best responses at every decision state.
    verification = verify_equilibrium(solution)
    print("Verification passed:", verification["passed"])

    # Display the policies and values at the initial state.
    initial_state = (0, 0, 0, 0)

    for seller in ("A", "B"):
        print(f"\nSeller {seller}")
        print("Probability of product 2:",
              solution["policy"][seller][initial_state])
        print("Expected discounted profit:",
              solution["value"][seller][initial_state])