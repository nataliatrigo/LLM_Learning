"""Simulate the solved two-seller game, one purchase at a time.

The solver works BACKWARD to calculate policies. This file moves FORWARD
through one possible history, using those policies at each state visited.
A path is one complete history of H purchases, starting from zero counts.
Multiple paths represent independent repetitions, not additional periods.

Solve and verify once, then simulate as many paths as needed. The horizon
stays fixed throughout each path: we do not solve a new H-period game after
every purchase or continue past the solved horizon.

This file uses Python's random module. A fixed seed reproduces the same
draws. Customer belief draws determine which seller is selected; the actual
product's success probability determines whether the outcome is a success.

Run the small example at the bottom with:
    .venv/bin/python two_seller_simulation.py

Or import the function into a Python session:
    from two_seller_game import solve_game, verify_equilibrium
    from two_seller_simulation import simulate_game

    solution = solve_game(horizon=5)
    verification = verify_equilibrium(solution)
    assert verification["passed"]
    results = simulate_game(solution, number_of_paths=100, seed=123)
    first_history = results["paths"][0]["history"]
    first_purchase = first_history[0]
"""

import random

from competition.two_seller_game import solve_game, verify_equilibrium


def simulate_game(solution, number_of_paths=1, seed=123):
    """Follow the supplied policies from the initial state through H purchases.

    Use the unchanged result of solve_game; verify it before calling this
    function if equilibrium play is required. Simulation does not itself
    establish equilibrium, and it does not modify the supplied solution.

    Each path records every purchase, the final state, and each seller's
    total discounted profit. Mean profits across many paths should approach
    solution['value'][seller][(0, 0, 0, 0)], with sampling error.
    All histories are kept in memory, so start with a small number of paths.
    """
    if (isinstance(number_of_paths, bool)
            or not isinstance(number_of_paths, int) or number_of_paths < 1):
        raise ValueError("Number of paths must be a positive integer.")

    random_draws = random.Random(seed)
    parameters = solution["parameters"]
    horizon = solution["horizon"]
    paths = []

    for path_number in range(1, number_of_paths + 1):
        state = (0, 0, 0, 0)
        history = []
        discounted_profit = {"A": 0.0, "B": 0.0}

        for elapsed_periods in range(horizon):
            successes_A, failures_A, successes_B, failures_B = state

            # STEP 1: The customer draws once from each current belief.
            draw_A = random_draws.betavariate(1 + successes_A, 1 + failures_A)
            draw_B = random_draws.betavariate(1 + successes_B, 1 + failures_B)

            # STEP 2: The seller with the higher draw gets the customer.
            if draw_A > draw_B:
                selected_seller = "A"
            elif draw_B > draw_A:
                selected_seller = "B"
            else:
                # Exact ties have probability zero mathematically. This handles
                # a possible numerical tie without favoring either seller.
                selected_seller = random_draws.choice(("A", "B"))

            # STEP 3: Look up the selected seller's policy at the current state.
            probability_product_2 = solution["policy"][selected_seller][state]
            if random_draws.random() < probability_product_2:
                chosen_product = 2
                cost = parameters["cost_2"]
                success_probability = parameters["success_2"]
            else:
                chosen_product = 1
                cost = parameters["cost_1"]
                success_probability = parameters["success_1"]

            # STEP 4: Generate the outcome using the ACTUAL product probability.
            # Do not use draw_A or draw_B as the outcome probability.
            success = random_draws.random() < success_probability

            # STEP 5: Only the selected seller's count changes.
            if selected_seller == "A":
                if success:
                    successes_A += 1
                else:
                    failures_A += 1
            else:
                if success:
                    successes_B += 1
                else:
                    failures_B += 1
            state_after = (successes_A, failures_A, successes_B, failures_B)

            # Revenue is paid on both success and failure. The other seller
            # earns zero. Discount by elapsed calendar periods, not selections.
            profit = {"A": 0.0, "B": 0.0}
            profit[selected_seller] = parameters["revenue"] - cost
            for seller in ("A", "B"):
                discounted_profit[seller] += (
                    parameters["discount"] ** elapsed_periods * profit[seller]
                )

            history.append({
                "purchase": elapsed_periods + 1,
                "purchases_remaining_before": horizon - elapsed_periods,
                "state_before": state,
                "demand_A": solution["demand_A"][state],
                "draw_A": draw_A,
                "draw_B": draw_B,
                "selected_seller": selected_seller,
                "probability_product_2": probability_product_2,
                "product": chosen_product,
                "success": success,
                "profit": profit,
                "state_after": state_after,
            })
            state = state_after

        paths.append({
            "path_number": path_number,
            "history": history,
            "final_state": state,
            "discounted_profit": discounted_profit,
        })

    mean_discounted_profit = {}
    for seller in ("A", "B"):
        total_profit = sum(path["discounted_profit"][seller] for path in paths)
        mean_discounted_profit[seller] = total_profit / number_of_paths

    return {
        "horizon": horizon,
        "seed": seed,
        "number_of_paths": number_of_paths,
        "paths": paths,
        "mean_discounted_profit": mean_discounted_profit,
    }


# This example runs only when you execute this file, not when you import it.
if __name__ == "__main__":
    solution = solve_game(horizon=5)
    verification = verify_equilibrium(solution)
    if not verification["passed"]:
        raise ValueError("The calculated policies did not pass verification.")

    results = simulate_game(solution, number_of_paths=1, seed=123)
    first_path = results["paths"][0]
    print("One history under the verified 5-purchase policies:")
    for purchase in first_path["history"]:
        outcome = "success" if purchase["success"] else "failure"
        print(
            f"Purchase {purchase['purchase']}: {purchase['state_before']} -> "
            f"seller {purchase['selected_seller']}, product {purchase['product']}, "
            f"{outcome} -> {purchase['state_after']}"
        )
    print("Discounted profits on this path:", first_path["discounted_profit"])
