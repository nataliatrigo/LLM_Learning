"""One entry point for forgetting experiments and state plots."""
from __future__ import annotations

import argparse
from importlib import import_module


# Import only the selected workflow; --help never starts a simulation.
COMMANDS = {
    "policy-summary": ("policy_summary", "main", "Tabulate K, policy thresholds and long-run state extrema"),
    "extend-initial": ("states", "extend_initial_main", "Run initial histories for additional p0 values"),
    "paper": ("run_experiments", "main", "Reproduce the paper experiments"),
    "monte-carlo": ("run_monte_carlo", "main", "Compare calendar-time simulations"),
    "paths": ("interaction_paths", "main", "Simulate individual interaction histories"),
    "grid": ("interaction_grid", "main", "Compare histories across parameter pairs"),
    "occupation": ("occupation_diagnostic", "main", "Study long-run state visitation"),
    "distributions": ("states", "distributions_main", "Refresh saved histograms with policy regions"),
    "initial": ("states", "initial_main", "Plot initial histories and visit counts"),
    "individual": ("states", "individual_main", "Plot a segment of a saved history"),
    "generate": ("states", "generate_main", "Simulate missing individual segments"),
    "overlap": ("states", "overlap_main", "Compare visited states with the policy"),
    "reachable": ("states", "reachable_main", "Plot the limiting reachable set"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(
            f"  {name:14} {description}"
            for name, (_, _, description) in COMMANDS.items()
        ) + "\n\nUse python -m forgetting COMMAND --help for its options.",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("options", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    module, function, _ = COMMANDS[args.command]
    getattr(import_module(f"forgetting.Code.experiments.{module}"), function)(args.options)


if __name__ == "__main__":
    main()
