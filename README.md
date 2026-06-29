# Dynamic Reputation with Hidden Product Choice

This repository studies a seller who privately chooses between a low-cost
product and a high-quality product while a user learns from binary outcomes.
The main analysis is the exact finite-horizon model with Thompson-sampling
demand.

## Main study: finite horizon

- [Model and simulation code](finite_horizon/main.py)
- [Reproduction instructions](finite_horizon/README.md)
- [Paper source](finite_horizon/paper/main.tex)
- [Compiled paper](finite_horizon/paper/main.pdf)
- [Main simulation plots](finite_horizon/outputs/plots/)
- [Forgetting-experiment plots](finite_horizon/outputs/plots_forgetting/)

The main long-horizon experiment uses \(T=2000\), retains policy summaries
through period 500, and simulates the first 1000 periods. Exact commands and a
smaller smoke test are provided in the finite-horizon README.

## Repository structure

```text
finite_horizon/
├── main.py
├── README.md
├── paper/
└── outputs/
    ├── plots/
    └── plots_forgetting/

other_experiments/
├── average_cost/
├── discounted/
└── finite_memory/
```

The secondary experiments are retained for completeness but are not required
to read or reproduce the main finite-horizon study. Large generated CSV files
and local archives are intentionally excluded from Git.
