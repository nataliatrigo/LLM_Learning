# Dynamic Reputation with Hidden Product Choice

This repository studies a seller who privately chooses between a low-cost
product and a high-quality product while a user learns from binary outcomes.
The active work is organized by model family. Generated outputs stay inside
each study folder so code, paper files, and figures do not get mixed together.

## Start Here

The two current analysis tracks are:

- [Finite-horizon study](finite_horizon/): original exact DP, simulations, and
  paper.
- [Discounted study](discounted/): discounted exact DP, fluid approximation,
  and discounted paper.

Secondary or older experiments are collected under
[other_experiments/](other_experiments/) or documented as such in their own
folder.

## Repository structure

```text
finite_horizon/
  main.py                  finite-horizon exact DP and simulations
  README.md                exact reproduction commands
  paper/                   manuscript source and compiled PDF
  outputs/                 generated figures and local data

discounted/
  README.md                map of the discounted study
  Paper/                   discounted manuscript
  DP/                      exact discounted finite-horizon DP
  Fluid/                   deterministic fluid solver and reparameterization
  BinaryBelief-Logit/      one-dimensional logit-demand experiment
  BinaryBelief/            older binary-belief diagnostics

other_experiments/
  README.md
  average_cost/
  finite_memory/
  finite_horizon_archives/
```

## Generated Files

Output folders are reproducible by running the commands in each study README.
Large data tables, caches, and local archives are ignored by Git. Curated
figures and compiled PDFs may be kept when they are useful for writing or
sharing.
