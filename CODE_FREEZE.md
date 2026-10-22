# Code freeze of the revision-confirmatory experiment

- Freeze commit (adds R1_CONFIRMATION_PROTOCOL.md and python/experiments/confirmation_analysis.py): `5dbaeaf53ddd53b0eeae203a163d5764d80c3279`
- Confirmation runs and analysis executed at commit `2fe2f8436766f307f7d5f61b5114d0dec074d0e1`. The only change between the two
  commits is the addition of two sweep drivers (accounting_check, endogeneity_check) to
  python/experiments/revision_sweeps.py; no policy, parameter, predictor, threshold or
  generative-model code changed.
- Confirmation results: data/results/confirmation.csv, confirmation_tests.csv,
  confirmation_equivalence.csv, confirmation_manifest.json.
- The freeze commit is also tagged `r1b-freeze`.
- After the confirmation, two ablation policies (`protection_predictive_route`,
  `protection_predictive_standby`) and the `p6_decomposition` sweep were added for an
  exploratory decomposition of P6 (data/results/p6_decomposition.csv). No existing policy was
  modified; P6 reproduces the confirmation runs exactly on all 128 seed-scenario pairs.
