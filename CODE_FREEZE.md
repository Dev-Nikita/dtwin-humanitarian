# Code freeze of the revision-confirmatory experiment

- Freeze commit (adds R1_CONFIRMATION_PROTOCOL.md and python/experiments/confirmation_analysis.py): `e1ab8207a2bf638ddde332345e8d691f1856d138`
- Confirmation runs and analysis executed at commit `8850528c55a6d5db1457ccb9c5a5b8be4a34b0f4`. The only change between the two
  commits is the addition of two sweep drivers (accounting_check, endogeneity_check) to
  python/experiments/revision_sweeps.py; no policy, parameter, predictor, threshold or
  generative-model code changed.
- Confirmation results: data/results/confirmation.csv, confirmation_tests.csv,
  confirmation_equivalence.csv, confirmation_manifest.json.
