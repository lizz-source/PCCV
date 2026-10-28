# Prepared release 1.0.0

- Retains the evaluated predictor, game and PCCV implementation for all reported inputs.
- Removes an unused local source-directory fallback and excludes the historical pilot importer.
- Adds explicit validation of PCCV's documented carrier count, budget and degree domain. Valid reported calls are numerically unchanged.
- Preserves the literal CSV family label null when reporting statistics, so the ten completely null games are excluded from the forty-game null-player summary. The allocation median mass fraction is 0.5381, replacing the earlier display value 0.3793.
- Adds publication figures, a deterministic real carrier example, and independent checks of display values and full-set ranks.
- Runs verification, analysis reproduction and retraining in isolated working copies, preserving the reference inputs, checkpoints and result tables.
- Adds MIT licensing for software, separate upstream-data notices, citation metadata, tests and a manual CI workflow.

No new biological experiment or real-data superiority result is introduced by release preparation. Full real-game PCCV rankings closely track permutation sampling, and incremental real-data PCCV gains remain unestablished.
