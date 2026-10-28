# Local release validation

`release_validation.json` records the fresh final-entry-point verification and five invariant tests. The run used an isolated working copy with the declared pinned CPU dependencies already available in the local environment. Fifty checkpoint reload / complete real-case checks passed. Verify mode does not retrain the real predictors; the separate same-seed replay is an archived and rechecked artifact.

Fresh hardware-dependent timings are execution checks, not replacements for the archived paper timing rows. `reference_snapshot_comparison.json` identifies the reporting-only differences from the immutable experimental archive. The manual GitHub Actions workflow has been prepared but was not run remotely during release preparation.
