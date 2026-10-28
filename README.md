# Efficient and Verifiable Carrier-Level Shapley Attribution for Pathway Sensitivity in Microbial Composition Models

This repository accompanies **Calibrated carrier attribution for pathway sensitivity in microbial composition models**. It contains the evaluated predictors, carrier deletion games, the probe-calibrated control variate estimator (PCCV), exact references and publication figure code.

The release starts from frozen abundance and annotation matrices. It does not reconstruct the complete historical raw-sequencing and genome-annotation workflow. See [data provenance](DATA_PROVENANCE.md), [methods](docs/METHODS.md) and [interpretation limits](docs/LIMITATIONS.md).

## Quick verification

Use Python 3.12 and the pinned CPU dependencies. On Windows, commands can use the virtual-environment interpreter directly, without changing PowerShell execution policy:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.venv\Scripts\python.exe scripts/check_release.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe run_pipeline.py --mode verify
```

On Linux with a compatible CPU environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python scripts/check_release.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_pipeline.py --mode verify
```

Verify reloads seven real-data models and forty controlled predictors, recomputes held-out metrics and three complete real attribution cases, and checks grouped/dense probability and gradient equivalence. It does not fit the real predictors again. Five fast tests independently check exact subset sums, null players, query accounting, the supported argument domain and dense/grouped probabilities.

Pipeline modes create a new working copy under `runs/` and leave the reference inputs, checkpoints and outputs intact. The printed `run_summary.json` contains pass/fail results, dependency versions and measured stage durations. An explicit `--output-dir` must name a new or empty directory. Runtime directories and local dependencies are ignored by Git.

## Reproduce and retrain

```text
python run_pipeline.py --mode reproduce
python run_pipeline.py --mode retrain
```

Reproduce reruns the controlled benchmarks, attribution comparisons, complete real-case analyses and sensitivity checks using archived real predictors. It also executes the controlled predictor fitting stage. Retrain additionally fits the six internal real predictors, one entire-study holdout predictor and the separate same-seed replay before the analyses. These modes take longer than verification. No duration or bitwise equality is guaranteed across hardware.

The reported same-seed replay is distinct from an independent training seed. Internal prediction comparisons use three training seeds; the entire-study holdout has one seed. Endpoint comparisons across models are not full cross-model PCCV experiments.

## Publication figures and real example

The ten final figures are in [publication/figures](publication/figures), with 600 dpi PNG and editable SVG. The [figure number map](publication/figure_number_map.json) relates source filenames to manuscript figures. To redraw from the archived reference tables:

```text
python publication/publication_figures.py
python publication/verify_display_values.py
```

Install Times New Roman for publication rendering. On Windows, the script checks the standard Fonts directory. On other systems it uses the font manager; `PCCV_TNR_FONT_DIR` can identify a directory containing `times.ttf`, `timesbd.ttf`, `timesi.ttf` and `timesbi.ttf`. The script does not silently substitute another family.

Figure display tables are in [publication/figure_data](publication/figure_data). The [complete real example](publication/case_study/Case_Study_S1.csv) gives all 78 carriers in CM.29_WGS and ko02020. Sample and pathway follow the first entries in the frozen plan; displayed species follow a disclosed rank-based rule. Both permutation sampling and PCCV estimates are retained, including negative contributions. This is a descriptive model example, not an additional biological validation.

Publication figures use the archived timing rows underlying the manuscript. Fresh verify timings are execution checks and may differ. The reporting reader preserves the literal family label `null`; this corrects the historical null-player summary to the forty nonnull mechanisms. See [change history](CHANGELOG.md).

## Contents and integrity

- `src/`: model, games, estimators, controlled experiments, audits and reporting.
- `inputs/`: frozen matrices, metadata and two historical provenance/implementation artifacts.
- `results/`, `study_holdout/results/`, `replay/results/`: reference checkpoints, partitions and individual results.
- `publication/`: final figure code, source display tables and complete example.
- `docs/`: method, protocol, reproducibility and scope notes.
- `verification/`: the local release-validation report. It is not a claim that GitHub CI has run.
- `DATA_MANIFEST.csv`, `SOURCE_MANIFEST.csv`, `RELEASE_MANIFEST.csv`: SHA-256 inventories.

`.gitattributes` disables line-ending conversion to preserve manifest bytes across Git checkouts. After an intentional source or data change, document the change and regenerate manifests with `python scripts/build_manifest.py`. A change to reference data establishes a different computational snapshot.

The manual GitHub Actions workflow installs the requirements and runs integrity checks, tests and verify mode on Ubuntu. It is provided for the repository owner to run after upload; the preparation run was performed locally on Windows CPU with the declared dependencies, not on a new remote machine.

## Software and data terms

Original software is distributed under the [MIT license](LICENSE). Data and third-party materials retain their applicable source terms; the software license does not relicense upstream study data. Read [NOTICE.md](NOTICE.md) and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) before redistributing or reusing those materials. Dependencies are installed separately and retain their own licenses.

Use [CITATION.cff](CITATION.cff) to identify this prepared software version. No publication DOI, public repository URL or completed journal acceptance is asserted. The owners can add the final repository URL and publication citation after those records exist.
