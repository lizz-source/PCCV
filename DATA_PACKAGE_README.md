# Supplementary Data S1

This package contains frozen inputs, ordered identifiers, metadata, checkpoints, splits and numerical reference results. Extract Supplementary Code S1 and Supplementary Data S1 into the same directory, preserving their internal paths. The complete GitHub-ready archive already combines both.

After installing the pinned requirements, run `python scripts/check_release.py`, `python -m unittest discover -s tests -v`, and `python run_pipeline.py --mode verify`. The data-only inventory is DATA_MANIFEST.csv; the complete release also supplies source and combined inventories. Publication display tables and the complete illustrative carrier table are under publication/.

The software MIT license does not relicense third-party inputs. Read NOTICE.md and DATA_PROVENANCE.md. The computational scope begins at the frozen matrices; the complete historical sequencing and genome-annotation lineage has not been reconstructed. All dates, seeds, sample exclusions and replay distinctions are documented in the accompanying methods and repository guides.
