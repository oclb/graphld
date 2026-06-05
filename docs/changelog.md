# Changelog

This changelog starts from `v1.2.0`. The `Unreleased` section tracks changes intended for the next release.

## Unreleased

### Changed

- Made annotation-loading tests and tracked examples self-contained by using explicit `data/test/rsid_position.csv` fixture paths instead of ignored root data files.
- Expanded `data/test/rsid_position.csv` to include allele columns matching the default positions-file shape expected by annotation loading.
- Updated `load_annotations()` so position-only calls can use three-column positions files, while allele columns are required only when `add_alleles=True`.
- Fixed `load_annotations()` to respect caller-supplied `file_pattern` values.
- Added explicit `Any` annotations and return types to multiprocessing hooks and public wrapper functions across graphREML, BLUP, clumping, simulation, and score-test gene annotation utilities.
- Simplified public wrapper docstrings for `run_graphREML`, `run_blup`, and `run_clump` so they point readers to the underlying implementation signatures instead of duplicating stale argument lists.
- Cleaned docstrings in allele merging, annotation loading, multiprocessing, and gene-to-variant annotation conversion utilities for clearer generated API documentation.

### Removed

- Removed internal release review notes from the public docs tree.
- Cleared legacy data-download helper content from the tracked `data/` helper files as part of data setup cleanup.

## v1.2.0 - 2026-05-22

Initial baseline for this changelog.
