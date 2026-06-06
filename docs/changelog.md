# Changelog

This changelog starts from `v1.2.0`.

## v1.2.1 - 2026-06-06

### Added

- Added score-test support for variant-level `.bed` annotation directories, including BED-only directories that use score-stat HDF5 row data for variant coordinates.
- Added `estest test --perturb-annot` to the wrapped score-test CLI.
- Exported `read_parquet_sumstats_multi` from the top-level `graphld` package.

### Changed

- Changed `LDClumper.clump()` and `run_clump()` to return the original input rows in original order with a boolean `is_index` column; `graphld clump` still writes only retained index variants.
- Harmonized `gaussian_likelihood_hessian()` with `gaussian_likelihood_gradient()` by adding `trace_estimator`, preserving `diagonal_method` as a deprecated keyword alias, and defaulting diagonal-only Hessian output to `xdiag`.
- Defined precomputed surrogate HDF5 maps in full LDGM row coordinates, with GraphREML translating them back to the active post-merge coordinate system.
- Updated BLUP so public `heritability` is interpreted as total heritability for the analyzed variant scope and distributed across matched LDGM effect dimensions.
- Removed GraphLD's package-import mutation of global NumPy floating-point warning settings.

### Fixed

- Fixed `graphld` CLI option precedence, `.vcf.gz` summary-stat dispatch, multi-trait parquet REML output naming, and simulate annotation-column forwarding.
- Fixed graphREML filtering and validation bugs affecting per-block max-chi-square filtering, binary annotation parameter alignment, and missing sample-size fallback behavior.
- Fixed `PrecisionOperator` subset PCG solves, inverse-diagonal paths, invalid diagonal update atomicity, selection-cache invalidation, copied solver state, and stale factor refresh across shared matrix aliases.
- Fixed simulation reproducibility and phasing issues by making seeded effect-size draws block-specific, loading worker LDGMs with metadata-row populations, and phasing annotation-based simulated effects/noise back to the annotation allele convention.
- Fixed score-test jackknife block boundaries, deterministic annotation-file ordering, empty trait-group handling, gene-set identifier detection, and genome-scale nearest-gene overflow.
- Fixed core I/O edge cases for VCF sample/FORMAT parsing, metadata-row variant partitioning, ID-only SNP-list merges, order-independent BED interval annotation, `load_annotations(file_pattern=...)`, position-only annotation loading, and `load_ldgm(snps_only=True)` variant metadata filtering.
- Hardened parquet summary-stat validation so incomplete trait `BETA`/`SE` pairs and missing variant identifiers fail with clear errors.
- Fixed surrogate-marker generation for blocks with no candidates by writing the initialized identity map.

### Removed

- Removed unsupported or stale public surfaces: the broken `create-geneset-annot` console script, the duplicate `graphld.heritability_testing` module, and the obsolete `graphld.genesets_gene_hdf5` workflow. Gene-level HDF5 conversion remains available through `estest convert`.

## v1.2.0 - 2026-05-22

Initial baseline for this changelog.
