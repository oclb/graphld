# GraphLD Agent Context

GraphLD implements graphREML and graphREML-ST, LDGM-backed LD matrix operations, enrichment score tests, simulation, LD clumping, BLUP weights, and surrogate marker utilities for GWAS summary statistics.

## Project Map

- `src/graphld/`: main Python package. Core modules include `precision.py`, `io.py`, `heritability.py`, `cli.py`, `blup.py`, `clumping.py`, `simulate.py`, and `surrogates.py`.
- `src/score_test/`: enrichment score-test package and CLI.
- `tests/`: pytest suite. Prefer small synthetic tests when tracked data fixtures are dirty.
- `data/test/`: intended small fixtures for LDGMs, sumstats, annotations, VCF, parquet, and score-test HDF5 files.
- `data/ldgms/`: default location for downloaded LDGMs. The usual metadata file is `data/ldgms/metadata.csv`; rows point to edgelist and snplist files.

## Agent Skills

Use the project-local skills for operational detail:

- `graphld-setup`: installation, dependency, data-download, and O2 reference guidance.
- `graphld-usage`: CLI/API usage patterns.

## CLI Surface

`graphld` subcommands are `reml`, `blup`, `clump`, `simulate`, and `surrogates`. `estest` handles enrichment score-test workflows with `test`, `show`, `add-meta`, `rm`, `mv`, and `convert`.

## Terminology

- LDGM: linkage disequilibrium graphical model, stored as metadata plus edgelist/snplist files.
- graphREML: REML method for heritability partitioning and enrichment estimation.
- Score test: enrichment test using precomputed derivative statistics in HDF5.
- Summary statistics: GWAS variant-level association data, commonly LDSC `.sumstats`, GWAS-VCF, or parquet.
- Annotation: variant, region, or gene-set feature used for enrichment analysis.
- Surrogates: marker substitutions used when GWAS variants are missing from LDGMs.

## Gotchas

Do not overwrite user data or generated results. This repo often has dirty `data/`, `results/`, `output/`, `scorestats/`, and fixture files during active analysis. Check file sizes and `git status` before treating local fixtures as a reliable baseline.
