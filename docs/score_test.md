# Enrichment Score Test

The enrichment score test is a fast method to test genomic or gene annotations for heritability enrichment conditional upon a null model.

## Overview

The test produces Z scores:

- **Positive score**: Heritability enrichment
- **Negative score**: Heritability depletion

Enrichments are conditional upon the null model, similar to the `tau` parameter in S-LDSC. The test does not produce point estimates (for that, run graphREML).

## Requirements

You need a file containing precomputed derivatives for each trait being tested. This can be:

- Downloaded from Zenodo via the Makefile (`make download_scorestats`)
- Created by running graphREML with the `--score-test-filename` flag

## Supported Annotation Formats

| Format | Description |
|--------|-------------|
| [LDSC `.annot`](file_formats.md#ldsc-format-annot) | Variant annotations |
| [UCSC `.bed`](file_formats.md#bed-format-bed) | Genomic regions |
| [GMT](file_formats.md#gmt-format-gmt) | Gene sets (symbols or IDs) |

## Basic Usage

### View Available Traits

```bash
uv run estest show path/to/precomputed/derivatives.h5
```

### Test Variant Annotations

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir /directory/containing/dot-annot/files/
```

### Test Genomic Regions

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir /directory/containing/dot-bed/files/
```

### Test Gene Annotations

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --gene-annot-dir /directory/containing/gmt/files/
```

### Options Reference

```
uv run estest test --help
```

| Option | Description |
|--------|-------------|
| `-a, --variant-annot-dir` | Directory containing .annot files |
| `-g, --gene-annot-dir` | Directory containing .gmt files |
| `--random-genes` | Comma-separated probabilities for random gene annotations |
| `--random-variants` | Comma-separated probabilities for random variant annotations |
| `--gene-table` | Path to gene table TSV (required for gene-level options) |
| `--nearest-weights` | Comma-separated weights for k-nearest genes |
| `--annotations` | Specific annotation names to test |
| `-n, --name` | Specific trait to process (default: all traits) |
| `-v, --verbose` | Enable verbose output |
| `--seed` | Random seed for reproducibility |

## Random Annotations

Test random annotations to verify the null distribution:

### Random Variant Annotations

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --random-variants 0.1,0.2,0.3
```

Creates random annotations with 10%, 20%, and 30% of variants.

### Random Gene Annotations

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --random-genes 0.1,0.2,0.3
```

Creates random annotations with 10%, 20%, and 30% of genes.

### Perturb Existing Annotations

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir /directory/containing/dot-annot/files/ \
    --perturb-annot 0.5  # 50% of annotation values sampled randomly
```

## Gene Set Testing

Gene sets can be tested for heritability enrichment under the Abstract Mediation Model (AMM; Weiner et al. 2022 AJHG). This tests whether variants in proximity to genes in the gene set are enriched for heritability.

### Basic Approach

Supply a GMT file to `--gene-annot-dir`.

### Faster Approach

Convert variant-level to gene-level score statistics first:

```bash
uv run estest convert variant_statistics.h5 gene_statistics.h5
```

This requires a gene positions file (provided in `data/genes.tsv` after running the Makefile).

Then run the test:

```bash
uv run estest \
    gene_statistics.h5 output_prefix \
    --gene-annot-dir /directory/containing/gmt/files/
```

Results are nearly identical to the variant-level test but much faster.

## Meta-Analysis Across Traits

Test whether an annotation is enriched across multiple traits:

### Add Meta-Analysis

```bash
# Add all traits
uv run estest add-meta statistics.h5 all_traits '*'

# Add specific traits
uv run estest add-meta statistics.h5 body_traits height bmi
```

Then run the score test as normal. The meta-analysis appears as a column in the output.

The meta-analysis uses precision-weighted linear combination of score statistics with jackknife standard errors. Non-independence across traits causes power loss but not false positives.

## Manipulating HDF5 Files

The `estest` command provides utilities for managing traits and meta-analyses in HDF5 files.

### Show Contents

Display all traits and meta-analyses in an HDF5 file:

```bash
uv run estest show statistics.h5
```

### Rename Traits or Meta-Analyses

Rename a trait or meta-analysis (auto-detects which):

```bash
uv run estest mv statistics.h5 old_name new_name
```

### Remove Traits or Meta-Analyses

Remove one or more traits or meta-analyses:

```bash
# Remove a single item
uv run estest rm statistics.h5 trait_name

# Remove multiple items
uv run estest rm statistics.h5 bmi height cancer

# Use wildcards
uv run estest rm statistics.h5 '*_EAS'

# Force removal without confirmation
uv run estest rm statistics.h5 'BMI*' -f
```

The command auto-detects whether names are traits or meta-analyses and supports wildcards (`*`).
