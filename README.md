# SparseLD

This module provides efficient sparse precision matrix operations with linkage disequilibrium graphical models (LDGMs) and GWAS likelihood computations. This is a partial Python implementation of the MATLAB API from the [LDGM repository](https://github.com/awohns/ldgm). Some of the likelihood functions are based on MATLAB functions contained in the [graphREML repository](https://github.com/huilisabrina/graphREML). 

For more information about LDGMs, see our [paper](https://pubmed.ncbi.nlm.nih.gov/37640881/):
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: 10.1038/s41588-023-01487-8

For documentation of the likelihood functions, see our [graphREML preprint](https://www.medrxiv.org/content/10.1101/2024.11.04.24316716v1):
>Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11.


Pre-computed LDGMs for the 1000 Genomes Project data are available at [Zenodo](https://zenodo.org/records/8157131). You can download them using the provided Makefile in the `data/` directory:

```bash
# Download LDGMs for all populations
cd data && make download

# Download only EUR population LDGMs
cd data && make download_eur

# Download all data including sample information and UK Biobank summary statistics
cd data && make download_all
```

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Matrix Operations](#matrix-operations)
  - [Likelihood Functions](#likelihood-functions)
  - [Simulation](#simulation)
- [File Formats](#file-formats)

## Overview

SparseLD is a Python package for working with sparse linkage disequilibrium (LD) matrices in genome-wide association studies (GWAS). The package provides:

- Fast and memory-efficient storage of sparse LD matrices using the CSR format
- Efficient operations for computing matrix-vector products and solving linear systems
- Functions for computing likelihoods and gradients of GWAS summary statistics
- Tools for simulating GWAS data under various genetic architectures
- Support for reading and writing LD matrices in various formats

## Installation

### Using uv (recommended)

For regular installation:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv sync
```

For development installation (includes test dependencies):
```bash
uv sync --dev --extra dev
```

Required system dependencies:
- SuiteSparse (for CHOLMOD): On Mac, install with `brew install suitesparse`

## Usage

### Matrix Operations

```python
import sparseld as sld
import numpy as np
from sparseld import PrecisionOperator

# Load LDGM data from a single file
ldgm: PrecisionOperator = sld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Load multiple LDGMs from a directory
ldgms: list[PrecisionOperator] = sld.load_ldgm(
    filepath="data/test",
    population="EUR"  # Try "EAS" or "EUR"
)

# Matrix-vector operations
vector = np.random.randn(ldgm.shape[0])
result = ldgm @ vector  # matrix-vector product
solution = ldgm.solve(result)  # solve linear system
assert np.allclose(solution, vector)

# Subset the rows/columns of the matrix so that it represents inv(correlation_matrix[indices, indices])
indices = np.array([0, 2, 4])
subsetted_ldgm = ldgm[indices]

# Modify the diagonal elements of the subsetted matrix
subsetted_ldgm.update_element(1, 1.23)  # add 1.23 to matrix[indices[1], indices[1]]
subsetted_ldgm.update_matrix(np.array([1.1, 2.2, 3.3]))  # add [1.1, 2.2, 3.3] to diagonal of matrix[indices, indices]

```

### Likelihood Functions

The package provides functions for computing the likelihood of GWAS summary statistics under a Gaussian model:

```python
beta ~ MVN(0, diag(sigmasq))
z|beta ~ MVN(R*beta, R/n)  # where R is the LD matrix, n is sample size
pz = inv(R) * z  # precision-premultiplied effect sizes
```

The main functions are:

- `gaussian_likelihood(pz, precision_op)`: Computes the log-likelihood
- `gaussian_likelihood_gradient(pz, precision_op, del_sigma_del_a=None)`: Computes the score (gradient) with respect to sigmasq or parameters `a`
- `gaussian_likelihood_hessian(pz, precision_op, del_sigma_del_a)`: Computes the average information matrix

The precision operator `precision_op` should contain the covariance matrix `M = sigmasq + P/n`, not just the precision matrix `P`.

Example usage:

```python
import sparseld as sld
import numpy as np
from sparseld.likelihood import gaussian_likelihood, gaussian_likelihood_gradient

# Load LDGM data
ldgm = sld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Create some example GWAS data
n_samples = 10_000
n_variants = ldgm.shape[0]
true_effects = np.random.normal(0, 0.1, size=n_variants)
z_scores = ldgm @ true_effects + np.random.normal(0, 1/np.sqrt(n_samples), size=n_variants)
pz = ldgm.solve(z_scores)  # precision-premultiplied z-scores

# Compute log-likelihood
ll = gaussian_likelihood(pz, ldgm)
print(f"Log-likelihood: {ll:.2f}")

# Compute gradient with respect to per-SNP variances
del_sigma = np.eye(n_variants)  # derivative matrix for per-SNP variances
gradient = gaussian_likelihood_gradient(pz, ldgm, del_sigma)
print(f"Gradient shape: {gradient.shape}")  # One value per SNP
```

### Simulation

The package provides tools for simulating GWAS summary statistics under a mixture model with multiple components. It supports the inclusion of arbitrary annotations, allele frequency-dependent architecture, and sparse architectures with effects drawn from a mixture of normal distributions:

```python
import sparseld as sld

# Load LDGM data
ldgm = sld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Create simulator
sim = sld.Simulate(
    sample_size=10_000,                  # GWAS sample size
    heritability=0.1,                   # Total trait heritability
    component_variance=[1.0, 0.1],      # Relative effect size variance for each component
    component_weight=[0.001, 0.01],       # Mixture weights (must sum to ≤ 1)
    alpha_param=-0.5                   # Allele frequency dependence parameter
)

# Simulate summary statistics for a list of LD blocks
sumstats = sim.simulate([ldgm])  # Returns list of DataFrames with Z-scores
```


## File Formats

### Edgelist File (.edgelist)

Tab-separated file containing one edge per line with columns:
1. Source variant index (0-based)
2. Target variant index (0-based)
3. Edge weight (correlation value)

### Snplist File (.snplist)

Tab-separated file with columns:
1. Variant ID (e.g., rs number)
2. Chromosome
3. Position
4. Reference allele
5. Alternative allele
