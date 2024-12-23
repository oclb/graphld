# SparseLD

This module provides efficient sparse precision matrix operations for LDGM (Linkage Disequilibrium Graph Model) data using scipy's LinearOperator interface.

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

# Load LDGM data
ldgm = sld.PrecisionOperator.load(
    filepath="path/to/block.edgelist",
    snplist_path="path/to/block.snplist"
)

# Matrix-vector operations
vector = np.random.randn(ldgm.shape[0])
result = ldgm @ vector  # matrix-vector product
solution = ldgm.solve(vector)  # solve linear system
```

### Likelihood Functions

The package provides functions for computing the likelihood of GWAS summary statistics under a Gaussian model:

```python
beta ~ MVN(0, diag(sigmasq))
z|beta ~ MVN(R*beta, R/n)  # where R is the LD matrix, n is sample size
pz = inv(R) * z  # precision-premultiplied effect sizes
```

The main functions are:

- `gaussian_likelihood(pz, precision_op)`: Computes the log-likelihood and log-determinant
- `gaussian_likelihood_gradient(pz, precision_op, del_sigma_del_a=None)`: Computes the score (gradient) with respect to sigmasq or parameters a
- `gaussian_likelihood_hessian(pz, precision_op, del_sigma_del_a)`: Computes the average information matrix

The precision operator `precision_op` should contain the covariance matrix `M = sigmasq + P/n`, not just the precision matrix `P`.

### Simulation

The package provides tools for simulating GWAS summary statistics under a mixture model with multiple components:

```python
import sparseld as sld

# Load LDGM data
ldgm = sld.PrecisionOperator.load(
    filepath="path/to/block.edgelist",
    snplist_path="path/to/block.snplist"
)

# Create simulator
sim = sld.Simulate(
    sample_size=10000,                  # GWAS sample size
    heritability=0.3,                   # Total trait heritability
    component_variance=[1.0, 0.1],      # Effect size variance for each component
    component_weight=[0.01, 0.1],       # Mixture weights (must sum to ≤ 1)
    alpha_param=-0.5,                   # Allele frequency dependence parameter
    annotation_dependent_polygenicity=True  # Use annotations to modify causal proportions
)

# Simulate summary statistics for a list of LD blocks
sumstats = sim.simulate([ldgm])  # Returns list of DataFrames with Z-scores
```

The simulator supports:
- Multiple mixture components with different effect size variances
- Annotation-dependent architectures through a customizable link function
- Allele frequency-dependent architectures via the alpha parameter
- Option to model annotations as affecting either polygenicity or effect size magnitude

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

## Dependencies

- Python 3.11
- numpy >= 2.2.0
- scipy >= 1.14.1
- polars >= 1.17.1
- scikit-sparse >= 0.4.12

### Development Dependencies

- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- pytest-xdist >= 3.3.1
- ruff >= 0.1.9
- hypothesis >= 6.82.6