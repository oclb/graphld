# GraphLD

This package is a partial Python implementation of the MATLAB API from the [LDGM repository](https://github.com/awohns/ldgm). Some of the likelihood functions are based on MATLAB functions contained in the [graphREML repository](https://github.com/huilisabrina/graphREML).

For more information about LDGMs, see our [paper](https://pubmed.ncbi.nlm.nih.gov/37640881/):
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: 10.1038/s41588-023-01487-8

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Matrix Operations](#matrix-operations)
  - [Likelihood Functions](#likelihood-functions)
  - [Simulation](#simulation)
- [File Formats](#file-formats)

## Installation

Required system dependencies:
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (for CHOLMOD): On Mac, install with `brew install suitesparse`

### Using uv (recommended)

For regular installation:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv sync
```

For development installation:
```bash
uv sync --dev --extra dev
```

### Downloading LDGM precision matrices
Pre-computed LDGMs for the 1000 Genomes Project data are available at [Zenodo](https://zenodo.org/records/8157131). You can download them using the provided Makefile in the `data/` directory:

```bash
# download LDGM precision matrices from the five continental ancestries in the 1000 Genomes Project
cd data && make download
```

The Makefile also contains `download_all` and `download_eur` targets.

## Usage

### Matrix Operations

The `PrecisionOperator` is a subclass of the SciPy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) interface. It represents either an LDGM precision matrix or its [Schur complement](https://en.wikipedia.org/wiki/Schur_complement), where the Schur complement of a precision matrix is the inverse of a submatrix of the correlation matrix. Thus, if one would
like to compute `correlation_matrix[indices, indices] @ vector`, one can use `ldgm[indices].solve(vector)`. If one would like to compute `inv(correlation_matrix[indices, indices]) @ vector`, one can use `ldgm[indices] @ vector`. See Section 5 of the supplementary material of [our paper](https://pubmed.ncbi.nlm.nih.gov/37640881/).

Example usage:

```python
import graphld as gld
import numpy as np
from graphld import PrecisionOperator

# Load LDGM data from a single file
ldgm: PrecisionOperator = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Load multiple LDGMs from a directory
ldgms: list[PrecisionOperator] = gld.load_ldgm(
    filepath="data/test",
    population="EUR"  # Try "EAS" or "EUR"
)

# Matrix-vector operations
vector = np.random.randn(ldgm.shape[0])
result = ldgm @ vector  # matrix-vector product
solution = ldgm.solve(result)  # solve linear system
assert np.allclose(solution, vector)
```

### Likelihood Functions

The package provides functions for computing the likelihood of GWAS summary statistics under a Gaussian model:

$$\beta \sim N(0, D)$$
$$z|\beta \sim N(n^{1/2}R\beta, R)$$
where $\beta$ is the effect-size vector in s.d-per-s.d. units, $D$ is a diagonal matrix of per-variant heritabilities, $z$ is the GWAS summary statistic vector, $R$ is the LD correlation matrix, and $n$ is the sample size. The likelihood functions operate on  precision-premultiplied GWAS summary statistics: 
$$pz = n^{-1/2} R^{-1}z \sim N(0, M), M = D + n^{-1}R^{-1}.$$ 

The following functions are available:
- `gaussian_likelihood(pz, M)`: Computes the log-likelihood
- `gaussian_likelihood_gradient(pz, M, del_M_del_a=None)`: Computes the score (gradient), either with respect to the diagonal elements of `M` (equivalently `D`), or with respect to parameters `a` whose partial derivatives are provided in `del_M_del_a`. 
- `gaussian_likelihood_hessian(pz, M, del_M_del_a)`: Computes the average information matrix, defined as the average of the observed and expected Fisher information. This is the second derivative of the log-likelihood with respect to the parameters `a` whose partial derivatives are provided in `del_M_del_a`.

Example usage:

```python
import graphld as gld
import numpy as np
from graphld.likelihood import gaussian_likelihood, gaussian_likelihood_gradient

# Load LDGM data
ldgm = gld.load_ldgm(
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


For background, see our [graphREML preprint](https://www.medrxiv.org/content/10.1101/2024.11.04.24316716v1):
>Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11.


### Simulation

The package provides tools for simulating GWAS summary statistics under a mixture model with multiple components. It supports the inclusion of arbitrary annotations, allele frequency-dependent architecture, and sparse architectures with effects drawn from a mixture of normal distributions:

```python
import graphld as gld

# Load LDGM data
ldgm = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Create simulator
sim = gld.Simulate(
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
