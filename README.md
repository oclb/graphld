# GraphLD

This repository provides a Python API for working with [linkage disequilibrium graphical models](https://github.com/awohns/ldgm) (LDGMs) via a convenient interface, the `PrecisionOperator`. You can use this API to perform computations involving GWAS summary statistics and the LD matrix.
For more information about LDGMs, see our [paper](https://pubmed.ncbi.nlm.nih.gov/37640881/):
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: 10.1038/s41588-023-01487-8
Some of the functions are translated from MATLAB functions contained in the [LDGM repository](https://github.com/awohns/ldgm/tree/main/MATLAB) and the [graphREML repository](https://github.com/huilisabrina/graphREML).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Matrix Operations](#matrix-operations)
  - [Likelihood Functions](#likelihood-functions)
  - [Simulation](#simulation)
- [Multiprocessing Framework](#multiprocessing-framework)
- [File Formats](#file-formats)

## Installation

Required system dependencies:
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (for CHOLMOD): On Mac, install with `brew install suitesparse`

### Using uv (recommended)

In the repo directory:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv sync
```

For development installation:
```bash
uv sync --dev --extra dev # editable with pytest dependencies
uv run pytest # test suite
```

### Downloading LDGMs
Pre-computed LDGMs for the 1000 Genomes Project data are available at [Zenodo](https://zenodo.org/records/8157131). You can download them using the provided Makefile in the `data/` directory:

```bash
# download LDGM precision matrices from the five continental ancestries in the 1000 Genomes Project
cd data && make download
```

The Makefile also contains `download_all` and `download_eur` targets.

## Usage

### Matrix Operations

`PrecisionOperator` subclasses the SciPy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) interface. It represents an LDGM precision matrix or its [Schur complement](https://en.wikipedia.org/wiki/Schur_complement). If one would
like to compute `correlation_matrix[indices, indices] @ vector`, one can use `ldgm[indices].solve(vector)`. To compute `inv(correlation_matrix[indices, indices]) @ vector`, use `ldgm[indices] @ vector`. See Section 5 of the supplementary material of [our paper](https://pubmed.ncbi.nlm.nih.gov/37640881/).

Example usage:

```python
import graphld as gld
import numpy as np
from graphld import PrecisionOperator

ldgm: PrecisionOperator = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

vector = np.random.randn(ldgm.shape[0])
precision_times_vector = ldgm @ vector
correlation_times_vector = ldgm.solve(result)
assert np.allclose(correlation_times_vector, vector)
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

For background, see our [graphREML preprint](https://www.medrxiv.org/content/10.1101/2024.11.04.24316716v1):
>Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O’Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11.


### Simulation

*[Under construction]* A parallelized simulation function is provided. It samples GWAS summary statistics from their asymptotic sampling distribution with effect sizes drawn from a flexible mixture distribution.

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

## Multiprocessing Framework

`ParallelProcessor` is a base class which can be used to implement parallel algorithms with LDGMs, wrapping Python's `multiprocessing` module. It splits work among processes, each of which loads a subset of LD blocks. The advantage of using this is that it handles for you the loading of LDGMs within worker processes.
 Three classes are provided:
- `ParallelProcessor`: Base class for parallel algorithms.
- `WorkerManager` and `SerialManager`: Convenience classes to manage processes either in parallel or in series.
- `SharedData`: Convenience class wrapping `Array` and `Value` from `multiprocessing`.


### Usage

Subclass `ParallelProcessor` and implement the following methods:
- `prepare_block_data` (optional): Prepare data specific to each block as a list with one entry per LD block.
- `create_shared_memory`: Create `SharedMemory` objects that can be used to communicate between processes and store results.
- `process_block`: Do some computation for a single LD block, storing results in the `SharedMemory` object.
- `supervise`: Start workers and handle communication using the `WorkerManager`, return results by reading from the `SharedMemory` object

Then call `ParallelProcessor.run` (for parallel processing) or `ParallelProcessor.run_series` (for serial processing/debugging).

An example can be found in `tests/test_multiprocessing.py`

## File Formats

### LDGM Metadata File (.csv)

CSV file containing information about LDGM blocks with columns:
1. `chrom`: Chromosome number
2. `chromStart`: Start position of the block
3. `chromEnd`: End position of the block
4. `name`: LDGM filename
5. `snplistName`: Name of the corresponding snplist file
6. `population`: Population identifier (e.g., EUR, EAS)
7. `numVariants`: Number of variants in the block
8. `numIndices`: Number of non-zero indices in the precision matrix
9. `numEntries`: Number of non-zero entries in the precision matrix
10. `info`: Additional information (optional)

Example usage:
```python
import graphld as gld
metadata = gld.read_ldgm_metadata(
    metadata_file="data/test/metadata.csv",
    populations="EUR",
    chromosomes=[1, 2],
)
```

### Edgelist File (.edgelist)

Tab-separated file containing one edge per line with columns:
1. Source variant index (0-based)
2. Target variant index (0-based)
3. Precision matrix entry

### Snplist File (.snplist)

Tab-separated file with columns:
1. Variant ID (e.g., rs number; NA for some variants)
2. Chromosome
3. Position
4. Reference allele
5. Alternative allele

It is recommended that you do not use the `variant ID` column for merging and instead use chromosome/position/ref/alt, as some variants lack RSIDs. 