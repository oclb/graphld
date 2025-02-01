# GraphLD

This repository provides an implementation of the graphREML method described in:
> Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11. DOI: [10.1101/2024.11.04.24316716](https://doi.org/10.1101/2024.11.04.24316716)

and a Python API for working with [linkage disequilibrium graphical models](https://github.com/awohns/ldgm) (LDGMs), described in:
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: [10.1038/s41588-023-01487-8](https://pubmed.ncbi.nlm.nih.gov/37640881/)

## Table of Contents
- [Installation](#installation)
- [Command Line Interface (CLI)](#command-line-interface-cli)
- [Usage](#usage)
  - [Heritability Estimation](#heritability-estimation)
  - [Matrix Operations](#matrix-operations)
  - [LD Clumping](#ld-clumping)
  - [Likelihood Functions](#likelihood-functions)
  - [Simulation](#simulation)
- [Multiprocessing Framework](#multiprocessing-framework)
- [File Formats](#file-formats)
- [See also](#see-also)

## Installation

Required system dependencies:
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (for CHOLMOD): On Mac, install with `brew install suitesparse`. SuiteSparse is wrapped in [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/).
- IntelMKL (for Intel chips): The performance of SuiteSparse is significantly improved by using IntelMKL instead of OpenBLAS, which will likely be the default. See Giulio Genovese's documentation [here](https://github.com/freeseek/score?tab=readme-ov-file#intel-mkl).

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
uv run pytest
```
### Using conda and pip install
Example codes are based on the O2 cluster in the Harvard Medical School computing system. 

- Create a conda `env` for `suitesparse` and activate it: you may need to revert or reinstall some Python packages
```bash
module load miniconda3/4.10.3
conda create -n suitesparse conda-forge::suitesparse python=3.11.0
conda activate suitesparse
```
- You may need to revert or reinstall some Python packages, if prompted. For example, to install the correct version of `numpy`, use:
```bash
pip install numpy==1.26.4
```
- Install `scikit-sparse`: you may need to add some `conda` channels (see below)
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install scikit-sparse
```
- Install and run graphLD
```bash
cd sparseld && pip install .
```
- Test if it works
```bash
graphld -h
```

### Using conda and pip install
Example codes are based on the O2 cluster in the Harvard Medical School computing system. 

- Create a conda `env` for `suitesparse` and activate it: you may need to revert or reinstall some Python packages
```bash
module load miniconda3/4.10.3
conda create -n suitesparse conda-forge::suitesparse python=3.11.0
conda activate suitesparse
```
- You may need to revert or reinstall some Python packages, if prompted. For example, to install the correct version of `numpy`, use:
```bash
pip install numpy==1.26.4
```
- Install `scikit-sparse`: you may need to add some `conda` channels (see below)
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install scikit-sparse
```
- Install and run graphLD
```bash
cd sparseld && pip install .
```
- Test if it works
```bash
graphld -h
```

### Downloading LDGMs
Pre-computed LDGMs for the 1000 Genomes Project data are available at [Zenodo](https://zenodo.org/records/8157131). You can download them using the provided Makefile in the `data/` directory:

```bash
cd data && make download
```

The Makefile also contains a `download_all` target to download additional data and a `download_eur` target to download European-ancestry LDGMs only.

## Command Line Interface (CLI)

The CLI has commands for `blup`, `clump`, `simulate`, and `reml`. After installing with `uv`, run (for example) `uv run graphld reml -h`. To run graphREML:

```bash
uv run graphld reml \
    /path/to/sumstats/file.sumstats \
    output_files_prefix \
    --annot /directory/containing/annot/files/ \
```
The summary statistics can be in VCF (`.vcf`) or  LDSC (`.sumstats`) format. The annotation files should be in LDSC (`.annot`) format. There will be three output files containing tables with point estimates and standard errors for the annotation-specific heritabilities, heritability enrichments, and model parameters. If you specify the same output file prefix for multiple traits, the estimates for each trait will be printed on a line of the same file.

## Usage

### Heritability Estimation

```python
from graphld.heritability import ModelOptions, MethodOptions, run_graphREML
from graphld.io import load_annotations
from graphld.ldsc_io import read_ldsc_sumstats

# Load LDSC-format summary statistics
sumstats = read_ldsc_sumstats("path/to/sumstats.sumstats")

# Load LDSC-format annotation data (*.annot)
annotations = load_annotations("path/to/annot/", chromosomes=[1])

# Default options
model_options = ModelOptions()
method_options = MethodOptions()

# Run heritability estimation
results = run_graphREML(
    model_options=model_options,
    method_options=method_options,
    summary_stats=sumstats,
    annotation_data=annotations,
    ldgm_metadata_path="path/to/ldgms/metadata.csv"
)

# Access results
print(f"Heritability: {results['heritability']}")
print(f"Heritability SE: {results['heritability_se']}")
print(f"Enrichment: {results['enrichment']}")
print(f"Enrichment SE: {results['enrichment_se']}")
```

The estimator returns a dictionary containing:
- `heritability`: Heritability estimates for each annotation
- `heritability_se`: Standard errors for heritability estimates
- `enrichment`: Enrichment values (relative to baseline)
- `enrichment_se`: Standard errors for enrichment values
- `likelihood_history`: Optimization history
- `params`: Final parameter values
- `param_se`: Standard errors for parameters

For a complete example, see [scripts/run_graphreml_height.py](scripts/run_graphreml_height.py).

### Matrix Operations

`PrecisionOperator` subclasses the SciPy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) interface. It represents an LDGM precision matrix or its [Schur complement](https://en.wikipedia.org/wiki/Schur_complement). If one would
like to compute `correlation_matrix[indices, indices] @ vector`, one can use `ldgm[indices].solve(vector)`. To compute `inv(correlation_matrix[indices, indices]) @ vector`, use `ldgm[indices] @ vector`. See Section 5 of the supplementary material of [our paper](https://pubmed.ncbi.nlm.nih.gov/37640881/).

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

### LD Clumping

LD clumping identifies independent index variants by iteratively selecting the variant with the highest $\chi^2$ statistic and pruning all variants in high LD with it.

```python
clumped = gld.LDClumper.clump(
    ldgm_metadata_path="data/metadata.csv"
    sumstats=sumstats_dataframe_with_z_scores
).filter(pl.col('is_index')) # is_index is added as a new column to the dataframe
```

### Likelihood Functions

The likelihood of GWAS summary statistics under an infinitesimal model is:

$$\beta \sim N(0, D)$$
$$z|\beta \sim N(n^{1/2}R\beta, R)$$
where $\beta$ is the effect-size vector in s.d-per-s.d. units, $D$ is a diagonal matrix of per-variant heritabilities, $z$ is the GWAS summary statistic vector, $R$ is the LD correlation matrix, and $n$ is the sample size. Our likelihood functions operate on  precision-premultiplied GWAS summary statistics: 
$$pz = n^{-1/2} R^{-1}z \sim N(0, M), M = D + n^{-1}R^{-1}.$$ 

The following functions are available:
- `gaussian_likelihood(pz, M)`: Computes the log-likelihood
- `gaussian_likelihood_gradient(pz, M, del_M_del_a=None)`: Computes the gradient of the log-likelihood, either with respect to the diagonal elements of `M` (equivalently `D`), or with respect to parameters `a` whose partial derivatives are provided in `del_M_del_a`. 
- `gaussian_likelihood_hessian(pz, M, del_M_del_a)`: Computes the average information matrix, defined as the average of the observed and expected Fisher information. This approximates the second derivative of the log-likelihood with respect to the parameters `a` whose partial derivatives are provided in `del_M_del_a`. The approximation is good when the gradient is nearly zero.

For background, see our [graphREML preprint](https://www.medrxiv.org/content/10.1101/2024.11.04.24316716v1):
>Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O’Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11.

### Simulation

A parallelized simulation function is provided which samples GWAS summary statistics from their asymptotic sampling distribution. Effect sizes are drawn from a flexible mixture distribution, which supports annotation-dependent and frequency-dependent architectures. Unlike the [MATLAB implementation](https://github.com/awohns/ldgm/blob/main/MATLAB/simulateSumstats.m), it does not support multiple ancestry groups.

```python
import graphld as gld

# Load LDGM data
ldgm = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

# Create simulator
sim = gld.Simulate(
    sample_size=10_000, 
    heritability=0.1,
    component_variance=[1.0, 0.1],     # Relative effect size variance for each component
    component_weight=[0.001, 0.01],    # Mixture weight of each component (must sum to ≤ 1)
    alpha_param=-0.5                   # Allele frequency dependence parameter
)

# Simulate summary statistics for a list of LD blocks
sumstats = sim.simulate([ldgm])  # Returns list of DataFrames with Z-scores
```

### Best Linear Unbiased Prediction (BLUP)
BLUP effect sizes can be computed using the following formula:
$$
E(\beta) = \sqrt{n} D (nD + R^{-1})^{-1} R^{-1}z
$$
where we approximate $R^{-1}$ with the LDGM precision matrix. A parallelized implementation is provided in `graphld.BLUP`.

## Multiprocessing Framework

`ParallelProcessor` is a base class which can be used to implement parallel algorithms with LDGMs, wrapping Python's `multiprocessing` module. It splits work among processes, each of which loads a subset of LD blocks. The advantage of using this is that it handles for you the loading of LDGMs within worker processes.
 Three classes are provided:
- `ParallelProcessor`: Base class for parallel algorithms.
- `WorkerManager` and `SerialManager`: Convenience classes to manage processes either in parallel or in series.
- `SharedData`: Convenience class wrapping `Array` and `Value` from `multiprocessing`.


### Usage

Subclass `ParallelProcessor` and implement the following methods:
- `prepare_block_data` (optional): Prepare data specific to each block as a list with one entry per LD block.
- `create_shared_memory`: Create `SharedData` objects that can be used to communicate between processes and store results.
- `process_block`: Do some computation for a single LD block, storing results in the `SharedData` object.
- `supervise`: Start workers and handle communication using the `WorkerManager`, return results by reading from the `SharedData` object

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

### SNP list File (.snplist)

Tab-separated file with columns:
1. `index`: corresponds to the index in the edgelist file. Multiple variants can have the same index.
2. `anc_alleles`: Inferred ancestral alleles
3. `EUR`, `EAS`, `AMR`, `SAS`, `AFR`: Derived allele frequencies in each 1000 Genomes superpopulation (optional)
4. `site_ids`: RSID for each variant
5. `position`: position in GRCh38 coordinates
6. `swap` (optional)
It is recommended that you do not use the `variant ID` column for merging and instead use chromosome/position/ref/alt, as some variants lack RSIDs. The file contains some variants which are not SNPs.

### LDSC Format Summary Statistics (.sumstats)
See [LDSC summary statistics file format](https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format). Read with `read_ldsc_sumstats`.

### GWAS-VCF (.vcf)
The [GWAS-VCF specification](https://github.com/MRCIEU/gwasvcf) is supported via the `read_gwas_vcf` function. It is a VCF file with the following mandatory FORMAT fields::

- `ES`: Effect size estimate
- `SE`: Standard error of effect size
- `LP`: -log10 p-value


### LDSC Format Annotations (.annot)
You can download BaselineLD model annotation files with GRCh38 coordinates from the Price lab Google Cloud bucket: https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE/GRCh38

Read annotation files with `load_annotations`.

## See Also

- Main LDGM repository, including a MATLAB API: [https://github.com/awohns/ldgm](https://github.com/awohns/ldgm)
- Original graphREML repository, with a MATLAB implementation: [https://github.com/huilisabrina/graphREML](https://github.com/huilisabrina/graphREML) (we recommend using the Python implementation, which is much faster)
- LD score regression repository: [https://github.com/bulik/ldsc](https://github.com/bulik/ldsc)
- Giulio Genovese has implemented a LDGM-VCF file format specification and a bcftools plugin written in C with partially overlapping features, available [here](https://github.com/freeseek/score).
- All of these rely heavily on sparse matrix operations implemented in [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).

