# GraphLD

This repository provides Python functions for working with [LDGMs](https://github.com/awohns/ldgm). Some of the functions are translated from MATLAB functions contained in the [LDGM repository](https://github.com/awohns/ldgm/tree/main/MATLAB) and the [graphREML repository](https://github.com/huilisabrina/graphREML).

For more information about LDGMs, see our [paper](https://pubmed.ncbi.nlm.nih.gov/37640881/):
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: 10.1038/s41588-023-01487-8

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

For regular installation:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv sync
```

For development installation:
```bash
uv sync --dev --extra dev
uv run pytest # test suite
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
z_scores = np.random.normal(0, 1, size=n_variants)
pz = ldgm.solve(z_scores)  # precision-premultiplied z-scores

# Compute log-likelihood
M = ldgm / n_samples
M.update_matrix(np.ones(n_variants)) # D = identity
ll = gaussian_likelihood(pz, M)
print(f"Log-likelihood: {ll:.2f}")

# Compute gradient with respect to per-SNP variances
del_sigma = np.eye(n_variants)  # derivative matrix for per-SNP variances
gradient = gaussian_likelihood_gradient(pz, ldgm, del_sigma)
print(f"Gradient shape: {gradient.shape}")  # One value per SNP
```


For background, see our [graphREML preprint](https://www.medrxiv.org/content/10.1101/2024.11.04.24316716v1):
>Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O’Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11.


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

## Multiprocessing Framework

We provide a base class, `ParallelProcessor`, which can be used to implement parallel algorithms with LDGMs. It splits work among processes, each of which loads a subset of LD blocks. It relies on Python's `multiprocessing` module. Three classes are provided:
- `ParallelProcessor`: Base class for parallel processing
- `WorkerManager`: Class for managing worker processes
- `SharedData`: Class for storing shared memory arrays 

### Usage

Subclass `ParallelProcessor` and implement its three required methods:
- `create_shared_memory`: Create `SharedMemory` objects that can be used to input data, output results, and communicate between processes
- `process_block`: Do some computation for a single LD block, storing results in the `SharedMemory` object
- `supervise`: Start workers and handle communication using the `WorkerManager`, return results by reading from the `SharedMemory` object
Optionally also implement:
- `prepare_block_data`: Prepare data specific to each block, to be passed to `process_block`

Then call `ParallelProcessor.run` with the following arguments:
- `ldgm_metadata_path`: Path to metadata file
- `populations`: Populations to process; None -> all
- `chromosomes`: Chromosomes to process; None -> all
- `num_processes`: Number of processes to use
- `worker_params`: Optional parameters passed to each worker process
- `**kwargs`: Additional arguments passed to `prepare_block_data`, `create_shared_memory`, and `supervise`

For example:
```python
from graphld.multiprocessing import ParallelProcessor, SharedData
import polars as pl

class MyProcessor(ParallelProcessor):
    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list, **kwargs) -> SharedData:
        """Create shared memory arrays for input and output data.
        
        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific data
            **kwargs: Additional arguments passed from run()
            
        Returns:
            SharedData object containing shared memory arrays
        """
        input_array_size = metadata['numIndices'].sum()
        output_array_size = len(metadata)
        return SharedData({
            'input': input_array_size,
            'output': output_array_size,
            'some_shared_scalar': None
        })

    @staticmethod
    def process_block(ldgm, flag, shared_data, block_offset, block_data, worker_params=None):
        """Process a single block of data.
        
        Args:
            ldgm: LDGM object for this block
            flag: Multiprocessing Value for worker control
            shared_data: SharedData containing arrays
            block_offset: Starting index for this block in shared arrays
            block_data: Data specific to this block
            worker_params: Optional parameters passed to each worker
        """
        block_slice = slice(block_offset, block_offset + ldgm.shape[0])

        # Note the slicing syntax
        input_data = shared_data['input', block_slice]
        
        solution = ldgm.solve(input_data)
        
        shared_data['output', block_slice] = solution
        
    @staticmethod
    def supervise(manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Supervise worker processes and handle results.
        
        Args:
            manager: WorkerManager instance
            shared_data: SharedData containing arrays
            block_data: List of block-specific data
            **kwargs: Additional arguments passed from run()

        Returns:
            Final results after all blocks processed
        """
        # If the workers need to communciate, you can call these functions in a loop
        manager.start_workers()
        manager.await_workers()
        
        # Return results array
        return np.array(shared_data['output'])

    @classmethod
    def prepare_block_data(cls, metadata, **kwargs) -> list:
        """Prepare data specific to each block.
        
        Args:
            metadata: Metadata DataFrame containing block information
            **kwargs: Additional arguments passed from run()
            
        Returns:
            List of data objects, one per block
        """
        block_data = []
        for block in metadata.iter_rows(named=True):
            # Create block-specific data (e.g., Polars DataFrame)
            df = pl.DataFrame({
                'position': range(block['numIndices']),
                'metadata': ['block_info'] * block['numIndices']
            })
            block_data.append(df)
        return block_data
```

You can then run your parallel processor with:
```python
# Create some worker-specific parameters
worker_params = {
    'max_iterations': 100,
    'tolerance': 1e-6
}

# Run parallel processing
results = MyProcessor.run(
    ldgm_metadata_path="path/to/metadata.csv",
    num_processes=12,
    populations="EUR",
    chromosomes=1,
    worker_params=worker_params,  # Optional parameters for each worker
    some_kwarg="value"  # Additional kwargs passed to all methods
)
```

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

# Read metadata with optional filtering
metadata = gld.read_ldgm_metadata(
    metadata_file="data/metadata.csv",
    populations=["EUR", "EAS"],       # Filter by populations
    chromosomes=[1, 2],               # Filter by chromosomes
    max_blocks=100                    # Limit number of blocks
)

# Partition variant data into LDGM blocks
blocks = gld.partition_variants(
    ldgm_metadata=metadata,           # LDGM block metadata
    variant_data=variants,            # Polars DataFrame with variant data
    chrom_col="chromosome",           # Optional column name for chromosome
    pos_col="position"                # Optional column name for position
)
```

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
