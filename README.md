# GraphLD

This repository implements the graphREML and graphREML-ST methods described in:
> Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11. DOI: [10.1101/2024.11.04.24316716](https://doi.org/10.1101/2024.11.04.24316716)

and provides a Python API for computationally efficient linkage disequilibrium (LD) matrix operations with [LD graphical models](https://github.com/awohns/ldgm) (LDGMs), described in:
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: [10.1038/s41588-023-01487-8](https://pubmed.ncbi.nlm.nih.gov/37640881/)

It also provides very fast utilities for simulating GWAS summary statistics, performing LD clumping, and computing polygenic risk scores.

## Installation
Using [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
git clone https://github.com/oclb/graphld.git
cd graphld && uv venv && uv sync
```

To download data (20GB; smaller options exist):

```bash
cd data && make download_all
```

`graphld` requires [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse). On Mac, install with `brew install suitesparse`. On Ubuntu/Debian, use `sudo apt-get install libsuitesparse-dev`. SuiteSparse is wrapped in [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/). For users with Intel chips, it is highly recommended to install Intel MKL, which can produce a 100x speedup with SuiteSparse vs. OpenBLAS (your likely default BLAS library). See Giulio Genovese's documentation [here](https://github.com/freeseek/score?tab=readme-ov-file#intel-mkl).

If you are only using the [heritability enrichment score test](#enrichment-score-test), no installation is needed; you can run the source file as a standalone script. With `uv` installed, run `uv run src/score_test/score_test.py --help`, or:

## Command Line Interface

The CLI has commands for:
- `estest`, which runs the enrichment score test
- `graphld reml`, which runs graphREML
- `graphld blup`, which computes best linear unbiased predictor weights
- `graphld clump`, which performs p-value thresholding and LD-based pruning
- `graphld simulate`, which simulates GWAS summary statistics
Use `uv run <command> -h` to see usage for a specific command. If you downloaded LDGMs using the Makefile, they will be located automatically.

## Python API

### Heritability Estimation

```python
import graphld as gld
import polars as pl

sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
annotations: pl.DataFrame = gld.load_annotations("data/test/", chromosome=1)

default_model_options = gld.ModelOptions()
default_method_options = gld.MethodOptions()

reml_results: dict = gld.run_graphREML(
    model_options=default_model_options,
    method_options=default_method_options,
    summary_stats=sumstats,
    annotation_data=annotations,
    ldgm_metadata_path="data/test/metadata.csv",
    populations="EUR"
)
```

The estimator returns a dictionary containing heritability, enrichment, and coefficient estimates for each annotation, together with standard errors and two-tailed log10 p-values.

### Matrix Operations

LD matrix operations can be performed using the `PrecisionOperator`, which subclasses the SciPy [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html). It represents an LDGM precision matrix or its [Schur complement](https://en.wikipedia.org/wiki/Schur_complement). If one would
like to compute `correlation_matrix[indices, indices] @ vector`, one can use `ldgm[indices].solve(vector)`. To compute `inv(correlation_matrix[indices, indices]) @ vector`, use `ldgm[indices] @ vector`. Importantly, you cannot do this indexing manually - the submatrix of an inverse is not the inverse of a submatrix. See Section 5 of the supplementary material of [our paper](https://pubmed.ncbi.nlm.nih.gov/37640881/). 

```python
ldgm: PrecisionOperator = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
)

vector = np.random.randn(ldgm.shape[0])
precision_times_vector = ldgm @ vector
correlation_times_vector = ldgm.solve(precision_times_vector)
assert np.allclose(correlation_times_vector, vector)
```

### I/O and merging
We provide convenient functions to load LDGMs and merge them with data (probably summary statistics). You'll likely wish to make use of a metadata file that contains information about LDGMs: 

```python
ldgm_metadata: pl.DataFrame = gld.read_ldgm_metadata("data/test/metadata.csv", populations=["EUR"])
```
Then, load some summary statistics and partition their rows across LD blocks:

```python
sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
partitioned_sumstats: List[pl.DataFrame] = gld.partition_variants(ldgm_metadata, sumstats)
```

You can now load LDGMs and merge them with summary statistics:

```python
merged_ldgms = []
for row, df in zip(ldgm_metadata.iter_rows(named=True), partitioned_sumstats):
    ldgm: PrecisionOperator = gld.load_ldgm(
        filepath="data/test/" + row['name'],
        snplist_path="data/test/" + row['snplistName']
    )
    ldgm, _ = gld.merge_snplists(ldgm, df)
    merged_ldgms.append(ldgm)
```
After doing this, each entry of `merged_ldgms` will contain in its `variant_info` dataframe all of the columns from your summary statistics, and you can perform operations like:

```python
for ldgm in merged_ldgms:
    # Retain only the first SNP that matches each row/col of the LDGM
    z_scores = ldgm.variant_info.group_by('index', maintain_order=True) \
        .agg(pl.col('Z').first()).select('Z').to_numpy()
    solution = ldgm.solve(z_scores)
```
### Likelihood Functions

The likelihood of GWAS summary statistics under an infinitesimal model is:

> β ~ N(0, D)
>
> z|β ~ N(n^(1/2) R β, R)

where β is the effect-size vector in s.d-per-s.d. units, D is a diagonal matrix of per-variant heritabilities, z is the GWAS summary statistic vector, R is the LD correlation matrix, and n is the sample size. Our likelihood functions operate on precision-premultiplied GWAS summary statistics:

> pz = n^(-1/2) R^(-1) z ~ N(0, M),  where M = D + n^(-1) R^(-1)

The following functions are available:
- `gaussian_likelihood(pz, M)`: Computes the log-likelihood
- `gaussian_likelihood_gradient(pz, M, del_M_del_a=None)`: Computes the gradient of the log-likelihood, either with respect to the diagonal elements of `M` (equivalently `D`), or with respect to parameters `a` whose partial derivatives are provided in `del_M_del_a`. 
- `gaussian_likelihood_hessian(pz, M, del_M_del_a)`: Computes an approximation to the Hessian of the log-likelihood with respect to `a`. This is minus the average of the Fisher information matrix and the observed information matrix, and it is a good approximation when the gradient is close to zero.

### LD Clumping

LD clumping identifies independent index variants by iteratively selecting the variant with the highest $\chi^2$ statistic and pruning all variants in high LD with it. Clumping + thresholding is a popular (though suboptimal) way of computing polygenic scores.

```python
sumstats_clumped: pl.DataFrame = gld.run_clump(
    sumstats=sumstats_dataframe_with_z_scores,
    z_col='Z',
    ldgm_metadata_path="data/test/metadata.csv",
    populations='EUR',
    rsq_threshold=0.1,
    chisq_threshold=30.0,
).filter(pl.col('is_index'))
```

### Simulation

Summary statistics can be simulated from their asymptotic distribution without individual-level genotype data. Effect sizes are drawn from a flexible mixture distribution, with support for annotation-dependent and frequency-dependent architectures. Unlike the [MATLAB implementation](https://github.com/awohns/ldgm/blob/main/MATLAB/simulateSumstats.m), it does not support multiple ancestry groups.

```python
# Define a custom link function at module level
def annot_to_h2(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 9 * x[:, 1]  # 10x more h2 for coding vs. noncoding

sumstats: pl.DataFrame = gld.run_simulate(
    sample_size=10000,
    heritability=0.5,
    component_variance=[1,10],
    component_weight=[.01,.001],
    alpha_param=-1,
    populations='EUR',
    annotations=annot_df,
    link_fn=annot_to_h2,
)
```
The effect size distribution is a mixture of normal distributions with variances specified by `component_variance` and weights specified by `component_weight`. If weights sum to less than one, remaining variants have no effect. The `alpha_param` parameter controls the frequency-dependent architecture and should normally range between -1 and 1. 

Annotations can be included; if so, the `link_fn` should be specified, mapping annotations to relative per-variant heritability. For example, if we have the all-ones annotation and a coding annotation, then the link function could map to 10x more per-variant heritability for coding vs. noncoding variants.

Custom `link_fn` functions must be defined at module level (not as lambda functions or nested functions) to work with multiprocessing.

### Best Linear Unbiased Prediction (BLUP)
Under the infinitesimal model, with per-s.d. effect sizes $\beta\sim N(0, D)$, the BLUP effect sizes are:

$$
E(\beta) = \sqrt{n} D (nD + R^{-1})^{-1} R^{-1}z
$$

where we approximate $R^{-1}$ with the LDGM precision matrix. A parallelized implementation is provided:

```python
sumstats_with_weights: pl.DataFrame = gld.run_blup(
    ldgm_metadata_path="data/metadata.csv",
    sumstats=sumstats_dataframe_with_z_scores,
    heritability=0.1
)
```
This function assumes that heritability is equally distributed among all variants with Z scores provided, i.e., with $D=m^{-1}h^2 I$.

### Parallel Processing

`ParallelProcessor` is a base class which can be used to implement parallel algorithms with LDGMs, wrapping Python's `multiprocessing` module. It splits work among processes, each of which loads a subset of LD blocks. The advantage of using this is that it handles for you the loading of LDGMs within worker processes. An example can be found in `tests/test_multiprocessing.py`.

## File Formats

### LDSC Format Summary Statistics (.sumstats)
See [LDSC summary statistics file format](https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format). Read with `read_ldsc_sumstats`.

### GWAS-VCF Summary Statistics (.vcf)
The [GWAS-VCF specification](https://github.com/MRCIEU/gwasvcf) is supported via the `read_gwas_vcf` function. It is a VCF file with the following mandatory FORMAT fields:

- `ES`: Effect size estimate
- `SE`: Standard error of effect size
- `LP`: -log10 p-value

### Parquet Summary Statistics (.parquet)
Parquet files produced by [kodama](https://github.com/quattro/linear-dag) are supported via `read_parquet_sumstats`. The format stores per-trait columns as `{trait}_BETA` and `{trait}_SE`, allowing multiple traits per file. Variant info columns (`site_ids`/`SNP`, `chrom`/`CHR`, `position`/`POS`, `ref`/`REF`, `alt`/`ALT`) are detected automatically.

When using `graphld reml` with a parquet file containing multiple traits, use `--name` to specify which traits to process:

```bash
# Process specific traits
uv run graphld reml sumstats.parquet output --name height,bmi ...

# Process all traits (omit --name)
uv run graphld reml sumstats.parquet output ...
```

### LDSC Format Annotations (.annot)
You can download BaselineLD model annotation files with GRCh38 coordinates from the Price lab Google Cloud bucket: https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE/GRCh38

Read annotation files with `load_annotations`.

### BED Format Annotations (.bed)
You can also read UCSC `.bed` annotation files with `load_annotations`, and they will be added to the annotation dataframe with one column per file.

### GMT Format Annotations (.gmt)
GMT files are tab-separated with one row per gene set and no header. The first entry is the gene set name, the second is a description, and the remaining entries are gene IDs. You can also read GMT files with `src/score_test/score_test_io.load_gene_annotations`.

## See Also

- Main LDGM repository, including a MATLAB API: [https://github.com/awohns/ldgm](https://github.com/awohns/ldgm)
- Original graphREML repository, with a MATLAB implementation: [https://github.com/huilisabrina/graphREML](https://github.com/huilisabrina/graphREML) (we recommend using the Python implementation, which is much faster)
- LD score regression repository: [https://github.com/bulik/ldsc](https://github.com/bulik/ldsc)
- Giulio Genovese has implemented a LDGM-VCF file format specification and a bcftools plugin written in C with partially overlapping features, available [here](https://github.com/freeseek/score).
- `graphld` relies on sparse matrix operations implemented in [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
