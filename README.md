# GraphLD

This repository provides an implementation of the graphREML method described in:
> Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11. DOI: [10.1101/2024.11.04.24316716](https://doi.org/10.1101/2024.11.04.24316716)

and a Python API for computationally efficient linkage disequilibrium (LD) matrix operations with [LD graphical models](https://github.com/awohns/ldgm) (LDGMs), described in:
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: [10.1038/s41588-023-01487-8](https://pubmed.ncbi.nlm.nih.gov/37640881/)

It also provides very fast utilities for simulating GWAS summary statistics and computing polygenic risk scores.

## Table of Contents
- [Installation](#installation)
- [Command Line Interface](#command-line-interface)
- [API](#api)
- [File Formats](#file-formats)
- [See also](#see-also)

## Installation

Required system dependencies:
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (for CHOLMOD): On Mac, install with `brew install suitesparse`. SuiteSparse is wrapped in [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/).
- IntelMKL (for Intel chips): The performance of SuiteSparse is significantly improved by using IntelMKL instead of OpenBLAS, which will likely be the default. See Giulio Genovese's documentation [here](https://github.com/freeseek/score?tab=readme-ov-file#intel-mkl).

If you are only using the [heritability enrichment score test](#enrichment-score-test), nothing needs to be installed. Clone the package and run the source file as a standalone script: `uv run graphld/score_test.py --help`.

### Using uv (recommended)

In the repo directory:
```bash
uv venv
source .venv/bin/activate
uv sync
```

For development installation:
```bash
uv sync --dev --extra dev # editable with pytest dependencies
uv run pytest # tests will fail if you haven't run `make download` (see below)
```
### Using conda and pip install
Although we recommend moving away from `conda`, it does have the advantage that you can `conda install` SuiteSparse. Example codes are based on the O2 cluster in the Harvard Medical School computing system. 

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
cd graphld && pip install .
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

The download should take 5-30 minutes. By default, this will download and extract LDGM precision matrices, baselineLD annotation files for heritability partitioning with graphREML, and precomputed score test statistics for the heritability enrichment score test. If you only want precision matrices, use `make download_precision`. If you only want enrichment score test statistics, use `make download_scorestats`. If you want to download GWAS summary statistics from Li et al. 2025, additionally run `make download_sumstats`. 

## Command Line Interface

The CLI has commands for:
- `blup`, which computes best linear unbiased predictor weights
- `clump`, which p-value thresholding and LD-based pruning
- `simulate`, which simulates GWAS summary statistics
- `reml`, which runs graphREML

The commands largely follow a common interface. Use `uv run graphld <command> -h` to see usage for a specific command. If you downloaded LDGMs using the Makefile, they will be located automatically.

There is a separate command for the [enrichment score test](#enrichment-score-test): `uv run estest -h`.

### Heritability Estimation

To run graphREML:

```bash
uv run graphld reml \
    /path/to/sumstats/file.sumstats \
    output_files_prefix \
    --annot-dir /directory/containing/annotation/files/ \
```
The summary statistics can be in VCF (`.vcf`) or  LDSC (`.sumstats`) format. The annotation directory should contain per-chromosome annotation files in [LDSC (`.annot`) format](https://github.com/bulik/ldsc/wiki/LD-File-Formats#annot). There can be multiple `.annot` files per chromosome, including some in the [`thin-annot` format]((https://github.com/bulik/ldsc/wiki/LD-Score-Estimation-Tutorial#partitioned-ld-scores)) (i.e., without variant IDs). It can additionally contain UCSC `.bed` files, not stratified per-chromosome; for each `.bed` file, a new binary annotation will be created with a `1` for variants whose GRCh38 coordinates match the `.bed` file.

There will be two output files: `output_files_prefix.tall.csv`, which contains heritability, enrichment, and coefficient estimates for each annotation; and `output_files_prefix.convergence.csv`, which contains information about the optimization process. If you specify the `--alt-output` flag, the `tall.csv` file will be replaced with three files, `.heritability.csv`, `.enrichment.csv`, and `.parameters.csv`, containing heritability, enrichment, and coefficient estimates for each annotation, respectively; these files have one line per model run and three columns per annotation, so that you can store the results of multiple runs or traits in one file. (Use it with `--name` to keep track of which line is which run.)

An important flag is `--intercept`, which specifies the expected inflation in the $\chi^2$ statistics and which cannot be estimated using graphREML. It is recommended to (1) run LD score regression and estimate the intercept and (2) specify that value with the `--intercept` flag. Not doing this leads to upward bias in the heritability estimates and downward bias (i.e., toward 1) in the enrichment estimates. For example, UK Biobank height has an intercept of around 1.5, and specifying this value changes the coding variant enrichment estimate from around 7 to around 11. Most traits will have smaller intercepts. You can skip this if you are OK with some downward bias.

Non-convergence can be an issue in this kind of algorithm. We've tried to ensure that the trust region procedure is robust, but you should inspect the `convergence.csv` file and check the log-likelihood changes. If these abruptly go from some large value (e.g., 10) to nearly zero (e.g., 1e-4), it indicates a problem with convergence. Try (1) increasing the number of samples in the stochastic trace estimator with  `--xtrace-num-samples 200`, (2) resetting the trust region at each iteration with `--reset-trust-region`, and (3) as a last resort, discarding LD blocks containing large-effect variants using `--max-chisq-threshold`, e.g., `--max-chisq-threshold 300`.

If some variants are missing from your GWAS summary statistics, graphREML automatically assigns 'surrogate markers' in high LD with those variants. This is a majority of variants if your summary statitics include HapMap3 SNPs only (~1.1M SNPs). This step can be very slow, but you can use precomputed surrogates to avoid re-doing it each time. To precompute surrogates, run `graphld surrogates` on your summary statistics. This creates a file with cached surrogate markers which can be passed to `graphREML` with the `--surrogates` flag. These should match your summary statistics approximately but do not need to match exactly; you can use the same surrogates across different sumstats files with slightly different SNPs.

### Enrichment Score Test

The enrichment score test is a fast way to test a large number of genomic annotations for heritability enrichment conditional upon some null model. It produces Z statistics, but not point estimates. It requires precomputed derivatives (first and second derivatives of the log-likelihood for each variant). Run graphREML as described above with whatever annotations you wish to include in the null model (i.e., not the ones being tested), and supply the `--score-test-filename` flag to create a file containing pre-computed derivatives. The file also includes all the annotations in the null model, which will make it somewhat large (>1GB). Data for multiple traits can be added to the same file, but they must share the same annotations.

To perform the score test:

```bash
uv run estest \
    path/to/precomputed/derivatives.h5 \
    path/to/output/file/prefix \
    -a /directory/containing/annotation/files/to/test \
```

Alternatively, use `uv run graphld/score_test.py` with the same arguments. All of the dependencies for this script are included as script dependencies, so there should be no installation needed. 

A file will be saved called `output_prefix.txt` containing one row per annotation and one column per trait, plus a column named `meta_analysis`. Each entry is a Z score, where a positive value indicates that an annotation is enriched for heritability conditional upon the null model (i.e., that it has a positive value of `tau`)..

Optional arguments:
-   `-a`, `--annotations_dir`: Directory containing the new annotation files (in `.annot` format; see above) that you want to test for enrichment.  Optionally, include UCSC `.bed` files (one for the whole genome). Default: "data/annotations/", which is populated by the Makefile.
-   `--annotations`: Optional comma-separated list with a subset of annotation names to be tested (header names in the `.annot` files).
-   `--trait_name`: Optional name of a single trait to be tested (specified with `--name` in the `graphld reml` run). If unspecified, all traits are tested and a meta-analysis is performed.
- `--add-random `: Optional comma-separated list of numbers between 0 and 1; for each entry, a random annotation will be created and tested with proportion of variants equal to the number specified. You can use this to validate that the test produces correctly calibrated Z scores.

## API

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
correlation_times_vector = ldgm.solve(result)
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
chrom,chromStart,chromEnd,name,snplistName,population,numVariants,numIndices,numEntries,info

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

$$\beta \sim N(0, D)$$
$$z|\beta \sim N(n^{1/2}R\beta, R)$$
where $\beta$ is the effect-size vector in s.d-per-s.d. units, $D$ is a diagonal matrix of per-variant heritabilities, $z$ is the GWAS summary statistic vector, $R$ is the LD correlation matrix, and $n$ is the sample size. Our likelihood functions operate on  precision-premultiplied GWAS summary statistics: 
$$pz = n^{-1/2} R^{-1}z \sim N(0, M), M = D + n^{-1}R^{-1}.$$ 

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
The effect size distribution is a mixture of normal distributions with variances specified by `component_variance` and weights specified by `component_weight`. If weights sum to less than one, remaining variants have no effect. The `alpha_param` parameter controls the frequency-dependent architecture and should normally range between -1 and 1. Annotations can be included; if so, the `link_fn` should be specified, mapping annotations to relative per-variant heritability. For example, if we have the all-ones annotation and a coding annotation, then the link function could be `lambda x: x[:, 0] + 9 * x[:, 1]`, giving 10x more per-variant heritability for coding vs. noncoding variants.

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

### Multiprocessing

`ParallelProcessor` is a base class which can be used to implement parallel algorithms with LDGMs, wrapping Python's `multiprocessing` module. It splits work among processes, each of which loads a subset of LD blocks. The advantage of using this is that it handles for you the loading of LDGMs within worker processes. An example can be found in `tests/test_multiprocessing.py`.

## File Formats

### LDSC Format Summary Statistics (.sumstats)
See [LDSC summary statistics file format](https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format). Read with `read_ldsc_sumstats`.

### GWAS-VCF Summary Statistics(.vcf)
The [GWAS-VCF specification](https://github.com/MRCIEU/gwasvcf) is supported via the `read_gwas_vcf` function. It is a VCF file with the following mandatory FORMAT fields:

- `ES`: Effect size estimate
- `SE`: Standard error of effect size
- `LP`: -log10 p-value

### LDSC Format Annotations (.annot)
You can download BaselineLD model annotation files with GRCh38 coordinates from the Price lab Google Cloud bucket: https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE/GRCh38

Read annotation files with `load_annotations`.

### BED Format Annotations (.bed)
You can also read UCSC `.bed` annotation files with `load_annotations`, and they will be added to the annotation dataframe with one column per file.

## See Also

- Main LDGM repository, including a MATLAB API: [https://github.com/awohns/ldgm](https://github.com/awohns/ldgm)
- Original graphREML repository, with a MATLAB implementation: [https://github.com/huilisabrina/graphREML](https://github.com/huilisabrina/graphREML) (we recommend using the Python implementation, which is much faster)
- LD score regression repository: [https://github.com/bulik/ldsc](https://github.com/bulik/ldsc)
- Giulio Genovese has implemented a LDGM-VCF file format specification and a bcftools plugin written in C with partially overlapping features, available [here](https://github.com/freeseek/score).
- All of these rely heavily on sparse matrix operations implemented in [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
