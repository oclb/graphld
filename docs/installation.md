# Installation

## Enrichment Score Test Only

**If you only need the [enrichment score test](score_test.md), no installation or dependencies are required.** With `uv` installed, simply run:

```bash
uv run src/score_test/score_test.py --help
```

Or make the script executable:
```bash
chmod +x src/score_test/score_test.py
./src/score_test/score_test.py --help
```

---

## Full Installation

The full `graphld` package (for graphREML, simulation, clumping, BLUP) requires [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) for sparse matrix operations.

### Prerequisites

**SuiteSparse**

On Mac:
```bash
brew install suitesparse
```

On Ubuntu/Debian:
```bash
sudo apt-get install libsuitesparse-dev
```

**Intel MKL (Recommended)**

For users with Intel chips, Intel MKL can produce a 100x speedup with SuiteSparse vs. OpenBLAS (your likely default BLAS library). See [Giulio Genovese's documentation](https://github.com/freeseek/score?tab=readme-ov-file#intel-mkl).

### Using uv (Recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. In the repo directory:

```bash
uv venv && uv sync
```

For development installation:
```bash
uv venv && uv sync --dev --extra dev  # editable with pytest dependencies
uv run pytest  # tests will fail if you haven't run `make download`
```

### Using conda and pip

Conda has the advantage that you can `conda install` SuiteSparse directly.

**Create conda environment:**

```bash
module load miniconda3/4.10.3  # if on a cluster
conda create -n suitesparse conda-forge::suitesparse python=3.11.0
conda activate suitesparse
```

You may need to revert or reinstall some Python packages:
```bash
pip install numpy==1.26.4
```

**Install scikit-sparse:**

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install scikit-sparse
```

**Install graphld:**

```bash
cd graphld && pip install .
```

**Test installation:**

```bash
graphld -h
```

## Downloading Data

Pre-computed LDGMs and data files are available from Zenodo. Download using the provided Makefile:

```bash
cd data && make download_all
```

The full download takes 30-60 minutes depending on connection speed.

### Recommended Downloads

| Use Case | Command | Size |
|----------|---------|------|
| Score test for gene sets only | `make download_gene_scores` | ~10 MB |
| Score test for variant annotations | `make download_scores` | ~6.5 GB |
| graphREML on European-ancestry data | `make download_reml` | ~2 GB |
| All populations / all features | `make download_all` | ~25 GB |

To try out graphREML or score test with example summary statistics, additionally run `make download_sumstats` (~7 GB).

### All Download Options

| Command | Description | Size |
|---------|-------------|------|
| `make download_all` | All data files | ~25 GB |
| `make download_reml` | UKBB precision + annotations + surrogates | ~2 GB |
| `make download_ukbb_precision` | UK Biobank LDGM precision matrices | ~1.5 GB |
| `make download_precision` | All LDGM precision matrices (all populations) | ~10 GB |
| `make download_annotations` | BaselineLD annotation files | ~400 MB |
| `make download_scores` | Score statistics (variant + gene level) | ~6.5 GB |
| `make download_gene_scores` | Gene-level score statistics only | ~10 MB |
| `make download_surrogates` | Surrogate markers + gene table | ~60 MB |
| `make download_sumstats` | GWAS summary statistics (Li et al. 2025) | ~7 GB |

### Data Sources

- **Precision matrices**: [Zenodo 8157131](https://zenodo.org/records/8157131) - LDGM precision matrices for 1000 Genomes populations
- **Annotations & sumstats**: [Zenodo 15085817](https://zenodo.org/records/15085817) - BaselineLD annotations and UK Biobank summary statistics
- **Score statistics**: [Zenodo 18102484](https://zenodo.org/records/18102484) - Pre-computed score statistics for enrichment testing

### Directory Structure

After downloading, the `data/` directory will contain:

```
data/
├── ldgms/              # LDGM precision matrices
├── baselineld/         # BaselineLD annotation files
├── scores/             # Score statistics (.h5 files)
├── surrogates/         # Surrogate marker files
├── genes.tsv           # Gene table (GRCh38)
└── rsid_position.csv   # SNP position mapping
```
