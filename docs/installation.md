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

## Downloading LDGMs

Pre-computed LDGMs for the 1000 Genomes Project data are available at [Zenodo](https://zenodo.org/records/8157131). Download using the provided Makefile:

```bash
cd data && make download
```

The download takes 5-30 minutes. Options:

| Command | Contents |
|---------|----------|
| `make download` | Full download (precision matrices, annotations, score statistics) |
| `make download_precision` | Precision matrices only |
| `make download_scorestats` | Enrichment score test statistics only |
| `make download_sumstats` | GWAS summary statistics from Li et al. 2025 |
