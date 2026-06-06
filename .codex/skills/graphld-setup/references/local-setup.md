# Local Setup

Use `uv` from the repo root:

```bash
uv sync --extra dev
```

Run commands through `uv run` so they use the project environment:

```bash
uv run graphld -h
uv run graphld reml -h
uv run estest --help
uv run --extra dev pytest tests/test_precision.py -q
```

Full GraphLD imports depend on `scikit-sparse`, which depends on SuiteSparse/CHOLMOD. GraphLD supports Python 3.11-3.13. On macOS, use Homebrew or another system Python in that range so source builds use the active Command Line Tools SDK instead of stale compiler flags from an older uv-managed Python.

On macOS, SuiteSparse is commonly installed with Homebrew:

```bash
brew install suitesparse
```

On Ubuntu/Debian, the usual system package is:

```bash
sudo apt-get install libsuitesparse-dev
```

If `scikit-sparse` fails to build or import, check that CHOLMOD headers and libraries are available before debugging GraphLD itself. If local package metadata looks stale, trust `pyproject.toml` and imports from `src/` over old `*.egg-info` directories.
