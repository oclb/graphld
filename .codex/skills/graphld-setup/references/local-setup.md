# Local Setup

Use `uv` from the repo root:

```bash
uv venv
uv sync --extra dev
```

Run commands through `uv run` so they use the project environment:

```bash
uv run graphld -h
uv run graphld reml -h
uv run estest --help
uv run --extra dev pytest tests/test_precision.py -q
```

Full GraphLD imports depend on `scikit-sparse`, which depends on SuiteSparse/CHOLMOD. On macOS, SuiteSparse is commonly installed with Homebrew:

```bash
brew install suitesparse
```

On Ubuntu/Debian, the usual system package is:

```bash
sudo apt-get install libsuitesparse-dev
```

If `scikit-sparse` fails to build or import, check that CHOLMOD headers and libraries are available before debugging GraphLD itself. If local package metadata looks stale, trust `pyproject.toml` and imports from `src/` over old `*.egg-info` directories.
