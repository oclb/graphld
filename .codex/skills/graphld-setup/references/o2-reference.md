# HMS O2 Reference

This is historical O2-specific guidance. Do not assume the user is on O2, and do not present these notes as generally applicable outside that cluster.

O2 setup previously needed these special steps:

1. Install or load conda.
2. Install SuiteSparse/CHOLMOD into conda.
3. Prefer Intel MKL BLAS for SuiteSparse performance.
4. Run `uv sync` with SuiteSparse include/library paths available.
5. Use `polars-lts-cpu` if default `polars` crashes with `Illegal instruction` on O2 compute nodes.
6. Keep `scikit-sparse` below `0.5` if GraphLD sees the newer `cholesky()` return API or hangs in worker processes.
7. Set runtime library/thread variables in jobs:

```bash
export LD_LIBRARY_PATH=<conda_base>/lib:$LD_LIBRARY_PATH
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Old O2 notes pinned `scikit-sparse==0.4.16`; the current local lock resolved `0.4.15`. The durable instruction is to avoid `0.5.x` unless GraphLD has been updated for that API.

The old O2 recipe also recommended `--run-in-serial` for graphREML jobs because multiprocessing could hang on that cluster. Treat that as an O2-specific operational note, not a general GraphLD rule.
