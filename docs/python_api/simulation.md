# Simulation

GraphLD can simulate GWAS summary statistics directly from their asymptotic distribution without individual-level genotype data.

Effect sizes are drawn from a flexible mixture distribution with support for annotation-dependent and frequency-dependent architectures.

```python
import numpy as np
import polars as pl
import graphld as gld


def annot_to_h2(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 9 * x[:, 1]


sumstats: pl.DataFrame = gld.run_simulate(
    sample_size=10000,
    heritability=0.5,
    component_variance=[1, 10],
    component_weight=[0.01, 0.001],
    alpha_param=-1,
    populations="EUR",
    annotations=annot_df,
    link_fn=annot_to_h2,
)
```

`component_variance` and `component_weight` define the mixture. If the weights sum to less than one, the remaining variants have zero effect. `alpha_param` controls frequency dependence and typically ranges from `-1` to `1`.

If annotations are included, `link_fn` should map annotations to relative per-variant heritability. Define custom link functions at module scope rather than as lambdas or nested functions so multiprocessing can import them.

See also:

- [Command Line Interface](../cli.md)
- [graphld.simulate API Reference](../api/simulate.md)
- [graphld package overview](../api/graphld.md)
