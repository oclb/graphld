# Best Linear Unbiased Prediction (BLUP)

Under the infinitesimal model, with per-s.d. effect sizes β ~ N(0, D), BLUP effect sizes are:

$$
E(\beta) = \sqrt{n} D (nD + R^{-1})^{-1} R^{-1}z
$$

where GraphLD approximates $R^{-1}$ with the LDGM precision matrix.

```python
import polars as pl
import graphld as gld

sumstats_with_weights: pl.DataFrame = gld.run_blup(
    ldgm_metadata_path="data/metadata.csv",
    sumstats=sumstats_dataframe_with_z_scores,
    heritability=0.1,
    sample_size=100000,
)
```

`heritability` is total trait heritability. This workflow assumes heritability is equally distributed among the matched LDGM effect indices, corresponding to $D = m^{-1} h^2 I$ with $\mathrm{trace}(D) = h^2$.

See also:

- [Command Line Interface](../cli.md)
- [graphld.blup API Reference](../api/blup.md)
- [graphld package overview](../api/graphld.md)
