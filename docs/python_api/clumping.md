# LD Clumping

LD clumping identifies independent index variants by iteratively selecting the variant with the highest chi-squared statistic and pruning variants in high LD with it.

```python
import polars as pl
import graphld as gld

sumstats_clumped: pl.DataFrame = gld.run_clump(
    sumstats=sumstats_dataframe_with_z_scores,
    z_col="Z",
    ldgm_metadata_path="data/test/metadata.csv",
    populations="EUR",
    rsq_threshold=0.1,
    chisq_threshold=30.0,
).filter(pl.col("is_index"))
```

Clumping plus thresholding is a common baseline for polygenic score construction, though it is typically less efficient than model-based methods.

See also:

- [Command Line Interface](../cli.md)
- [graphld.clumping API Reference](../api/clumping.md)
- [graphld package overview](../api/graphld.md)
