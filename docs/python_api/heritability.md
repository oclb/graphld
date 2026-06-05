# Heritability Estimation

Use `graphld.run_graphREML` to estimate heritability, enrichment, and annotation coefficients from summary statistics and LDGMs.

```python
import graphld as gld
import polars as pl

sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
annotations: pl.DataFrame = gld.load_annotations(
    "data/test/annot",
    chromosome=1,
    positions_file="data/test/rsid_position.csv",
)

default_model_options = gld.ModelOptions()
default_method_options = gld.MethodOptions()

reml_results: dict = gld.run_graphREML(
    model_options=default_model_options,
    method_options=default_method_options,
    summary_stats=sumstats,
    annotation_data=annotations,
    ldgm_metadata_path="data/test/metadata.csv",
    populations="EUR",
)
```

The returned dictionary contains heritability, enrichment, and coefficient estimates for each annotation, together with standard errors and two-tailed log10 p-values.

See also:

- [Command Line Interface](../cli.md)
- [graphld.heritability API Reference](../api/heritability.md)
- [graphld package overview](../api/graphld.md)
