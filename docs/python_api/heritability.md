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

## Options and convergence

Pass `ModelOptions` and `MethodOptions` to configure the model and fit. For example:

```python
model_options = gld.ModelOptions(link_function="softplus")
method_options = gld.MethodOptions(
    num_iterations=100,
    num_processes=4,
    num_jackknife_blocks=100,
)
```

Inspect the returned status before interpreting estimates:

```python
log = reml_results["log"]
print(log["converged"], log["optimizer_status"], log["uncertainty_status"])
```

Convergence establishes local checks. Uncertainty at unresolved endpoints is
provisional; singular information or invalid delete calculations make it unavailable.
The default tolerance, `0.001`, controls precise score and objective checks.
The retained `trust_region_*`, `max_trust_iterations`, `reset_trust_region`,
`minimum_likelihood_increase`, and `convergence_window` settings have no effect.

The optional `MethodOptions(information_penalty=1.0)` adds an information penalty;
its bias reduction and uncertainty calibration have not been established.
See the [methods write-up](../methods/graphreml_optimization/README.md) for details.

See also:

- [Command Line Interface](../cli.md)
- [graphld.heritability API Reference](../api/heritability.md)
- [graphld package overview](../api/graphld.md)
