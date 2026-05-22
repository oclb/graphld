# I/O and Merging

GraphLD provides helpers to load LDGMs, read metadata, partition summary statistics across LD blocks, and merge variants onto LDGM indices.

Start by loading metadata:

```python
import graphld as gld
import polars as pl

ldgm_metadata: pl.DataFrame = gld.read_ldgm_metadata(
    "data/test/metadata.csv",
    populations=["EUR"],
)
```

Then partition summary statistics across LD blocks:

```python
sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
partitioned_sumstats = gld.partition_variants(ldgm_metadata, sumstats)
```

Load LDGMs and merge them with summary statistics:

```python
merged_ldgms = []
for row, df in zip(ldgm_metadata.iter_rows(named=True), partitioned_sumstats):
    ldgm = gld.load_ldgm(
        filepath="data/test/" + row["name"],
        snplist_path="data/test/" + row["snplistName"],
    )
    ldgm, _ = gld.merge_snplists(ldgm, df)
    merged_ldgms.append(ldgm)
```

After merging, each `ldgm.variant_info` table carries the summary-statistics columns for matched variants:

```python
for ldgm in merged_ldgms:
    z_scores = (
        ldgm.variant_info.group_by("index", maintain_order=True)
        .agg(pl.col("Z").first())
        .select("Z")
        .to_numpy()
    )
    solution = ldgm.solve(z_scores)
```

See also:

- [File Formats](../file_formats.md)
- [graphld.io API Reference](../api/io.md)
- [graphld.ldsc_io API Reference](../api/ldsc_io.md)
- [graphld.vcf_io API Reference](../api/vcf_io.md)
- [graphld.parquet_io API Reference](../api/parquet_io.md)
- [graphld package overview](../api/graphld.md)
