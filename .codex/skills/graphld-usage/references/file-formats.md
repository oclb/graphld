# File Formats

GraphLD accepts these major input families:

- LDGM metadata CSV plus edgelist/snplist files.
- LDSC `.sumstats` files with either `Z` or `Beta`/`se`, plus `SNP`, `N`, `A1`, and `A2`.
- GWAS-VCF summary statistics.
- Parquet summary statistics, including multi-trait parquet files.
- LDSC `.annot` variant annotations.
- BED region annotations.
- GMT gene-set annotations with a gene table.
- HDF5 score-stat files for `estest`.

Fixture locations:

```text
data/test/
tests/score_test_data/
```

Always check fixture integrity before using these paths:

```bash
wc -c data/test/metadata.csv data/test/example.sumstats data/test/test.scores.h5
git status --short -- data/test tests/score_test_data
```
