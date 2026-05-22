# Data Downloads

Do not download large GraphLD data unless the user explicitly asks. Prefer tracked fixtures in `data/test/` for examples and tests, but first check whether the files are present and non-empty.

The default downloaded LDGM location is:

```text
data/ldgms/metadata.csv
```

Metadata rows point to LDGM edgelist files and snplist files in the same directory. README/docs describe Makefile targets such as `download_reml`, `download_scores`, `download_gene_scores`, `download_surrogates`, `download_sumstats`, and `download_all`; verify `data/Makefile` before relying on those targets in the current checkout.
