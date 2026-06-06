# Data Downloads

Do not download large GraphLD data unless the user wants setup for a workflow that needs it. When data needs are unclear, first ask:

> Score test only, or do you need LDGMs for graphREML, BLUP, clumping, simulation, or surrogate-marker work?

Then choose the smallest existing `data/Makefile` target that fits. Before running a download target, mention the approximate size and that it downloads external data.

Tracked tests should use `data/test/`. If a test fails because downloaded data under `data/` is missing, treat that as test/configuration drift rather than a reason to fetch large datasets.

## Workflow Map

Run targets from `data/`, for example `cd data && make download_gene_scores`.

| User workflow | Follow-up question | Target | Approx. size |
| --- | --- | --- | --- |
| Quick validation, examples backed by fixtures, or tests | Check whether `data/test/` has the needed fixture. | No download by default | 0 |
| Score test only | Gene-set tests or variant-annotation tests? | `download_gene_scores` for gene-set tests; `download_scores` for variant annotations | 10 MB or 6.5 GB |
| graphREML | Confirm UKBB/EUR-style bundled data is appropriate. | `download_reml` | 2 GB |
| BLUP, clumping, simulation, or surrogate-marker generation with UKBB/EUR LDGMs | Confirm UKBB/EUR LDGMs are sufficient. | `download_ukbb_precision` | 1.5 GB |
| LDGM work needing non-UKBB or multi-ancestry 1000 Genomes LDGMs | Confirm all-population LDGMs are needed. | `download_precision` | 10 GB |
| Any LDGM workflow where the user also wants bundled GWAS summary statistics | Ask after selecting the LDGM target. | Add `download_sumstats` | 7 GB |
| User explicitly wants every data product | Confirm broad download intent. | `download_all` | 25 GB |

Chromosome selection is an analysis-time filter after installing the needed LDGM target.

The default downloaded LDGM location is:

```text
data/ldgms/metadata.csv
```

Metadata rows point to LDGM edgelist files and snplist files in the same directory. README/docs describe Makefile targets such as `download_reml`, `download_ukbb_precision`, `download_precision`, `download_scores`, `download_gene_scores`, `download_sumstats`, and `download_all`; verify `data/Makefile` before relying on those targets in the current checkout.
