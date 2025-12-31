# File Formats

GraphLD supports various input and output file formats for summary statistics, annotations, and gene sets.

## Summary Statistics

### LDSC Format (.sumstats)

The standard [LDSC summary statistics format](https://github.com/bulik/ldsc/wiki/Summary-Statistics-File-Format).

Read with:
```python
import graphld as gld
sumstats = gld.read_ldsc_sumstats("path/to/file.sumstats")
```

### GWAS-VCF Format (.vcf)

The [GWAS-VCF specification](https://github.com/MRCIEU/gwasvcf) is supported. Required FORMAT fields:

| Field | Description |
|-------|-------------|
| `ES` | Effect size estimate |
| `SE` | Standard error of effect size |
| `LP` | -log10 p-value |

Read with:
```python
import graphld as gld
sumstats = gld.read_gwas_vcf("path/to/file.vcf")
```

### Parquet Format (.parquet)

Parquet files produced by [kodama](https://github.com/quattro/linear-dag) are supported. The format stores per-trait columns as `{trait}_BETA` and `{trait}_SE`, allowing multiple traits per file.

Variant info columns are detected automatically:

- `site_ids` or `SNP`
- `chrom` or `CHR`
- `position` or `POS`
- `ref` or `REF`
- `alt` or `ALT`

Read with:
```python
import graphld as gld
sumstats = gld.read_parquet_sumstats("path/to/file.parquet")
```

#### CLI Usage with Multi-Trait Parquet

```bash
# Process specific traits
uv run graphld reml sumstats.parquet output --name height,bmi

# Process all traits
uv run graphld reml sumstats.parquet output
```

## Annotations

### LDSC Format (.annot)

Per-chromosome annotation files in [LDSC format](https://github.com/bulik/ldsc/wiki/LD-File-Formats#annot).

Download BaselineLD model annotations (GRCh38) from the [Price lab Google Cloud bucket](https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE/GRCh38).

Both standard and `thin-annot` formats (without variant IDs) are supported.

Read with:
```python
import graphld as gld
annotations = gld.load_annotations("path/to/annot_dir/", chromosome=1)
```

### BED Format (.bed)

UCSC BED files containing genomic regions. Each `.bed` file creates a binary annotation with `1` for variants whose GRCh38 coordinates match.

BED files should not be stratified per-chromosome. Place them in the annotation directory and they will be processed automatically.

Read with:
```python
import graphld as gld
annotations = gld.load_annotations("path/to/annot_dir/", chromosome=1)
```

### GMT Format (.gmt)

Gene Matrix Transposed format for gene sets. Tab-separated with:

1. Gene set name
2. Description
3. Gene IDs or symbols (remaining columns)

No header row.

Example:
```
PATHWAY_A    Description of pathway A    GENE1    GENE2    GENE3
PATHWAY_B    Description of pathway B    GENE4    GENE5
```

Read with:
```python
from score_test.score_test_io import load_gene_annotations
gene_annotations = load_gene_annotations("path/to/file.gmt")
```

## LDGM Files

### Edge List Format

LDGM precision matrices are stored as edge lists. Files are named like:
```
1kg_chr1_16103_2888443.EAS.edgelist
```

### SNP List Format

Associated SNP lists contain variant information:
```
1kg_chr1_16103_2888443.snplist
```

### Metadata CSV

The metadata file contains information about all LDGM blocks:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome |
| `chromStart` | Block start position |
| `chromEnd` | Block end position |
| `name` | Edge list filename |
| `snplistName` | SNP list filename |
| `population` | Population code (e.g., EUR, EAS) |
| `numVariants` | Number of variants |
| `numIndices` | Number of matrix indices |
| `numEntries` | Number of non-zero entries |

Read with:
```python
import graphld as gld
metadata = gld.read_ldgm_metadata("path/to/metadata.csv", populations=["EUR"])
```
