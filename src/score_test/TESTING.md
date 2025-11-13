# Testing Guide

## Test Files

### Test Data (`test_data/`)
- **genes_test.tsv**: Sample gene table with 15 genes from chromosome 22
- **test_symbols.gmt**: GMT file with gene sets using gene symbols (OSM, NCF4, etc.)
- **test_ids.gmt**: GMT file with gene sets using Ensembl IDs (ENSG00000099985, etc.)

### Test Modules
- **test_score_test_io.py**: Tests for all I/O operations
- **test_genesets.py**: Tests for gene set loading and conversion

## Running Tests

```bash
# Run all tests
pytest src/graphld_score_test/

# Run specific test file
pytest src/graphld_score_test/test_score_test_io.py -v

# Run specific test class
pytest src/graphld_score_test/test_genesets.py::TestLoadGeneSetsFromGmt -v

# Run with coverage
pytest src/graphld_score_test/ --cov=src/graphld_score_test --cov-report=html
```

## Test Coverage

### `test_score_test_io.py`
- ✅ `load_variant_data`: Basic loading, file not found error
- ✅ `load_trait_data`: Basic loading, correct data shapes
- ✅ `load_annotations`: Single/multiple chromosomes, binary conversion, multiple files per chromosome, no files found error
- ✅ `load_variant_annotations`: All columns, filtered columns
- ✅ `load_gene_annotations`: Basic loading and conversion from GMT
- ✅ `create_random_gene_annotations`: Basic creation, reproducibility with seed
- ✅ `create_random_variant_annotations`: Basic creation, value validation, reproducibility with seed

### `test_genesets.py`
- ✅ `_is_gene_id`: Ensembl ID detection, gene symbol detection
- ✅ `load_gene_table`: Basic loading, chromosome filtering, midpoint calculation
- ✅ `load_gene_sets_from_gmt`: Symbol-based sets, ID-based sets, multiple files, no files error
- ✅ `convert_gene_sets_to_variant_annotations`: Basic conversion, ID-based conversion, empty sets

## Test Data Creation

The test data was created as follows:

```bash
# Extract chr22 genes from main gene table
awk '$6=="22"' data/genes.tsv | head -15 > src/graphld_score_test/test_data/genes_test.tsv

# Add header
head -1 data/genes.tsv > temp_header.tsv
cat temp_header.tsv src/graphld_score_test/test_data/genes_test.tsv > temp_combined.tsv
mv temp_combined.tsv src/graphld_score_test/test_data/genes_test.tsv
```

GMT files were manually created with:
- **test_symbols.gmt**: 2 gene sets with 3 genes each (symbols)
- **test_ids.gmt**: 2 gene sets with 3 genes each (Ensembl IDs)

## Expected Test Results

All tests should pass. Key validations:
- HDF5 files are correctly read with proper data types
- Annotation files are merged horizontally when multiple files exist
- Binary annotations (0/1) are converted to boolean type
- Gene sets are correctly parsed from GMT format
- Gene-to-variant conversion produces valid LDSC-format output
- Both gene symbols and Ensembl IDs are properly handled
