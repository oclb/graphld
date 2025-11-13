# GraphLD Score Test

Standalone score test module for testing annotation enrichment in GWAS data.

## Files

- **score_test.py**: Main score test implementation and CLI (standalone executable)
- **score_test_io.py**: All I/O operations (HDF5, annotations)
- **genesets.py**: Gene set handling and gene-to-variant conversion

## Running as Standalone Script

The module can be run as a standalone script without installing the package. It uses PEP 723 inline script metadata to declare dependencies.

### Using `uv run` (recommended)

`uv` will automatically install dependencies in an isolated environment:

```bash
# Test with variant-level annotations
uv run src/graphld_score_test/score_test.py variant_stats.h5 -a /path/to/annot/dir

# Test with gene-level annotations (GMT format)
uv run src/graphld_score_test/score_test.py variant_stats.h5 -g /path/to/gmt/dir

# Test with random gene annotations
uv run src/graphld_score_test/score_test.py variant_stats.h5 --random-genes 0.1,0.01 --seed 42

# Test with random variant annotations
uv run src/graphld_score_test/score_test.py variant_stats.h5 --random-variants 0.1,0.01 --seed 42
```

### Using Python directly

If dependencies are already installed:

```bash
cd src/graphld_score_test
python3 score_test.py ../../variant_stats.h5 --random-variants 0.1,0.2
```

## Usage Examples

```bash
# Test multiple random variant annotations
uv run src/graphld_score_test/score_test.py scorestats/bld_chr22.variants.h5 \
  --random-variants .1,.2,.3,.4 --seed 42

# Test multiple random gene annotations
uv run src/graphld_score_test/score_test.py scorestats/bld_chr22.variants.h5 \
  --random-genes .1,.2,.3,.4 --seed 42

# Save results to file
uv run src/graphld_score_test/score_test.py scorestats/bld_chr22.variants.h5 \
  --random-variants .1,.2 --seed 42 -o results/my_test

# Verbose output
uv run src/graphld_score_test/score_test.py scorestats/bld_chr22.variants.h5 \
  --random-variants .1,.2 --seed 42 -v
```

## Dependencies

All dependencies are declared in the script metadata and automatically installed by `uv run`:

- click
- h5py
- numpy
- polars
- scipy
