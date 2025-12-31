# Code Review Notes for Release

This document tracks issues found during code review that require careful consideration before fixing. These are NOT simple linting fixes - they represent potential logic changes or architectural decisions.

## Coverage Gaps

### Modules with 0% Coverage (Need Tests)
1. **graphld/genesets.py** (88 statements) - Gene set annotation utilities
2. **graphld/genesets_gene_hdf5.py** (207 statements) - Gene-level HDF5 operations  
3. **graphld/heritability_testing.py** (509 statements) - Heritability testing module

### Modules with Low Coverage (<50%)
1. **graphld/cli.py** (47%) - Command line interface; many branches untested
2. **score_test/convert_scores.py** (38%) - Score conversion utilities

## Potential Issues to Review

### graphld package

#### Star Imports (heritability.py, heritability_testing.py)
- Lines 8, 16, 17 in heritability.py: `from typing import *`, `from .io import *`, `from .likelihood import *`
- This causes F403/F405 linting errors and makes the code harder to understand
- **Decision needed**: Replace with explicit imports (may require significant refactoring)

#### Undefined Name References (io.py)
- Line 15: `PrecisionOperator` referenced before it's imported
- This is a forward reference issue that may cause runtime errors in some contexts
- **Decision needed**: Add proper TYPE_CHECKING import or restructure

#### Unused Variable (ldsc_io.py)
- Line 77: `columns_to_read` is assigned but never used
- **Decision needed**: Remove or use the variable

### score_test package

#### Potential Logic Bug (score_test_io.py, lines 292-293)
```python
if not add_positions:
    annotations = annotations.rename({"BP": "POS"})
```
The logic appears inverted. The docstring says "If True, rename BP to POS" but the code renames when `add_positions=False`.
- **Decision needed**: Verify intended behavior and fix if necessary

#### Incorrect Docstring (score_test.py, run_score_test)
The docstring says "Unlike run_score_test, this function does not adjust..." but this IS `run_score_test`. Copy-paste error.
- **Decision needed**: Update docstring

#### Type Mismatch in MetaAnalysis (meta_analysis.py)
Class attributes are declared as `np.ndarray` but `__init__` sets them to integers (`0`).
- **Decision needed**: Initialize as empty arrays or change type hints

#### Dead Code: GenomeAnnot Class (score_test.py, lines 266-280)
Class only raises `NotImplementedError`. 
- **Decision needed**: Remove or implement

#### Unused Function Parameters (score_test_io.py, load_gene_annotations)
Parameters `gene_table_path` and `nearest_weights` are passed in but the loaded `gene_table` is never used.
- **Decision needed**: Review if these should be used or removed

## Recommended Changes

### High Priority (Potential Bugs)
1. Fix inverted logic in `load_annotations` (`add_positions` parameter) in score_test_io.py
2. Fix docstring in `run_score_test` that references itself incorrectly
3. Add `Optional` to nullable dataclass fields in `TraitData`
4. Fix forward reference for `PrecisionOperator` in io.py

### Medium Priority (Code Quality)
1. Replace star imports in heritability.py and heritability_testing.py with explicit imports
2. Add missing type annotations, especially in meta_analysis.py
3. Standardize on Python 3.10+ style type hints (`list[str]` instead of `List[str]`)
4. Refactor the `main()` function in score_test.py (143 lines) into smaller functions
5. Remove or properly implement `GenomeAnnot` class

### Low Priority / Nice to Have
1. Fix naming conventions (`noBlocks` -> `num_blocks` in score_test.py)
2. Remove unused imports across all files (already partially done with ruff --fix)
3. Consolidate duplicate import try/except patterns across score_test modules
4. Add tests for 0% coverage modules

## Notes on Specific Files

### heritability.py / heritability_testing.py
These files are near-duplicates with substantial shared code. Consider:
- Why are there two versions?
- Should they be merged or one deprecated?
- The `_testing` version appears to be a variant for score test output

### genesets.py / genesets_gene_hdf5.py
Both deal with gene sets but have different approaches:
- genesets.py: Basic gene set utilities
- genesets_gene_hdf5.py: HDF5-based storage for gene-level data
Both have 0% test coverage and need review.

### multiprocessing_template.py
Complex multiprocessing framework. The `ParallelProcessor` base class is well-designed but:
- Documentation could be improved
- Some methods have `**kwargs` without clear documentation of expected parameters

### cli.py (both packages)
Both CLI modules have low coverage and complex argument handling:
- graphld/cli.py: 47% coverage, 1124 lines
- score_test/cli.py: 65% coverage, 478 lines
Consider integration tests for CLI commands.

## Files Already Fixed (Linting)

The following were automatically fixed with `ruff --fix`:
- Trailing whitespace (W291)
- Blank lines with whitespace (W293)  
- Missing newline at end of file (W292)
- f-strings without placeholders (F541)
- Some unused imports (F401)
