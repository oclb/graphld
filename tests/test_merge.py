"""Tests for merge_snplists functionality."""

import polars as pl
import pytest
from scipy.sparse import csr_matrix

from sparseld import PrecisionOperator
from sparseld.io import merge_snplists, merge_alleles


def test_merge_alleles():
    """Test merge_alleles function."""
    # Test exact match
    phase = merge_alleles(
        pl.Series(['A']), pl.Series(['T']), 
        pl.Series(['A']), pl.Series(['T'])
    )
    assert phase[0] == 1

    # Test flipped match
    phase = merge_alleles(
        pl.Series(['A']), pl.Series(['T']), 
        pl.Series(['T']), pl.Series(['A'])
    )
    assert phase[0] == -1

    # Test mismatch
    phase = merge_alleles(
        pl.Series(['A']), pl.Series(['T']), 
        pl.Series(['A']), pl.Series(['G'])
    )
    assert phase[0] == 0

    # Test case insensitivity
    phase = merge_alleles(
        pl.Series(['a']), pl.Series(['t']), 
        pl.Series(['A']), pl.Series(['T'])
    )
    assert phase[0] == 1

    # Test multiple variants
    phase = merge_alleles(
        pl.Series(['A', 'C', 'G']), pl.Series(['T', 'G', 'A']),
        pl.Series(['A', 'G', 'G']), pl.Series(['T', 'C', 'A'])
    )
    assert phase.to_list() == [1, -1, 1]


def test_merge_snplists():
    """Test merge_snplists function."""
    # Create test data
    # Block 1: 3 variants
    v1 = pl.DataFrame({
        'index': [0, 1, 2],
        'site_ids': ['rs1', 'rs2', 'rs3'],
        'position': [100, 200, 300],
        'chr': [1, 1, 1],
        'anc_alleles': ['A', 'C', 'GTC'],  # rs3 is an indel
        'deriv_alleles': ['T', 'G', 'G']
    })
    m1 = csr_matrix((3, 3))
    
    # Block 2: 2 variants
    v2 = pl.DataFrame({
        'index': [0, 1],
        'site_ids': ['rs4', 'rs5'],
        'position': [400, 500],
        'chr': [1, 1],
        'anc_alleles': ['A', 'T'],
        'deriv_alleles': ['G', 'C']
    })
    m2 = csr_matrix((2, 2))
    
    # Create PrecisionOperators
    op1 = PrecisionOperator(m1, v1)
    op2 = PrecisionOperator(m2, v2)
    
    # Create test summary statistics
    sumstats = pl.DataFrame({
        'SNP': ['rs1', 'rs2', 'rs3', 'rs4', 'rs6'],  # rs6 doesn't exist in blocks
        'A1': ['A', 'C', 'GTC', 'A', 'A'],
        'A2': ['T', 'G', 'G', 'G', 'C'],
        'CHR': [1, 1, 1, 1, 1],
        'POS': [100, 200, 300, 400, 600]
    })
    
    # Test matching by variant ID
    merged = merge_snplists([op1, op2], sumstats)
    assert len(merged) == 2
    assert len(merged[0]) == 3  # rs1, rs2, rs3
    assert len(merged[1]) == 1  # rs4
    assert op1._which_indices.tolist() == [0, 1, 2]
    assert op2._which_indices.tolist() == [0]
    
    # Test matching by position
    merged = merge_snplists([op1, op2], sumstats, match_by_position=True,
                           chr_col='CHR', pos_col='POS')
    assert len(merged) == 2
    assert len(merged[0]) == 3  # positions 100, 200, 300
    assert len(merged[1]) == 1  # position 400
    
    # Test allele matching
    merged = merge_snplists([op1, op2], sumstats, ref_allele_col='A1', alt_allele_col='A2')
    assert len(merged) == 2
    assert len(merged[0]) == 3  # All variants match alleles
    assert len(merged[1]) == 1  # rs4 matches alleles
    
    # Test VCF format
    sumstats_vcf = sumstats.rename({'SNP': 'ID'})
    merged = merge_snplists([op1, op2], sumstats_vcf, table_format='vcf')
    assert len(merged) == 2
    assert len(merged[0]) == 3
    assert len(merged[1]) == 1


def test_merge_snplists_errors():
    """Test error handling in merge_snplists."""
    # Create minimal test data
    v1 = pl.DataFrame({
        'index': [0],
        'site_ids': ['rs1'],
        'position': [100],
        'chr': [1],
        'anc_alleles': ['A'],
        'deriv_alleles': ['T']
    })
    m1 = csr_matrix((1, 1))
    op1 = PrecisionOperator(m1, v1)
    
    # Test missing variant ID column
    sumstats = pl.DataFrame({'CHR': [1], 'POS': [100]})
    with pytest.raises(ValueError, match=r'must contain SNP column.*Found columns: CHR, POS'):
        merge_snplists([op1], sumstats)
    
    # Test missing position columns
    sumstats = pl.DataFrame({'SNP': ['rs1']})
    with pytest.raises(ValueError, match=r'must contain CHR and POS columns.*Found columns: SNP'):
        merge_snplists([op1], sumstats, match_by_position=True)
