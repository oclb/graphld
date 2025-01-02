"""Tests for merge_snplists functionality."""


from pathlib import Path
import tempfile

import polars as pl
import pytest
from scipy.sparse import csr_matrix

from graphld import PrecisionOperator
from graphld.io import (
    load_ldgm,
    merge_alleles,
    merge_snplists,
    create_ldgm_metadata,
    read_ldgm_metadata,
    partition_variants
)


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
    op1 = PrecisionOperator(m1, v1)

    # Create test summary statistics
    sumstats = pl.DataFrame({
        'SNP': ['rs1', 'rs2', 'rs3', 'rs4', 'rs6'],  # rs4, rs6 don't exist in block
        'A1': ['A', 'C', 'GTC', 'A', 'A'],
        'A2': ['T', 'G', 'G', 'G', 'C'],
        'CHR': [1, 1, 1, 1, 1],
        'POS': [100, 200, 300, 400, 600],
        'BETA': [0.1, 0.2, 0.3, 0.4, 0.5],
        'SE': [0.01, 0.02, 0.03, 0.04, 0.05]
    })

    # Test matching by variant ID
    merged_op = merge_snplists(op1, sumstats)
    assert merged_op.shape[0] == 3  # rs1, rs2, rs3
    assert len(merged_op.variant_info) == 3
    assert list(merged_op.variant_info['site_ids']) == ['rs1', 'rs2', 'rs3']

    # Test matching by position
    merged_op = merge_snplists(op1, sumstats, match_by_position=True,
                              pos_col='POS')
    assert merged_op.shape[0] == 3  # positions 100, 200, 300
    assert len(merged_op.variant_info) == 3
    assert list(merged_op.variant_info['position']) == [100, 200, 300]

    # Test allele matching
    merged_op = merge_snplists(op1, sumstats, ref_allele_col='A1', alt_allele_col='A2')
    assert merged_op.shape[0] == 3  # All variants match alleles
    assert len(merged_op.variant_info) == 3

    # Test VCF format
    sumstats_vcf = sumstats.rename({'SNP': 'ID'})
    merged_op = merge_snplists(op1, sumstats_vcf, table_format='vcf')
    assert merged_op.shape[0] == 3
    assert len(merged_op.variant_info) == 3

    # Test appending columns
    merged_op = merge_snplists(op1, sumstats, add_cols=['BETA', 'SE'])
    assert merged_op.shape[0] == 3
    assert len(merged_op.variant_info) == 3
    assert 'BETA' in merged_op.variant_info.columns
    assert 'SE' in merged_op.variant_info.columns
    assert list(merged_op.variant_info['BETA']) == [0.1, 0.2, 0.3]
    assert list(merged_op.variant_info['SE']) == [0.01, 0.02, 0.03]

    # Test is_representative column with duplicate indices
    v2 = pl.DataFrame({
        'index': [0, 0, 1, 1, 2],  # Duplicated indices
        'site_ids': ['rs1', 'rs1_dup', 'rs2', 'rs2_dup', 'rs3'],
        'position': [100, 100, 200, 200, 300],
        'chr': [1, 1, 1, 1, 1],
        'anc_alleles': ['A', 'A', 'C', 'C', 'GTC'],
        'deriv_alleles': ['T', 'T', 'G', 'G', 'G']
    })
    m2 = csr_matrix((5, 5))
    op2 = PrecisionOperator(m2, v2)
    
    # Create matching sumstats
    sumstats2 = pl.DataFrame({
        'SNP': ['rs1', 'rs1_dup', 'rs2', 'rs2_dup', 'rs3'],
        'A1': ['A', 'A', 'C', 'C', 'GTC'],
        'A2': ['T', 'T', 'G', 'G', 'G'],
        'CHR': [1, 1, 1, 1, 1],
        'POS': [100, 100, 200, 200, 300],
        'BETA': [0.1, 0.15, 0.2, 0.25, 0.3],
        'SE': [0.01, 0.015, 0.02, 0.025, 0.03]
    })
    
    merged_op = merge_snplists(op2, sumstats2)
    assert 'is_representative' in merged_op.variant_info.columns
    assert len(merged_op.variant_info) == 5  # Should keep all variants
    # First occurrence of each index should be marked as representative
    assert list(merged_op.variant_info['is_representative']) == [1, 0, 1, 0, 1]
    # All variants should be retained in order
    assert list(merged_op.variant_info['site_ids']) == ['rs1', 'rs1_dup', 'rs2', 'rs2_dup', 'rs3']

    # Test allelic columns
    merged_op = merge_snplists(op1, sumstats,
                              ref_allele_col='A1',
                              alt_allele_col='A2',
                              add_allelic_cols=['BETA'])
    assert 'phase' in merged_op.variant_info.columns
    assert 'BETA' in merged_op.variant_info.columns
    # Check that BETA has been phased correctly
    expected_beta = [0.1, 0.2, 0.3]  # Original values
    actual_beta = list(merged_op.variant_info['BETA'])
    assert all(abs(a - e) < 1e-10 for a, e in zip(actual_beta, expected_beta))


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

    # Test position matching without required columns
    sumstats = pl.DataFrame({'SNP': ['rs1']})
    with pytest.raises(ValueError, match=r'must contain POS column.*Found columns: SNP'):
        merge_snplists(op1, sumstats, match_by_position=True)

    # Test error on missing append columns
    with pytest.raises(ValueError, match="Requested columns not found in sumstats"):
        merge_snplists(op1, sumstats, add_cols=['NONEXISTENT'])


def test_load_ldgm():
    """Test loading LDGM data from files and directories."""
    import numpy as np

    # Test loading from single files

    operator = load_ldgm(
        filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
        snplist_path="data/test/1kg_chr1_16103_2888443.snplist"
    )
    assert operator.shape[0] == operator.shape[1]  # Square matrix
    assert np.all(operator.matrix.diagonal() != 0)  # No zeros on diagonal

    # Test loading from directory
    operators = load_ldgm(
        filepath="data/test",
        population="EAS"
    )
    assert isinstance(operators, list)
    assert len(operators) > 0
    for op in operators:
        assert op.shape[0] == op.shape[1]
        assert np.all(op.matrix.diagonal() != 0)

    # Test loading from directory with different population
    operators = load_ldgm(
        filepath="data/test",
        population="EUR"
    )
    assert isinstance(operators, list)
    assert len(operators) > 0
    for op in operators:
        assert op.shape[0] == op.shape[1]
        assert np.all(op.matrix.diagonal() != 0)

    # Test that population filter works
    operators_eas = load_ldgm(filepath="data/test", population="EAS")
    operators_eur = load_ldgm(filepath="data/test", population="EUR")

    # Should have same number of files
    assert len(operators_eas) == len(operators_eur)

    # Different populations should have different data
    assert operators_eas[0].shape != operators_eur[0].shape


def test_create_ldgm_metadata():
    """Test creation of LDGM metadata file."""
    # Get test directory
    test_dir = Path(__file__).parent.parent / "data/test"
    
    # Create metadata
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        df = create_ldgm_metadata(test_dir, tmp.name)
        
        # Check DataFrame properties
        assert len(df) == 4, "Expected 4 LDGM files"
        assert all(col in df.columns for col in [
            'chrom', 'chromStart', 'chromEnd', 'name', 'snplistName',
            'population', 'numVariants', 'numIndices', 'numEntries', 'info'
        ])
        
        # Check file properties
        assert set(df['population'].unique()) == {'EUR', 'EAS'}
        assert all(df['chrom'] == 1)
        assert df['chromStart'].is_sorted()
        
        # Read back and verify
        df2 = read_ldgm_metadata(tmp.name)
        assert df.equals(df2)
        
        # Check numeric properties
        assert all(df['numVariants'] > 0)
        assert all(df['numIndices'] > 0)
        assert all(df['numEntries'] > 0)
        assert all(df['numIndices'] <= df['numVariants'])
        assert all(df['numEntries'] >= df['numIndices'])  # At least diagonal entries


def test_read_ldgm_metadata_validation():
    """Test validation in read_ldgm_metadata."""
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        # Create invalid metadata file
        invalid_df = pl.DataFrame({
            'chrom': [1],
            'name': ['test.edgelist']  # Missing required columns
        })
        invalid_df.write_csv(tmp.name)
        
        # Check that validation fails
        with pytest.raises(ValueError, match="Missing required columns"):
            read_ldgm_metadata(tmp.name)


def test_read_ldgm_metadata_filtering():
    """Test filtering options in read_ldgm_metadata."""
    # Get test directory
    test_dir = Path(__file__).parent.parent / "data/test"
    
    # Create metadata file
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
        df = create_ldgm_metadata(test_dir, tmp.name)
        
        # Test population filtering
        df_eur = read_ldgm_metadata(tmp.name, populations='EUR')
        assert len(df_eur) == 2
        assert all(df_eur['population'] == 'EUR')
        
        df_both = read_ldgm_metadata(tmp.name, populations=['EUR', 'EAS'])
        assert df_both.equals(df)  # Should match original
        
        with pytest.raises(ValueError, match="No blocks found for populations"):
            read_ldgm_metadata(tmp.name, populations='AFR')
            
        # Test chromosome filtering
        df_chr1 = read_ldgm_metadata(tmp.name, chromosomes=1)
        assert len(df_chr1) == 4
        assert all(df_chr1['chrom'] == 1)
        
        with pytest.raises(ValueError, match="No blocks found for chromosomes"):
            read_ldgm_metadata(tmp.name, chromosomes=2)
            
        # Test max_blocks
        df_limited = read_ldgm_metadata(tmp.name, max_blocks=2)
        assert len(df_limited) == 2
        assert df_limited.equals(df.head(2))
        
        # Test combined filtering
        df_combined = read_ldgm_metadata(
            tmp.name,
            populations='EUR',
            chromosomes=1,
            max_blocks=1
        )
        assert len(df_combined) == 1
        assert df_combined['population'][0] == 'EUR'
        assert df_combined['chrom'][0] == 1


def test_partition_variants():
    """Test partitioning of variant data into LDGM blocks."""
    # Create test metadata
    metadata = pl.DataFrame({
        'chrom': [1, 1, 2],
        'chromStart': [100, 300, 200],
        'chromEnd': [300, 500, 400],
        'name': ['block1', 'block2', 'block3'],
        'snplistName': ['snp1', 'snp2', 'snp3'],
        'population': ['EUR', 'EUR', 'EUR'],
        'numVariants': [10, 10, 10],
        'numIndices': [5, 5, 5],
        'numEntries': [20, 20, 20],
        'info': ['', '', '']
    })
    
    # Create test variant data with different column names
    variants = pl.DataFrame({
        'CHR': [1, 1, 1, 1, 2, 2],
        'POS': [150, 250, 350, 450, 250, 350],
        'REF': ['A', 'C', 'G', 'T', 'A', 'C'],
        'ALT': ['T', 'G', 'C', 'A', 'G', 'T']
    })
    
    # Test automatic column name detection
    partitioned = partition_variants(metadata, variants)
    assert len(partitioned) == 3
    
    # Check block contents
    assert len(partitioned[0]) == 2  # Block 1: pos 150, 250
    assert len(partitioned[1]) == 2  # Block 2: pos 350, 450
    assert len(partitioned[2]) == 2  # Block 3: pos 250, 350 on chr 2
    
    # Test with explicit column names
    variants2 = pl.DataFrame({
        'chromosome': [1, 1, 1, 1, 2, 2],
        'position': [150, 250, 350, 450, 250, 350],
        'REF': ['A', 'C', 'G', 'T', 'A', 'C'],
        'ALT': ['T', 'G', 'C', 'A', 'G', 'T']
    })
    partitioned2 = partition_variants(
        metadata,
        variants2,
        chrom_col='chromosome',
        pos_col='position'
    )
    assert len(partitioned2) == 3
    for df1, df2 in zip(partitioned, partitioned2):
        assert len(df1) == len(df2)
    
    # Test with string chromosome
    variants3 = variants.with_columns(
        pl.col('CHR').cast(str).alias('CHR')
    )
    partitioned3 = partition_variants(metadata, variants3)
    assert len(partitioned3) == 3
    for df1, df3 in zip(partitioned, partitioned3):
        assert len(df1) == len(df3)
    
    # Test error cases
    bad_variants = pl.DataFrame({
        'CHROM': [1, 2],  # Wrong column name
        'POS': [100, 200]
    })
    with pytest.raises(ValueError, match="Could not find chromosome column"):
        partition_variants(metadata, bad_variants)
        
    bad_variants = pl.DataFrame({
        'CHR': [1, 2],
        'POSITION': [100, 200]  # Wrong column name
    })
    with pytest.raises(ValueError, match="Could not find position column"):
        partition_variants(metadata, bad_variants)
