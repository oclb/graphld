"""Tests for genesets module."""

import sys
from pathlib import Path

# Add src/score_test to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "score_test"))

import numpy as np
import polars as pl
import pytest

from genesets import (
    load_gene_table,
    load_gene_sets_from_gmt,
    convert_gene_sets_to_variant_annotations,
    _is_gene_id,
)


TEST_DATA_DIR = Path(__file__).parent / "score_test_data"


class TestIsGeneId:
    """Tests for _is_gene_id function."""
    
    def test_ensembl_id(self):
        """Test that Ensembl IDs are correctly identified."""
        assert _is_gene_id("ENSG00000099985")
        assert _is_gene_id("ENSG00000100365")
    
    def test_gene_symbol(self):
        """Test that gene symbols are not identified as IDs."""
        assert not _is_gene_id("OSM")
        assert not _is_gene_id("NCF4")
        assert not _is_gene_id("CACNA1I")


class TestLoadGeneTable:
    """Tests for load_gene_table function."""
    
    def test_load_gene_table_basic(self):
        """Test basic loading of gene table."""
        gene_table_path = TEST_DATA_DIR / "genes_test.tsv"
        
        gene_table = load_gene_table(str(gene_table_path))
        
        # Assertions
        assert isinstance(gene_table, pl.DataFrame)
        assert 'gene_id' in gene_table.columns
        assert 'gene_name' in gene_table.columns
        assert 'CHR' in gene_table.columns
        assert 'start' in gene_table.columns
        assert 'end' in gene_table.columns
        assert 'midpoint' in gene_table.columns
        assert len(gene_table) > 0
    
    def test_load_gene_table_with_chromosome_filter(self):
        """Test loading gene table with chromosome filter."""
        gene_table_path = TEST_DATA_DIR / "genes_test.tsv"
        
        gene_table = load_gene_table(str(gene_table_path), chromosomes=[22])
        
        # Assertions
        assert len(gene_table) > 0
        assert all(gene_table['CHR'].cast(pl.Int64) == 22)
    
    def test_load_gene_table_midpoint_calculation(self):
        """Test that midpoint is correctly calculated."""
        gene_table_path = TEST_DATA_DIR / "genes_test.tsv"
        
        gene_table = load_gene_table(str(gene_table_path))
        
        # Check that midpoint is average of start and end
        for row in gene_table.iter_rows(named=True):
            expected_midpoint = (row['start'] + row['end']) / 2
            assert abs(row['midpoint'] - expected_midpoint) < 0.1


class TestLoadGeneSetsFromGmt:
    """Tests for load_gene_sets_from_gmt function."""
    
    def test_load_gene_sets_symbols(self):
        """Test loading gene sets with gene symbols."""
        gmt_dir = TEST_DATA_DIR
        
        # Create a temporary directory with only the symbols GMT file
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy(TEST_DATA_DIR / "test_symbols.gmt", tmp_dir)
            
            gene_sets = load_gene_sets_from_gmt(tmp_dir)
            
            # Assertions
            assert isinstance(gene_sets, dict)
            assert 'test_set_1' in gene_sets
            assert 'test_set_2' in gene_sets
            assert set(gene_sets['test_set_1']) == {'OSM', 'NCF4', 'CACNA1I'}
            assert set(gene_sets['test_set_2']) == {'LIF', 'TRIOBP', 'SHISAL1'}
    
    def test_load_gene_sets_ids(self):
        """Test loading gene sets with Ensembl IDs."""
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy(TEST_DATA_DIR / "test_ids.gmt", tmp_dir)
            
            gene_sets = load_gene_sets_from_gmt(tmp_dir)
            
            # Assertions
            assert isinstance(gene_sets, dict)
            assert 'test_set_ids_1' in gene_sets
            assert 'test_set_ids_2' in gene_sets
            assert 'ENSG00000099985' in gene_sets['test_set_ids_1']
            assert 'ENSG00000100365' in gene_sets['test_set_ids_1']
    
    def test_load_gene_sets_multiple_files(self):
        """Test loading gene sets from multiple GMT files."""
        gene_sets = load_gene_sets_from_gmt(str(TEST_DATA_DIR))
        
        # Should load from both test_symbols.gmt and test_ids.gmt
        assert len(gene_sets) >= 4  # At least 2 from each file
        assert 'test_set_1' in gene_sets
        assert 'test_set_ids_1' in gene_sets
    
    def test_load_gene_sets_no_files(self, tmp_path):
        """Test that FileNotFoundError is raised when no GMT files are found."""
        with pytest.raises(FileNotFoundError, match="No .gmt files found"):
            load_gene_sets_from_gmt(str(tmp_path))


class TestConvertGeneSetsToVariantAnnotations:
    """Tests for convert_gene_sets_to_variant_annotations function."""
    
    def test_convert_gene_sets_basic(self):
        """Test basic conversion of gene sets to variant annotations."""
        # Create simple test data
        variant_data = pl.DataFrame({
            'CHR': [22, 22, 22, 22],
            'POS': [30000000, 30250000, 36870000, 39600000],
            'RSID': ['rs1', 'rs2', 'rs3', 'rs4']
        })
        
        gene_table = pl.DataFrame({
            'gene_id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'gene_name': ['GENE1', 'GENE2', 'GENE3'],
            'CHR': ['22', '22', '22'],
            'start': [30000000, 30240000, 36860000],
            'end': [30010000, 30250000, 36880000],
            'midpoint': [30005000, 30245000, 36870000]
        })
        
        gene_sets = {
            'set1': ['GENE1', 'GENE2'],
            'set2': ['GENE3']
        }
        
        nearest_weights = np.array([1.0])
        
        result = convert_gene_sets_to_variant_annotations(
            gene_sets, variant_data, gene_table, nearest_weights
        )
        
        # Assertions
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 4
        assert 'CHR' in result.columns
        assert 'BP' in result.columns
        assert 'SNP' in result.columns
        assert 'CM' in result.columns
        assert 'set1' in result.columns
        assert 'set2' in result.columns
        
        # Check that annotations are numeric
        assert result['set1'].dtype in (pl.Float32, pl.Float64)
        assert result['set2'].dtype in (pl.Float32, pl.Float64)
    
    def test_convert_gene_sets_with_ids(self):
        """Test conversion using Ensembl IDs instead of symbols."""
        variant_data = pl.DataFrame({
            'CHR': [22, 22],
            'POS': [30000000, 36870000],
            'RSID': ['rs1', 'rs2']
        })
        
        gene_table = pl.DataFrame({
            'gene_id': ['ENSG00000001', 'ENSG00000002'],
            'gene_name': ['GENE1', 'GENE2'],
            'CHR': ['22', '22'],
            'start': [30000000, 36860000],
            'end': [30010000, 36880000],
            'midpoint': [30005000, 36870000]
        })
        
        # Use Ensembl IDs in gene sets
        gene_sets = {
            'set_with_ids': ['ENSG00000001', 'ENSG00000002']
        }
        
        nearest_weights = np.array([1.0])
        
        result = convert_gene_sets_to_variant_annotations(
            gene_sets, variant_data, gene_table, nearest_weights
        )
        
        # Assertions
        assert 'set_with_ids' in result.columns
        assert len(result) == 2
    
    def test_convert_gene_sets_empty_set(self):
        """Test conversion with an empty gene set."""
        variant_data = pl.DataFrame({
            'CHR': [22],
            'POS': [30000000],
            'RSID': ['rs1']
        })
        
        gene_table = pl.DataFrame({
            'gene_id': ['ENSG1'],
            'gene_name': ['GENE1'],
            'CHR': ['22'],
            'start': [30000000],
            'end': [30010000],
            'midpoint': [30005000]
        })
        
        gene_sets = {
            'empty_set': []
        }
        
        nearest_weights = np.array([1.0])
        
        result = convert_gene_sets_to_variant_annotations(
            gene_sets, variant_data, gene_table, nearest_weights
        )
        
        # Should still create the annotation column, but all zeros
        assert 'empty_set' in result.columns
        assert all(result['empty_set'] == 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
