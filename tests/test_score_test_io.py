"""Tests for score_test_io module."""

from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from score_test.score_test_io import (
    load_variant_data,
    load_trait_data,
    load_annotations,
    load_variant_annotations,
    load_gene_annotations,
    create_random_gene_annotations,
    create_random_variant_annotations,
)


class TestLoadVariantData:
    """Tests for load_variant_data function."""
    
    def test_load_variant_data_basic(self, tmp_path):
        """Test basic loading of variant data from HDF5."""
        # Create test HDF5 file
        hdf5_path = tmp_path / "test_variants.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            variants_group = f.create_group('variants')
            variants_group.create_dataset('CHR', data=np.array([1, 1, 2, 2]))
            variants_group.create_dataset('POS', data=np.array([1000, 2000, 3000, 4000]))
            variants_group.create_dataset('RSID', data=np.array([b'rs1', b'rs2', b'rs3', b'rs4']))
            variants_group.create_dataset('annotations', data=np.random.randn(4, 3))
            variants_group.create_dataset('jackknife_blocks', data=np.array([0, 0, 1, 1]))
        
        # Load data
        df = load_variant_data(str(hdf5_path))
        
        # Assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert 'CHR' in df.columns
        assert 'POS' in df.columns
        assert 'RSID' in df.columns
        assert 'annotations' in df.columns
        assert 'jackknife_blocks' in df.columns
        assert df['CHR'].to_list() == [1, 1, 2, 2]
        assert df['RSID'].to_list() == ['rs1', 'rs2', 'rs3', 'rs4']
    
    def test_load_variant_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_variant_data("/nonexistent/path.h5")


class TestLoadTraitData:
    """Tests for load_trait_data function."""
    
    def test_load_trait_data_basic(self, tmp_path):
        """Test basic loading of trait data from HDF5."""
        # Create test HDF5 file
        hdf5_path = tmp_path / "test_traits.h5"
        
        n_variants = 100
        n_params = 5
        n_blocks = 10
        
        with h5py.File(hdf5_path, 'w') as f:
            traits_group = f.create_group('traits')
            trait_group = traits_group.create_group('test_trait')
            trait_group.create_dataset('parameters', data=np.random.randn(n_params))
            trait_group.create_dataset('jackknife_parameters', data=np.random.randn(n_blocks, n_params))
            trait_group.create_dataset('gradient', data=np.random.randn(n_variants))
            trait_group.create_dataset('hessian', data=np.random.randn(n_variants))
        
        # Load data
        data = load_trait_data(str(hdf5_path), 'test_trait')
        
        # Assertions
        assert isinstance(data, dict)
        assert 'parameters' in data
        assert 'jackknife_parameters' in data
        assert 'gradient' in data
        assert 'hessian' in data
        assert data['parameters'].shape == (n_params,)
        assert data['jackknife_parameters'].shape == (n_blocks, n_params)
        assert data['gradient'].shape == (n_variants,)
        assert data['hessian'].shape == (n_variants,)


class TestLoadAnnotations:
    """Tests for load_annotations function."""
    
    def test_load_annotations_single_chromosome(self, tmp_path):
        """Test loading annotations for a single chromosome."""
        # Create test annotation file
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        annot_file = annot_dir / "test.22.annot"
        df = pl.DataFrame({
            'CHR': [22, 22, 22],
            'BP': [1000, 2000, 3000],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'CM': [0.0, 0.0, 0.0],
            'annot1': [1, 0, 1],
            'annot2': [0, 1, 0]
        })
        df.write_csv(annot_file, separator='\t')
        
        # Load annotations
        result = load_annotations(str(annot_dir), chromosome=22, add_positions=False)
        
        # Assertions
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert 'CHR' in result.columns
        assert 'POS' in result.columns  # BP renamed to POS when add_positions=False
        assert 'SNP' in result.columns
        assert 'annot1' in result.columns
        assert 'annot2' in result.columns
    
    def test_load_annotations_multiple_chromosomes(self, tmp_path):
        """Test loading annotations for multiple chromosomes."""
        # Create test annotation files
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        for chrom in [1, 2]:
            annot_file = annot_dir / f"test.{chrom}.annot"
            df = pl.DataFrame({
                'CHR': [chrom] * 3,
                'BP': [1000, 2000, 3000],
                'SNP': [f'rs{chrom}_{i}' for i in range(3)],
                'CM': [0.0, 0.0, 0.0],
                'annot1': [1, 0, 1]
            })
            df.write_csv(annot_file, separator='\t')
        
        # Load annotations for both chromosomes
        result = load_annotations(str(annot_dir), chromosome=None, add_positions=False)
        
        # Assertions
        assert len(result) == 6  # 3 variants * 2 chromosomes
        assert set(result['CHR'].unique()) == {1, 2}
    
    def test_load_annotations_binary_conversion(self, tmp_path):
        """Test that binary columns are converted to boolean."""
        # Create test annotation file
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        annot_file = annot_dir / "test.1.annot"
        df = pl.DataFrame({
            'CHR': [1, 1, 1],
            'BP': [1000, 2000, 3000],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'CM': [0.0, 0.0, 0.0],
            'binary_annot': [1, 0, 1],  # Should be converted to boolean
            'continuous_annot': [0.5, 1.5, 2.5]  # Should stay numeric
        })
        df.write_csv(annot_file, separator='\t')
        
        # Load annotations
        result = load_annotations(str(annot_dir), chromosome=1, add_positions=False)
        
        # Assertions
        assert result['binary_annot'].dtype == pl.Boolean
        assert result['continuous_annot'].dtype in (pl.Float32, pl.Float64)
    
    def test_load_annotations_no_files_found(self, tmp_path):
        """Test that ValueError is raised when no annotation files are found."""
        annot_dir = tmp_path / "empty_annot"
        annot_dir.mkdir()
        
        with pytest.raises(ValueError, match="No annotation files found"):
            load_annotations(str(annot_dir), chromosome=1)
    
    def test_load_annotations_multiple_files_per_chromosome(self, tmp_path):
        """Test loading multiple annotation files for the same chromosome."""
        # Create test annotation files
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        # First file with some annotations
        annot_file1 = annot_dir / "baseline.1.annot"
        df1 = pl.DataFrame({
            'CHR': [1, 1],
            'BP': [1000, 2000],
            'SNP': ['rs1', 'rs2'],
            'CM': [0.0, 0.0],
            'annot1': [1, 0]
        })
        df1.write_csv(annot_file1, separator='\t')
        
        # Second file with different annotations (same variants)
        annot_file2 = annot_dir / "custom.1.annot"
        df2 = pl.DataFrame({
            'CHR': [1, 1],
            'BP': [1000, 2000],
            'SNP': ['rs1', 'rs2'],
            'CM': [0.0, 0.0],
            'annot2': [0, 1]
        })
        df2.write_csv(annot_file2, separator='\t')
        
        # Load annotations
        result = load_annotations(str(annot_dir), chromosome=1, add_positions=False)
        
        # Assertions
        assert len(result) == 2
        assert 'annot1' in result.columns
        assert 'annot2' in result.columns
        # CHR, BP, SNP, CM should not be duplicated
        assert result.columns.count('CHR') == 1
        assert result.columns.count('SNP') == 1


class TestLoadVariantAnnotations:
    """Tests for load_variant_annotations function."""
    
    def test_load_variant_annotations_all_columns(self, tmp_path):
        """Test loading all annotation columns."""
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        annot_file = annot_dir / "test.1.annot"
        df = pl.DataFrame({
            'CHR': [1, 1],
            'BP': [1000, 2000],
            'SNP': ['rs1', 'rs2'],
            'CM': [0.0, 0.0],
            'annot1': [1, 0],
            'annot2': [0, 1]
        })
        df.write_csv(annot_file, separator='\t')
        
        df_annot, annot_names = load_variant_annotations(str(annot_dir))
        
        assert len(annot_names) == 2
        assert 'annot1' in annot_names
        assert 'annot2' in annot_names
        assert len(df_annot) == 2
    
    def test_load_variant_annotations_filtered(self, tmp_path):
        """Test loading specific annotation columns."""
        annot_dir = tmp_path / "annot"
        annot_dir.mkdir()
        
        annot_file = annot_dir / "test.1.annot"
        df = pl.DataFrame({
            'CHR': [1, 1],
            'BP': [1000, 2000],
            'SNP': ['rs1', 'rs2'],
            'CM': [0.0, 0.0],
            'annot1': [1, 0],
            'annot2': [0, 1],
            'annot3': [1, 1]
        })
        df.write_csv(annot_file, separator='\t')
        
        df_annot, annot_names = load_variant_annotations(str(annot_dir), ['annot1', 'annot3'])
        
        assert len(annot_names) == 2
        assert 'annot1' in annot_names
        assert 'annot3' in annot_names
        assert 'annot2' not in annot_names


class TestLoadGeneAnnotations:
    """Tests for load_gene_annotations function."""
    
    def test_load_gene_annotations_basic(self, tmp_path):
        """Test basic loading and conversion of gene annotations."""
        # Create test GMT file
        gmt_dir = tmp_path / "gmt"
        gmt_dir.mkdir()
        gmt_file = gmt_dir / "test.gmt"
        gmt_file.write_text("set1\tDescription\tGENE1\tGENE2\n")
        
        # Create test gene table
        gene_table_path = tmp_path / "genes.tsv"
        gene_df = pl.DataFrame({
            'gene_id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'gene_id_version': ['ENSG1.1', 'ENSG2.1', 'ENSG3.1'],
            'gene_name': ['GENE1', 'GENE2', 'GENE3'],
            'start': [1000, 2000, 3000],
            'end': [1500, 2500, 3500],
            'CHR': ['22', '22', '22']
        })
        gene_df.write_csv(gene_table_path, separator='\t')
        
        # Create variant data
        variant_data = pl.DataFrame({
            'CHR': [22, 22, 22],
            'POS': [1250, 2250, 3250],
            'RSID': ['rs1', 'rs2', 'rs3']
        })
        
        weights = np.array([1.0])
        
        df_annot, annot_names = load_gene_annotations(
            str(gmt_dir), variant_data, str(gene_table_path), weights
        )
        
        assert len(annot_names) == 1
        assert 'set1' in annot_names
        assert 'CHR' in df_annot.columns
        assert 'SNP' in df_annot.columns
        assert len(df_annot) == 3


class TestCreateRandomGeneAnnotations:
    """Tests for create_random_gene_annotations function."""
    
    def test_create_random_gene_annotations_basic(self, tmp_path):
        """Test creating random gene annotations."""
        # Create test gene table
        gene_table_path = tmp_path / "genes.tsv"
        gene_df = pl.DataFrame({
            'gene_id': [f'ENSG{i}' for i in range(10)],
            'gene_id_version': [f'ENSG{i}.1' for i in range(10)],
            'gene_name': [f'GENE{i}' for i in range(10)],
            'start': [i * 1000 for i in range(10)],
            'end': [i * 1000 + 500 for i in range(10)],
            'CHR': ['22'] * 10
        })
        gene_df.write_csv(gene_table_path, separator='\t')
        
        # Create variant data
        variant_data = pl.DataFrame({
            'CHR': [22] * 10,
            'POS': [i * 1000 + 250 for i in range(10)],
            'RSID': [f'rs{i}' for i in range(10)]
        })
        
        weights = np.array([1.0])
        probs = [0.5, 0.2]
        
        np.random.seed(42)
        df_annot, annot_names = create_random_gene_annotations(
            variant_data, str(gene_table_path), weights, probs
        )
        
        assert len(annot_names) == 2
        assert 'random_gene_0' in annot_names
        assert 'random_gene_1' in annot_names
        assert len(df_annot) == 10
        assert 'CHR' in df_annot.columns
        assert 'SNP' in df_annot.columns
    
    def test_create_random_gene_annotations_reproducible(self, tmp_path):
        """Test that random gene annotations are reproducible with seed."""
        # Create test gene table
        gene_table_path = tmp_path / "genes.tsv"
        gene_df = pl.DataFrame({
            'gene_id': [f'ENSG{i}' for i in range(5)],
            'gene_id_version': [f'ENSG{i}.1' for i in range(5)],
            'gene_name': [f'GENE{i}' for i in range(5)],
            'start': [i * 1000 for i in range(5)],
            'end': [i * 1000 + 500 for i in range(5)],
            'CHR': ['22'] * 5
        })
        gene_df.write_csv(gene_table_path, separator='\t')
        
        variant_data = pl.DataFrame({
            'CHR': [22] * 5,
            'POS': [i * 1000 + 250 for i in range(5)],
            'RSID': [f'rs{i}' for i in range(5)]
        })
        
        weights = np.array([1.0])
        probs = [0.5]
        
        # Generate twice with same seed
        np.random.seed(42)
        df1, names1 = create_random_gene_annotations(variant_data, str(gene_table_path), weights, probs)
        
        np.random.seed(42)
        df2, names2 = create_random_gene_annotations(variant_data, str(gene_table_path), weights, probs)
        
        assert names1 == names2
        assert df1['random_gene_0'].to_list() == df2['random_gene_0'].to_list()


class TestCreateRandomVariantAnnotations:
    """Tests for create_random_variant_annotations function."""
    
    def test_create_random_variant_annotations_basic(self):
        """Test creating random variant annotations."""
        variant_data = pl.DataFrame({
            'CHR': [1, 1, 2, 2],
            'POS': [1000, 2000, 3000, 4000],
            'RSID': ['rs1', 'rs2', 'rs3', 'rs4']
        })
        
        probs = [0.5, 0.2]
        
        np.random.seed(42)
        df_annot, annot_names = create_random_variant_annotations(variant_data, probs)
        
        assert len(annot_names) == 2
        assert 'random_variant_0' in annot_names
        assert 'random_variant_1' in annot_names
        assert len(df_annot) == 4
        assert 'CHR' in df_annot.columns
        assert 'BP' in df_annot.columns
        assert 'SNP' in df_annot.columns
        assert 'CM' in df_annot.columns
    
    def test_create_random_variant_annotations_values(self):
        """Test that random variant annotations have correct values."""
        variant_data = pl.DataFrame({
            'CHR': [1] * 100,
            'POS': list(range(100)),
            'RSID': [f'rs{i}' for i in range(100)]
        })
        
        probs = [0.5]
        
        np.random.seed(42)
        df_annot, annot_names = create_random_variant_annotations(variant_data, probs)
        
        # Check that values are 0 or 1
        values = df_annot['random_variant_0'].to_list()
        assert all(v in [0.0, 1.0] for v in values)
        
        # Check that approximately half are 1 (with some tolerance)
        proportion = sum(values) / len(values)
        assert 0.3 < proportion < 0.7  # Allow some variance
    
    def test_create_random_variant_annotations_reproducible(self):
        """Test that random variant annotations are reproducible with seed."""
        variant_data = pl.DataFrame({
            'CHR': [1, 1, 1],
            'POS': [1000, 2000, 3000],
            'RSID': ['rs1', 'rs2', 'rs3']
        })
        
        probs = [0.5]
        
        # Generate twice with same seed
        np.random.seed(42)
        df1, names1 = create_random_variant_annotations(variant_data, probs)
        
        np.random.seed(42)
        df2, names2 = create_random_variant_annotations(variant_data, probs)
        
        assert names1 == names2
        assert df1['random_variant_0'].to_list() == df2['random_variant_0'].to_list()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
