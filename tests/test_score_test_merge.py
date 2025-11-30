"""Test for VariantAnnot.merge with 2D annotations array."""

import numpy as np
import polars as pl
import pytest
import h5py
from pathlib import Path

from score_test.score_test import VariantAnnot, TraitData, run_score_test
from score_test.score_test_io import load_trait_data, load_variant_data, create_random_variant_annotations


def test_variant_annot_merge_with_2d_annotations(tmp_path):
    """Test that VariantAnnot.merge handles 2D annotations array correctly."""
    # Create test HDF5 file with 2D annotations
    hdf5_path = tmp_path / "test_merge.h5"
    
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    with h5py.File(hdf5_path, 'w') as f:
        # Use new format
        f.attrs['metadata'] = ''
        f.attrs['data_type'] = 'variant'
        f.attrs['keys'] = ['RSID', 'POS']
        
        # Create row_data group
        row_data_group = f.create_group('row_data')
        row_data_group.create_dataset('CHR', data=np.ones(n_variants, dtype=int))
        row_data_group.create_dataset('POS', data=np.arange(n_variants))
        # Split annotations into separate columns
        annot_data = np.random.randn(n_variants, n_params)
        for i in range(n_params):
            row_data_group.create_dataset(f'annot_{i}', data=annot_data[:, i])
        row_data_group.create_dataset('jackknife_blocks', data=np.repeat(np.arange(n_blocks), n_variants // n_blocks))
        row_data_group.create_dataset('RSID', data=np.array([f'rs{i}' for i in range(n_variants)], dtype='S10'))
        
        # Create traits group
        traits_group = f.create_group('traits')
        trait_group = traits_group.create_group('test_trait')
        params_group = trait_group.create_group('parameters')
        params_group.create_dataset('parameters', data=np.random.randn(n_params))
        params_group.create_dataset('jackknife_parameters', data=np.random.randn(n_blocks, n_params))
        trait_group.create_dataset('gradient', data=np.random.randn(n_variants))
        trait_group.create_dataset('hessian', data=np.random.randn(n_variants))
        
        f.create_group('groups')
    
    # Load variant and trait data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    np.random.seed(42)
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # This should not raise an error
    grad, hessian, model_annot, test_annot, block_boundaries = variant_annot.merge(trait_data)
    
    # Verify shapes
    assert grad.shape == (n_variants,)
    assert hessian.shape == (n_variants,)
    assert model_annot.shape == (n_variants, n_params)
    assert test_annot.shape == (n_variants, len(probs))
    assert len(block_boundaries) == n_blocks + 1


def test_run_score_test_with_2d_annotations(tmp_path):
    """Test full run_score_test with 2D annotations array."""
    # Create test HDF5 file
    hdf5_path = tmp_path / "test_score.h5"
    
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    with h5py.File(hdf5_path, 'w') as f:
        # Use new format
        f.attrs['metadata'] = ''
        f.attrs['data_type'] = 'variant'
        f.attrs['keys'] = ['RSID', 'POS']
        
        # Create row_data group
        row_data_group = f.create_group('row_data')
        row_data_group.create_dataset('CHR', data=np.ones(n_variants, dtype=int))
        row_data_group.create_dataset('POS', data=np.arange(n_variants))
        annot_data = np.random.randn(n_variants, n_params)
        for i in range(n_params):
            row_data_group.create_dataset(f'annot_{i}', data=annot_data[:, i])
        row_data_group.create_dataset('jackknife_blocks', data=np.repeat(np.arange(n_blocks), n_variants // n_blocks))
        row_data_group.create_dataset('RSID', data=np.array([f'rs{i}' for i in range(n_variants)], dtype='S10'))
        
        # Create traits group
        traits_group = f.create_group('traits')
        trait_group = traits_group.create_group('test_trait')
        params_group = trait_group.create_group('parameters')
        params_group.create_dataset('parameters', data=np.random.randn(n_params))
        params_group.create_dataset('jackknife_parameters', data=np.random.randn(n_blocks, n_params))
        trait_group.create_dataset('gradient', data=np.random.randn(n_variants))
        trait_group.create_dataset('hessian', data=np.random.randn(n_variants))
        
        f.create_group('groups')
    
    # Load variant and trait data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    np.random.seed(42)
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run score test - this should not raise an error
    point_estimates, jackknife_estimates = run_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    # Verify output shapes
    assert point_estimates.shape == (1, len(probs))
    assert jackknife_estimates.shape == (n_blocks, len(probs))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
