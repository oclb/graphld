"""Tests for approximate score test functionality."""

import numpy as np
import polars as pl
import pytest
import h5py
from pathlib import Path

from score_test.score_test import (
    VariantAnnot, TraitData, run_score_test, run_approx_score_test
)
from score_test.score_test_io import (
    load_trait_data, load_variant_data, create_random_variant_annotations
)


def create_test_hdf5_with_hessian(hdf5_path, n_variants=100, n_params=5, n_blocks=10):
    """Helper to create test HDF5 file with hessian."""
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


def create_test_hdf5_without_hessian(hdf5_path, n_variants=100, n_params=5, n_blocks=10):
    """Helper to create test HDF5 file without hessian."""
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
        # Note: no hessian dataset
        
        f.create_group('groups')


def test_run_approx_score_test_basic(tmp_path):
    """Test basic approximate score test functionality."""
    hdf5_path = tmp_path / "test_approx.h5"
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    np.random.seed(42)
    create_test_hdf5_with_hessian(hdf5_path, n_variants, n_params, n_blocks)
    
    # Load data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run approximate score test
    point_estimates, jackknife_estimates, _ = run_approx_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    # Verify output shapes
    assert point_estimates.shape == (1, len(probs))
    assert jackknife_estimates.shape == (n_blocks, len(probs))
    
    # Verify jackknife estimates sum to point estimate (approximately)
    # The sum should be close to point_estimate * n_blocks
    jackknife_sum = np.sum(jackknife_estimates, axis=0)
    expected_sum = point_estimates.ravel() * (n_blocks - 1)
    np.testing.assert_allclose(jackknife_sum, expected_sum, rtol=1e-10)


def test_approx_vs_exact_score_test(tmp_path):
    """Test that approximate and exact score tests produce different results."""
    hdf5_path = tmp_path / "test_comparison.h5"
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    np.random.seed(42)
    create_test_hdf5_with_hessian(hdf5_path, n_variants, n_params, n_blocks)
    
    # Load data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run both tests
    point_approx, jackknife_approx, _ = run_approx_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    point_exact, jackknife_exact, _ = run_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    # Point estimates should be the same (both use full data)
    np.testing.assert_allclose(point_approx, point_exact, rtol=1e-10)
    
    # Jackknife estimates should differ (exact corrects for parameter uncertainty)
    # They should not be identical
    assert not np.allclose(jackknife_approx, jackknife_exact, rtol=1e-10)


def test_approx_score_test_without_hessian(tmp_path):
    """Test that approximate score test works without hessian data."""
    hdf5_path = tmp_path / "test_no_hessian.h5"
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    np.random.seed(42)
    create_test_hdf5_without_hessian(hdf5_path, n_variants, n_params, n_blocks)
    
    # Load data
    variant_table = load_variant_data(str(hdf5_path))
    
    # Verify hessian is not in columns
    assert 'hessian' not in variant_table.columns
    
    # Load trait data - should work even without hessian
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run approximate score test - should work without hessian
    point_estimates, jackknife_estimates, _ = run_approx_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    # Verify output shapes
    assert point_estimates.shape == (1, len(probs))
    assert jackknife_estimates.shape == (n_blocks, len(probs))


def test_exact_score_test_requires_hessian(tmp_path):
    """Test that exact score test fails without hessian data."""
    hdf5_path = tmp_path / "test_no_hessian_exact.h5"
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    np.random.seed(42)
    create_test_hdf5_without_hessian(hdf5_path, n_variants, n_params, n_blocks)
    
    # Load data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run exact score test - should fail without hessian
    with pytest.raises(Exception):  # Will raise KeyError or similar
        run_score_test(
            trait_data=trait_data,
            annot=variant_annot,
        )


def test_approx_score_test_consistency(tmp_path):
    """Test that approximate score test gives consistent results with same seed."""
    hdf5_path = tmp_path / "test_consistency.h5"
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    np.random.seed(42)
    create_test_hdf5_with_hessian(hdf5_path, n_variants, n_params, n_blocks)
    
    # Load data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations with same seed
    np.random.seed(123)
    probs = [0.1, 0.2]
    variant_annot1 = create_random_variant_annotations(trait_data.df, probs)
    
    np.random.seed(123)
    variant_annot2 = create_random_variant_annotations(trait_data.df, probs)
    
    # Run approximate score test twice
    point1, jackknife1, _ = run_approx_score_test(trait_data=trait_data, annot=variant_annot1)
    point2, jackknife2, _ = run_approx_score_test(trait_data=trait_data, annot=variant_annot2)
    
    # Results should be identical
    np.testing.assert_array_equal(point1, point2)
    np.testing.assert_array_equal(jackknife1, jackknife2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
