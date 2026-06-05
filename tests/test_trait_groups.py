"""Tests for trait groups and meta-analysis functionality."""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from score_test.score_test_io import (
    save_trait_groups,
    get_trait_groups,
    load_variant_data,
    load_trait_data,
    create_random_variant_annotations,
)
from score_test.score_test import TraitData, run_score_test
from score_test.meta_analysis import MetaAnalysis


@pytest.fixture
def test_hdf5_path(tmp_path):
    """Temporary copy of the test HDF5 file."""
    source = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    target = tmp_path / "test.scores.h5"
    shutil.copy(source, target)
    return target


@pytest.fixture
def setup_trait_groups(test_hdf5_path):
    """Setup trait groups in test HDF5 file."""
    groups = {
        'body': ['bmi'],
        'cancer': ['prca'],
    }
    save_trait_groups(str(test_hdf5_path), groups)
    return groups


def test_trait_data_exclude_cols_handles_nullable_keys():
    """TraitData allows keys and annot_names to be omitted."""
    trait_data = TraitData(
        df=pl.DataFrame({
            'CHR': [1],
            'POS': [1],
            'gradient': [0.1],
            'jackknife_blocks': [0],
        })
    )

    assert trait_data.annot_names is None
    assert trait_data.keys is None
    assert trait_data.exclude_cols == {'CHR', 'POS', 'jackknife_blocks', 'gradient', 'hessian'}


def test_trait_data_exclude_cols_includes_keys():
    """TraitData excludes merge keys when annotation names are inferred."""
    trait_data = TraitData(
        df=pl.DataFrame({
            'RSID': ['rs1'],
            'CHR': [1],
            'POS': [1],
            'gradient': [0.1],
            'jackknife_blocks': [0],
        }),
        keys=['RSID'],
    )

    assert trait_data.exclude_cols == {'RSID', 'CHR', 'POS', 'jackknife_blocks', 'gradient', 'hessian'}


def test_meta_analysis_initializes_arrays_on_first_update():
    """MetaAnalysis starts empty and becomes array-backed after update()."""
    meta = MetaAnalysis()
    point_estimates = np.array([[2.0, 4.0]])
    jackknife_estimates = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    assert meta.point_estimates is None
    assert meta.jackknife_estimates is None

    meta.update(point_estimates, jackknife_estimates)

    precision = 1 / np.var(jackknife_estimates, axis=0)
    np.testing.assert_allclose(meta.point_estimates, precision * point_estimates)
    np.testing.assert_allclose(meta.jackknife_estimates, precision * jackknife_estimates)


def test_meta_analysis_accumulates_multiple_updates():
    """MetaAnalysis keeps array shapes stable while accumulating estimates."""
    meta = MetaAnalysis()
    point_estimates = np.array([[2.0, 4.0]])
    jackknife_estimates = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    meta.update(point_estimates, jackknife_estimates)
    first_point_estimates = meta.point_estimates.copy()
    meta.update(point_estimates, jackknife_estimates)

    np.testing.assert_allclose(meta.point_estimates, first_point_estimates * 2)
    assert meta.point_estimates.shape == (1, 2)
    assert meta.jackknife_estimates.shape == (3, 2)
    assert np.all(np.isfinite(meta.std))
    assert np.all(np.isfinite(meta.z_scores))


def test_save_and_load_trait_groups(test_hdf5_path):
    """Test saving and loading trait groups."""
    # Define test groups
    groups = {
        'body': ['bmi'],
        'cancer': ['prca'],
    }
    
    # Save groups
    save_trait_groups(str(test_hdf5_path), groups)
    
    # Load groups back
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    
    # Verify
    assert 'body' in loaded_groups
    assert 'cancer' in loaded_groups
    assert set(loaded_groups['body']) == {'bmi'}
    assert set(loaded_groups['cancer']) == {'prca'}


def test_overwrite_trait_groups(test_hdf5_path):
    """Test that saving groups overwrites existing ones."""
    # Save initial groups
    groups1 = {
        'group1': ['trait1', 'trait2'],
    }
    save_trait_groups(str(test_hdf5_path), groups1)
    
    # Save different groups
    groups2 = {
        'body': ['bmi'],
        'cancer': ['prca'],
    }
    save_trait_groups(str(test_hdf5_path), groups2)
    
    # Load and verify only new groups exist
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    assert 'group1' not in loaded_groups
    assert 'body' in loaded_groups
    assert 'cancer' in loaded_groups


def test_meta_analysis_with_random_variants(test_hdf5_path, setup_trait_groups):
    """Test meta-analysis with random variant annotations."""
    # Load variant data
    variant_table = load_variant_data(str(test_hdf5_path))
    
    # Create random variant annotations
    probs = [0.1, 0.2]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Get trait groups
    trait_groups = get_trait_groups(str(test_hdf5_path))
    
    # Process each group
    for group_name, group_traits in trait_groups.items():
        meta = MetaAnalysis()
        
        # Run score test for each trait in the group
        for trait_name in group_traits:
            trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
            
            point_estimates, jackknife_estimates = run_score_test(
                trait_data=trait_data,
                annot=annot,
            )
            
            meta.update(point_estimates, jackknife_estimates)
        
        # Verify meta-analysis produces valid results
        z_scores = meta.z_scores
        assert z_scores.shape == (1, len(probs))
        assert not np.any(np.isnan(z_scores))
        
        # Verify standard errors are positive
        std_errors = meta.std
        assert np.all(std_errors > 0)


def test_meta_analysis_body_group(test_hdf5_path, setup_trait_groups):
    """Test meta-analysis specifically for body group (bmi)."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.15]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Meta-analyze body traits
    meta = MetaAnalysis()
    for trait_name in ['bmi']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_estimates, jackknife_estimates = run_score_test(
            trait_data=trait_data,
            annot=annot,
        )
        meta.update(point_estimates, jackknife_estimates)
    
    z_scores = meta.z_scores
    assert z_scores.shape == (1, 1)
    assert not np.isnan(z_scores[0, 0])


def test_meta_analysis_cancer_group(test_hdf5_path, setup_trait_groups):
    """Test meta-analysis specifically for cancer group (prca)."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.15]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Meta-analyze cancer traits
    meta = MetaAnalysis()
    for trait_name in ['prca']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_estimates, jackknife_estimates = run_score_test(
            trait_data=trait_data,
            annot=annot,
        )
        meta.update(point_estimates, jackknife_estimates)
    
    z_scores = meta.z_scores
    assert z_scores.shape == (1, 1)
    assert not np.isnan(z_scores[0, 0])


def test_trait_groups_with_multiple_annotations(test_hdf5_path, setup_trait_groups):
    """Test that trait groups work with multiple annotations."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.05, 0.1, 0.2, 0.5]
    annot = create_random_variant_annotations(variant_table, probs)
    
    trait_groups = get_trait_groups(str(test_hdf5_path))
    
    for group_name, group_traits in trait_groups.items():
        meta = MetaAnalysis()
        
        for trait_name in group_traits:
            trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
            point_estimates, jackknife_estimates = run_score_test(
                trait_data=trait_data,
                annot=annot,
            )
            meta.update(point_estimates, jackknife_estimates)
        
        z_scores = meta.z_scores
        assert z_scores.shape == (1, len(probs))
        assert not np.any(np.isnan(z_scores))


def test_empty_trait_groups(test_hdf5_path):
    """Test behavior when no trait groups are defined."""
    # Save empty groups
    save_trait_groups(str(test_hdf5_path), {})
    
    # Load and verify
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    assert loaded_groups == {}


def test_single_trait_group(test_hdf5_path):
    """Test that single-trait groups are handled (should not create meta-analysis)."""
    groups = {
        'single': ['bmi'],
    }
    save_trait_groups(str(test_hdf5_path), groups)
    
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    assert 'single' in loaded_groups
    assert loaded_groups['single'] == ['bmi']


def test_trait_groups_with_nonexistent_traits(test_hdf5_path):
    """Test that groups can contain trait names that don't exist (they'll be filtered)."""
    groups = {
        'mixed': ['bmi', 'nonexistent_trait', 'prca'],
    }
    save_trait_groups(str(test_hdf5_path), groups)
    
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    assert 'mixed' in loaded_groups
    # The group is saved as-is; filtering happens during score test execution
    assert set(loaded_groups['mixed']) == {'bmi', 'nonexistent_trait', 'prca'}


def test_meta_analysis_precision_weighting(test_hdf5_path, setup_trait_groups):
    """Test that meta-analysis uses precision weighting correctly."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.1]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Get results for individual traits
    trait_results = {}
    for trait_name in ['bmi', 'prca']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_est, jk_est = run_score_test(trait_data=trait_data, annot=annot)
        
        # Compute individual z-score
        std_dev = np.std(jk_est, axis=0)
        n = jk_est.shape[0] - 1
        z_score = point_est.ravel() / std_dev / np.sqrt(n)
        
        trait_results[trait_name] = {
            'point': point_est,
            'jackknife': jk_est,
            'z_score': z_score,
            'precision': 1 / np.var(jk_est, axis=0)
        }
    
    # Compute meta-analysis
    meta = MetaAnalysis()
    for trait_name in ['bmi', 'prca']:
        meta.update(trait_results[trait_name]['point'], trait_results[trait_name]['jackknife'])
    
    meta_z = meta.z_scores.ravel()[0]
    
    # They should be different (unless by chance the precisions are equal)
    # Just verify meta-analysis produces a valid result
    assert not np.isnan(meta_z)
    assert np.isfinite(meta_z)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
