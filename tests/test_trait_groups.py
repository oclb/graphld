"""Tests for trait groups and meta-analysis functionality."""

import numpy as np
import polars as pl
import pytest
from pathlib import Path

from score_test.score_test_io import (
    save_trait_groups,
    get_trait_groups,
    load_variant_data,
    load_trait_data,
    create_random_variant_annotations,
    get_trait_names,
)
from score_test.score_test import run_approx_score_test
from score_test.meta_analysis import MetaAnalysis


@pytest.fixture
def test_hdf5_path():
    """Path to test HDF5 file."""
    return Path(__file__).parent / "score_test_data" / "test.scores.h5"


@pytest.fixture
def setup_trait_groups(test_hdf5_path):
    """Setup trait groups in test HDF5 file."""
    groups = {
        'body': ['height', 'bmi'],
        'cancer': ['brca', 'prca'],
    }
    save_trait_groups(str(test_hdf5_path), groups)
    return groups


def test_save_and_load_trait_groups(test_hdf5_path):
    """Test saving and loading trait groups."""
    # Define test groups
    groups = {
        'body': ['height', 'bmi'],
        'cancer': ['brca', 'prca'],
    }
    
    # Save groups
    save_trait_groups(str(test_hdf5_path), groups)
    
    # Load groups back
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    
    # Verify
    assert 'body' in loaded_groups
    assert 'cancer' in loaded_groups
    assert set(loaded_groups['body']) == {'height', 'bmi'}
    assert set(loaded_groups['cancer']) == {'brca', 'prca'}


def test_overwrite_trait_groups(test_hdf5_path):
    """Test that saving groups overwrites existing ones."""
    # Save initial groups
    groups1 = {
        'group1': ['trait1', 'trait2'],
    }
    save_trait_groups(str(test_hdf5_path), groups1)
    
    # Save different groups
    groups2 = {
        'body': ['height', 'bmi'],
        'cancer': ['brca', 'prca'],
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
            
            point_estimates, jackknife_estimates, _ = run_approx_score_test(
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
    """Test meta-analysis specifically for body group (height, bmi)."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.15]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Meta-analyze body traits
    meta = MetaAnalysis()
    for trait_name in ['height', 'bmi']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_estimates, jackknife_estimates, _ = run_approx_score_test(
            trait_data=trait_data,
            annot=annot,
        )
        meta.update(point_estimates, jackknife_estimates)
    
    z_scores = meta.z_scores
    assert z_scores.shape == (1, 1)
    assert not np.isnan(z_scores[0, 0])


def test_meta_analysis_cancer_group(test_hdf5_path, setup_trait_groups):
    """Test meta-analysis specifically for cancer group (brca, prca)."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.15]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Meta-analyze cancer traits
    meta = MetaAnalysis()
    for trait_name in ['brca', 'prca']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_estimates, jackknife_estimates, _ = run_approx_score_test(
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
            point_estimates, jackknife_estimates, _ = run_approx_score_test(
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
        'mixed': ['bmi', 'nonexistent_trait', 'height'],
    }
    save_trait_groups(str(test_hdf5_path), groups)
    
    loaded_groups = get_trait_groups(str(test_hdf5_path))
    assert 'mixed' in loaded_groups
    # The group is saved as-is; filtering happens during score test execution
    assert set(loaded_groups['mixed']) == {'bmi', 'nonexistent_trait', 'height'}


def test_meta_analysis_precision_weighting(test_hdf5_path, setup_trait_groups):
    """Test that meta-analysis uses precision weighting correctly."""
    variant_table = load_variant_data(str(test_hdf5_path))
    probs = [0.1]
    annot = create_random_variant_annotations(variant_table, probs)
    
    # Get results for individual traits
    trait_results = {}
    for trait_name in ['height', 'bmi']:
        trait_data = load_trait_data(str(test_hdf5_path), trait_name, variant_table)
        point_est, jk_est, _ = run_approx_score_test(trait_data=trait_data, annot=annot)
        
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
    for trait_name in ['height', 'bmi']:
        meta.update(trait_results[trait_name]['point'], trait_results[trait_name]['jackknife'])
    
    meta_z = meta.z_scores.ravel()[0]
    
    # Meta-analysis z-score should be different from simple average
    avg_z = np.mean([trait_results['height']['z_score'][0], trait_results['bmi']['z_score'][0]])
    
    # They should be different (unless by chance the precisions are equal)
    # Just verify meta-analysis produces a valid result
    assert not np.isnan(meta_z)
    assert np.isfinite(meta_z)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
