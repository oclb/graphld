"""Tests for heritability estimation."""

import numpy as np
import pytest
from typing import Optional, Union, List
from graphld import load_ldgm, Simulate, read_ldgm_metadata
from graphld.heritability import ModelOptions, MethodOptions, run_graphREML, partition_variants, _get_softmax_link_function
from graphld.io import read_ldgm_metadata
import polars as pl
import os

def test_run_graphREML(metadata_path, create_annotations, create_sumstats):
    """Test heritability estimation with simulated data."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Run GraphREML
    model = ModelOptions(params=np.zeros((1,1)),
                sample_size=1000
    )
    method = MethodOptions(
        match_by_position=True
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert result is not None


def test_softmax_link_function():
    """Test softmax link function against MATLAB output."""
    link_fn, link_fn_grad = _get_softmax_link_function(denominator=10)
    
    # Test case from MATLAB: linkFnGrad((1:3),(2:4)')
    annot = np.array([1, 2, 3])
    theta = np.array([[2], [3], [4]])
    grad = link_fn_grad(annot, theta)
    assert grad.shape == (3,)
    assert np.allclose(grad, [0.1, 0.2, 0.3])

    # Test case from MATLAB: linkFn((1:3),(2:4)')
    val = link_fn(annot, theta)
    assert np.allclose(val, 2.0)


def test_max_z_squared_threshold(metadata_path, create_annotations, create_sumstats):
    """Test that blocks with high Z² values are discarded."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Print initial Z² values
    print("\nInitial Z² values:")
    for i, block in enumerate(partition_variants(read_ldgm_metadata(metadata_path, populations='EUR'), sumstats)):
        max_z2 = float(np.max(block.select('Z').to_numpy() ** 2))
        print(f"Block {i}: max Z² = {max_z2}")

    # Spike in a large Z-score in the first block
    first_pos = sumstats.select('POS').head(1).item()
    sumstats = sumstats.with_columns(pl.when(pl.col('POS') == first_pos)
                                   .then(10.0)  # This will give a Z² of 100
                                   .otherwise(pl.col('Z'))
                                   .alias('Z'))
    
    # Print Z² values after spiking
    print("\nZ² values after spiking:")
    for i, block in enumerate(partition_variants(read_ldgm_metadata(metadata_path, populations='EUR'), sumstats)):
        max_z2 = float(np.max(block.select('Z').to_numpy() ** 2))
        print(f"Block {i}: max Z² = {max_z2}")
    
    # Set threshold to filter out the block with the large Z-score
    threshold = 50.0  # This will filter out blocks with Z² > 50
    
    # Run GraphREML with Z² threshold
    model = ModelOptions(params=np.zeros((1,1)),
                sample_size=1000
    )
    method = MethodOptions(
        match_by_position=True,
        max_chisq_threshold=threshold,
        verbose=True,  # Enable verbose output
        num_iterations=1,  # Only need one iteration for this test
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert result is not None


def test_variant_specific_statistics(metadata_path, create_annotations, create_sumstats):
    """Test computation of variant-specific gradient and Hessian diagonal."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Run GraphREML with variant-specific statistics computation enabled
    model = ModelOptions(
        params=np.zeros((1,1)),
        sample_size=1000,
    )
    method = MethodOptions(
        match_by_position=True,
        num_iterations=1,  # Only need one iteration for this test
        verbose=True,
        run_serial=True,  # Run in serial mode for debugging
        compute_variant_stats=True  # Enable computation of variant-specific statistics
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR'
    )

    # Verify that variant-specific statistics were computed
    assert result is not None
    assert 'variant_gradient' in result
    assert 'variant_hessian_diag' in result
    
    # Check that values are finite (not NaN or Inf)
    assert np.all(np.isfinite(result['variant_gradient']))
    assert np.all(np.isfinite(result['variant_hessian_diag']))
    
    # Check that gradient and Hessian have the same shape
    assert result['variant_gradient'].shape == result['variant_hessian_diag'].shape
    
    # Verify that gradient and Hessian are reasonable (just check they're not all zeros)
    assert not np.allclose(result['variant_gradient'], 0)
    assert not np.allclose(result['variant_hessian_diag'], 0)
