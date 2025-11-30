"""Tests for heritability estimation."""

import os
import tempfile

import h5py
import numpy as np
import polars as pl

from graphld.heritability import (
    MethodOptions,
    ModelOptions,
    _get_softmax_link_function,
    partition_variants,
    run_graphREML,
)
from graphld.io import read_ldgm_metadata


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
    link_fn, link_fn_grad, link_fn_hess = _get_softmax_link_function(denominator=10)

    # Test case from MATLAB: linkFnGrad((1:3),(2:4)')
    annot = np.array([[1, 2, 3]])
    theta = np.array([[0.2], [0.3], [0.4]])

    # Test case from MATLAB: linkFn((1:3),(2:4)')
    val = link_fn(annot, theta)
    assert np.allclose(val, 0.2127, rtol=1e-3), f"Expected 0.2127, got {val}"

    grad = link_fn_grad(annot, theta)
    assert np.allclose(grad, [0.0881,0.1762,0.2642], rtol=1e-3)

    hess = link_fn_hess(annot, theta)
    assert np.allclose(hess, [0.0105, 0.0420, 0.0945], rtol=1e-3)

    annot = np.array([[1,2,3],[1,1,1]])
    val = link_fn_grad(np.array(1), annot @ theta)
    print(val)
    expected = [[0.0881], [0.0711]]
    assert np.allclose(val, expected, rtol=1e-3), f"Expected {expected}, got {val}"


def test_max_z_squared_threshold(metadata_path, create_annotations, create_sumstats):
    """Test that blocks with high Z² values are discarded."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Print initial Z² values
    print("\nInitial Z² values:")
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    for i, block in enumerate(partition_variants(metadata, sumstats)):
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
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    for i, block in enumerate(partition_variants(metadata, sumstats)):
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

    # Create a temporary HDF5 file for variant-specific statistics output
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        variant_stats_path = tmp.name

    # Run GraphREML with variant-specific statistics computation enabled
    model = ModelOptions(
        params=np.zeros((1,1)),
        sample_size=1000,
    )
    method = MethodOptions(
        match_by_position=False,
        num_iterations=1,  # Only need one iteration for this test
        verbose=True,
        run_serial=True,  # Run in serial mode for debugging
        score_test_hdf5_file_name=variant_stats_path,
        score_test_hdf5_trait_name='test',
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

    # Verify that the HDF5 file exists and has the required structure
    assert os.path.exists(variant_stats_path), f"HDF5 file {variant_stats_path} does not exist"

    with h5py.File(variant_stats_path, 'r') as f:
        # Verify that the file contains required blocks
        assert 'traits' in f, "HDF5 file missing 'traits' group"
        assert 'row_data' in f, "HDF5 file missing 'row_data' group"

    # Return the path to the HDF5 file for use in the score test
    return variant_stats_path


def test_score_test(metadata_path, create_annotations, create_sumstats):
    """Test the score test functionality using the refactored run_score_test function."""
    # Generate the HDF5 file with variant statistics
    variant_stats_path = test_variant_specific_statistics(
        metadata_path,
        create_annotations,
        create_sumstats
    )

    try:
        # Import the necessary functions from score_test
        from score_test.score_test_io import load_trait_data, load_row_data
        from score_test.score_test import run_score_test, VariantAnnot

        # Load variant and trait data
        variant_table = load_row_data(variant_stats_path)
        trait_data = load_trait_data(variant_stats_path, trait_name='test', variant_table=variant_table)

        # Create a simple test annotation dataframe
        annotations = create_annotations(metadata_path, 'EUR')
        test_annot_name = "test_score_annot"

        # Rename SNP to RSID for consistency with new format
        annotations = annotations.rename({'SNP': 'RSID'})
        
        # Add a random annotation column
        np.random.seed(42)
        annotations = annotations.with_columns(
            pl.Series(name=test_annot_name, values=np.random.normal(0, 1, annotations.height))
        )

        # Create VariantAnnot object
        annot = VariantAnnot(annotations, [test_annot_name])

        # Run the score test with the loaded dataframes
        point_estimates, jackknife_estimates = run_score_test(
            trait_data=trait_data,
            annot=annot,
        )

        # Verify the results
        assert point_estimates is not None
        assert jackknife_estimates is not None

    finally:
        # Clean up the variant statistics file
        if os.path.exists(variant_stats_path):
            os.unlink(variant_stats_path)
