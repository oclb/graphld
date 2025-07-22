"""Test LD clumping functionality."""

import numpy as np
import polars as pl

from graphld import LDClumper


def test_clumping(create_sumstats):
    """Test LD clumping with simulated summary statistics."""

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create sumstats from metadata
    metadata_path = "data/test/metadata.csv"
    sumstats = create_sumstats(metadata_path, populations="EUR")

    assert isinstance(sumstats, pl.DataFrame)
    print(sumstats.head())

    # Run clumping with reasonable thresholds
    rsq_threshold = 0.2  # Relatively stringent LD threshold
    chisq_threshold = 5.0  # Roughly p < 2.5e-2

    # Test default options (match by position, Z column)
    clumped = LDClumper.clump(
        sumstats,
        ldgm_metadata_path=metadata_path,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False
    )

    # Basic sanity checks
    assert(len(clumped) == np.sum(sumstats.select(pl.col('POS').is_first_distinct()).to_numpy()))
    assert 'is_index' in clumped.columns
    assert clumped.select('is_index').to_numpy().dtype == bool

    # Check that we found some index variants
    n_index = clumped.select(pl.col('is_index')).sum().item()
    assert n_index > 0, "No index variants found"

    # Check that index variants have high chi-square values
    index_variants = clumped.filter(pl.col('is_index'))
    chisq = index_variants.select(pl.col('Z')**2).to_numpy()
    assert np.all(chisq >= chisq_threshold), "Found index variant below chi-square threshold"

    # Test matching by variant ID
    clumped_by_id = LDClumper.clump(
        sumstats,
        ldgm_metadata_path=metadata_path,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False,
        match_by_position=False,
        variant_id_col='SNP'
    )

    # Results should be identical since variant IDs match positions
    assert np.array_equal(
        clumped.select('is_index').to_numpy(),
        clumped_by_id.select('is_index').to_numpy()
    ), "Results differ when matching by variant ID"

    # Test alternative Z score column
    clumped_alt_z = LDClumper.clump(
        sumstats,
        ldgm_metadata_path=metadata_path,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False,
        z_col='Z'
    )

    # Results should be identical since Z scores are the same
    assert np.array_equal(
        clumped.select('is_index').to_numpy(),
        clumped_alt_z.select('is_index').to_numpy()
    ), "Results differ with alternative Z score column"
