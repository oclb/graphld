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
    assert len(clumped) == len(sumstats)
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


def test_clumping_preserves_input_rows(create_sumstats):
    """Clumping results should align with the caller's input rows."""
    metadata_path = "data/test/metadata.csv"
    sumstats = create_sumstats(metadata_path, populations="EUR").head(5)
    lead_snp = sumstats.row(4, named=True)["SNP"]
    sumstats = sumstats.with_columns(
        pl.when(pl.col("SNP") == lead_snp).then(10.0).otherwise(0.0).alias("Z")
    )
    out_of_block = sumstats.head(1).with_columns(
        pl.lit("out_of_block").alias("SNP"),
        pl.lit(1, dtype=pl.Int64).alias("CHR"),
        pl.lit(999_999_999, dtype=pl.Int64).alias("POS"),
        pl.lit(0.0).alias("Z"),
    )
    unsorted_sumstats = pl.concat(
        [
            sumstats.slice(3, 1),
            out_of_block,
            sumstats.slice(1, 1),
            sumstats.slice(0, 1),
            sumstats.slice(4, 1),
            sumstats.slice(2, 1),
        ]
    ).with_columns(
        pl.int_range(pl.len()).alias("input_order"),
        (pl.int_range(pl.len()) + 100).alias("row_nr"),
        pl.lit("caller-row").alias("__graphld_clump_row_nr"),
        pl.lit(True).alias("__graphld_clump_is_index"),
    )

    clumped = LDClumper.clump(
        unsorted_sumstats,
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        chisq_threshold=1.0,
        run_in_serial=True,
    )
    clumped_parallel = LDClumper.clump(
        unsorted_sumstats,
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        chisq_threshold=1.0,
        num_processes=2,
        run_in_serial=False,
    )

    assert len(clumped) == len(unsorted_sumstats)
    assert clumped.to_dict(as_series=False) == clumped_parallel.to_dict(as_series=False)
    assert clumped.select("input_order").to_series().to_list() == list(range(6))
    assert clumped.select("row_nr").to_series().to_list() == list(range(100, 106))
    assert clumped.select("__graphld_clump_row_nr").to_series().to_list() == (
        ["caller-row"] * 6
    )
    assert clumped.select("__graphld_clump_is_index").to_series().to_list() == (
        [True] * 6
    )
    assert clumped.select("SNP").to_series().to_list() == (
        unsorted_sumstats.select("SNP").to_series().to_list()
    )
    assert clumped.select("is_index").to_numpy().dtype == bool
    assert clumped.filter(pl.col("SNP") == lead_snp).select("is_index").item()
    assert not clumped.filter(pl.col("SNP") == "out_of_block").select("is_index").item()


def test_clumping_retains_rows_when_all_blocks_are_empty(create_sumstats):
    """Rows on chromosomes absent from metadata should return as non-index rows."""
    metadata_path = "data/test/metadata.csv"
    sumstats = (
        create_sumstats(metadata_path, populations="EUR")
        .head(3)
        .with_columns(
            pl.lit(2, dtype=pl.Int64).alias("CHR"),
            pl.int_range(pl.len()).alias("input_order"),
        )
    )

    clumped = LDClumper.clump(
        sumstats,
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        run_in_serial=True,
    )

    assert len(clumped) == len(sumstats)
    assert clumped.select("input_order").to_series().to_list() == [0, 1, 2]
    assert not clumped.select("is_index").to_series().any()
