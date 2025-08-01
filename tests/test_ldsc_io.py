"""Tests for LDSC sumstats file reading functionality."""

import polars as pl

from graphld.ldsc_io import read_ldsc_sumstats


def test_read_ldsc_sumstats_with_beta_se():
    """Test reading LDSC sumstats file with Beta and SE."""
    # Use a test sumstats file with Beta and SE
    df = read_ldsc_sumstats(
        "data/test/example.sumstats",
        add_positions=True,
        positions_file="data/test/rsid_position.csv"
    )

    # Check basic DataFrame structure
    assert isinstance(df, pl.DataFrame)

    # Check required columns
    required_cols = {'SNP', 'N', 'Z', 'REF', 'ALT', 'CHR', 'POS'}
    missing_cols = required_cols - set(df.columns)
    assert required_cols.issubset(set(df.columns)), f"Missing columns: {missing_cols}"

    # Check data types
    assert df.schema['SNP'] == pl.Utf8
    assert df.schema['N'] == pl.Int64
    assert df.schema['Z'] == pl.Float64
    assert df.schema['REF'] == pl.Utf8
    assert df.schema['ALT'] == pl.Utf8
    assert df.schema['CHR'] == pl.Int64
    assert df.schema['POS'] == pl.Int64

    # Check that we have data
    assert df.height > 0
