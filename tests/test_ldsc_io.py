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
    assert df.schema['N'] == pl.Float64
    assert df.schema['Z'] == pl.Float64
    assert df.schema['REF'] == pl.Utf8
    assert df.schema['ALT'] == pl.Utf8
    assert df.schema['CHR'] == pl.Int64
    assert df.schema['POS'] == pl.Int64

    # Check that we have data
    assert df.height > 0


def test_read_ldsc_sumstats_with_float_n(tmp_path):
    """Test reading LDSC sumstats file where N column contains floating-point values."""
    # Create a test file with floating-point N values
    sumstats_file = tmp_path / "float_n.sumstats"
    sumstats_file.write_text(
        "SNP\tA1\tA2\tBeta\tse\tN\n"
        "rs10399793\tT\tC\t-0.001\t0.003\t58749.13\n"
        "rs2462492\tC\tT\t-0.005\t0.003\t58749.50\n"
        "rs3107975\tT\tC\t0.009\t0.015\t58750.00\n"
    )

    # Create a minimal positions file
    positions_file = tmp_path / "positions.csv"
    positions_file.write_text(
        "chrom,site_ids,position\n"
        "1,rs10399793,49298\n"
        "1,rs2462492,54676\n"
        "1,rs3107975,55326\n"
    )

    df = read_ldsc_sumstats(
        str(sumstats_file),
        add_positions=True,
        positions_file=str(positions_file)
    )

    # Check that the file was read successfully
    assert isinstance(df, pl.DataFrame)
    assert df.height == 3

    # Check that N is Float64 and contains the expected values
    assert df.schema['N'] == pl.Float64
    n_values = df['N'].to_list()
    assert abs(n_values[0] - 58749.13) < 0.01
    assert abs(n_values[1] - 58749.50) < 0.01
    assert abs(n_values[2] - 58750.00) < 0.01
