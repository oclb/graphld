"""Tests for LDSC sumstats file reading functionality."""

import polars as pl
import pytest
import numpy as np
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
    required_cols = {'SNP', 'N', 'Z', 'A1', 'A2', 'CHR', 'POS'}
    assert set(df.columns) == required_cols
    
    # Check data types
    assert df.schema['SNP'] == pl.Utf8
    assert df.schema['N'] == pl.Int64
    assert df.schema['Z'] == pl.Float64
    assert df.schema['A1'] == pl.Utf8
    assert df.schema['A2'] == pl.Utf8
    assert df.schema['CHR'] == pl.Int64
    assert df.schema['POS'] == pl.Int64
    
    # Check that we have data
    assert df.height > 0
    