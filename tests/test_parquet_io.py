"""Tests for parquet summary statistics file reading functionality."""

import numpy as np
import polars as pl
import pytest

from graphld.parquet_io import get_parquet_traits, read_parquet_sumstats, read_parquet_sumstats_multi


def test_get_parquet_traits():
    """Test getting trait names from a parquet file."""
    traits = get_parquet_traits("data/test/example_multi_trait.parquet")
    
    assert isinstance(traits, list)
    assert len(traits) == 3
    assert set(traits) == {'height', 'bmi', 'weight'}


def test_read_parquet_sumstats_default_trait():
    """Test reading parquet sumstats with default (first) trait."""
    df = read_parquet_sumstats("data/test/example_multi_trait.parquet")
    
    # Check basic DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    
    # Check required columns exist
    required_cols = {'SNP', 'CHR', 'POS', 'REF', 'ALT', 'Z'}
    assert required_cols.issubset(set(df.columns))
    
    # Check data types
    assert df.schema['SNP'] == pl.Utf8
    assert df.schema['Z'] == pl.Float64


def test_read_parquet_sumstats_specific_trait():
    """Test reading a specific trait from parquet file."""
    df = read_parquet_sumstats("data/test/example_multi_trait.parquet", trait="bmi")
    
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0
    assert 'Z' in df.columns
    
    # Z scores should be finite
    assert df['Z'].is_finite().all()


def test_read_parquet_sumstats_invalid_trait():
    """Test that reading an invalid trait raises an error."""
    with pytest.raises(ValueError, match="Trait 'nonexistent' not found"):
        read_parquet_sumstats("data/test/example_multi_trait.parquet", trait="nonexistent")


def test_read_parquet_sumstats_multi():
    """Test reading multiple traits from parquet file."""
    result = read_parquet_sumstats_multi("data/test/example_multi_trait.parquet")
    
    assert isinstance(result, dict)
    assert len(result) == 3
    assert set(result.keys()) == {'height', 'bmi', 'weight'}
    
    for trait_name, df in result.items():
        assert isinstance(df, pl.DataFrame)
        assert 'Z' in df.columns
        assert df.height > 0


def test_read_parquet_sumstats_multi_subset():
    """Test reading a subset of traits from parquet file."""
    result = read_parquet_sumstats_multi(
        "data/test/example_multi_trait.parquet",
        traits=['height', 'bmi']
    )
    
    assert len(result) == 2
    assert set(result.keys()) == {'height', 'bmi'}


def test_read_parquet_sumstats_multi_invalid_trait():
    """Test that reading invalid traits raises an error."""
    with pytest.raises(ValueError, match="Traits not found"):
        read_parquet_sumstats_multi(
            "data/test/example_multi_trait.parquet",
            traits=['height', 'nonexistent']
        )


def test_read_parquet_sumstats_z_score_computation():
    """Test that Z scores are correctly computed from BETA/SE."""
    df = read_parquet_sumstats("data/test/example_multi_trait.parquet", trait="height")
    
    # Read raw data to verify Z computation
    raw = pl.read_parquet("data/test/example_multi_trait.parquet")
    expected_z = (raw['height_BETA'] / raw['height_SE']).cast(pl.Float64)
    
    # Z scores should match BETA/SE (filtering out any non-finite values)
    finite_mask = expected_z.is_finite()
    expected_finite = expected_z.filter(finite_mask).to_numpy()
    actual_z = df['Z'].to_numpy()
    
    assert len(actual_z) == len(expected_finite)
    np.testing.assert_allclose(actual_z, expected_finite, rtol=1e-5)


def test_read_parquet_sumstats_with_temporary_file(tmp_path):
    """Test reading a parquet file created on the fly."""
    # Create a test parquet file
    parquet_file = tmp_path / "test.parquet"
    
    df = pl.DataFrame({
        'site_ids': ['rs1', 'rs2', 'rs3'],
        'chrom': [1, 1, 2],
        'position': [1000, 2000, 3000],
        'ref': ['A', 'C', 'G'],
        'alt': ['T', 'G', 'A'],
        'trait1_BETA': [0.1, -0.2, 0.3],
        'trait1_SE': [0.05, 0.04, 0.06],
        'trait2_BETA': [0.2, 0.1, -0.1],
        'trait2_SE': [0.03, 0.02, 0.04],
    })
    df.write_parquet(str(parquet_file))
    
    # Test trait detection
    traits = get_parquet_traits(str(parquet_file))
    assert set(traits) == {'trait1', 'trait2'}
    
    # Test reading
    result = read_parquet_sumstats(str(parquet_file), trait='trait1')
    assert result.height == 3
    assert 'Z' in result.columns
    
    # Verify Z computation
    expected_z = [0.1/0.05, -0.2/0.04, 0.3/0.06]
    np.testing.assert_allclose(result['Z'].to_numpy(), expected_z, rtol=1e-5)
