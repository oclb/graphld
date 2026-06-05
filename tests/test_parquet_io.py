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


def _write_parquet(tmp_path, name, **columns):
    path = tmp_path / name
    pl.DataFrame(columns).write_parquet(str(path))
    return str(path)


def test_incomplete_trait_missing_se(tmp_path):
    """Trait with a BETA column but no SE column raises with a clear message."""
    path = _write_parquet(
        tmp_path,
        "missing_se.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        trait1_BETA=[0.1, 0.2],
        # trait1_SE intentionally absent
    )
    with pytest.raises(ValueError, match="Trait 'trait1' is incomplete"):
        read_parquet_sumstats(path, trait='trait1')
    with pytest.raises(ValueError, match=r"trait1_SE"):
        read_parquet_sumstats(path, trait='trait1')


def test_incomplete_trait_missing_beta(tmp_path):
    """Trait with an SE column but no BETA column raises with a clear message."""
    path = _write_parquet(
        tmp_path,
        "missing_beta.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        trait1_SE=[0.05, 0.04],
    )
    with pytest.raises(ValueError, match=r"trait1_BETA"):
        read_parquet_sumstats(path, trait='trait1')


def test_incomplete_trait_default_selection_raises(tmp_path):
    """When trait=None and the first available trait is incomplete, raise."""
    path = _write_parquet(
        tmp_path,
        "default_incomplete.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        # Only one half — get_parquet_traits returns ['trait1'] but it's incomplete.
        trait1_BETA=[0.1, 0.2],
    )
    with pytest.raises(ValueError, match="Trait 'trait1' is incomplete"):
        read_parquet_sumstats(path)


def test_no_variant_identifier(tmp_path):
    """File with a complete trait but no SNP- or POS-alias column raises."""
    path = _write_parquet(
        tmp_path,
        "no_id.parquet",
        trait1_BETA=[0.1, 0.2],
        trait1_SE=[0.05, 0.04],
    )
    with pytest.raises(ValueError, match="no usable variant identifier"):
        read_parquet_sumstats(path, trait='trait1')


def test_only_snp_identifier_ok(tmp_path):
    """SNP alone is sufficient; the result includes SNP and Z (no POS)."""
    path = _write_parquet(
        tmp_path,
        "snp_only.parquet",
        SNP=['rs1', 'rs2'],
        trait1_BETA=[0.1, 0.2],
        trait1_SE=[0.05, 0.04],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert 'SNP' in df.columns
    assert 'Z' in df.columns
    assert 'POS' not in df.columns
    np.testing.assert_allclose(df['Z'].to_numpy(), [2.0, 5.0], rtol=1e-6)


def test_only_pos_identifier_ok(tmp_path):
    """POS alone (with optional CHR) is sufficient; no SNP column required."""
    path = _write_parquet(
        tmp_path,
        "pos_only.parquet",
        chrom=[1, 1],
        position=[1000, 2000],
        trait1_BETA=[0.1, -0.2],
        trait1_SE=[0.05, 0.04],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert 'POS' in df.columns
    assert 'CHR' in df.columns
    assert 'SNP' not in df.columns
    assert 'Z' in df.columns


def test_missing_alleles_ok(tmp_path):
    """REF/ALT are optional; their absence does not block reading."""
    path = _write_parquet(
        tmp_path,
        "no_alleles.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        trait1_BETA=[0.1, 0.2],
        trait1_SE=[0.05, 0.04],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert 'SNP' in df.columns
    assert 'POS' in df.columns
    assert 'REF' not in df.columns
    assert 'ALT' not in df.columns
    assert 'Z' in df.columns


def test_a1_a2_alleles_mapped_to_alt_ref(tmp_path):
    """A1/A2 aliases map to ALT/REF respectively (GWAS-VCF convention)."""
    path = _write_parquet(
        tmp_path,
        "a1_a2.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        A1=['T', 'G'],
        A2=['A', 'C'],
        trait1_BETA=[0.1, 0.2],
        trait1_SE=[0.05, 0.04],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert df['ALT'].to_list() == ['T', 'G']
    assert df['REF'].to_list() == ['A', 'C']


def test_no_traits_in_file(tmp_path):
    """File with no BETA/SE columns at all raises early."""
    path = _write_parquet(
        tmp_path,
        "no_traits.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
    )
    with pytest.raises(ValueError, match="No traits found"):
        read_parquet_sumstats(path)


def test_multi_propagates_incomplete_trait(tmp_path):
    """`read_parquet_sumstats_multi` surfaces per-trait incomplete-pair errors."""
    path = _write_parquet(
        tmp_path,
        "multi_incomplete.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        good_BETA=[0.1, 0.2],
        good_SE=[0.05, 0.04],
        bad_BETA=[0.1, 0.2],
        # bad_SE absent
    )
    with pytest.raises(ValueError, match="Trait 'bad' is incomplete"):
        read_parquet_sumstats_multi(path)


def test_all_nan_beta_se_returns_empty(tmp_path):
    """All-null BETA/SE produces an empty DataFrame after Z-finite filtering."""
    path = _write_parquet(
        tmp_path,
        "all_nan.parquet",
        SNP=['rs1', 'rs2'],
        POS=[100, 200],
        trait1_BETA=[None, None],
        trait1_SE=[None, None],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert df.height == 0
    assert 'Z' in df.columns


def test_alias_priority_pinned(tmp_path):
    """When multiple aliases coexist, the first in alias-tuple order wins."""
    path = _write_parquet(
        tmp_path,
        "alias_priority.parquet",
        site_ids=['rs1', 'rs2'],
        SNP=['ignored1', 'ignored2'],
        position=[100, 200],
        POS=[999, 999],
        ref=['A', 'C'],
        REF=['Z', 'Z'],
        trait1_BETA=[0.1, 0.2],
        trait1_SE=[0.05, 0.04],
    )
    df = read_parquet_sumstats(path, trait='trait1')
    assert df['SNP'].to_list() == ['rs1', 'rs2']  # site_ids wins over SNP
    assert df['POS'].to_list() == [100, 200]      # position wins over POS
    assert df['REF'].to_list() == ['A', 'C']      # ref wins over REF


def test_multi_export_from_top_level():
    """`read_parquet_sumstats_multi` is importable from the top-level package."""
    import graphld

    assert hasattr(graphld, 'read_parquet_sumstats_multi')
    assert 'read_parquet_sumstats_multi' in graphld.__all__
