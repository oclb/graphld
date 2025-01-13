"""Test CLI functionality."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from graphld.cli import _blup, _clump, _simulate


@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path("data/test")


@pytest.fixture
def metadata_path(test_data_dir):
    """Get test metadata path."""
    return test_data_dir / "metadata.csv"


@pytest.fixture
def sumstats_path(test_data_dir):
    """Get test sumstats path."""
    return test_data_dir / "test_sumstats.sumstats"


def test_blup(metadata_path, sumstats_path):
    """Test BLUP command."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        _blup(
            sumstats=str(sumstats_path),
            out=tmp.name,
            metadata=str(metadata_path),
            num_samples=1000,
            heritability=0.5,
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False
        )
        
        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert 'Z' in result.columns  # BLUP outputs Z-scores
        assert len(result) > 0


def test_clump(metadata_path, sumstats_path):
    """Test clumping command."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        _clump(
            sumstats=str(sumstats_path),
            out=tmp.name,
            metadata=str(metadata_path),
            num_samples=None,
            min_chisq=1.0,  # Lower threshold to get some results
            max_rsq=0.9,    # Higher threshold to get some results
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False
        )
        
        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert len(result) > 0  # Should have some results with relaxed thresholds


def test_simulate(metadata_path):
    """Test simulation command."""
    with tempfile.NamedTemporaryFile(suffix=".sumstats") as tmp:
        _simulate(
            sumstats=tmp.name,
            metadata=str(metadata_path),
            heritability=0.5,
            component_variance=[1.0],
            component_weight=[1.0],
            alpha_param=-0.5,
            annotation_dependent_polygenicity=False,
            random_seed=42,
            annotation_columns=None,
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False
        )
        
        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert 'Z' in result.columns
        assert len(result) > 0


def test_invalid_sumstats_format():
    """Test error handling for invalid sumstats format."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError, match="Input file must end in .vcf or .sumstats"):
            _blup(
                sumstats=tmp.name,
                out="out.csv",
                metadata="data/test/metadata.csv",
                num_samples=1000,
                heritability=0.5,
                num_processes=None,
                run_in_serial=True,
                chromosome=None,
                population=None,
                verbose=False
            )
