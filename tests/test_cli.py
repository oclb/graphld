"""Test CLI functionality."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from graphld.cli import _blup, _clump, _reml, _simulate


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
    return test_data_dir / "example.sumstats"


@pytest.fixture
def annotation_dir(test_data_dir):
    """Get test annotation directory."""
    return test_data_dir / "annot"


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
            verbose=False,
            quiet=True
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
            verbose=False,
            quiet=True
        )

        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert len(result) > 0  # Should have some results with relaxed thresholds


def test_simulate(metadata_path):
    """Test simulation command."""
    with tempfile.NamedTemporaryFile(suffix=".sumstats") as tmp:
        _simulate(
            sumstats_out=tmp.name,
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
            verbose=False,
            sample_size=1000
        )

        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert 'Z' in result.columns
        assert 'POS' in result.columns
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
                verbose=False,
                quiet=True
            )


def test_reml_basic(metadata_path, create_annotations, create_sumstats):
    """Test basic REML functionality."""
    # Create test data using fixtures
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')
    annotations = annotations.rename({'POS': 'BP'})

    # Ensure CHR is Int64 in both DataFrames
    sumstats = sumstats.with_columns(pl.col('CHR').cast(pl.Int64))
    annotations = annotations.with_columns(pl.col('CHR').cast(pl.Int64))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "test"

        # Write sumstats to temporary file
        sumstats_file = Path(tmpdir) / "sumstats.csv"
        sumstats.write_csv(sumstats_file, separator='\t')

        # Write annotations to temporary file
        annot_dir = Path(tmpdir) / "annot"
        annot_dir.mkdir()
        annotations.write_csv(annot_dir / "baselineLD.22.annot", separator='\t')

        _reml(
            type("Args", (), {
                "sumstats": str(sumstats_file),
                "annot_dir": str(annot_dir),
                "out": str(out_prefix),
                "metadata": str(metadata_path),
                "num_samples": 1000,
                "name": "test",
                "intercept": 1.0,
                "num_iterations": 2,  # Small number for testing
                "convergence_tol": 0.001,
                "convergence_window": 2,
                "run_in_serial": True,
                "num_processes": None,
                "verbose": False,
                "quiet": True,
                "num_jackknife_blocks": 100,
                "match_by_position": True,
                "reset_trust_region": False,
                "chromosome": None,
                "xtrace_num_samples": 100,
                "max_chisq_threshold": None,
                "alt_output": False,  # Use default tall format
                "maximum_missingness": 1.0,
                "variant_stats_output": None,
                "population": 'EUR',
                "annotation_columns": None,
                "score_test_filename": None,
                "binary_annotations_only": False,
            })()
        )

        # Check that output files exist
        assert (out_prefix.with_suffix(".tall.csv")).exists()
        assert (out_prefix.with_suffix(".convergence.csv")).exists()

        # Also test alt_output format
        _reml(
            type("Args", (), {
                "sumstats": str(sumstats_file),
                "annot_dir": str(annot_dir),
                "out": str(out_prefix) + "_alt",
                "metadata": str(metadata_path),
                "num_samples": 1000,
                "name": "test",
                "intercept": 1.0,
                "num_iterations": 2,  # Small number for testing
                "convergence_tol": 0.001,
                "convergence_window": 2,
                "run_in_serial": True,
                "num_processes": None,
                "verbose": False,
                "quiet": True,
                "num_jackknife_blocks": 100,
                "match_by_position": True,
                "reset_trust_region": False,
                "chromosome": None,
                "population": "EUR",
                "xtrace_num_samples": 100,
                "max_chisq_threshold": None,
                "alt_output": True,  # Test alternative output format
                "maximum_missingness": 1.0,
                "variant_stats_output": None,
                "annotation_columns": None,
                "score_test_filename": None,
                "binary_annotations_only": False,
            })()
        )

        # Check that alt output files exist
        alt_prefix = Path(str(out_prefix) + "_alt")
        assert (alt_prefix.with_suffix(".heritability.csv")).exists()
        assert (alt_prefix.with_suffix(".enrichment.csv")).exists()
        assert (alt_prefix.with_suffix(".parameters.csv")).exists()
        assert (alt_prefix.with_suffix(".convergence.csv")).exists()
