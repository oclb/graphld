"""Tests for score test CLI functionality."""

import subprocess
import pytest
from pathlib import Path
import polars as pl
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from score_test.score_test_io import save_trait_groups, get_trait_groups


def test_score_test_cli_random_variants():
    """Test score test CLI with random variants."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    
    # Run the command
    result = subprocess.run(
        ["uv", "run", "estest", str(test_data), "--random-variants", ".1,.2"],
        capture_output=True,
        text=True,
    )
    
    # Check it succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Check output contains expected columns (may be truncated as "annotatio")
    assert "annotat" in result.stdout
    
    # Check we have some expected traits (not all are visible due to truncation)
    assert "bmi" in result.stdout
    assert "brca" in result.stdout
    
    # Check we have 2 random variants with 9 columns 
    # (annotation + 6 traits_z + 2 groups)
    assert "shape: (2, 9)" in result.stdout
    
    # Check that at least one group is present
    assert "body" in result.stdout or "cancer" in result.stdout


def test_score_test_cli_random_variants_with_output(tmp_path):
    """Test score test CLI with random variants and output file."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    output_file = tmp_path / "results"
    
    # Run the command (positional argument, not --output)
    result = subprocess.run(
        ["uv", "run", "estest", str(test_data), str(output_file), "--random-variants", ".1,.2"],
        capture_output=True,
        text=True,
    )
    
    # Check it succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Check output file was created
    output_txt = Path(str(output_file) + ".txt")
    assert output_txt.exists(), f"Output file {output_txt} was not created"
    
    # Read and verify output
    df = pl.read_csv(output_txt, separator='\t')
    
    # Check columns
    assert "annotation" in df.columns
    
    # Check we have 2 annotations (random_variant_0 and random_variant_1)
    assert len(df) == 2
    
    # Check traits are present (with _Z suffix)
    for trait in ["bmi", "brca", "cad", "edu", "height", "prca"]:
        assert f"{trait}_Z" in df.columns
    
    # Check that group columns are present (with _Z suffix)
    assert "body_Z" in df.columns
    assert "cancer_Z" in df.columns


def test_score_test_cli_multiple_random_variants():
    """Test score test CLI with multiple random variant probabilities."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    
    # Run with more annotations
    result = subprocess.run(
        ["uv", "run", "estest", str(test_data), "--random-variants", ".05,.1,.2,.5"],
        capture_output=True,
        text=True,
    )
    
    # Check it succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Check output contains expected number of rows (4 annotations, 9 columns)
    assert "shape: (4, 9)" in result.stdout


def test_score_test_meta_analysis():
    """Test that meta-analysis is computed when multiple traits are present."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    
    # Run the command
    result = subprocess.run(
        ["uv", "run", "estest", str(test_data), "--random-variants", ".1"],
        capture_output=True,
        text=True,
    )
    
    # Check it succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Check that group columns are present (groups were added to test.scores.h5)
    # Groups: body, cancer
    assert "body" in result.stdout or "cancer" in result.stdout


def test_trait_groups_io():
    """Test reading and writing trait groups."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    
    # Read existing groups
    groups = get_trait_groups(str(test_data))
    
    # Should have the groups we added
    assert 'body' in groups
    assert 'cancer' in groups
    
    # Verify group contents
    assert set(groups['body']) == {'height', 'bmi'}
    assert set(groups['cancer']) == {'brca', 'prca'}


def test_score_test_with_trait_groups(tmp_path):
    """Test that custom trait groups produce meta-analysis columns."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    output_file = tmp_path / "results"
    
    # Run the command with output
    result = subprocess.run(
        ["uv", "run", "estest", str(test_data), str(output_file), "--random-variants", ".1"],
        capture_output=True,
        text=True,
    )
    
    # Check it succeeded
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    
    # Read output file
    output_txt = Path(str(output_file) + ".txt")
    df = pl.read_csv(output_txt, separator='\t')
    
    # Check that group columns are present (with _Z suffix)
    assert 'body_Z' in df.columns
    assert 'cancer_Z' in df.columns
    
    # Check individual trait columns are also present (with _Z suffix)
    assert 'bmi_Z' in df.columns
    assert 'height_Z' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
