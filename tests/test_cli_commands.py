"""Tests for estest CLI commands."""

import subprocess
import pytest
from pathlib import Path
import shutil
import tempfile


@pytest.fixture
def temp_hdf5(tmp_path):
    """Create a temporary copy of test HDF5 file."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    temp_file = tmp_path / "test.scores.h5"
    shutil.copy(test_data, temp_file)
    return str(temp_file)


def test_show_command(temp_hdf5):
    """Test the show command displays traits and meta-analyses."""
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Traits (6):" in result.stdout
    assert "bmi" in result.stdout
    assert "prca" in result.stdout
    assert "Meta-analyses (2):" in result.stdout
    assert "body: bmi" in result.stdout
    assert "cancer: prca" in result.stdout
    assert "Variant datasets:" in result.stdout


def test_show_command_verbose(temp_hdf5):
    """Test the show command with verbose flag."""
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5, "-v"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Detailed trait information:" in result.stdout


def test_add_meta_command(temp_hdf5):
    """Test adding a meta-analysis group."""
    result = subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "test_group", "bmi", "t2d"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Added meta-analysis 'test_group'" in result.stdout
    
    # Verify it was added
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5],
        capture_output=True,
        text=True,
    )
    assert "test_group: bmi, t2d" in result.stdout


def test_add_meta_duplicate_trait_name(temp_hdf5):
    """Test that adding a meta-analysis with a trait name fails."""
    result = subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "bmi", "t2d", "scz"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "already a trait name" in result.stderr


def test_add_meta_nonexistent_trait(temp_hdf5):
    """Test that adding a meta-analysis with nonexistent traits fails."""
    result = subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "test_group", "bmi", "nonexistent"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "do not exist" in result.stderr
    assert "nonexistent" in result.stderr


def test_add_meta_single_trait(temp_hdf5):
    """Test that adding a meta-analysis with only one trait fails."""
    result = subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "test_group", "bmi"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "at least 2 traits" in result.stderr


def test_rm_meta_command(temp_hdf5):
    """Test removing a meta-analysis group."""
    # First add a group
    subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "test_group", "bmi", "t2d"],
        capture_output=True,
    )
    
    # Remove it
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "test_group", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Removed 1 meta-analysis(es): test_group" in result.stdout
    
    # Verify it was removed
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5],
        capture_output=True,
        text=True,
    )
    assert "test_group" not in result.stdout


def test_rm_meta_default(temp_hdf5):
    """Test that rm defaults to removing meta-analysis."""
    # Add a group
    subprocess.run(
        ["uv", "run", "estest", "add-meta", temp_hdf5, "test_group", "bmi", "t2d"],
        capture_output=True,
    )
    
    # Remove without --meta flag (should default to meta)
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "test_group", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Removed 1 meta-analysis(es): test_group" in result.stdout


def test_rm_nonexistent_meta(temp_hdf5):
    """Test removing a nonexistent meta-analysis fails."""
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "nonexistent", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "No traits or meta-analyses match pattern" in result.stderr


def test_rm_trait_command(temp_hdf5):
    """Test removing a trait."""
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "scz", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Removed 1 trait(s): scz" in result.stdout
    
    # Verify it was removed
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5],
        capture_output=True,
        text=True,
    )
    assert "Traits (5):" in result.stdout
    assert "scz" not in result.stdout


def test_rm_trait_updates_meta_analyses(temp_hdf5):
    """Test that removing a trait updates meta-analyses."""
    # Remove bmi (part of body meta-analysis)
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "bmi", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Removed 1 trait(s): bmi" in result.stdout
    # Body meta-analysis should be removed (only 1 trait left) but message suppressed with -f
    
    # Verify
    result = subprocess.run(
        ["uv", "run", "estest", "show", temp_hdf5],
        capture_output=True,
        text=True,
    )
    assert "body" not in result.stdout
    assert "bmi" not in result.stdout


def test_rm_nonexistent_trait(temp_hdf5):
    """Test removing a nonexistent trait fails."""
    result = subprocess.run(
        ["uv", "run", "estest", "rm", temp_hdf5, "nonexistent", "-f"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "No traits or meta-analyses match pattern" in result.stderr


def test_legacy_behavior_still_works(temp_hdf5):
    """Test that legacy direct invocation still works."""
    result = subprocess.run(
        ["uv", "run", "estest", temp_hdf5, "--random-variants", ".1"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "shape:" in result.stdout
    assert "bmi" in result.stdout


def test_convert_command_help():
    """Test that convert command help works."""
    result = subprocess.run(
        ["uv", "run", "estest", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Convert variant-level statistics" in result.stdout
    assert "gene-table" in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
