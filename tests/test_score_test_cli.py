"""Tests for score test CLI functionality."""

import os
import shutil
import subprocess
import pytest
from pathlib import Path
import numpy as np
import polars as pl
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from score_test.score_test_io import get_trait_groups, load_variant_data, save_trait_groups

REPO_ROOT = Path(__file__).parent.parent


def run_estest_from_cwd(tmp_path, *args):
    return subprocess.run(
        ["uv", "run", "--project", str(REPO_ROOT), "estest", *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_score_test_script_help_without_pythonpath():
    """Test documented direct script invocation without installing the package."""
    script_path = REPO_ROOT / "src" / "score_test" / "score_test.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "--variant-annot-dir" in result.stdout


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
    assert "prca" in result.stdout
    
    # Check we have 2 random variants with 9 columns 
    # (annotation + 6 traits_z + 2 groups)
    assert "shape: (2, 9)" in result.stdout
    
    # Check that at least one group is present
    assert "body" in result.stdout or "cancer" in result.stdout


def test_score_test_legacy_random_variants_from_non_repo_cwd(tmp_path):
    """Test legacy estest invocation does not require default gene table."""
    test_data = REPO_ROOT / "data" / "test" / "test.scores.h5"

    result = run_estest_from_cwd(
        tmp_path, test_data, "--random-variants", ".1"
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "shape: (1, 9)" in result.stdout


def test_score_test_subcommand_random_variants_from_non_repo_cwd(tmp_path):
    """Test estest test does not require default gene table."""
    test_data = REPO_ROOT / "data" / "test" / "test.scores.h5"

    result = run_estest_from_cwd(
        tmp_path, "test", test_data, "--random-variants", ".1"
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "shape: (1, 9)" in result.stdout


def test_score_test_random_genes_requires_gene_table_from_non_repo_cwd(tmp_path):
    """Test variant-level gene workflows still require a real gene table."""
    test_data = REPO_ROOT / "data" / "test" / "test.scores.h5"

    result = run_estest_from_cwd(
        tmp_path, "test", test_data, "--random-genes", ".1"
    )

    assert result.returncode == 2
    assert "--gene-table path 'data/genes.tsv' does not exist" in result.stderr
    assert "--random-genes" in result.stderr


def test_score_test_gene_annot_dir_requires_gene_table_from_non_repo_cwd(tmp_path):
    """Test gene annotation conversion still requires a real gene table."""
    test_data = REPO_ROOT / "data" / "test" / "test.scores.h5"
    gmt_dir = tmp_path / "gmt"
    gmt_dir.mkdir()
    (gmt_dir / "test.gmt").write_text("set1\tDescription\tGENE1\n")

    result = run_estest_from_cwd(
        tmp_path, "test", test_data, "--gene-annot-dir", gmt_dir
    )

    assert result.returncode == 2
    assert "--gene-table path 'data/genes.tsv' does not exist" in result.stderr
    assert "--gene-annot-dir" in result.stderr


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
    for trait in ["bmi", "prca", "scz", "stroke", "t1d", "t2d"]:
        assert f"{trait}_Z" in df.columns
    
    # Check that group columns are present (with _Z suffix)
    assert "body_Z" in df.columns
    assert "cancer_Z" in df.columns


def test_score_test_cli_bed_variant_annotations(tmp_path):
    """Test score test CLI with BED-only variant annotations."""
    source_data = REPO_ROOT / "data" / "test" / "test.scores.h5"
    test_data = tmp_path / "test.scores.h5"
    shutil.copy(source_data, test_data)
    output_file = tmp_path / "bed_results"

    variant_data = load_variant_data(str(test_data))
    first_variant = variant_data.select("CHR", "POS").row(0, named=True)

    annot_dir = tmp_path / "bed"
    annot_dir.mkdir()
    bed_file = annot_dir / "regions.bed"
    bed_file.write_text(
        f"chr{first_variant['CHR']}\t{first_variant['POS']}\t"
        f"{first_variant['POS'] + 1}\n"
    )

    result = run_estest_from_cwd(
        tmp_path,
        "test",
        test_data,
        output_file,
        "--variant-annot-dir",
        annot_dir,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    output_txt = Path(str(output_file) + ".txt")
    df = pl.read_csv(output_txt, separator='\t')
    assert df["annotation"].to_list() == ["regions"]
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
    assert set(groups['body']) == {'bmi'}
    assert set(groups['cancer']) == {'prca'}


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
    assert 'prca_Z' in df.columns


def test_score_test_empty_processed_trait_group_outputs_nan(tmp_path):
    """Test stale trait groups that filter to zero processed traits."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    temp_hdf5 = tmp_path / "test.scores.h5"
    output_file = tmp_path / "results"
    shutil.copy(test_data, temp_hdf5)
    save_trait_groups(str(temp_hdf5), {"stale": ["missing_trait"]})

    result = subprocess.run(
        ["uv", "run", "estest", str(temp_hdf5), str(output_file), "--random-variants", ".1"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "Warning: Meta-analysis group 'stale' has no processed traits" in result.stderr

    output_txt = Path(str(output_file) + ".txt")
    df = pl.read_csv(output_txt, separator='\t')
    assert "stale_Z" in df.columns
    assert all(np.isnan(value) for value in df["stale_Z"].to_list())


def test_score_test_selected_stale_trait_group_outputs_nan(tmp_path):
    """Test explicitly selected stale trait groups do not load missing traits."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    temp_hdf5 = tmp_path / "test.scores.h5"
    output_file = tmp_path / "results"
    shutil.copy(test_data, temp_hdf5)
    save_trait_groups(str(temp_hdf5), {"stale": ["missing_trait"]})

    result = subprocess.run(
        [
            "uv", "run", "estest", str(temp_hdf5), str(output_file),
            "--random-variants", ".1", "--name", "stale",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "Warning: Meta-analysis group 'stale' has no processed traits" in result.stderr

    output_txt = Path(str(output_file) + ".txt")
    df = pl.read_csv(output_txt, separator='\t')
    assert "stale_Z" in df.columns
    assert all(np.isnan(value) for value in df["stale_Z"].to_list())


def test_score_test_selected_trait_outputs_nan_for_filtered_groups(tmp_path):
    """Test unprocessed groups are retained as NaN columns under --name."""
    test_data = Path(__file__).parent.parent / "data" / "test" / "test.scores.h5"
    output_file = tmp_path / "results"

    result = subprocess.run(
        [
            "uv", "run", "estest", str(test_data), str(output_file),
            "--random-variants", ".1", "--name", "bmi",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
    assert "Warning: Meta-analysis group 'cancer' has no processed traits" in result.stderr

    output_txt = Path(str(output_file) + ".txt")
    df = pl.read_csv(output_txt, separator='\t')
    assert "body_Z" in df.columns
    assert "cancer_Z" in df.columns
    assert not np.isnan(df["body_Z"].to_list()[0])
    assert all(np.isnan(value) for value in df["cancer_Z"].to_list())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
