"""Tests for graphld.genesets module."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from graphld.genesets import (
    _compute_positions,
    _get_gene_variant_matrix,
    _get_nearest_genes,
    _is_gene_id,
    convert_gene_sets_to_variant_annotations,
    load_gene_sets_from_gmt,
    load_gene_table,
)


class TestIsGeneId:
    """Tests for _is_gene_id function."""

    def test_ensembl_id(self):
        assert _is_gene_id("ENSG00000139618") is True
        assert _is_gene_id("ENSG00000141510.12") is True

    def test_gene_symbol(self):
        assert _is_gene_id("BRCA2") is False
        assert _is_gene_id("TP53") is False
        assert _is_gene_id("MYC") is False


class TestLoadGeneSetsFromGmt:
    """Tests for load_gene_sets_from_gmt function."""

    def test_load_single_gmt(self, tmp_path):
        """Test loading a single GMT file."""
        gmt_content = "pathway1\tdescription1\tGENE1\tGENE2\tGENE3\n"
        gmt_content += "pathway2\tdescription2\tGENE4\tGENE5\n"

        gmt_file = tmp_path / "test.gmt"
        gmt_file.write_text(gmt_content)

        gene_sets = load_gene_sets_from_gmt(str(tmp_path))

        assert len(gene_sets) == 2
        assert gene_sets["pathway1"] == ["GENE1", "GENE2", "GENE3"]
        assert gene_sets["pathway2"] == ["GENE4", "GENE5"]

    def test_load_multiple_gmt_files(self, tmp_path):
        """Test loading multiple GMT files from directory."""
        gmt1 = tmp_path / "set1.gmt"
        gmt1.write_text("pathway_a\tdesc\tGENE1\tGENE2\n")

        gmt2 = tmp_path / "set2.gmt"
        gmt2.write_text("pathway_b\tdesc\tGENE3\tGENE4\n")

        gene_sets = load_gene_sets_from_gmt(str(tmp_path))

        assert len(gene_sets) == 2
        assert "pathway_a" in gene_sets
        assert "pathway_b" in gene_sets

    def test_no_gmt_files_raises(self, tmp_path):
        """Test that FileNotFoundError is raised when no GMT files exist."""
        with pytest.raises(FileNotFoundError, match="No .gmt files found"):
            load_gene_sets_from_gmt(str(tmp_path))

    def test_empty_lines_skipped(self, tmp_path):
        """Test that empty lines and short lines are handled."""
        gmt_content = "pathway1\tdesc\tGENE1\tGENE2\n"
        gmt_content += "\n"  # Empty line
        gmt_content += "short\tdesc\n"  # Too short (no genes)
        gmt_content += "pathway2\tdesc\tGENE3\n"

        gmt_file = tmp_path / "test.gmt"
        gmt_file.write_text(gmt_content)

        gene_sets = load_gene_sets_from_gmt(str(tmp_path))

        # Only pathways with genes should be loaded
        assert len(gene_sets) == 2
        assert "pathway1" in gene_sets
        assert "pathway2" in gene_sets


class TestGetNearestGenes:
    """Tests for _get_nearest_genes function."""

    def test_simple_case(self):
        """Test with simple sorted positions."""
        var_pos = np.array([100, 200, 300, 400])
        gene_pos = np.array([150, 350])
        num_nearest = 2

        nearest = _get_nearest_genes(var_pos, gene_pos, num_nearest)

        assert nearest.shape == (4, 2)
        # Each variant should have indices of both genes (only 2 genes)
        assert set(nearest[0].tolist()) == {0, 1}

    def test_unsorted_raises(self):
        """Test that unsorted positions raise ValueError."""
        var_pos = np.array([200, 100, 300])
        gene_pos = np.array([150, 350])

        with pytest.raises(ValueError, match="Variant positions must be sorted"):
            _get_nearest_genes(var_pos, gene_pos, 1)

    def test_gene_unsorted_raises(self):
        """Test that unsorted gene positions raise ValueError."""
        var_pos = np.array([100, 200, 300])
        gene_pos = np.array([350, 150])

        with pytest.raises(ValueError, match="Gene positions must be sorted"):
            _get_nearest_genes(var_pos, gene_pos, 1)


class TestGetGeneVariantMatrix:
    """Tests for _get_gene_variant_matrix function."""

    def test_output_shape(self):
        """Test that output matrix has correct shape."""
        var_pos = np.array([100, 200, 300, 400, 500])
        gene_pos = np.array([150, 350, 450])
        weights = np.array([1.0, 0.5])

        matrix = _get_gene_variant_matrix(var_pos, gene_pos, weights)

        assert matrix.shape == (5, 3)

    def test_weights_applied(self):
        """Test that weights are correctly applied."""
        var_pos = np.array([100, 200])
        gene_pos = np.array([100, 200, 300])
        weights = np.array([1.0, 0.5])

        matrix = _get_gene_variant_matrix(var_pos, gene_pos, weights)

        # First variant at 100 should have gene at 100 as nearest (weight 1.0)
        # and gene at 200 as second nearest (weight 0.5)
        row0 = matrix[0].toarray().ravel()
        assert row0[0] == 1.0  # Gene at 100
        assert row0[1] == 0.5  # Gene at 200


class TestComputePositions:
    """Tests for _compute_positions function."""

    def test_single_chromosome(self):
        """Test positions on a single chromosome."""
        table = pl.DataFrame({
            "CHR": [1, 1, 1],
            "POS": [100, 200, 300],
        })

        positions = _compute_positions(table)

        assert len(positions) == 3
        assert positions[0] == 1e9 + 100
        assert positions[1] == 1e9 + 200
        assert positions[2] == 1e9 + 300

    def test_multiple_chromosomes(self):
        """Test positions across chromosomes."""
        table = pl.DataFrame({
            "CHR": [1, 2, 22],
            "POS": [100, 200, 300],
        })

        positions = _compute_positions(table)

        assert positions[0] == 1e9 + 100
        assert positions[1] == 2e9 + 200
        assert positions[2] == 22e9 + 300


class TestConvertGeneSetsToVariantAnnotations:
    """Tests for convert_gene_sets_to_variant_annotations function."""

    def test_basic_conversion(self):
        """Test basic gene set to variant annotation conversion."""
        gene_sets = {
            "pathway1": ["GENE1", "GENE2"],
            "pathway2": ["GENE3"],
        }

        variant_table = pl.DataFrame({
            "CHR": [1, 1, 1, 1],
            "POS": [100, 200, 300, 400],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
        })

        gene_table = pl.DataFrame({
            "CHR": [1, 1, 1],
            "POS": [150, 250, 350],
            "gene_name": ["GENE1", "GENE2", "GENE3"],
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
        })

        weights = np.array([1.0])

        result = convert_gene_sets_to_variant_annotations(
            gene_sets, variant_table, gene_table, weights
        )

        # Check output structure
        assert "CHR" in result.columns
        assert "BP" in result.columns
        assert "SNP" in result.columns
        assert "CM" in result.columns
        assert "pathway1" in result.columns
        assert "pathway2" in result.columns

        # Check that we have correct number of rows
        assert len(result) == 4

    def test_ensembl_id_gene_sets(self):
        """Test with Ensembl ID gene sets."""
        gene_sets = {
            "pathway1": ["ENSG00000001", "ENSG00000002"],
        }

        variant_table = pl.DataFrame({
            "CHR": [1, 1],
            "POS": [100, 200],
            "SNP": ["rs1", "rs2"],
        })

        gene_table = pl.DataFrame({
            "CHR": [1, 1],
            "POS": [150, 250],
            "gene_name": ["GENE1", "GENE2"],
            "gene_id": ["ENSG00000001", "ENSG00000002"],
        })

        weights = np.array([1.0])

        result = convert_gene_sets_to_variant_annotations(
            gene_sets, variant_table, gene_table, weights
        )

        assert "pathway1" in result.columns


class TestLoadGeneTable:
    """Tests for load_gene_table function."""

    def test_load_gene_table(self, tmp_path):
        """Test loading a gene table TSV file."""
        gene_table_content = "gene_id\tgene_id_version\tgene_name\tstart\tend\tCHR\n"
        gene_table_content += "ENSG00000001\tENSG00000001.1\tGENE1\t1000\t2000\t1\n"
        gene_table_content += "ENSG00000002\tENSG00000002.1\tGENE2\t3000\t4000\t1\n"
        gene_table_content += "ENSG00000003\tENSG00000003.1\tGENE3\t5000\t6000\t2\n"

        gene_file = tmp_path / "genes.tsv"
        gene_file.write_text(gene_table_content)

        result = load_gene_table(str(gene_file))

        assert len(result) == 3
        assert "gene_id" in result.columns
        assert "gene_name" in result.columns
        assert "POS" in result.columns  # Midpoint column

    def test_filter_by_chromosome(self, tmp_path):
        """Test filtering gene table by chromosome."""
        gene_table_content = "gene_id\tgene_id_version\tgene_name\tstart\tend\tCHR\n"
        gene_table_content += "ENSG00000001\tENSG00000001.1\tGENE1\t1000\t2000\t1\n"
        gene_table_content += "ENSG00000002\tENSG00000002.1\tGENE2\t3000\t4000\t2\n"
        gene_table_content += "ENSG00000003\tENSG00000003.1\tGENE3\t5000\t6000\t22\n"

        gene_file = tmp_path / "genes.tsv"
        gene_file.write_text(gene_table_content)

        result = load_gene_table(str(gene_file), chromosomes=[1, 2])

        assert len(result) == 2
        chrs = result["CHR"].unique().to_list()
        assert "22" not in chrs
