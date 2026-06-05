import subprocess
import sys

import numpy as np
import polars as pl
from score_test.genesets import (
    _get_gene_variant_matrix,
    _get_nearest_genes,
    convert_gene_to_variant_annotations,
    gene_variant_matrix,
)

def _load_get_nearest_genes():
    return _get_nearest_genes


def _brute_force_nearest(var_pos: np.ndarray, gene_pos: np.ndarray, k: int) -> np.ndarray:
    # For each variant, compute distances to all genes and pick k with smallest distance.
    # Tie-break by smaller gene index for determinism.
    nvar, ngene = len(var_pos), len(gene_pos)
    assert k <= ngene, "k cannot exceed number of genes"
    out = np.empty((nvar, k), dtype=np.int32)
    for i, v in enumerate(var_pos):
        d = np.abs(gene_pos - v)
        # argsort by (distance, gene_index)
        order = np.lexsort((np.arange(ngene), d))
        out[i, :] = order[:k]
    return out


def test_simple_k1():
    get_nearest_genes = _load_get_nearest_genes()

    var_pos = np.array([10, 20, 30])
    gene_pos = np.array([5, 25, 40])
    k = 1

    expected = _brute_force_nearest(var_pos, gene_pos, k)
    got = get_nearest_genes(var_pos, gene_pos, k)

    assert isinstance(got, np.ndarray)
    assert got.dtype == np.int32
    assert got.shape == (len(var_pos), k)
    np.testing.assert_array_equal(got, expected)


def test_simple_k2():
    get_nearest_genes = _load_get_nearest_genes()

    var_pos = np.array([10, 20, 30, 100])
    gene_pos = np.array([5, 25, 40, 90])
    k = 2

    expected = _brute_force_nearest(var_pos, gene_pos, k)
    got = get_nearest_genes(var_pos, gene_pos, k)

    assert got.shape == (len(var_pos), k)
    np.testing.assert_array_equal(got, expected)


def test_random_matches_bruteforce(seed: int = 123):
    get_nearest_genes = _load_get_nearest_genes()

    rng = np.random.default_rng(seed)
    # Make unique, unsorted positions to mimic realistic inputs
    gene_pos = np.unique(rng.integers(0, 10_000, size=50))
    var_pos = np.unique(rng.integers(0, 10_000, size=60))
    k = 3

    expected = _brute_force_nearest(var_pos, gene_pos, k)
    got = get_nearest_genes(var_pos, gene_pos, k)

    assert got.shape == (len(var_pos), k)
    np.testing.assert_array_equal(got, expected)


def test_genome_scale_distances_do_not_overflow():
    get_nearest_genes = _load_get_nearest_genes()

    var_pos = np.array([22_000_000_000], dtype=np.int64)
    gene_pos = np.array([1_000_000_000, 22_100_000_000], dtype=np.int64)

    got = get_nearest_genes(var_pos, gene_pos, 1)

    np.testing.assert_array_equal(got, np.array([[1]], dtype=np.int32))


def test_gene_variant_matrix_prefers_same_chromosome_before_fallback():
    variant_table = pl.DataFrame({
        "CHR": [1],
        "POS": [999_900_000],
    })
    gene_table = pl.DataFrame({
        "CHR": [1, 2],
        "POS": [100, 100],
    })

    matrix = gene_variant_matrix(variant_table, gene_table, np.array([1.0, 0.5]))
    row = matrix[0].toarray().ravel()

    np.testing.assert_array_equal(row, np.array([1.0, 0.5]))


def test_score_test_low_level_matrix_preserves_fractional_weights():
    var_pos = np.array([100])
    gene_pos = np.array([100, 200])

    matrix = _get_gene_variant_matrix(var_pos, gene_pos, np.array([0.4, 0.2]))
    row = matrix[0].toarray().ravel()

    np.testing.assert_array_equal(row, np.array([0.4, 0.2]))


def test_score_test_gene_annotation_conversion_uses_chromosome_aware_matrix():
    from score_test.score_test import GeneAnnot, VariantAnnot

    gene_annot = GeneAnnot({"chr2_pathway": ["GENE2"]})
    variant_table = pl.DataFrame({
        "CHR": [1],
        "POS": [999_900_000],
        "RSID": ["rs_boundary"],
    })
    gene_table = pl.DataFrame({
        "CHR": [1, 2],
        "POS": [100, 100],
        "gene_name": ["GENE1", "GENE2"],
        "gene_id": ["ENSG1", "ENSG2"],
    })

    annot = convert_gene_to_variant_annotations(
        gene_annot, variant_table, gene_table, np.array([1.0, 0.5])
    )

    assert isinstance(annot, VariantAnnot)
    assert annot.df["chr2_pathway"][0] == 0.5


def test_score_test_genesets_imports_standalone_without_graphld():
    script = (
        "import genesets, sys; "
        "assert 'graphld' not in sys.modules; "
        "print(genesets._get_nearest_genes([10], [5], 1).tolist())"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd="src/score_test",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[[0]]"
