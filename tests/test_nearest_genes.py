import importlib.util
import pathlib
from typing import Callable

import numpy as np

# TODO
def _load_get_nearest_genes() -> Callable:
    # Load scoretest/nearest_genes.py as a module and return get_nearest_genes
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scoretest" / "nearest_genes.py"
    spec = importlib.util.spec_from_file_location("nearest_genes", mod_path)
    assert spec and spec.loader, f"Could not load module from {mod_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "get_nearest_genes")


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
