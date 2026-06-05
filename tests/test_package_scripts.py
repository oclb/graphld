"""Tests for packaged console-script entry points."""

import importlib
import importlib.util
import tomllib
from pathlib import Path

import numpy as np
import polars as pl


def test_packaged_console_scripts_are_supported() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    scripts = pyproject["project"]["scripts"]

    assert scripts == {
        "graphld": "graphld.cli:main",
        "estest": "score_test.cli:main",
    }
    assert "create-geneset-annot" not in scripts

    for target in scripts.values():
        module_name, object_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, object_name))


def test_stale_heritability_testing_module_is_not_packaged() -> None:
    assert importlib.util.find_spec("graphld.heritability_testing") is None


def test_gene_hdf5_module_imports_shared_geneset_helpers() -> None:
    module = importlib.import_module("graphld.genesets_gene_hdf5")

    assert callable(module.compute_gene_variant_matrix_from_data)


def test_gene_hdf5_matrix_helper_matches_shared_geneset_matrix() -> None:
    module = importlib.import_module("graphld.genesets_gene_hdf5")
    shared = importlib.import_module("score_test.genesets")
    variant_data = pl.DataFrame({
        "CHR": [1],
        "POS": [999_900_000],
    })
    gene_table = pl.DataFrame({
        "CHR": [1, 2],
        "POS": [100, 100],
    })
    weights = np.array([1.0, 0.5])

    got = module.compute_gene_variant_matrix_from_data(
        variant_data,
        gene_table,
        weights,
    )
    expected = shared.gene_variant_matrix(variant_data, gene_table, weights)

    np.testing.assert_array_equal(got.toarray(), expected.toarray())
