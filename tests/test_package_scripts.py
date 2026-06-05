"""Tests for packaged console-script entry points."""

import importlib
import importlib.util
import tomllib
from pathlib import Path


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
