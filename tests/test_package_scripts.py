"""Tests for packaged console-script entry points."""

import importlib
import importlib.util
import shutil
import subprocess
import sys
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


def test_stale_gene_hdf5_module_is_not_packaged() -> None:
    assert importlib.util.find_spec("graphld.genesets_gene_hdf5") is None


def test_console_scripts_help_from_non_repo_cwd(tmp_path: Path) -> None:
    """Packaged entry points should not depend on the caller's cwd."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]

    for script in scripts:
        script_path = Path(sys.executable).parent / script
        if not script_path.exists():
            path_from_env = shutil.which(script)
            assert path_from_env is not None, (
                f"Could not find installed {script!r} script"
            )
            script_path = Path(path_from_env)
        assert script_path.exists(), f"Could not find installed {script!r} script"

        result = subprocess.run(
            [str(script_path), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
