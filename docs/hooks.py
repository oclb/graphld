"""MkDocs hooks for dynamic configuration."""
import tomllib
from pathlib import Path


def on_config(config, **kwargs):
    """Read version from pyproject.toml and set it in the config."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject.get("project", {}).get("version", "unknown")
        config["extra"]["version"] = version
        # Update site name to include version
        if "site_name" in config and version != "unknown":
            base_name = config["site_name"].split(" v")[0]  # Remove old version if present
            config["site_name"] = f"{base_name} v{version}"
    return config
