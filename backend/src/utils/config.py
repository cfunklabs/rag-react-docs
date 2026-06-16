import tomllib
from pathlib import Path


def load_pyproject(path: str = "pyproject.toml") -> dict:
    with open(Path(path), "rb") as f:
        return tomllib.load(f)


def load_colors() -> dict:
    return load_pyproject()["tool"]["console"]["colors"]
