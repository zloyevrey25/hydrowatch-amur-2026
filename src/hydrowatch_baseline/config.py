from __future__ import annotations

from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def resolve_project_path(value: str, project_root: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path

