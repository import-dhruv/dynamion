"""Task spec loading and validation.

Task specs live as YAML or JSON. Loading validates against the Pydantic
models in :mod:`dynamion.core`; :func:`schema_json` emits the equivalent
JSON Schema so editors and CI can validate without importing the package.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .core import TaskSpec

VALID_SUFFIXES = {".yaml", ".yml", ".json"}

#: Task specs are named explicitly rather than discovered by extension.
#: A task directory legitimately contains other YAML and JSON -- hidden
#: grader material, golden behaviour fixtures, agent configs -- and
#: treating those as task specs fails the whole load. Discovery by
#: ``*.yaml`` glob is exactly the kind of quiet coupling that makes a
#: task directory un-extensible.
SPEC_FILENAMES = {"task.yaml", "task.yml", "task.json"}


def is_spec_path(path: Path) -> bool:
    """True when ``path`` names a task spec rather than task material."""
    if path.name in SPEC_FILENAMES:
        return True
    return path.name.endswith(".task.yaml") or path.name.endswith(".task.yml")


def load_task(path: Path) -> TaskSpec:
    """Load and validate a single task spec from disk."""
    path = Path(path)
    if path.suffix not in VALID_SUFFIXES:
        raise ValueError(f"task spec must be .yaml/.yml/.json, got {path.suffix!r}")

    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: task spec must be a mapping at the top level")

    try:
        return TaskSpec.model_validate(data)
    except Exception as exc:
        raise ValueError(f"{path}: invalid task spec\n{exc}") from exc


def load_tasks(root: Path) -> list[TaskSpec]:
    """Load every task under ``root``, recursively.

    Duplicate ``id@version`` pairs are rejected: two tasks claiming the
    same identity would make leaderboard rows ambiguous.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"tasks directory not found: {root}")

    paths = sorted(p for p in root.rglob("*") if p.is_file() and is_spec_path(p))
    if not paths:
        raise ValueError(f"no task specs found under {root}")

    tasks = [load_task(p) for p in paths]

    seen: dict[str, Path] = {}
    for path, task in zip(paths, tasks, strict=True):
        if task.slug in seen:
            raise ValueError(
                f"duplicate task id {task.slug!r}: {seen[task.slug]} and {path}"
            )
        seen[task.slug] = path

    return tasks


def schema_json() -> str:
    """Return the JSON Schema for a task spec."""
    return json.dumps(TaskSpec.model_json_schema(), indent=2)


__all__ = ["VALID_SUFFIXES", "is_spec_path", "load_task", "load_tasks", "schema_json"]
