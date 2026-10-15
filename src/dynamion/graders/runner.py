"""Construct a grader from a task spec."""

from __future__ import annotations

from pathlib import Path

from ..core import GraderSpec, TaskSpec
from .base import AggregateGrader, Grader
from .rubric import RubricGraderRunner
from .tests import TestGraderRunner


def build_single_grader(
    grader_id: str,
    spec: GraderSpec,
    *,
    source_root: Path | None = None,
    environment=None,
) -> Grader:
    if spec.type == "tests":
        return TestGraderRunner(
            grader_id, spec, source_root=source_root, environment=environment
        )

    if spec.type == "rubric":
        return RubricGraderRunner(grader_id, spec)

    raise ValueError(f"unsupported grader type: {spec.type!r}")


def build_grader(
    task: TaskSpec, *, source_root: Path | None = None, environment=None
) -> Grader:
    """Build the aggregate grader for a task.

    ``source_root`` is the directory containing the task spec; hidden test
    paths in the spec are resolved relative to it. ``environment`` is the
    execution environment; when omitted, grading runs on the host.
    """
    graders = [
        build_single_grader(
            f"{task.id}#{index}", spec, source_root=source_root, environment=environment
        )
        for index, spec in enumerate(task.graders)
    ]
    return AggregateGrader(graders)


__all__ = ["build_grader", "build_single_grader"]
