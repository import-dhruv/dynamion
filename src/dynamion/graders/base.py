"""Grader interface.

A grader consumes a workspace and a patch and returns a score in [0, 1].
Graders never receive the agent, the runner, or the prompt that was given
to the model: they see only the finished artefact. That restriction is
what makes grading reproducible offline.
"""

from __future__ import annotations

import abc
from pathlib import Path

from ..core import GraderResult, Patch


class Grader(abc.ABC):
    """Base class for all graders."""

    name: str = "base"

    @abc.abstractmethod
    def run(self, workdir: Path, patch: Patch) -> GraderResult:
        """Score the state of ``workdir`` and return a normalised result."""

    def inject_hidden_tests(self, workdir: Path) -> None:
        """Hook called after the agent finishes, before grading.

        No-op by default. Test graders override this to withhold test
        files from the agent during its turn.
        """

    @property
    def grader_id(self) -> str:
        return self.name


class AggregateGrader(Grader):
    """Weighted combination of several graders.

    Weighted mean rather than a strict AND: an attempt that passes all
    hidden tests but satisfies no rubric criteria should not score the
    same as one that does the opposite. Both are partial success, and the
    distinction between "implemented but undocumented" and "documented but
    unimplemented" is exactly the signal a benchmark should report.
    """

    name = "aggregate"

    def __init__(self, graders: list[Grader]) -> None:
        if not graders:
            raise ValueError("AggregateGrader requires at least one grader")
        self.graders = graders

    def run(self, workdir: Path, patch: Patch) -> GraderResult:
        results = [g.run(workdir, patch) for g in self.graders]

        total_weight = sum(r.weight for r in results)
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(r.score * r.weight for r in results) / total_weight

        return GraderResult(
            grader_id=self.name,
            type="tests",
            score=score,
            weight=1.0,
            passed=score >= 1.0,
            detail={
                "components": [
                    {
                        "grader_id": r.grader_id,
                        "score": r.score,
                        "weight": r.weight,
                        "passed": r.passed,
                        "error": r.error,
                        **r.detail,
                    }
                    for r in results
                ]
            },
        )

    def inject_hidden_tests(self, workdir: Path) -> None:
        for g in self.graders:
            g.inject_hidden_tests(workdir)

    @property
    def grader_id(self) -> str:
        return self.name


__all__ = ["Grader", "AggregateGrader"]
