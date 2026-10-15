"""Test grader: the reward oracle.

Design points worth defending:

**Hidden tests are injected, never visible.** The grader copies
``hidden_paths`` into the workspace only after the agent's turn is over.
An agent that can read its own tests can satisfy them without solving the
task, and the resulting score is indistinguishable from a real solve.

**A grader error is not a task failure.** If pytest is missing, the
container dies, or collection fails, that is an infrastructure fault.
Reporting it as a failed task corrupts the leaderboard, so it raises
instead of returning 0.0. The check runs *before* any counting, because
a collection error prints ``ERROR collecting ...`` and a naive per-line
fallback counts those lines as failed tests.

**Partial credit comes from per-test counts.** Exit codes collapse a
9-of-10 fix and a 0-of-10 failure into the same bucket.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ..core import GraderResult, Patch, TestGrader
from ..env.base import run_locally
from .base import Grader


class GraderEnvironmentError(RuntimeError):
    """The grading environment itself failed, as opposed to the task."""


class TestGraderRunner(Grader):
    """Execute hidden tests and score the attempt."""

    # Stops pytest trying to collect this as a test class, since its name
    # starts with "Test" and it has a constructor.
    __test__ = False

    name = "tests"

    def __init__(
        self,
        grader_id: str,
        spec: TestGrader,
        *,
        source_root: Path | None = None,
        keep_workspace: bool = False,
        environment=None,
    ) -> None:
        self._grader_id = grader_id
        self.spec = spec
        self.source_root = source_root
        self.keep_workspace = keep_workspace
        self.environment = environment

    @property
    def grader_id(self) -> str:
        return self._grader_id

    def inject_hidden_tests(self, workdir: Path) -> None:
        """Copy hidden test files into the workspace after the agent's turn.

        Each entry in ``hidden_paths`` is either a bare source path (copied
        to the same relative path in the workspace) or an explicit
        ``source=destination`` pair. The explicit form matters: a hidden
        test usually has to land in the same directory as the visible
        tests so that relative imports and fixtures resolve identically.
        Silently placing it elsewhere yields a test run that collects
        nothing, which reads as a total failure rather than a config bug.
        """
        if not self.spec.hidden_paths:
            return
        if self.source_root is None:
            raise GraderEnvironmentError(
                f"grader {self.grader_id!r} declares hidden_paths but no source_root "
                "was provided"
            )

        for entry in self.spec.hidden_paths:
            if "=" in entry:
                src_rel, dst_rel = entry.split("=", 1)
            else:
                src_rel = dst_rel = entry

            src = self.source_root / src_rel
            if not src.exists():
                raise GraderEnvironmentError(f"hidden test missing: {src}")

            dst = workdir / dst_rel
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def _resolve_command(self, python: str | None = None) -> list[str]:
        """Expand ``${PYTHON}`` and any task-declared env placeholders.

        Task specs must not hard-code an interpreter path. ``python3`` on
        the developer's machine may be a virtualenv with unrelated plugins,
        which breaks the run in a way that looks like an agent failure.

        The interpreter that ``${PYTHON}`` maps to depends on where the
        command runs. Inside a container the host's interpreter path does
        not exist, so the environment supplies its own.
        """
        if python is None:
            if self.environment is not None:
                python = self.environment.python
            else:
                import sys

                python = sys.executable

        return [token.replace("${PYTHON}", python) for token in self.spec.command]

    def _execute(self, workdir: Path, timeout: int):
        """Run the test command, inside the container when configured.

        Hidden tests are already injected at this point, which is what
        allows the workspace to be mounted read-only: the agent's turn is
        over, and the grader cannot be edited by the thing it is grading.
        """
        if self.environment is not None:
            return self.environment.run_grading(
                self._resolve_command(), workdir=workdir, timeout=timeout
            )
        return run_locally(self._resolve_command(), cwd=workdir, timeout=timeout)

    def run(self, workdir: Path, patch: Patch) -> GraderResult:
        try:
            self.inject_hidden_tests(workdir)
        except GraderEnvironmentError:
            raise
        except OSError as exc:
            raise GraderEnvironmentError(f"failed to inject hidden tests: {exc}") from exc

        try:
            proc = self._execute(workdir, self.spec.timeout_seconds)
        except GraderEnvironmentError:
            raise
        except FileNotFoundError as exc:
            raise GraderEnvironmentError(
                f"test command not found: {self.spec.command[0]!r}"
            ) from exc

        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.exit_code

        output = f"{stdout}\n{stderr}"

        # Infrastructure faults are detected before any counting, not only
        # when counting yields nothing. A collection error prints "ERROR
        # collecting ...", and the per-line fallback below happily counts
        # those ERROR lines as failed tests -- producing a confident 0/2
        # for a run that never executed.
        fault = _detect_environment_fault(output)
        if fault is not None:
            raise GraderEnvironmentError(
                f"grader {self.grader_id!r} could not run tests: {fault}"
            )

        passed, total = _count_results(output, self.spec.result_pattern)

        if total == 0:
            # Tests ran but nothing was parseable; fall back to the exit
            # code so a fully passing run still scores 1.0.
            score = 1.0 if returncode == 0 else 0.0
            passed, total = (int(returncode == 0), 1)
        elif self.spec.partial_credit:
            score = passed / total
        else:
            score = 1.0 if passed == total else 0.0

        return GraderResult(
            grader_id=self.grader_id,
            type="tests",
            score=score,
            weight=self.spec.weight,
            passed=passed == total and total > 0,
            detail={
                "passed": passed,
                "total": total,
                "returncode": returncode,
                "stdout": _tail(stdout),
            },
        )


#: Signatures that indicate the test run never actually happened, as
#: opposed to tests running and failing.
_ENVIRONMENT_FAULTS = (
    "ModuleNotFoundError",
    "ImportError",
    "PluginValidationError",
    "No module named",
    "command not found",
    "no tests ran",
    "ERROR collecting",
    "Interrupted",
)


def _detect_environment_fault(output: str) -> str | None:
    """Return a description if ``output`` indicates an infra failure."""
    for marker in _ENVIRONMENT_FAULTS:
        if marker in output:
            for line in output.splitlines():
                if marker in line:
                    return line.strip()
    return None


_PYTEST_SUMMARY = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_COUNT_FAILED = re.compile(r"(\d+)\s+failed", re.IGNORECASE)


def _last_count(pattern: re.Pattern[str], output: str) -> int | None:
    """Return the final match of ``pattern`` in ``output``.

    The last match, not the first. pytest prints its summary last, but
    failure output can contain an earlier accidental match: an assertion
    like ``assert 2 == 1`` immediately followed by a ``FAILED`` line
    satisfies ``(\\d+)\\s+failed`` and fabricates a count. Taking the first
    match turned a real 4-of-17 run into a reported 4-of-5.
    """
    matches = pattern.findall(output)
    if not matches:
        return None
    return int(matches[-1])


def _count_results(output: str, pattern: str | None) -> tuple[int, int]:
    """Return ``(passed, total)`` parsed from test output.

    Prefers the pytest summary line, which is lowercase under ``-q``
    (``3 passed, 2 failed``) and so must be matched case-insensitively --
    a case-sensitive version read a fully-failing run as ``0/0`` and then
    reported 0.0 for a reason that had nothing to do with the agent.
    """
    if pattern:
        found = re.search(pattern, output)
        if found:
            passed = int(found.group(1))
            failed = _last_count(_COUNT_FAILED, output) or 0
            return passed, passed + failed

    summary_passed = _last_count(_PYTEST_SUMMARY, output)
    if summary_passed is not None:
        failed = _last_count(_COUNT_FAILED, output) or 0
        return summary_passed, summary_passed + failed

    # A run where everything failed prints "4 failed" and never mentions
    # "passed" at all, so a total must still be recovered from it.
    summary_failed = _last_count(_COUNT_FAILED, output)
    if summary_failed is not None:
        return 0, summary_failed

    passed = len(re.findall(r"\bPASSED\b|PASSED\s+\S+", output))
    failed = len(re.findall(r"\bFAILED\b|\bERROR\b", output))
    if passed or failed:
        return passed, passed + failed
    return 0, 0


def _tail(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


__all__ = ["TestGraderRunner", "GraderEnvironmentError"]