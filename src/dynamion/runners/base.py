"""Agent runner interface.

The contract every agent adapter implements:

    ``run(task_prompt, workdir, budget) -> Trajectory``

plus an optional :meth:`Runner.environment_overrides` hook for agents that
need tooling injected into the sandbox.

The important design constraint is that a runner never sees the grader's
hidden paths. An adapter that can read the hidden tests can be prompted
to optimise against them, and every score it produces is then worthless.
Keeping that boundary here rather than in each adapter means a new adapter
cannot accidentally cross it.
"""

from __future__ import annotations

import abc
import datetime as dt
import subprocess
from pathlib import Path

from ..core import Budget, Patch, TokenUsage, Trajectory, TrajectoryEvent


class BudgetExhausted(Exception):
    """Raised when an attempt hits a configured limit."""


class RunnerError(Exception):
    """Raised when an agent cannot be run at all (as opposed to failing)."""


#: Substrings that identify an agent CLI rejecting its own arguments or
#: missing from PATH. These are invocation faults, not model behaviour: an
#: agent that never started has not solved anything, and grading it as a
#: failed attempt silently corrupts results.
_INVOCATION_FAULTS = (
    "unrecognized arguments",
    "no such file or directory",
    "command not found",
    "invalid choice",
    "error: option",
    "usage:",
)


class Runner(abc.ABC):
    """Base class for agent adapters."""

    #: Stable identifier used in reports and the leaderboard.
    name: str = "base"

    #: Model identifier, e.g. ``ollama/qwen2.5-coder:7b``.
    model: str = ""

    #: Whether this adapter has actually been executed against its real
    #: CLI. Unverified adapters ship so the harness is extensible, but
    #: their numbers are not measurements and the leaderboard says so.
    verified: bool = False

    def __init__(self, *, timeout_grace_seconds: int = 30) -> None:
        self.timeout_grace_seconds = timeout_grace_seconds

    @abc.abstractmethod
    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        """Return the argv used to invoke the agent.

        Kept separate from :meth:`run` so that the exact invocation can be
        logged and reproduced without executing it.
        """

    @abc.abstractmethod
    def parse_trajectory(self, raw_stdout: str, started: dt.datetime) -> Trajectory:
        """Turn the agent's raw output into a structured trajectory."""

    def environment_overrides(self, workdir: Path) -> dict[str, str]:
        """Environment variables the sandbox needs for this agent."""
        return {}

    def run(
        self,
        task_prompt: str,
        workdir: Path,
        budget: Budget,
        *,
        task_id: str,
    ) -> Trajectory:
        """Execute the agent against a prompt and return its trajectory.

        ``run`` is intentionally not abstract: every adapter needs the same
        subprocess supervision, so implementing it once here avoids five
        near-identical copies with five different timeout bugs.
        """
        started = dt.datetime.now(dt.UTC)
        command = self.build_command(task_prompt, workdir, budget)

        env_overrides = self.environment_overrides(workdir)
        try:
            proc = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=budget.max_seconds + self.timeout_grace_seconds,
                env=_merge_env(env_overrides),
                check=False,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            code = -1

        traj = self.parse_trajectory(stdout, started)

        if stderr:
            traj.events.append(
                TrajectoryEvent(
                    turn=traj.n_edits,
                    kind="error",
                    content=stderr[-4000:],
                    timestamp=dt.datetime.now(dt.UTC),
                )
            )

        if code == -1:
            traj.exit_reason = "timeout"
        elif traj.exit_reason is None:
            traj.exit_reason = "completed" if code == 0 else "error"

        traj.ended_at = dt.datetime.now(dt.UTC)

        self._check_invocation(command, code, stderr, traj.duration_seconds or 0.0)
        return traj

    @staticmethod
    def _check_invocation(
        command: list[str], code: int, stderr: str, elapsed: float
    ) -> None:
        """Raise if the agent CLI failed to start.

        An agent that exits in under two seconds having produced no work
        has not attempted the task. Reporting it as a 0.0 turns a
        configuration mistake into a benchmark result.
        """
        if code == 0:
            return

        lowered = stderr.lower()
        for marker in _INVOCATION_FAULTS:
            if marker in lowered:
                raise RunnerError(
                    f"agent invocation failed ({marker!r}); command was: "
                    f"{' '.join(command)}\n{stderr[:500]}"
                )

        if elapsed < 2.0 and not stderr.strip():
            raise RunnerError(
                f"agent exited immediately with code {code} and produced no output; "
                f"command was: {' '.join(command)}"
            )

    def compute_patch(self, workdir: Path, trajectory: Trajectory) -> Patch:
        """Capture the repository diff produced during the run.

        Untracked files are included via ``git add --intent-to-add`` first.
        Without that step ``git diff HEAD`` is blind to anything the agent
        created, and this failure is silent and severe: a model whose main
        output is a *new* file produces an empty patch, an empty
        ``files_changed`` list, and — because the rubric judge reads the
        diff — a rubric score of zero for work that may be entirely
        correct. Observed on a real run.
        """
        run_locally_quiet(["git", "add", "--intent-to-add", "--all"], cwd=workdir)

        try:
            proc = subprocess.run(
                ["git", "diff", "--no-color", "HEAD"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            diff = proc.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            diff = ""

        files: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                files.append(line[6:].strip())
        return Patch(unified_diff=diff, files_changed=files, diff_bytes=len(diff.encode()))


def run_locally_quiet(command: list[str], cwd: Path) -> None:
    """Run a command, discarding failure. Used for index bookkeeping."""
    import contextlib

    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError, OSError):
        subprocess.run(
            command, cwd=cwd, capture_output=True, timeout=60, check=False
        )


def _merge_env(overrides: dict[str, str]) -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.update(overrides)
    return env


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = ["BudgetExhausted", "Runner", "RunnerError", "TokenUsage"]
