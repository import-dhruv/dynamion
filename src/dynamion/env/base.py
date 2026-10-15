"""Execution environment interface.

The harness runs a task in an :class:`Environment`. Two exist:

* :class:`~dynamion.env.docker.DockerEnvironment` — hermetic, reproducible
* :class:`~dynamion.env.local.LocalEnvironment` — runs in place, no isolation

The local environment is not a fallback for convenience; it is a
deliberately weaker mode that reports itself as such. A score produced
without isolation is a different measurement from a score produced with
it, and mixing the two in one leaderboard would be dishonest.
"""

from __future__ import annotations

import abc
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a command executed inside an environment."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


class EnvironmentError_(RuntimeError):
    """Raised when the execution environment itself fails.

    Named with a trailing underscore to avoid shadowing the builtin
    ``EnvironmentError`` (= ``OSError``) at call sites that catch it.
    """


class Environment(abc.ABC):
    """Where and how a task executes."""

    #: Whether commands are network-isolated. Recorded in run metadata so
    #: a score can be interpreted correctly.
    hermetic: bool = False

    #: Interpreter that ``${PYTHON}`` resolves to when a task is run here.
    #: Inside a container this must be a path that exists in the image.
    python: str = "python3"

    @abc.abstractmethod
    def prepare(self, workdir: Path, setup_commands: list[str]) -> None:
        """Materialise the workspace and run any setup commands."""

    @abc.abstractmethod
    def run(
        self,
        command: list[str],
        *,
        workdir: Path,
        network: bool = False,
        timeout: int = 900,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute ``command`` inside the environment."""

    def describe(self) -> dict[str, str]:
        """Metadata recorded alongside a score for reproducibility."""
        return {"environment": type(self).__name__, "hermetic": str(self.hermetic)}

    def cleanup(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release resources. Safe to call more than once."""


def run_locally(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command in the current process environment."""
    import os

    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return CommandResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            exit_code=-1,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            duration_seconds=time.monotonic() - start,
        )
    except FileNotFoundError as exc:
        raise EnvironmentError_(f"command not found: {command[0]!r}") from exc


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = ["Environment", "EnvironmentError_", "CommandResult", "run_locally"]
