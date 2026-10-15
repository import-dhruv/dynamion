"""Non-isolated execution environment.

Runs commands directly on the host. This exists so the harness is usable
on a machine without Docker, and it is honest about what it is:
``hermetic`` is False and that fact is recorded in the run metadata.

A score produced here is not comparable with a score produced inside a
container, because in one case the grader could reach the network and in
the other it could not. Mixing the two in a single leaderboard would
quietly change what the numbers mean, so reports must be split by mode.
"""

from __future__ import annotations

from pathlib import Path

from ..core import Environment as EnvironmentSpec
from .base import CommandResult, Environment, run_locally


class LocalEnvironment(Environment):
    """Execute in the current process environment, without isolation."""

    hermetic = False

    def __init__(self, spec: EnvironmentSpec) -> None:
        import sys

        self.spec = spec
        # The host interpreter is correct here and only here.
        self.python = sys.executable

    def prepare(self, workdir: Path, setup_commands: list[str]) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        for command in setup_commands:
            result = run_locally(
                ["/bin/sh", "-lc", command],
                cwd=workdir,
                timeout=1800,
                env_overrides=self.spec.env,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"setup command failed: {command}\n{result.stderr[:600]}"
                )

    def run(
        self,
        command: list[str],
        *,
        workdir: Path,
        network: bool = False,
        timeout: int = 900,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        # `network` is accepted and ignored. It is accepted rather than
        # rejected so that a caller written for Docker runs unchanged here
        # -- but the parameter has no effect, which is exactly why hermetic
        # is False and gets recorded.
        return run_locally(
            command,
            cwd=workdir,
            timeout=timeout,
            env_overrides={**self.spec.env, **(env or {})},
        )

    def describe(self) -> dict[str, str]:
        return {
            "environment": "LocalEnvironment",
            "hermetic": "false",
            "image": "host",
            "grading_network": "unrestricted",
            "workdir": self.spec.workdir,
        }


__all__ = ["LocalEnvironment"]
