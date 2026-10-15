"""Aider adapter.

Notes from smoke testing that shaped this adapter, kept here because they
are easy to regress:

* ``--no-git`` is mandatory. Aider commits by default, which both
  destroys the baseline ``git diff`` is computed against and lets the
  agent "undo" its own edits by reverting a commit.
* There is no ``--cwd`` flag in aider 0.86.x. The working directory is set
  via ``subprocess(cwd=...)`` in :meth:`Runner.run`.
* ``--map-tokens 0`` disables the repo map. There is no boolean
  ``--no-map-tokens`` flag; passing one makes aider exit with
  "unrecognized arguments" and, before the invocation guard was added,
  that surfaced as a plausible-looking 0.00 score.
* ``--no-stream`` removes ANSI escape codes, which otherwise corrupt the
  trajectory parser and the stored transcript.
* Local Ollama models report no token counts, so usage is parsed when
  present and left at zero otherwise rather than guessed.
* Weak models (qwen2.5-coder:3b) sometimes create a *new* file with the
  intended name instead of editing the named one. That shows up in
  ``files_changed`` as a pure addition and is a real failure mode, not a
  harness bug.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from ..core import Budget, TokenUsage, Trajectory, TrajectoryEvent
from .base import Runner

_EDIT_BLOCK = re.compile(r"^(?P<file>[\w./-]+\.\w+)\s*$")
_TOKENS = re.compile(r"Tokens:\s*(?P<sent>[\d,]+)\s*sent,\s*(?P<recv>[\d,]+)\s*received")
_APPLIED = re.compile(r"Applied edit to\s+(?P<file>[\w./-]+)")


class AiderRunner(Runner):
    """Run Aider headlessly against a task prompt."""

    name = "aider"

    #: Exercised end to end against a local Ollama model, so its scores
    #: are measured rather than assumed.
    verified = True

    def __init__(
        self,
        model: str = "ollama/qwen2.5-coder:7b",
        *,
        executable: str = "aider",
        repo_map: bool = False,
        extra_args: list[str] | None = None,
        timeout_grace_seconds: int = 30,
    ) -> None:
        super().__init__(timeout_grace_seconds=timeout_grace_seconds)
        self.model = model
        self.executable = executable
        self.repo_map = repo_map
        self.extra_args = extra_args or []

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        command = [
            self.executable,
            "--model",
            self.model,
            "--message",
            task_prompt,
            "--yes-always",
            "--no-git",
            "--no-stream",
            "--no-check-update",
            "--no-show-model-warnings",
            "--no-fancy-input",
            "--no-notifications",
        ]
        if not self.repo_map:
            # Repo maps cost tokens on every turn. For long-horizon tasks
            # they are usually worth enabling; they are opt-in here so the
            # cost difference between configurations is explicit.
            command.extend(["--map-tokens", "0"])
        command.extend(self.extra_args)
        return command

    def environment_overrides(self, workdir: Path) -> dict[str, str]:
        return {}

    def parse_trajectory(self, raw_stdout: str, started: dt.datetime) -> Trajectory:
        events: list[TrajectoryEvent] = []
        usage = TokenUsage()
        turn = 0
        now = started

        for line in raw_stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            tokens = _TOKENS.search(stripped)
            if tokens:
                usage = TokenUsage(
                    input_tokens=int(tokens.group("sent").replace(",", "")),
                    output_tokens=int(tokens.group("recv").replace(",", "")),
                )
                continue

            applied = _APPLIED.search(stripped)
            if applied:
                turn += 1
                events.append(
                    TrajectoryEvent(
                        turn=turn,
                        kind="edit",
                        content=applied.group("file"),
                        timestamp=now,
                    )
                )
                continue

            if _EDIT_BLOCK.match(stripped) and not stripped.startswith(" "):
                turn += 1
                events.append(
                    TrajectoryEvent(
                        turn=turn, kind="tool_call", content=stripped, timestamp=now
                    )
                )
                continue

            events.append(
                TrajectoryEvent(turn=turn, kind="message", content=stripped, timestamp=now)
            )

        return Trajectory(
            task="",
            agent=f"{self.name}:{self.model}",
            events=events,
            usage=usage,
            started_at=started,
        )


__all__ = ["AiderRunner"]