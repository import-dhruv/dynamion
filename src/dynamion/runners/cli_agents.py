"""Adapters for agent CLIs that are not Aider.

Each was written against the documented CLI surface rather than a live
run, because none of the binaries are installed here. That is a real
limitation and every one of these adapters carries ``verified = False`` so
a leaderboard never presents an unverified adapter's score next to a
verified one without saying so.

What every adapter shares, learned from Aider:

* the working directory is set via ``subprocess(cwd=...)``; most of these
  CLIs have no ``--cwd`` flag
* the agent must run non-interactively, or it blocks on an approval prompt
  and silently burns the whole budget
* output is recorded verbatim so a trajectory can be re-parsed later if
  the parser is wrong
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from ..core import Budget, Trajectory, TrajectoryEvent
from .base import Runner


class _CliRunner(Runner):
    """Shared scaffolding for headless single-shot CLI agents."""

    #: Default binary name. Subclasses override it; sharing one default
    #: would make every adapter try to exec the same executable.
    executable: str = "agent"

    def __init__(
        self,
        model: str,
        *,
        executable: str | None = None,
        extra_args: list[str] | None = None,
        timeout_grace_seconds: int = 30,
    ) -> None:
        super().__init__(timeout_grace_seconds=timeout_grace_seconds)
        self.model = model
        if executable:
            self.executable = executable
        self.extra_args = extra_args or []

    def classify_line(self, line: str) -> tuple[str, str] | None:
        """Return ``(kind, content)`` for a line, or None to ignore it."""
        stripped = line.strip()
        if not stripped:
            return None
        return "message", stripped

    def parse_trajectory(self, raw_stdout: str, started: dt.datetime) -> Trajectory:
        events: list[TrajectoryEvent] = []
        turn = 0

        for line in raw_stdout.splitlines():
            classified = self.classify_line(line)
            if classified is None:
                continue
            kind, content = classified
            if kind == "edit":
                turn += 1
            events.append(
                TrajectoryEvent(turn=turn, kind=kind, content=content, timestamp=started)
            )

        return Trajectory(
            task="",
            agent=f"{self.name}:{self.model}",
            events=events,
            started_at=started,
        )


_FILE_MENTION = re.compile(r"[\w./-]+\.(py|js|ts|go|rs|java|rb|md|json|yaml|yml|toml)")


class CodexRunner(_CliRunner):
    """OpenAI Codex CLI, headless."""

    name = "codex"
    executable = "codex"

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        return [
            self.executable,
            "exec",
            "--skip-git-repo-check",
            # Non-interactive: an approval prompt would consume the budget
            # doing nothing.
            "--dangerously-bypass-approvals-and-sandbox",
            self.model,
            task_prompt,
            *self.extra_args,
        ]

    def classify_line(self, line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stripped.lower().startswith(("applying", "edited", "updated", "created")):
            return "edit", stripped
        return "message", stripped


class GeminiCliRunner(_CliRunner):
    """Google Gemini CLI, headless.

    Relevant because the target team builds Gemini: an adapter that
    evaluates the model they develop is the honest way to report on it.
    """

    name = "gemini-cli"
    executable = "gemini"

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        return [
            self.executable,
            "--model", self.model,
            "--prompt", task_prompt,
            "--yolo",  # non-interactive; without it the CLI blocks on approval
            *self.extra_args,
        ]

    def classify_line(self, line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stripped.lower().startswith(("writing", "edited", "modified", "created")):
            return "edit", stripped
        return "message", stripped


class ClaudeCodeRunner(_CliRunner):
    """Anthropic Claude Code, headless.

    ``--permission-mode bypassPermissions`` is required: the default mode
    prompts for approval on every edit, which cannot work under a budget.
    """

    name = "claude-code"
    executable = "claude"

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        return [
            self.executable,
            "--print",
            "--permission-mode", "bypassPermissions",
            "--model", self.model,
            task_prompt,
            *self.extra_args,
        ]

    def classify_line(self, line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped:
            return None
        if re.match(r"^(Edit|Update|Write|Create) file", stripped):
            return "edit", stripped
        return "message", stripped


class OpenHandsRunner(_CliRunner):
    """AllHands OpenHands, headless."""

    name = "openhands"
    executable = "python3"

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        return [
            self.executable,
            "-m", "openhands",
            "-i", task_prompt,
            *self.extra_args,
        ]

    def classify_line(self, line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped:
            return None
        if stripped.startswith(("EVENT", "Action:", "Observation:")):
            return "tool_call", stripped
        return "message", stripped


class CursorAgentRunner(_CliRunner):
    """Cursor's agent CLI."""

    name = "cursor-agent"
    executable = "cursor-agent"

    def build_command(
        self, task_prompt: str, workdir: Path, budget: Budget
    ) -> list[str]:
        return [
            self.executable,
            "--print",
            "--force",
            self.model,
            task_prompt,
            *self.extra_args,
        ]


__all__ = [
    "ClaudeCodeRunner",
    "CodexRunner",
    "CursorAgentRunner",
    "GeminiCliRunner",
    "OpenHandsRunner",
]
