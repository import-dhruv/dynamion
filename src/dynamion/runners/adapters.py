"""Adapter lookup.

New agents register here. The map is intentionally explicit rather than
discovered, so that the set of supported agents is readable in one place
and a typo fails loudly rather than silently returning ``None``.
"""

from __future__ import annotations

from .aider import AiderRunner
from .base import Runner
from .cli_agents import (
    ClaudeCodeRunner,
    CodexRunner,
    CursorAgentRunner,
    GeminiCliRunner,
    OpenHandsRunner,
)

_REGISTRY: dict[str, type[Runner]] = {
    "aider": AiderRunner,
    "claude-code": ClaudeCodeRunner,
    "codex": CodexRunner,
    "cursor-agent": CursorAgentRunner,
    "gemini-cli": GeminiCliRunner,
    "openhands": OpenHandsRunner,
}


def get_adapter(name: str) -> type[Runner]:
    """Return the runner class registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown adapter {name!r}; available: {available}") from None


def register_adapter(name: str, cls: type[Runner]) -> None:
    _REGISTRY[name] = cls


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["available_adapters", "get_adapter", "register_adapter"]
