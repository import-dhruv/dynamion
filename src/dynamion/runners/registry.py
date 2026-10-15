"""Agent registry.

Agents are described by small YAML files so that adding one does not
require touching the harness. Keeping configuration out of code means a
leaderboard row can always be traced back to a file in the repository,
which is what makes a result auditable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import Runner


@dataclass(frozen=True)
class AgentConfig:
    """One configured agent, as declared in ``agents/*.yaml``."""

    name: str
    adapter: str
    model: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.adapter}:{self.model}"


def load_agent_configs(directory: Path) -> list[AgentConfig]:
    """Load every ``*.yaml`` agent config in ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"agents directory not found: {directory}")

    configs: list[AgentConfig] = []
    for path in sorted(directory.glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: agent config must be a mapping")

        missing = {"name", "adapter", "model"} - data.keys()
        if missing:
            raise ValueError(f"{path}: missing required keys {sorted(missing)}")

        configs.append(
            AgentConfig(
                name=data["name"],
                adapter=data["adapter"],
                model=data["model"],
                args=data.get("args") or {},
            )
        )

    if not configs:
        raise ValueError(f"no agent configs found in {directory}")
    return configs


def build_runner(spec: str | AgentConfig, *, model: str | None = None) -> Runner:
    """Instantiate a runner from an agent name or an explicit config."""
    from .adapters import get_adapter

    if isinstance(spec, AgentConfig):
        config = spec
    else:
        if model is None:
            raise ValueError(
                f"agent {spec!r} needs a model: pass --model or use an agents/*.yaml config"
            )
        config = AgentConfig(name=spec, adapter=spec, model=model)

    if model:
        config = AgentConfig(
            name=config.name,
            adapter=config.adapter,
            model=model,
            args=config.args,
        )

    return get_adapter(config.adapter)(model=config.model, **config.args)


__all__ = ["AgentConfig", "build_runner", "load_agent_configs"]
