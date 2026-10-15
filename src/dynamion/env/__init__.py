"""Execution environments.

``LocalEnvironment`` (no isolation) and ``DockerEnvironment`` (hermetic).
The distinction is recorded in every run record, because a score obtained
without network isolation measures something different from one obtained
with it.
"""

from .base import CommandResult, Environment, EnvironmentError_, run_locally
from .docker import DockerEnvironment
from .local import LocalEnvironment


def build_environment(spec, *, mode: str = "local", **kwargs) -> Environment:
    """Create an environment by mode name."""
    if mode == "docker":
        return DockerEnvironment(spec, **kwargs)
    if mode == "local":
        return LocalEnvironment(spec, **kwargs)
    raise ValueError(f"unknown environment mode: {mode!r} (expected 'docker' or 'local')")


__all__ = [
    "CommandResult",
    "DockerEnvironment",
    "Environment",
    "EnvironmentError_",
    "LocalEnvironment",
    "build_environment",
    "run_locally",
]
