"""Verify the isolation claims the README makes.

These are the load-bearing assertions of the whole benchmark: that
grading cannot reach the network, and cannot be edited by the thing it
is grading. If either stops holding, every published number becomes
uninterpretable, so they are tested rather than documented.

The control matters as much as the test. Under ``--env local`` the
network assertion must FAIL, which proves the assertion is actually
capable of detecting the thing it claims to detect. A test that passes
vacuously is worse than no test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamion.core import Environment as EnvironmentSpec
from dynamion.env import build_environment
from dynamion.env.base import EnvironmentError_

pytestmark = pytest.mark.docker

#: Image produced by `DockerEnvironment.prepare` on the bundled tasks.
#: Overridden by the CI workflow, which builds it explicitly.
IMAGE = "dynamion-test-env"


@pytest.fixture
def workspace(tmp_path_factory):
    """A workspace Docker can actually mount.

    Not ``tmp_path``: Docker Desktop does not share /tmp with containers,
    so a mount from there is denied outright.
    """
    base = Path.home() / "dynamion-ws" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(str(tmp_path_factory.mktemp("ws")))
    try:
        path.relative_to(Path.home())
        return path
    except ValueError:
        pass

    import shutil

    target = base / path.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(path, target)
    return target


def _run_probe(workspace, body: str) -> str:
    """Execute ``body`` inside the container's grading mount and return stdout.

    The probe has to run *inside* the container: a test asserting on the
    host process proves nothing about the container's isolation.
    """
    spec = EnvironmentSpec(image="python:3.11.9-slim", workdir="/workspace")
    env = build_environment(spec, mode="docker")
    env.image_id = IMAGE

    probe = workspace / "_probe.py"
    probe.write_text(body, encoding="utf-8")
    try:
        result = env.run_grading(
            ["python3", "_probe.py"], workdir=workspace, timeout=120
        )
    finally:
        probe.unlink(missing_ok=True)
    return result.stdout.strip()


def test_network_is_blocked_during_grading(workspace):
    """A DNS lookup must fail: no egress while grading."""
    output = _run_probe(
        workspace,
        "import socket\n"
        "try:\n"
        "    socket.gethostbyname('example.com')\n"
        "    print('NETWORK_REACHABLE')\n"
        "except OSError:\n"
        "    print('NETWORK_BLOCKED')\n",
    )
    assert output == "NETWORK_BLOCKED", (
        "grading had network egress; scores would not be reproducible"
    )


def test_workspace_is_read_only_during_grading(workspace):
    """The grader must not be able to write into the workspace."""
    output = _run_probe(
        workspace,
        "try:\n"
        "    open('GRADER_WAS_EDITABLE', 'w').write('tampered')\n"
        "    print('WORKSPACE_WRITABLE')\n"
        "except OSError:\n"
        "    print('WORKSPACE_READ_ONLY')\n",
    )
    assert output == "WORKSPACE_READ_ONLY", (
        "workspace was writable at grading time; the agent could edit the "
        "tests deciding whether it passed"
    )


def test_local_mode_is_not_isolated(workspace):
    """The control: the same probe must succeed locally.

    Without this, the isolation tests above could pass vacuously -- for
    example if the probe silently failed to run.
    """
    spec = EnvironmentSpec(image="unused", workdir="/workspace")
    env = build_environment(spec, mode="local")

    probe = workspace / "_control.py"
    probe.write_text(
        "import socket\n"
        "try:\n"
        "    socket.gethostbyname('example.com')\n"
        "    print('NETWORK_REACHABLE')\n"
        "except OSError:\n"
        "    print('NETWORK_BLOCKED')\n",
        encoding="utf-8",
    )
    try:
        result = env.run([env.python, "_control.py"], workdir=workspace, timeout=120)
    finally:
        probe.unlink(missing_ok=True)

    assert "NETWORK_REACHABLE" in result.stdout, (
        "local mode unexpectedly had no network; the isolation tests would "
        "then be proving nothing"
    )


def test_agent_turn_can_reach_an_allowlisted_host(workspace):
    """The agent phase must still be able to reach a model server.

    The other half of the control: if the agent turn were also fully
    offline, grading would be isolated for the wrong reason and no
    local-model agent could run.
    """
    spec = EnvironmentSpec(image="python:3.11.9-slim", workdir="/workspace")
    env = build_environment(spec, mode="docker")
    env.image_id = IMAGE
    result = env.run(
        ["python3", "-c", "print('AGENT_NETWORK_OK')"],
        workdir=workspace,
        network=True,
        timeout=120,
    )
    assert "AGENT_NETWORK_OK" in result.stdout


def test_docker_environment_reports_hermetic():
    spec = EnvironmentSpec(image="python:3.11.9-slim", workdir="/workspace")
    env = build_environment(spec, mode="docker")
    assert env.hermetic is True
    assert env.describe()["grading_network"] == "none"


def test_local_environment_admits_it_is_not_hermetic():
    """Honesty check: local mode must not claim isolation it lacks."""
    spec = EnvironmentSpec(image="unused", workdir="/workspace")
    env = build_environment(spec, mode="local")
    assert env.hermetic is False
    assert env.describe()["grading_network"] == "unrestricted"


def test_setup_is_baked_into_a_derived_image():
    """`docker run --rm` discards its filesystem, so setup must commit.

    Regression guard: without the commit step, `pip install` during setup
    vanishes and grading fails with "No module named pytest" -- an
    infrastructure fault that looks exactly like a model failure.
    """
    spec = EnvironmentSpec(image="python:3.11.9-slim", workdir="/workspace")
    env = build_environment(spec, mode="docker")
    env.build()
    before = env.image_id
    env.prepare(Path.cwd(), ["true"])
    assert env.image_id != before
    assert str(env.image_id).endswith("-setup")


def test_unknown_environment_mode_is_rejected():
    spec = EnvironmentSpec(image="unused", workdir="/workspace")
    with pytest.raises(ValueError, match="unknown environment mode"):
        build_environment(spec, mode="chroot")


def test_missing_image_raises_environment_error(workspace):
    """An unavailable image must raise, not score zero."""
    spec = EnvironmentSpec(
        image="dynamion/definitely-does-not-exist:v0", workdir="/workspace"
    )
    env = build_environment(spec, mode="docker")
    with pytest.raises(EnvironmentError_):
        env.build()


def test_grading_mount_is_read_only(workspace):
    """The grading invocation itself must carry the `:ro` flag."""
    spec = EnvironmentSpec(image="python:3.11.9-slim", workdir="/workspace")
    env = build_environment(spec, mode="docker")
    env.image_id = IMAGE

    captured: dict[str, list[str]] = {}

    import dynamion.env.docker as docker_mod

    original = docker_mod.run_locally

    def spy(argv, **kwargs):
        captured["argv"] = argv
        from dynamion.env.base import CommandResult

        return CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.0)

    docker_mod.run_locally = spy
    try:
        env.run_grading(["true"], workdir=workspace)
    finally:
        docker_mod.run_locally = original

    argv = captured["argv"]
    assert "none" in argv[argv.index("--network") + 1], "grading must be offline"
    mounts = [a for a in argv if a.startswith(str(workspace))]
    assert mounts and mounts[0].endswith(":ro"), "workspace must mount read-only"