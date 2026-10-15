"""Docker-backed execution environment.

Three properties this buys, each of which the plan's anti-contamination
story depends on:

**Hermetic grading.** The grader runs with ``--network none``. Without
this a test can reach the network mid-run, and the same commit scores
differently on a different day. Reproducibility is the whole point.

**Pinned base image.** The image is recorded by digest in the run
metadata. Tag-pinned images drift, and a benchmark whose environment
moves under it cannot be re-run by anyone else.

**Read-only grading mount.** The workspace is mounted read-only during
grading. That matters: an agent with write access at grading time could
edit the hidden tests it is about to be scored against, and the
resulting score would be indistinguishable from a real solve.

The agent's turn is *not* network-isolated by default, because a local
model served over HTTP has to be reachable. Callers pass
``network=True`` for the agent turn and ``network=False`` for grading, and
``allow_hosts`` is added to ``/etc/hosts`` so an agent can still reach a
model server without unrestricted egress.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..core import Environment as EnvironmentSpec
from .base import CommandResult, Environment, EnvironmentError_, run_locally

#: Image used when a task spec does not pin one.
DEFAULT_IMAGE = "python:3.11.9-slim"

#: Host alias so a containerised agent can reach a model server on the
#: host. Only added when the caller opts in.
HOST_ALIAS = "host.docker.internal"


class DockerEnvironment(Environment):
    """Run a task inside a pinned container."""

    hermetic = True

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        allow_hosts: list[str] | None = None,
        keep_container: bool = False,
        image: str | None = None,
        build_context: Path | None = None,
    ) -> None:
        self.spec = spec
        self.image = image or spec.image or DEFAULT_IMAGE
        self.allow_hosts = allow_hosts or spec.network_allowlist
        self.keep_container = keep_container
        #: Directory the Dockerfile is built from; defaults to the CWD.
        self.build_context = Path(build_context or Path.cwd())
        self._prepared = False
        self.image_digest: str | None = None
        self.image_id: str | None = None

    # -- image ------------------------------------------------------------

    def _image_tag(self) -> str:
        """A cache tag derived from the spec, so rebuilds are cheap.

        Derived from the whole spec rather than the image name alone so
        that changing the workdir or seed set does not silently reuse a
        stale image.
        """
        payload = json.dumps(self.spec.model_dump(mode="json"), sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"dynamion-env:{digest}"

    def build(self) -> str:
        """Build (or pull) the pinned image and return its tag.

        Two paths: a task may supply a Dockerfile, in which case the image
        is built from the digest-pinned base, or it may reference a
        published image directly.
        """
        tag = self._image_tag()

        if self._inspect(tag):
            self.image_id = tag
            # Still resolve the digest on the cached path. A run record
            # that says the digest is "unresolved" is exactly the record
            # that cannot be reproduced later, and the cache is the common
            # case -- every run after the first hits this branch.
            self.image_digest = self._resolve_digest(tag)
            return tag

        if self.spec.dockerfile:
            self._build_from_dockerfile(tag)
        else:
            self._pull(self.image)

        self.image_digest = self._resolve_digest(
            tag if self.spec.dockerfile else self.image, fallback_tag=tag
        )
        self.image_id = tag
        return tag

    def _resolve_digest(self, reference: str, fallback_tag: str | None = None) -> str:
        """Best available content reference for ``reference``.

        For a digest-pinned image this is the digest itself. For a locally
        built or retagged image there is no repo digest, so the image id
        is used -- still an exact content reference, and better than
        reporting nothing.
        """
        digest = self._repo_digest(reference)
        if digest:
            return digest
        if fallback_tag:
            digest = self._repo_digest(fallback_tag)
            if digest:
                return digest
        return self._image_id(reference or fallback_tag or "") or "unknown"

    def _build_from_dockerfile(self, tag: str) -> None:
        dockerfile = Path(self.spec.dockerfile or "")
        if not dockerfile.is_absolute():
            dockerfile = (self.build_context / dockerfile).resolve()
        if not dockerfile.exists():
            raise EnvironmentError_(f"dockerfile not found: {dockerfile}")

        argv = ["docker", "build", "-f", str(dockerfile), "-t", tag]
        for key, value in self.spec.build_args.items():
            argv.extend(["--build-arg", f"{key}={value}"])
        argv.append(str(self.build_context))

        result = run_locally(argv, cwd=self.build_context, timeout=3600)
        if result.exit_code != 0:
            raise EnvironmentError_(
                f"docker build failed for {dockerfile}:\n{result.stderr[-800:]}"
            )

    def _pull(self, image: str) -> None:
        result = run_locally(["docker", "pull", image], cwd=Path.cwd(), timeout=1800)
        if result.exit_code != 0:
            raise EnvironmentError_(
                f"could not pull image {image!r}: {result.stderr[:400]}"
            )
        run_locally(
            ["docker", "tag", image, self._image_tag()],
            cwd=Path.cwd(),
            timeout=120,
        )

    def _inspect(self, tag: str) -> bool:
        result = run_locally(
            ["docker", "image", "inspect", tag], cwd=Path.cwd(), timeout=60
        )
        return result.exit_code == 0

    def _repo_digest(self, image: str) -> str | None:
        result = run_locally(
            ["docker", "image", "inspect", "--format={{index .RepoDigests 0}}", image],
            cwd=Path.cwd(),
            timeout=60,
        )
        return result.stdout.strip() if result.exit_code == 0 else None

    def _image_id(self, image: str) -> str | None:
        result = run_locally(
            ["docker", "image", "inspect", "--format={{.Id}}", image],
            cwd=Path.cwd(),
            timeout=60,
        )
        return result.stdout.strip() if result.exit_code == 0 else None

    # -- setup ------------------------------------------------------------

    def prepare(self, workdir: Path, setup_commands: list[str]) -> None:
        """Materialise the environment, baking setup commands into an image.

        ``docker run --rm`` discards its filesystem on exit, so a
        ``pip install`` performed during setup is gone by the time grading
        runs and every later command fails with "No module named pytest".
        Setup therefore happens once in a long-lived container whose
        result is committed as a derived image. That image is then reused
        for the agent turn and for grading, which also makes grading fast
        and removes setup cost from the per-task budget.
        """
        base_tag = self.build()

        if not setup_commands:
            self.image_id = base_tag
            self._prepared = True
            return

        derived = f"{base_tag}-setup"
        if not self._inspect(derived):
            self._build_derived(derived, setup_commands)

        self.image_id = derived
        self._prepared = True

    def _build_derived(self, derived_tag: str, setup_commands: list[str]) -> None:
        base_tag = self.image_id
        if base_tag is None:
            raise EnvironmentError_("base image was not built before setup")

        container = f"dynamion-setup-{os.getpid()}-{derived_tag.split(':')[-1][-8:]}"

        run_locally(["docker", "rm", "-f", container], cwd=Path.cwd(), timeout=120)

        start = run_locally(
            [
                "docker", "run", "-d",
                "--name", container,
                "--network", "bridge",
                "--workdir", "/",
                base_tag,
                "sleep", "infinity",
            ],
            cwd=Path.cwd(),
            timeout=300,
        )
        if start.exit_code != 0:
            raise EnvironmentError_(
                f"could not start setup container: {start.stderr[:400]}"
            )

        try:
            for command in setup_commands:
                result = run_locally(
                    ["docker", "exec", container, "/bin/sh", "-lc", command],
                    cwd=Path.cwd(),
                    timeout=1800,
                )
                if result.exit_code != 0:
                    raise EnvironmentError_(
                        f"setup command failed: {command}\n{result.stderr[:600]}"
                    )

            commit = run_locally(
                ["docker", "commit", container, derived_tag],
                cwd=Path.cwd(),
                timeout=900,
            )
            if commit.exit_code != 0:
                raise EnvironmentError_(
                    f"could not commit setup image: {commit.stderr[:400]}"
                )
        finally:
            run_locally(["docker", "rm", "-f", container], cwd=Path.cwd(), timeout=120)

    # -- execution --------------------------------------------------------

    def run(
        self,
        command: list[str],
        *,
        workdir: Path,
        network: bool = False,
        timeout: int = 900,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute ``command`` in the container.

        ``network=False`` is the default and the important case: the
        grading path passes nothing and gets no egress.
        """
        workdir = Path(workdir).resolve()
        if not workdir.exists():
            raise EnvironmentError_(f"workdir does not exist: {workdir}")

        tag = self.image_id or self.build()

        argv = [
            "docker", "run", "--rm",
            "--network", "none" if not network else "bridge",
            "--add-host", f"{HOST_ALIAS}:host-gateway",
            "--workdir", self.spec.workdir,
            "--volume", f"{workdir}:{self.spec.workdir}",
            "-w", self.spec.workdir,
        ]
        for host in self.allow_hosts:
            argv.extend(["--add-host", f"{host}:host-gateway"])
        for key, value in {**self.spec.env, **(env or {})}.items():
            argv.extend(["--env", f"{key}={value}"])
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            argv.extend(["--env", f"{key}="])

        argv.append(tag)
        argv.extend(command)
        return run_locally(argv, cwd=workdir, timeout=timeout)

    def run_grading(
        self,
        command: list[str],
        *,
        workdir: Path,
        timeout: int = 900,
    ) -> CommandResult:
        """Run a grader command with the workspace mounted read-only.

        Read-only is the point. The agent cannot reach grading time, so
        it cannot edit the tests that decide whether it passed.
        """
        workdir = Path(workdir).resolve()
        tag = self.image_id or self.build()

        argv = [
            "docker", "run", "--rm",
            "--network", "none",
            "--add-host", f"{HOST_ALIAS}:host-gateway",
            "--workdir", self.spec.workdir,
            "--volume", f"{workdir}:{self.spec.workdir}:ro",
            "-w", self.spec.workdir,
        ]
        for key, value in self.spec.env.items():
            argv.extend(["--env", f"{key}={value}"])
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            argv.extend(["--env", f"{key}="])
        argv.append(tag)
        argv.extend(command)

        return run_locally(argv, cwd=workdir, timeout=timeout)

    # -- metadata ---------------------------------------------------------

    def describe(self) -> dict[str, str]:
        return {
            "environment": "DockerEnvironment",
            "hermetic": "true",
            "image": self.image,
            "image_digest": self.image_digest or "unresolved",
            "image_digest_pinned": "true" if "@sha256:" in self.image else "false",
            "grading_network": "none",
            "workdir": self.spec.workdir,
        }

    def cleanup(self) -> None:
        return None


__all__ = ["DEFAULT_IMAGE", "HOST_ALIAS", "DockerEnvironment"]