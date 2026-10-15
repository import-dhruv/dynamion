"""Benchmark orchestration: tasks x agents -> leaderboard.

Each run is written to disk as an individual JSON record before the next
one starts, so a crash mid-benchmark loses at most the run in flight
rather than the whole matrix.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path

from .core import FailureMode, Patch, Score, TaskSpec
from .env import build_environment
from .graders.runner import build_grader
from .runners.registry import AgentConfig, build_runner, load_agent_configs

#: Exception types that mean "the code does not match the interface" as
#: opposed to "the code computed the wrong answer". A NameError or
#: AttributeError is an API mismatch; an AssertionError is a logic error.
_API_ERRORS = (
    "AttributeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "TypeError",
)

#: Signs the agent edited the tests rather than the code.
_TEST_EDIT_MARKERS = ("test_", "conftest", "/tests/")


def _normalise_components(result) -> list[dict]:
    """Return grader components as plain dicts.

    ``detail`` may hold GraderResult models when the aggregate grader
    passed its own result through, and calling ``.get()`` on a model
    raises rather than returning a default -- which turned every
    single-grader run into a crash instead of a label.
    """
    raw = result.detail.get("components") or []
    if not raw:
        return [result.model_dump()]
    return [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in raw]


def classify_failure(result, trajectory, patch: Patch | None = None) -> FailureMode:
    """Assign a failure mode from grader output, patch and trajectory.

    Deliberately conservative. Each rule is chosen to be evidence a
    reviewer can point at, and an unrecognised pattern yields UNKNOWN
    rather than a confident label, because a wrong label in a taxonomy is
    worse than an honest gap.

    Order matters. Infrastructure faults are checked first: a run where
    the harness broke usually has zero edits simply because the agent
    never got a working workspace, and reporting that as GAVE_UP blames
    the model for a failure of our own making.
    """
    if result.passed:
        return FailureMode.NONE

    if trajectory.exit_reason == "timeout":
        return FailureMode.TIMEOUT

    components = _normalise_components(result)
    errors = [c.get("error") for c in components if c.get("error")]

    if any("environment" in (e or "").lower() for e in errors):
        return FailureMode.ENVIRONMENT_ISSUE
    if any("timed out" in (e or "") for e in errors):
        return FailureMode.TIMEOUT

    if trajectory.n_edits == 0:
        return FailureMode.GAVE_UP

    output = " ".join(str(c.get("stdout", "")) for c in components)

    # Editing the tests is the classic way to fake progress. Checked
    # before the generic gaming markers because it is stronger evidence.
    if patch is not None:
        touched = [f for f in patch.files_changed if any(m in f for m in _TEST_EDIT_MARKERS)]
        if touched:
            return FailureMode.TEST_MISREAD

    gaming_markers = ("skip", "xfail", "pytest.skip", "@unittest.skip", "return true")
    if any(marker in output.lower() for marker in gaming_markers):
        return FailureMode.GRADER_GAMING

    # An exception type in the failure output means the code did not line
    # up with the interface it was supposed to use.
    if any(err in output for err in _API_ERRORS):
        return FailureMode.WRONG_API

    if not output.strip():
        # Edits were made and grading reported a failure, but there is no
        # output to reason about. Any specific label here would be a guess.
        return FailureMode.UNKNOWN

    return FailureMode.WRONG_APPROACH


def run_benchmark(
    tasks: list[TaskSpec],
    agents: list[str],
    *,
    agents_dir: Path,
    results_dir: Path,
    model: str | None = None,
    dry_run: bool = False,
    env_mode: str = "local",
    allow_hosts: list[str] | None = None,
) -> list[dict]:
    """Run every (task, agent) pair and return leaderboard rows."""
    configs: list[AgentConfig] = []
    if agents and all(Path(a).suffix in {".yaml", ".yml"} for a in agents):
        import yaml as _yaml

        for path in agents:
            raw = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            configs.append(
                AgentConfig(
                    name=raw["name"],
                    adapter=raw["adapter"],
                    model=model or raw["model"],
                    args=raw.get("args") or {},
                )
            )
    else:
        available = {c.name: c for c in load_agent_configs(agents_dir)}
        for name in agents:
            if name in available:
                config = available[name]
            else:
                if model is None:
                    raise ValueError(
                        f"agent {name!r} not found in {agents_dir} and no --model given"
                    )
                config = AgentConfig(name=name, adapter=name, model=model)
            if model:
                config = AgentConfig(config.name, config.adapter, model, config.args)
            configs.append(config)

    verified_map: dict[str, bool] = {}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    scores: list[Score] = []

    for config in configs:
        for task in tasks:
            print(f"  -> {config.label} | {task.slug}")
            if dry_run:
                continue

            workspace = results_dir / "workspaces" / task.id / config.name
            workspace.mkdir(parents=True, exist_ok=True)
            _seed_workspace(workspace, task)

            environment = build_environment(
                task.environment, mode=env_mode, allow_hosts=allow_hosts or []
            )
            environment.prepare(workspace, task.environment.setup_commands)

            runner = build_runner(config)
            trajectory = runner.run(
                task.prompt, workspace, task.budget, task_id=task.slug
            )
            trajectory.task = task.slug
            patch = runner.compute_patch(workspace, trajectory)

            grader = build_grader(task, environment=environment)
            result = grader.run(workspace, patch)

            score = Score(
                task=task.slug,
                agent=trajectory.agent,
                resolved=result.passed,
                score=result.score,
                graders=[result],
                failure_mode=classify_failure(result, trajectory, patch),
                wall_seconds=trajectory.duration_seconds or 0.0,
                tokens=trajectory.usage.total,
            )
            scores.append(score)
            verified_map[config.label] = bool(getattr(type(runner), "verified", False))

            record_path = results_dir / "runs" / f"{stamp}__{config.name}__{task.id}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                json.dumps(
                    {
                        "score": score.model_dump(mode="json"),
                        "trajectory": trajectory.model_dump(mode="json"),
                        "patch": patch.model_dump(mode="json"),
                        "environment": environment.describe(),
                        "adapter": config.adapter,
                        "adapter_verified": verified_map[config.label],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return summarise(scores, verified=verified_map)


def _seed_workspace(workspace: Path, task) -> None:
    """Populate a workspace from the task spec's declared setup."""
    import shutil
    import subprocess

    for rel, source in task.environment.seed_repos.items():
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"seed repo missing for {task.slug}: {src}")
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    if not (workspace / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)

    # Agent scratch files are not part of the solution. Leaving them in the
    # patch means the rubric judge reads Aider's own chat transcript as
    # though it were the agent's work.
    ignore = workspace / ".gitignore"
    ignore.write_text(".aider.*\n__pycache__/\n*.pyc\n.hypothesis/\n", encoding="utf-8")

    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=h@dynamion.dev", "-c", "user.name=harness",
         "commit", "-qm", "baseline"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )


def summarise(scores: list[Score], verified: dict[str, bool] | None = None) -> list[dict]:
    """Aggregate scores into leaderboard rows.

    ``verified`` maps an agent label to whether its adapter was actually
    executed against the real CLI. Surfaced on every row so a score from
    an unexercised adapter is never silently compared with a measured one.
    """
    verified = verified or {}
    rows: dict[str, list[Score]] = {}
    for score in scores:
        rows.setdefault(score.agent, []).append(score)

    leaderboard: list[dict] = []
    for agent, group in rows.items():
        resolved = sum(1 for s in group if s.resolved)
        leaderboard.append(
            {
                "agent": agent,
                "n": len(group),
                "resolved": resolved,
                "resolve_rate": resolved / len(group) if group else 0.0,
                "mean_score": statistics.fmean(s.score for s in group),
                # Progress against each task's untouched seed. This is the
                # figure that separates models; raw pass rate does not,
                # because a buggy seed often satisfies part of the
                # contract by accident.
                "mean_progress": _mean_progress(group),
                "total_tokens": sum(s.tokens for s in group),
                "total_wall_seconds": sum(s.wall_seconds for s in group),
                "failure_modes": _count_modes(group),
                "adapter_verified": verified.get(agent, False),
            }
        )

    leaderboard.sort(key=lambda r: (-r["resolve_rate"], -r["mean_score"]))
    return leaderboard


def _mean_progress(scores: list[Score]) -> float | None:
    """Mean baseline-relative progress, or None if no baseline is recorded."""
    values: list[float] = []
    for score in scores:
        for grader in score.graders:
            for component in grader.detail.get("components", [grader.detail]):
                if "progress" in component:
                    values.append(component["progress"])
    return statistics.fmean(values) if values else None


def _count_modes(scores: list[Score]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in scores:
        key = s.failure_mode.value
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["classify_failure", "run_benchmark", "summarise"]