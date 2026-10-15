"""Command-line interface.

Subcommands are intentionally small and composable:

    dynamion validate          check every task spec against the schema
    dynamion list              show the task inventory
    dynamion schema            emit JSON Schema for editors/CI
    dynamion run               run one task with one agent and grade it
    dynamion bench             run a matrix of tasks x agents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import Score
from .tasks import load_task, load_tasks, schema_json

DEFAULT_TASKS_DIR = Path("tasks")


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        tasks = load_tasks(Path(args.tasks_dir))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for task in tasks:
        warnings = task_warnings(task)
        status = "ok" if not warnings else "warn"
        print(f"[{status}] {task.slug}  {task.title}")
        for warning in warnings:
            print(f"       - {warning}")
    print(f"\n{len(tasks)} task(s) validated.")
    return 0


def task_warnings(task) -> list[str]:
    """Non-fatal quality issues.

    These encode the review bar from CONTRIBUTING.md: a task without
    provenance or a human baseline is technically valid but not
    publishable.
    """
    issues: list[str] = []

    if task.source.kind == "derived" and not task.source.upstream:
        issues.append("derived task has no upstream recorded")

    if task.source.contamination_checked_at is None:
        issues.append("no contamination check recorded")

    if task.human_baseline is None:
        issues.append("no human baseline")

    if not task.expected_pitfalls:
        issues.append("no expected pitfalls documented")

    for grader in task.graders:
        if grader.type == "tests" and not grader.hidden_paths:
            issues.append(f"test grader {grader.type!r} exposes its tests to the agent")
        if grader.type == "rubric" and grader.self_consistency_k < 3:
            issues.append("rubric grader uses self_consistency_k < 3")

    image = task.environment.unpinned_reference()
    if image:
        issues.append(
            f"image {image!r} is tag-pinned, not digest-pinned; a published "
            "score would not be reproducible if the tag moves"
        )
    if task.environment.dockerfile and not task.environment.is_digest_pinned:
        issues.append("task uses a Dockerfile but its base image is not digest-pinned")

    return issues


def _cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks_dir))
    print(f"{'TASK':<34} {'DIFF':<8} {'BUDGET':<10} {'GRADERS':<14} TITLE")
    for task in tasks:
        minutes = f"{task.budget.max_minutes:.0f}m"
        graders = ",".join(g.type for g in task.graders)
        print(
            f"{task.slug:<34} {task.difficulty.value:<8} {minutes:<10} "
            f"{graders:<14} {task.title}"
        )
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    if args.out:
        Path(args.out).write_text(schema_json() + "\n", encoding="utf-8")
        print(f"wrote schema to {args.out}")
    else:
        print(schema_json())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .env import build_environment
    from .env.base import EnvironmentError_
    from .graders.rubric import JudgeUnavailable
    from .graders.runner import build_grader
    from .graders.tests import GraderEnvironmentError
    from .runners.base import RunnerError
    from .runners.registry import build_runner

    task = load_task(Path(args.task))
    runner = build_runner(args.agent, model=args.model)

    workspace = Path(args.workspace).resolve()
    print(f"task   : {task.slug}")
    print(f"agent  : {runner.name}:{runner.model}")
    print(f"budget : {task.budget.max_turns} turns / {task.budget.max_minutes:.0f} min")
    print(f"workdir: {workspace}\n")

    try:
        environment = build_environment(
            task.environment,
            mode=args.env,
            allow_hosts=args.allow_host or [],
            build_context=Path(args.task).parent,
        )
        environment.prepare(workspace, task.environment.setup_commands)
        print(f"env    : {environment.describe()}\n")
    except EnvironmentError_ as exc:
        print(f"\nerror: environment setup failed\n  {exc}", file=sys.stderr)
        return 5

    try:
        trajectory = runner.run(
            task.prompt, workspace, task.budget, task_id=task.slug
        )
    except RunnerError as exc:
        print(f"\nerror: agent could not be invoked\n  {exc}", file=sys.stderr)
        return 4

    trajectory.task = task.slug
    patch = runner.compute_patch(workspace, trajectory)

    grader = build_grader(
        task, source_root=Path(args.task).parent, environment=environment
    )
    try:
        grader_result = grader.run(workspace, patch)
    except GraderEnvironmentError as exc:
        # The task was not attempted as far as the grader is concerned.
        # Reporting this as a score would be wrong, and a traceback would
        # be unhelpful: it is an environment problem with a known fix.
        print(
            f"\nerror: grading environment failed, no score recorded\n  {exc}\n\n"
            "This is an infrastructure fault, not a task failure. Check that the\n"
            "test command runs standalone inside the task workspace.",
            file=sys.stderr,
        )
        return 3
    except JudgeUnavailable as exc:
        print(f"\nerror: rubric judge unavailable\n  {exc}", file=sys.stderr)
        return 3

    score = Score(
        task=task.slug,
        agent=trajectory.agent,
        resolved=grader_result.passed,
        score=grader_result.score,
        graders=[grader_result],
        wall_seconds=trajectory.duration_seconds or 0.0,
        tokens=trajectory.usage.total,
    )

    print("\n--- result ---")
    print(f"score    : {score.score:.2f}  resolved={score.resolved}")
    print(f"edits    : {trajectory.n_edits}")
    print(f"files    : {', '.join(patch.files_changed) or '(none)'}")
    print(f"wall     : {score.wall_seconds:.0f}s")

    if args.out:
        record = {
            "score": score.model_dump(mode="json"),
            "trajectory": trajectory.model_dump(mode="json"),
            "patch": patch.model_dump(mode="json"),
            "environment": environment.describe(),
            "adapter": args.agent,
            "adapter_verified": bool(getattr(type(runner), "verified", False)),
        }
        Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"run record written to {args.out}")

    return 0 if score.resolved else 2


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark
    from .env.base import EnvironmentError_

    tasks = load_tasks(Path(args.tasks_dir))
    if args.only:
        wanted = set(args.only)
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            print(f"error: no tasks matched {sorted(wanted)}", file=sys.stderr)
            return 1

    agents = args.agents
    print(f"benchmark: {len(tasks)} task(s) x {len(agents)} agent(s)")
    print(f"env      : {args.env}\n")

    try:
        summary = run_benchmark(
            tasks,
            agents,
            agents_dir=Path(args.agents_dir),
            results_dir=Path(args.results_dir),
            model=args.model,
            dry_run=args.dry_run,
            env_mode=args.env,
            allow_hosts=args.allow_host or [],
        )
    except EnvironmentError_ as exc:
        print(f"\nerror: environment setup failed\n  {exc}", file=sys.stderr)
        return 5

    print("\n=== leaderboard ===")
    print(f"{'AGENT':<38} {'RESOLVED':<16} {'SCORE':<10} TOKENS")
    for row in summary:
        verified = " [verified]" if row.get("adapter_verified") else " [unverified]"
        print(
            f"{row['agent'] + verified:<38} "
            f"{row['resolved']}/{row['n']} ({row['resolve_rate']:.0%})".ljust(55)
            + f"{row['mean_score']:<10.3f} {row['total_tokens']}"
        )

    out = Path(args.results_dir) / "leaderboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nresults written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamion",
        description="Long-horizon software-engineering evaluation for coding agents.",
    )
    parser.add_argument("--version", action="version", version=f"dynamion {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate task specs")
    p_validate.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    p_validate.set_defaults(func=_cmd_validate)

    p_list = sub.add_parser("list", help="list available tasks")
    p_list.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    p_list.set_defaults(func=_cmd_list)

    p_schema = sub.add_parser("schema", help="emit the task JSON Schema")
    p_schema.add_argument("--out", help="write to a file instead of stdout")
    p_schema.set_defaults(func=_cmd_schema)

    p_run = sub.add_parser("run", help="run a single task with a single agent")
    p_run.add_argument("task", help="path to a task spec")
    p_run.add_argument("--agent", default="aider")
    p_run.add_argument("--model", default="ollama/qwen2.5-coder:7b")
    p_run.add_argument("--workspace", required=True, help="prepared workspace directory")
    p_run.add_argument(
        "--env",
        choices=["local", "docker"],
        default="local",
        help="execution environment; 'docker' grades with no network access",
    )
    p_run.add_argument(
        "--allow-host",
        action="append",
        help="host reachable from the container (repeatable); use for a model server",
    )
    p_run.add_argument("--out", help="write the full run record as JSON")
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="run tasks x agents and rank them")
    p_bench.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    p_bench.add_argument("--agents-dir", default="agents")
    p_bench.add_argument("--results-dir", default="results")
    p_bench.add_argument("--agents", nargs="+", default=["aider-local"])
    p_bench.add_argument("--model", default=None, help="override model for all agents")
    p_bench.add_argument("--only", nargs="+", help="restrict to these task ids")
    p_bench.add_argument(
        "--env",
        choices=["local", "docker"],
        default="local",
        help="execution environment; 'docker' grades with no network access",
    )
    p_bench.add_argument(
        "--allow-host",
        action="append",
        help="host reachable from the container (repeatable); use for a model server",
    )
    p_bench.add_argument(
        "--dry-run", action="store_true", help="resolve config without executing agents"
    )
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())