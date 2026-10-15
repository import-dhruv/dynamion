"""Unit tests for the domain types and task loading.

The validators here are the harness's first line of defence. A task spec
that is accepted but misconfigured produces a score that looks valid and
is wrong, so the important cases are the ones that must be *rejected*.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dynamion.core import (
    Budget,
    Environment,
    RubricCriterion,
    RubricGrader,
    SourceProvenance,
    TaskSpec,
)
from dynamion.tasks import load_tasks, schema_json

REPO_TASKS = Path(__file__).resolve().parents[1] / "tasks"

BUNDLED_TASK_IDS = {
    "retry-backoff",
    "idempotent-ledger",
    "legacy-dateutil",
    "inplace-refactor",
    "cache-coalesce",
}


def minimal_task(**overrides) -> dict:
    base = {
        "id": "demo-task",
        "version": "1.0.0",
        "title": "A demo task",
        "prompt": "x" * 60,
        "environment": {"image": "python:3.11.9-slim"},
        "graders": [{"type": "tests", "command": ["pytest"]}],
        "source": {"kind": "fresh_repo"},
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ strict


def test_unknown_field_is_rejected():
    """A typo must fail loudly, not silently drop configuration.

    ``extra="forbid"`` is the single most important property here: a
    benchmark that silently ignores ``hidden_path`` (singular) would run
    with the tests visible and report a meaningless number.
    """
    with pytest.raises(ValidationError, match="hidden_path"):
        TaskSpec.model_validate(minimal_task(hidden_path=["x.py"]))


def test_short_prompt_is_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(minimal_task(prompt="fix it"))


def test_bad_id_is_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(minimal_task(id="Not A Valid Id"))


def test_bad_version_is_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(minimal_task(version="1.0"))


def test_task_requires_at_least_one_grader():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(minimal_task(graders=[]))


def test_relative_workdir_is_rejected():
    """A relative workdir silently changes where a container mounts."""
    with pytest.raises(ValidationError, match="absolute"):
        Environment(image="python:3.11.9-slim", workdir="workspace")


# ------------------------------------------------------------------ budget


def test_budget_rejects_zero_seconds():
    with pytest.raises(ValidationError):
        Budget(max_seconds=0)


def test_budget_reports_minutes():
    assert Budget(max_seconds=1800).max_minutes == 30.0


# ------------------------------------------------------------------ rubric


def test_duplicate_criterion_ids_are_rejected():
    """Duplicate ids make a rubric score ambiguous.

    Rejected at construction, not at grading time: catching it later
    would mean a run had already been scored against an ambiguous rubric.
    """
    with pytest.raises(ValidationError, match="unique"):
        RubricGrader(
            criteria=[
                RubricCriterion(id="clarity", description="x"),
                RubricCriterion(id="clarity", description="y"),
            ],
            judge_model="ollama/qwen2.5-coder:7b",
        )


def test_rubric_total_weight():
    spec = RubricGrader(
        criteria=[
            RubricCriterion(id="a", description="x", weight=2.0),
            RubricCriterion(id="b", description="y", weight=3.0),
        ],
        judge_model="judge",
    )
    assert spec.total_weight == 5.0


# ------------------------------------------------------------- digest pins


def test_digest_pinning_is_detectable():
    pinned = Environment(image="python:3.11.9-slim@sha256:abc")
    assert pinned.is_digest_pinned is True
    assert pinned.unpinned_reference() is None

    tagged = Environment(image="python:3.11.9-slim")
    assert tagged.is_digest_pinned is False
    assert tagged.unpinned_reference() == "python:3.11.9-slim"


# ----------------------------------------------------------------- loading


def test_load_bundled_tasks():
    tasks = load_tasks(REPO_TASKS)
    assert {t.id for t in tasks} == BUNDLED_TASK_IDS


def test_task_material_is_not_mistaken_for_a_spec(tmp_path):
    """Only `task.yaml` is a spec; other YAML/JSON in a task dir is material.

    A task directory legitimately holds hidden grader tests, golden
    behaviour fixtures and agent configs. Discovering specs by ``*.yaml``
    made the loader try to validate a golden JSON file as a task, which
    failed the entire load rather than just skipping the file.
    """
    task = tmp_path / "demo"
    (task / "hidden").mkdir(parents=True)
    (task / "agents").mkdir()
    (task / "task.yaml").write_text(yaml.safe_dump(minimal_task()), encoding="utf-8")
    # Material that must be ignored, not validated.
    (task / "hidden" / "test_hidden.py").write_text("def test_x(): pass\n")
    (task / "hidden" / "behaviour-golden.json").write_text('{"a": 1}', encoding="utf-8")
    (task / "agents" / "aider.yaml").write_text("name: x\n", encoding="utf-8")
    (task / "notes.yaml").write_text("title: not a task\n", encoding="utf-8")

    assert [t.id for t in load_tasks(tmp_path)] == ["demo-task"]


def test_dotted_spec_names_are_accepted(tmp_path):
    task = tmp_path / "demo"
    task.mkdir()
    (task / "retry.task.yaml").write_text(
        yaml.safe_dump(minimal_task()), encoding="utf-8"
    )
    assert [t.id for t in load_tasks(tmp_path)] == ["demo-task"]


def test_duplicate_task_ids_are_rejected(tmp_path):
    payload = yaml.safe_dump(minimal_task())
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.yaml").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(tmp_path)


def test_missing_tasks_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "nope")


def test_empty_tasks_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no task specs"):
        load_tasks(tmp_path)


def test_schema_is_valid_json():
    import json

    parsed = json.loads(schema_json())
    assert "properties" in parsed or "$defs" in parsed


# --------------------------------------------- bundled task quality gates


def test_bundled_tasks_have_hidden_tests():
    """A test grader with no hidden paths measures compliance, not skill."""
    for task in load_tasks(REPO_TASKS):
        for grader in task.graders:
            if grader.type == "tests":
                assert grader.hidden_paths, f"{task.slug} exposes its tests"


def test_bundled_tasks_record_provenance():
    for task in load_tasks(REPO_TASKS):
        assert isinstance(task.source, SourceProvenance)
        assert task.source.contamination_checked_at is not None, (
            f"{task.slug} has no contamination check"
        )


def test_bundled_tasks_digest_pin_their_images():
    for task in load_tasks(REPO_TASKS):
        assert task.environment.is_digest_pinned, f"{task.slug} image is not pinned"


def test_bundled_rubric_judges_use_self_consistency():
    for task in load_tasks(REPO_TASKS):
        for grader in task.graders:
            if grader.type == "rubric":
                assert grader.self_consistency_k >= 3, (
                    f"{task.slug}: single-pass judging is too noisy"
                )


def test_bundled_tasks_do_not_hardcode_an_interpreter():
    """A hard-coded path breaks on any other machine or in a container."""
    for task in load_tasks(REPO_TASKS):
        for grader in task.graders:
            if grader.type == "tests":
                assert "${PYTHON}" in grader.command, (
                    f"{task.slug} hard-codes an interpreter: {grader.command}"
                )


def test_bundled_tasks_map_hidden_tests_to_an_existing_source():
    """Every ``source=destination`` entry must point at a real file.

    A hidden path that does not resolve silently produces a run that
    collects nothing and scores 0.0 -- indistinguishable from total
    failure.
    """
    for task in load_tasks(REPO_TASKS):
        spec_path = next(
            p
            for p in REPO_TASKS.rglob("task.yaml")
            if task.id in p.parts
        )
        root = spec_path.parent
        for grader in task.graders:
            if grader.type != "tests":
                continue
            for entry in grader.hidden_paths:
                source = entry.split("=", 1)[0]
                assert (root / source).exists(), (
                    f"{task.slug}: hidden path does not resolve: {source}"
                )

def test_bundled_tasks_record_a_baseline():
    """A task without a recorded baseline reports no progress figure.

    Without it, every agent gets a free floor and the score cannot
    separate a weak model from a strong one.
    """
    for task in load_tasks(REPO_TASKS):
        for grader in task.graders:
            if grader.type == "tests":
                assert grader.baseline is not None, f"{task.slug} has no baseline"
                assert grader.baseline.passed <= grader.baseline.total
