"""Core domain types for the harness.

Every type here is a frozen-by-convention Pydantic model so that a task
spec, a run, and a score can round-trip through JSON without loss. That
property is what lets us re-score historical trajectories offline: a
grader consumes a recorded :class:`Run`, never a live agent.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"


class StrictModel(BaseModel):
    """Base for all serialisable models.

    ``extra="forbid"`` is deliberate. A task spec that silently ignores an
    unknown key is a task spec that silently loses configuration, and a
    benchmark that loses configuration produces numbers nobody can trust.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


class Budget(StrictModel):
    """Resource ceilings for a single agent attempt.

    Both limits are enforced independently. A model that loops until it
    hits the wall-clock limit burns far more tokens than one that runs a
    bounded number of turns, so reporting cost-per-solve requires the
    turn limit to exist even when it rarely binds.
    """

    max_turns: int = Field(default=30, ge=1, le=500)
    max_seconds: int = Field(default=1800, ge=10, le=86400)
    max_tokens: int | None = Field(default=None, ge=1)

    @property
    def max_minutes(self) -> float:
        return self.max_seconds / 60.0


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class Environment(StrictModel):
    """How to build the sandbox a task runs in.

    ``image`` should be digest-pinned (``python:3.11.9-slim@sha256:...``)
    rather than tag-pinned. Tags move; a benchmark whose environment can
    drift underneath it cannot be re-run by anyone else.

    Alternatively, supply ``dockerfile`` and the harness will build the
    image itself, which is how a task pins OS-level dependencies that no
    published image has. ``image`` then names the base to build FROM and
    must also be digest-pinned.
    """

    image: str
    workdir: str = "/workspace"
    setup_commands: list[str] = Field(default_factory=list)
    #: Sibling directories copied into /workspace, e.g. a repo snapshot.
    seed_repos: dict[str, str] = Field(default_factory=dict)
    #: Endpoints the agent may reach. Empty means fully offline.
    network_allowlist: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    #: Per-image content hash recorded in the run log for reproducibility.
    digest: str | None = None
    #: Path to a Dockerfile, relative to the task spec.
    dockerfile: str | None = None
    #: Build arguments passed to the Dockerfile.
    build_args: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _workdir_is_absolute(self) -> Environment:
        if not self.workdir.startswith("/"):
            raise ValueError(f"workdir must be an absolute path, got {self.workdir!r}")
        return self

    @property
    def is_digest_pinned(self) -> bool:
        """True when the image reference cannot drift.

        Checked rather than assumed: a tag-pinned image is the single
        easiest way for a published benchmark number to stop being
        reproducible six months later.
        """
        return "@sha256:" in self.image

    def unpinned_reference(self) -> str | None:
        """Return the image reference if it is not digest-pinned."""
        return None if self.is_digest_pinned else self.image


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------


class TestGrader(StrictModel):
    """Runs hidden tests and returns a fractional score in [0, 1].

    Partial credit is derived from per-test pass counts rather than a
    binary exit code, because a long-horizon task almost never resolves
    all-or-nothing and a binary signal throws away most of the signal.
    """

    #: Stops pytest collecting this model as a test class: it shares a
    #: name prefix with the thing it describes and has a constructor.
    __test__ = False

    type: Literal["tests"] = "tests"
    command: list[str]
    #: Paths withheld from the agent, injected only at grading time.
    #: Each entry is ``source`` or ``source=destination`` relative to the
    #: task directory. The explicit form matters: a hidden test usually
    #: has to land beside the visible tests so relative imports resolve.
    hidden_paths: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=900, ge=1)
    weight: float = Field(default=1.0, ge=0.0)
    #: Regex used to count individual results, e.g. pytest's report line.
    result_pattern: str | None = r"(\d+) passed"
    partial_credit: bool = True


class RubricCriterion(StrictModel):
    """One scored, human-readable dimension of a rubric.

    ``anchor_low``/``anchor_high`` describe what each end of the scale
    means. Without anchors, judges drift and scores stop being comparable
    across runs.
    """

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    weight: float = Field(default=1.0, ge=0.0)
    anchor_low: str = Field(default="Not satisfied")
    anchor_high: str = Field(default="Fully satisfied")
    required: bool = False


class RubricGrader(StrictModel):
    """LLM-as-judge over explicit criteria.

    ``judge_model`` must differ from the model under evaluation, and
    ``self_consistency_k`` > 1 is strongly preferred: a single judge pass
    is noisy enough that its scores are close to meaningless.
    """

    type: Literal["rubric"] = "rubric"
    criteria: list[RubricCriterion] = Field(min_length=1)
    judge_model: str
    judge_provider: str = "ollama"
    judge_base_url: str | None = None
    self_consistency_k: int = Field(default=3, ge=1, le=16)
    #: Which artefact the judge reads: the diff, the prompt, or both.
    judge_input: list[Literal["diff", "prompt", "trajectory"]] = Field(
        default_factory=lambda: ["diff"]
    )
    weight: float = Field(default=1.0, ge=0.0)
    #: Judges are told to ignore this; see the note in CONTRIBUTING.md.
    anti_gaming_note: str | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> RubricGrader:
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric criterion ids must be unique")
        return self

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


GraderSpec = Annotated[TestGrader | RubricGrader, Field(discriminator="type")]


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class SourceProvenance(StrictModel):
    """Where the task came from.

    Required, not optional. The weakest criticism of public coding
    benchmarks is that solved tasks may have been in pretraining data.
    Recording provenance lets a reviewer judge the contamination risk
    rather than guess at it.
    """

    kind: Literal["fresh_repo", "seeded_bug", "synthetic_feature", "derived"]
    upstream: str | None = None
    commit: str | None = None
    notes: str = ""
    #: Search performed for public copies of the task text, with a date.
    contamination_checked_at: dt.datetime | None = None


class HumanBaseline(StrictModel):
    """A human solving the same task under the same budget."""

    solver: str
    seconds: int = Field(ge=1)
    notes: str = ""
    recorded_at: dt.datetime | None = None


class TaskSpec(StrictModel):
    """A single long-horizon engineering task.

    The separation of ``prompt`` (what the agent sees) from ``hidden_paths``
    (what only the grader sees) is the anti-gaming boundary. If a task's
    prompt contains its own tests, a solved task and a gamed task are
    indistinguishable.
    """

    schema_version: str = SCHEMA_VERSION
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    prompt: str = Field(min_length=40)
    difficulty: Difficulty = Difficulty.MEDIUM
    tags: list[str] = Field(default_factory=list)
    environment: Environment
    graders: list[GraderSpec] = Field(min_length=1)
    budget: Budget = Field(default_factory=Budget)
    source: SourceProvenance
    human_baseline: HumanBaseline | None = None
    expected_pitfalls: list[str] = Field(default_factory=list)
    time_created: dt.datetime | None = None

    @model_validator(mode="after")
    def _has_scoring_path(self) -> TaskSpec:
        if not self.graders:
            raise ValueError("task must define at least one grader")
        return self

    @property
    def slug(self) -> str:
        return f"{self.id}@{self.version}"


# --------------------------------------------------------------------------
# Runs, trajectories, scores
# --------------------------------------------------------------------------


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCall(StrictModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str | None = None
    wall_seconds: float = Field(default=0.0, ge=0.0)


class TrajectoryEvent(StrictModel):
    """One timestamped entry in the agent's work log."""

    turn: int = Field(ge=0)
    kind: Literal["message", "tool_call", "tool_result", "edit", "error", "note"]
    content: str = ""
    timestamp: dt.datetime


class Trajectory(StrictModel):
    """Everything an agent did, recorded so a run can be audited."""

    task: str
    agent: str
    events: list[TrajectoryEvent] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    started_at: dt.datetime
    ended_at: dt.datetime | None = None
    exit_reason: Literal["completed", "budget_exhausted", "error", "timeout"] | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def n_edits(self) -> int:
        return sum(1 for e in self.events if e.kind == "edit")


class Patch(StrictModel):
    """The agent's net change to the repository."""

    unified_diff: str = ""
    files_changed: list[str] = Field(default_factory=list)
    diff_bytes: int = Field(default=0, ge=0)


class GraderResult(StrictModel):
    grader_id: str
    type: Literal["tests", "rubric"]
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    detail: dict[str, Any] = Field(default_factory=dict)
    passed: bool = False
    error: str | None = None


class FailureMode(StrEnum):
    """Taxonomy for why an attempt did not fully succeed.

    Ordered roughly from most to least common in observed runs. The
    distinction that matters most in practice is between
    ``WRONG_APPROACH`` and ``GAVE_UP``: the first is a capability limit,
    the second is usually a budget or interface problem.

    Every member must be reachable from :func:`dynamion.bench.classify_failure`;
    a test asserts this, because a declared-but-unreachable mode is a dead
    constant that makes a taxonomy look complete while reporting nothing.
    """

    NONE = "none"
    WRONG_APPROACH = "wrong_approach"
    WRONG_API = "wrong_api"
    TEST_MISREAD = "test_misread"
    ENVIRONMENT_ISSUE = "environment_issue"
    GAVE_UP = "gave_up"
    GRADER_GAMING = "grader_gaming"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class Score(StrictModel):
    """The result of grading one attempt."""

    task: str
    agent: str
    resolved: bool = False
    score: float = Field(ge=0.0, le=1.0)
    graders: list[GraderResult] = Field(default_factory=list)
    failure_mode: FailureMode = FailureMode.NONE
    wall_seconds: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = None

    @property
    def cost_per_solve(self) -> float | None:
        """USD per resolved task, or None when nothing resolved."""
        if not self.resolved or self.cost_usd is None or self.score <= 0:
            return None
        return self.cost_usd / self.score


class Run(StrictModel):
    """A complete, replayable record of one attempt."""

    id: str
    task: str
    agent: str
    trajectory: Trajectory
    patch: Patch
    started_at: dt.datetime
    harness_version: str = SCHEMA_VERSION