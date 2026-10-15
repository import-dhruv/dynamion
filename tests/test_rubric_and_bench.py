"""Unit tests for the rubric grader and leaderboard aggregation.

The rubric grader is the component most likely to produce confidently
wrong numbers, because an LLM judge will return a well-formed score for
an ill-formed question. These tests therefore check the failure paths:
judge unavailability, malformed responses, and the self-judging case that
would otherwise quietly invalidate a whole comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamion.bench import _count_modes, classify_failure, summarise
from dynamion.core import (
    FailureMode,
    GraderResult,
    Patch,
    RubricCriterion,
    RubricGrader,
    Score,
    Trajectory,
    TrajectoryEvent,
)
from dynamion.graders.rubric import (
    JudgeUnavailable,
    RubricGraderRunner,
    check_judge_independence,
)

PATCH = Patch(unified_diff="--- a\n+++ b\n+added line\n", files_changed=["x.py"])


def rubric_spec(**kw) -> RubricGrader:
    defaults = {
        "criteria": [
            RubricCriterion(id="clarity", description="Is it clear?", weight=1.0),
            RubricCriterion(id="typing", description="Is it typed?", weight=1.0),
        ],
        "judge_model": "ollama/judge-model",
        "self_consistency_k": 3,
    }
    defaults.update(kw)
    return RubricGrader(**defaults)


class StubJudge(RubricGraderRunner):
    """A grader whose judge is replaced by a scripted callable."""

    def __init__(self, spec, responses, **kw):
        super().__init__("rubric", spec, **kw)
        self._responses = responses
        self.calls = 0

    def _call_judge(self, prompt):
        self.calls += 1
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


# ------------------------------------------------------------ independence


def test_self_judging_is_rejected():
    with pytest.raises(JudgeUnavailable, match="same as the model"):
        check_judge_independence(
            "ollama/qwen2.5-coder:7b", "ollama/qwen2.5-coder:7b"
        )


def test_different_models_are_allowed():
    check_judge_independence("ollama/judge", "ollama/qwen2.5-coder:7b")


def test_provider_prefix_is_ignored_when_comparing():
    """The comparison must not be defeated by a provider prefix."""
    with pytest.raises(JudgeUnavailable):
        check_judge_independence("openai/gpt-4o", "gpt-4o")


# ------------------------------------------------------- self-consistency


def test_scores_are_averaged_across_samples():
    responses = [
        {"scores": {"clarity": 4, "typing": 0}, "reasoning": "a"},
        {"scores": {"clarity": 0, "typing": 0}, "reasoning": "b"},
        {"scores": {"clarity": 4, "typing": 0}, "reasoning": "c"},
    ]
    grader = StubJudge(rubric_spec(), responses)
    result = grader.run(Path("."), PATCH)

    # clarity: (1+0+1)/3 = 0.667 ; typing: 0
    assert result.detail["criteria"]["clarity"] == pytest.approx(2 / 3, abs=1e-6)
    assert result.detail["criteria"]["typing"] == 0.0
    assert result.score == pytest.approx(1 / 3, abs=1e-6)
    assert result.detail["n_samples"] == 3


def test_spread_is_reported():
    """Disagreement between samples is itself a finding."""
    responses = [
        {"scores": {"clarity": 4, "typing": 4}},
        {"scores": {"clarity": 0, "typing": 0}},
        {"scores": {"clarity": 4, "typing": 4}},
    ]
    result = StubJudge(rubric_spec(), responses).run(Path("."), PATCH)
    assert result.detail["spread"]["clarity"] > 0.0


def test_unavailable_judge_raises_rather_than_scoring_zero():
    """A judge that cannot be reached is not the same as a bad attempt."""
    grader = StubJudge(rubric_spec(), [JudgeUnavailable("connection refused")])
    with pytest.raises(JudgeUnavailable, match="no usable samples"):
        grader.run(Path("."), PATCH)


def test_partial_sample_failure_still_scores():
    responses = [
        {"scores": {"clarity": 4, "typing": 4}},
        JudgeUnavailable("timeout"),
        {"scores": {"clarity": 4, "typing": 4}},
    ]
    result = StubJudge(rubric_spec(), responses).run(Path("."), PATCH)
    assert result.detail["n_samples"] == 2
    assert result.score == 1.0
    assert result.error is not None


def test_malformed_json_response_is_survived():
    """Well-formed JSON with no scores is a failed judgement, not a zero.

    Counting it as an empty sample would dilute the real ones and drag
    every criterion toward 0, turning a flaky judge into a confident low
    score.
    """
    responses = [{"not": "scores"}, {"scores": {"clarity": 4, "typing": 4}}]
    result = StubJudge(rubric_spec(self_consistency_k=2), responses).run(Path("."), PATCH)
    assert result.detail["n_samples"] == 1
    assert result.error is not None
    assert result.score == 1.0, "the surviving good sample must not be diluted"


def test_required_criterion_gates_passing():
    spec = rubric_spec(
        criteria=[
            RubricCriterion(id="security", description="safe?", required=True),
            RubricCriterion(id="style", description="nice?"),
        ]
    )
    result = StubJudge(spec, [{"scores": {"security": 0, "style": 4}}]).run(
        Path("."), PATCH
    )
    assert result.detail["required_met"] is False
    assert result.passed is False


def test_prompt_includes_diff_and_anti_gaming_note():
    spec = rubric_spec(anti_gaming_note="Do not reward comments.")
    grader = StubJudge(spec, [{"scores": {"clarity": 0, "typing": 0}}])
    prompt = grader.build_prompt("", PATCH)
    assert "added line" in prompt
    assert "Do not reward comments." in prompt


# ------------------------------------------------------------------- bench


def _traj(**kw) -> Trajectory:
    import datetime as dt

    base = {"task": "t", "agent": "aider:local", "started_at": dt.datetime.now(dt.UTC)}
    base.update(kw)
    return Trajectory(**base)


def _edited_traj():
    """A trajectory where the agent actually changed something."""
    import datetime as dt

    return _traj(
        events=[
            TrajectoryEvent(
                turn=1,
                kind="edit",
                content="store.py",
                timestamp=dt.datetime.now(dt.UTC),
            )
        ]
    )


def _result(score=0.5, passed=False, components=None, stdout="1 failed, 1 passed") -> GraderResult:
    """A plausible failing grader result.

    ``stdout`` defaults to a real pytest summary because the classifier
    treats empty output as "no evidence" and returns UNKNOWN; a test that
    omits it would exercise the fallback rather than the rule it means to.
    """
    if components is None:
        components = [{"stdout": stdout, "error": None, "passed": None, "score": score}]
    return GraderResult(
        grader_id="g",
        type="tests",
        score=score,
        passed=passed,
        detail={"components": components},
    )


def test_classify_passed_is_none():
    assert classify_failure(_result(passed=True), _traj()) is FailureMode.NONE


def test_classify_timeout():
    assert classify_failure(_result(), _traj(exit_reason="timeout")) is FailureMode.TIMEOUT


def test_classify_no_edits_is_gave_up():
    assert classify_failure(_result(), _traj()) is FailureMode.GAVE_UP


def test_classify_environment_fault_precedes_gave_up():
    """Harness breakage must not be reported as the model giving up.

    A run where the environment broke usually has zero edits simply
    because the agent never got a working workspace. Reporting that as
    GAVE_UP blames the model for a failure of our own making.
    """
    components = [{"stdout": "", "error": "grading environment exploded"}]
    assert (
        classify_failure(_result(components=components), _traj())
        is FailureMode.ENVIRONMENT_ISSUE
    )


def test_classify_wrong_api():
    """An exception type means the code did not match the interface."""
    components = [{"stdout": "E   AttributeError: 'Foo' has no attribute 'bar'"}]
    assert (
        classify_failure(_result(components=components), _edited_traj())
        is FailureMode.WRONG_API
    )


def test_classify_wrong_approach_on_total_failure():
    assert (
        classify_failure(_result(score=0.0), _edited_traj()) is FailureMode.WRONG_APPROACH
    )


def test_classify_test_misread_from_patch():
    """Editing the tests to pass is the clearest form of misreading them."""
    patch = Patch(unified_diff="", files_changed=["tests/test_thing.py"])
    assert (
        classify_failure(_result(), _edited_traj(), patch) is FailureMode.TEST_MISREAD
    )


def test_classify_conftest_edit_counts_as_test_misread():
    patch = Patch(unified_diff="", files_changed=["conftest.py"])
    assert (
        classify_failure(_result(), _edited_traj(), patch) is FailureMode.TEST_MISREAD
    )


def test_editing_source_files_is_not_test_misread():
    """Guard against the rule over-firing on ordinary work."""
    patch = Patch(unified_diff="", files_changed=["ledgerpay/store.py"])
    assert (
        classify_failure(_result(score=0.0), _edited_traj(), patch)
        is FailureMode.WRONG_APPROACH
    )


def test_classify_gaming():
    """Skipped tests are the classic way to fake a pass."""
    components = [{"stdout": "1 skipped: pytest.skip used", "error": None}]
    assert (
        classify_failure(_result(components=components), _edited_traj())
        is FailureMode.GRADER_GAMING
    )


def test_classify_unknown_when_no_evidence():
    components = [{"stdout": "", "error": None}]
    assert (
        classify_failure(_result(components=components), _edited_traj())
        is FailureMode.UNKNOWN
    )


def test_every_declared_failure_mode_is_reachable():
    """A category nothing can produce is a dead constant, not a taxonomy.

    Three of the six originally declared modes were never assigned by any
    code path, which made the taxonomy look complete while silently
    reporting UNKNOWN for every real failure.
    """
    produced = {
        classify_failure(_result(passed=True), _traj()),
        classify_failure(_result(), _traj()),
        classify_failure(_result(), _traj(exit_reason="timeout")),
        classify_failure(_result(), _edited_traj()),
        classify_failure(_result(components=[{"stdout": "NameError: x"}]), _edited_traj()),
        classify_failure(
            _result(),
            _edited_traj(),
            Patch(unified_diff="", files_changed=["tests/test_x.py"]),
        ),
        classify_failure(
            _result(components=[{"stdout": "", "error": "environment broken"}]),
            _edited_traj(),
        ),
        classify_failure(
            _result(components=[{"stdout": "1 skipped"}]), _edited_traj()
        ),
        classify_failure(_result(components=[{"stdout": ""}]), _edited_traj()),
    }
    unreachable = set(FailureMode) - produced
    assert not unreachable, f"declared but never assigned: {sorted(unreachable)}"


def test_summarise_ranks_by_resolve_rate():
    scores = [
        Score(task="t1", agent="strong", resolved=True, score=1.0, tokens=10),
        Score(task="t2", agent="strong", resolved=False, score=0.4, tokens=10),
        Score(task="t1", agent="weak", resolved=False, score=0.2, tokens=5),
        Score(task="t2", agent="weak", resolved=False, score=0.1, tokens=5),
    ]
    board = summarise(scores)
    assert board[0]["agent"] == "strong"
    assert board[0]["resolve_rate"] == 0.5
    assert board[1]["resolve_rate"] == 0.0
    assert board[0]["total_tokens"] == 20


def test_summarise_surfaces_verification_status():
    """An unexercised adapter must not be billed as a measurement."""
    scores = [Score(task="t", agent="a", score=0.5)]
    board = summarise(scores, verified={"a": False})
    assert board[0]["adapter_verified"] is False


def test_summarise_counts_failure_modes():
    scores = [
        Score(task="t1", agent="a", resolved=False, score=0.0, failure_mode=FailureMode.GAVE_UP),
        Score(task="t2", agent="a", resolved=False, score=0.0, failure_mode=FailureMode.GAVE_UP),
    ]
    assert _count_modes(scores) == {"gave_up": 2}


def test_cost_per_solve_requires_a_resolve():
    unresolved = Score(task="t", agent="a", resolved=False, cost_usd=1.0, score=0.0)
    assert unresolved.cost_per_solve is None