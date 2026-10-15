"""Unit tests for the test grader: the reward oracle.

The grader turns one attempt into one number that ends up on a
leaderboard. Every case below is a way that number can come out wrong,
and most of them fail *silently* -- a grader that scores 0.0 when it
should score 0.7 looks exactly like a model failure.

The two distinctions that matter most and are easy to get backwards:

* an infrastructure fault is not a task failure
* partial credit must be derived from per-test counts, not exit codes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynamion.core import Patch, TestGrader
from dynamion.graders.tests import (
    GraderEnvironmentError,
    TestGraderRunner,
    _count_results,
    _detect_environment_fault,
)

PATCH = Patch(unified_diff="")


def make_workspace(tmp_path: Path, body: str, name: str = "mod_under_test.py") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def spec_for(tmp_path: Path, *, pattern: str | None = r"(\d+) passed", **kw) -> TestGrader:
    # The interpreter is the ${PYTHON} placeholder, exactly as authored
    # task specs do it, so resolution is exercised on every run.
    return TestGrader(
        command=["${PYTHON}", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        timeout_seconds=120,
        result_pattern=pattern,
        **kw,
    )


# ------------------------------------------------------------------ parsing


def test_count_results_from_pytest_summary():
    passed, total = _count_results("3 passed, 2 failed in 0.12s", r"(\d+) passed")
    assert (passed, total) == (3, 5)


def test_count_results_all_passing():
    assert _count_results("7 passed in 0.01s", r"(\d+) passed") == (7, 7)


def test_count_results_all_failing():
    """`-q` prints lowercase "failed" and never mentions "passed"."""
    assert _count_results("4 failed in 0.01s", r"(\d+) passed") == (0, 4)


def test_count_results_does_not_mistake_assertion_output_for_a_summary():
    """Regression guard: a real 4-of-17 run was reported as 4-of-5.

    An assertion line ending in a digit, immediately followed by pytest's
    ``FAILED`` line, satisfies ``(\\d+)\\s+failed``. Taking the first
    match therefore invented a count of 1 and reported 4/5 -- a
    plausible-looking score that was simply wrong.
    """
    output = (
        "FAILED tests/test_retry.py::test_max_attempts_one_disables_retry"
        " - assert 2 == 1\n"
        "FAILED tests/test_retry.py::test_policy_usable_without_http_client"
        " - AssertionError\n"
        "13 failed, 4 passed in 0.07s\n"
    )
    assert _count_results(output, r"(\d+) passed") == (4, 17)


def test_count_results_falls_back_to_outcome_lines():
    output = (
        "PASSED tests/test_a.py::test_one\n"
        "PASSED tests/test_a.py::test_two\n"
        "FAILED tests/test_a.py::test_three\n"
    )
    assert _count_results(output, pattern=None) == (2, 3)


def test_count_returns_zero_when_nothing_parsed():
    assert _count_results("total garbage", pattern=r"(\d+) passed") == (0, 0)


# ------------------------------------------------------ fault vs. failure


@pytest.mark.parametrize(
    "output",
    [
        "ModuleNotFoundError: No module named 'pytest'",
        "PluginValidationError: Plugin 'tavern' for hook",
        "ERROR collecting tests/test_x.py",
        "no tests ran in 0.00s",
        "bash: pytest: command not found",
    ],
)
def test_detects_infrastructure_faults(output):
    assert _detect_environment_fault(output) is not None


def test_normal_pytest_output_is_not_a_fault():
    """A genuinely failing run must not be mistaken for a broken one."""
    assert _detect_environment_fault("3 failed, 2 passed in 0.10s") is None


def test_collection_error_raises_rather_than_scoring_zero(tmp_path):
    """An ImportError must not be scored as a failing test run.

    Regression guard, and the most important case in this file. A broken
    import makes pytest print "ERROR collecting ...", and the per-line
    counting fallback treats those ERROR lines as failed tests. Because
    that yields a non-zero total, an earlier version of the fault check
    -- which only ran when counting returned zero -- let the run through
    and reported a confident 0/2.

    Real instance: an agent removed ``AccountClosed`` from the package,
    the hidden test module failed to import, and the harness attributed
    the resulting collection error to the model's inability rather than
    raising. Scores below the task baseline should be a prompt to ask
    whether the grader ran at all.
    """
    workdir = make_workspace(tmp_path, "x = 1\n")
    (workdir / "tests").mkdir()
    (workdir / "tests" / "test_broken.py").write_text(
        "from nonexistent_module import thing\n\ndef test_never_runs():\n    assert True\n"
    )

    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    with pytest.raises(GraderEnvironmentError, match="could not run tests"):
        runner.run(workdir, PATCH)


def test_agent_breaking_a_public_import_is_not_scored_as_failure(tmp_path):
    """The same fault, reached the way it actually happened.

    The agent edited a package's ``__init__`` such that a name the hidden
    tests import no longer exists. Nothing about that is a measurement of
    the task; the grader never ran.
    """
    workdir = make_workspace(tmp_path, "x = 1\n")
    (workdir / "pkg").mkdir()
    (workdir / "pkg" / "__init__.py").write_text("from .mod import REMOVED_NAME\n")
    (workdir / "pkg" / "mod.py").write_text("# agent deleted the symbol\n")
    (workdir / "tests").mkdir()
    (workdir / "tests" / "test_pkg.py").write_text(
        "from pkg import REMOVED_NAME\n\ndef test_x():\n    assert True\n"
    )

    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    with pytest.raises(GraderEnvironmentError):
        runner.run(workdir, PATCH)


# ------------------------------------------------------- missing test file


def test_missing_hidden_test_raises_not_zero(tmp_path):
    source_root = tmp_path / "src"
    (source_root / "hidden").mkdir(parents=True)
    workdir = make_workspace(tmp_path / "ws", "x = 1\n")

    runner = TestGraderRunner("hidden", spec_for(tmp_path), source_root=source_root)
    runner.spec = runner.spec.model_copy(
        update={"hidden_paths": ["hidden/does_not_exist.py"]}
    )

    with pytest.raises(GraderEnvironmentError, match="hidden test missing"):
        runner.run(workdir, PATCH)


def test_hidden_paths_without_source_root_raises(tmp_path):
    runner = TestGraderRunner("hidden", spec_for(tmp_path), source_root=None)
    runner.spec = runner.spec.model_copy(update={"hidden_paths": ["hidden/x.py"]})
    with pytest.raises(GraderEnvironmentError, match="no source_root"):
        runner.run(make_workspace(tmp_path / "ws", "x = 1\n"), PATCH)


# ------------------------------------------------------------- injection


def test_hidden_test_is_injected_after_the_agent_turn(tmp_path):
    source_root = tmp_path / "src"
    (source_root / "hidden").mkdir(parents=True)
    (source_root / "hidden" / "test_hidden.py").write_text("def test_x(): pass\n")

    workdir = make_workspace(tmp_path / "ws", "x = 1\n")
    assert not (workdir / "tests").exists(), "workspace must not contain tests yet"

    runner = TestGraderRunner("hidden", spec_for(tmp_path), source_root=source_root)
    runner.spec = runner.spec.model_copy(
        update={"hidden_paths": ["hidden/test_hidden.py=tests/test_hidden.py"]}
    )

    runner.inject_hidden_tests(workdir)
    assert (workdir / "tests" / "test_hidden.py").exists()


def test_explicit_destination_mapping_is_honoured(tmp_path):
    """A hidden test must land where the grader command expects it.

    Regression guard: mapping every hidden test to its source path meant
    the command ran against a file that did not exist, every run collected
    zero tests, and the harness reported a plausible-looking 0.00.
    """
    source_root = tmp_path / "src"
    (source_root / "hidden").mkdir(parents=True)
    (source_root / "hidden" / "test_hidden.py").write_text(
        "def test_ok():\n    assert True\n"
    )

    workdir = make_workspace(tmp_path / "ws", "x = 1\n")
    runner = TestGraderRunner("hidden", spec_for(tmp_path), source_root=source_root)
    runner.spec = runner.spec.model_copy(
        update={"hidden_paths": ["hidden/test_hidden.py=tests/test_hidden.py"]}
    )
    runner.inject_hidden_tests(workdir)

    result = runner.run(workdir, PATCH)
    assert result.score == 1.0, "correctly-placed hidden test should pass"
    assert result.detail["passed"] == 1


# --------------------------------------------------------- partial credit


def _grading_workspace(tmp_path: Path) -> Path:
    """A workspace with 4 tests, 3 of which pass against the code."""
    workdir = make_workspace(
        tmp_path / "ws",
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a + b\n\n"
        "def mul(a, b):\n    return a * b\n\n"
        "def div(a, b):\n    return a // b\n",
    )
    (workdir / "tests").mkdir()
    (workdir / "tests" / "test_calc.py").write_text(
        "from mod_under_test import add, sub, mul, div\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n"
        "def test_sub():\n    assert sub(3, 2) == 1\n"
        "def test_mul():\n    assert mul(2, 3) == 6\n"
        "def test_div():\n    assert div(6, 2) == 3\n"
    )
    return workdir


def test_partial_credit_is_fractional(tmp_path):
    workdir = _grading_workspace(tmp_path)
    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    result = runner.run(workdir, PATCH)
    # add, mul and div are correct; sub is wrong.
    assert result.detail["passed"] == 3
    assert result.detail["total"] == 4
    assert result.score == pytest.approx(0.75), (
        "a partly-finished fix must not collapse to the same score as no attempt"
    )
    assert result.passed is False


def test_partial_credit_can_be_disabled(tmp_path):
    workdir = _grading_workspace(tmp_path)
    runner = TestGraderRunner("hidden", spec_for(tmp_path, partial_credit=False))
    result = runner.run(workdir, PATCH)
    assert result.score == 0.0
    assert result.passed is False


def test_fully_correct_scores_one(tmp_path):
    workdir = _grading_workspace(tmp_path)
    (workdir / "mod_under_test.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n\n"
        "def mul(a, b):\n    return a * b\n\n"
        "def div(a, b):\n    return a // b\n"
    )
    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    result = runner.run(workdir, PATCH)
    assert result.score == 1.0
    assert result.passed is True


def test_no_attempt_scores_zero(tmp_path):
    workdir = make_workspace(tmp_path / "ws", "x = 1\n")
    (workdir / "tests").mkdir()
    (workdir / "tests" / "test_x.py").write_text("def test_x():\n    assert False\n")
    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    result = runner.run(workdir, PATCH)
    assert result.score == 0.0


# ------------------------------------------------------------ interpreter


def test_python_placeholder_resolves_to_host_interpreter(tmp_path):
    import sys

    runner = TestGraderRunner("hidden", spec_for(tmp_path))
    cmd = runner._resolve_command()
    assert cmd[0] == sys.executable, "local mode must use the host interpreter"
    assert "${PYTHON}" not in " ".join(cmd)


def test_python_placeholder_follows_the_environment(tmp_path):
    """Inside a container the host interpreter path does not exist."""

    class FakeEnv:
        python = "python3"
        hermetic = True

        def run_grading(self, *a, **k):  # pragma: no cover - not exercised
            raise AssertionError

    runner = TestGraderRunner("hidden", spec_for(tmp_path), environment=FakeEnv())
    assert runner._resolve_command()[0] == "python3"