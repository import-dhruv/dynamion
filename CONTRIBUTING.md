# Contributing

This project has two kinds of contributor: people who improve the
harness, and people who author tasks. Task authoring carries most of the
weight, because a weak task makes every number the harness produces
meaningless.

## Task review bar

A task is merged only after all four gates below are cleared. The gates
exist to catch the four most common reasons a benchmark stops being
believable.

### 1. Realism check

**Question:** would a competent engineer plausibly be asked to do this in
a real codebase?

A task must involve comprehension of existing code, span more than one
file or concept, and have a reason to exist beyond producing test output.
"If a string contains these characters, return the index" is not a task.
"Make this payment ledger idempotent under retry" is.

**Rejecting signal:** the task can be solved without reading the seed
workspace at all.

### 2. Solvability proof

**Question:** have you solved it yourself, blind, under the same budget?

Record the wall-clock time. That number is the human baseline, and without
it difficulty calibration is guesswork — "hard" is not a property you can
assert, it is one you can only compare against a human.

An unsolvable task is worse than a trivial one, because it looks like
evidence about model capability when it is evidence about your grader.

### 3. Anti-gaming review

**Question:** what is the cheapest way to score 1.0 without solving the
task?

Try it. Then close the hole. The standard exploits are:

| Exploit | Countermeasure |
|---|---|
| Rewrite or delete the tests | Hidden tests are injected *after* the agent's turn; the visible tests are a copy |
| `return True`, `pytest.skip`, `@unittest.skip` | Behavioural assertions that a stub cannot satisfy |
| Patch the grader's own dependencies | Hidden paths never exist in the agent's workspace |
| Detect the test environment and branch | Vary observable inputs; never branch on environment identity |
| Edit `conftest.py` or fixtures | Ship fixtures inside the hidden test file |
| Delete a public symbol to dodge an assertion | An API-surface test asserting every re-exported name still exists |

The last row was added after an agent rewrote a module from scratch and
dropped four of eight exported names, breaking the package import.

### 4. Difficulty calibration

**Question:** do two models of clearly different ability separate on it?

Run at least a weak and a strong model. If both score identically, the
task is measuring nothing — most often because it is too easy, or because
the grader keys on a detail that does not matter.

Partial credit is preferred over binary scoring. A long-horizon task
almost never resolves all-or-nothing, and a binary signal discards most of
the information about *how close* an agent got.

## Authoring a task

```
tasks/<task-id>/
├── task.yaml            # the spec
├── workspace/           # the agent's view: seed code + visible tests
└── hidden/              # injected only at grading time
```

```bash
dynamion validate      # schema + quality warnings
dynamion list
```

### Grader rules

- **Test graders must have `hidden_paths`.** A grader whose tests the
  agent can read measures compliance, not capability.
- **Map hidden tests explicitly.** Use `source=destination` so a hidden
  test lands beside the visible tests. Copying it to its source path
  instead means the command runs a file that does not exist, collects
  nothing, and reports a plausible-looking `0.00`.
- **Rubric graders need `self_consistency_k >= 3`.** Single-pass judging is
  too noisy to distinguish agents.
- **The judge model must not be the model under evaluation.** Enforced by
  `check_judge_independence`.
- **Every criterion needs anchors.** `anchor_low` and `anchor_high` say
  what each end of the scale means. Without them judges drift and scores
  stop being comparable across runs.
- **Never let infrastructure failure score as task failure.** A broken
  runner is `GraderEnvironmentError`, not `0.0`.
- **Do not hard-code an interpreter.** Use `${PYTHON}`; it resolves to
  the host interpreter locally and `python3` inside a container.

### Prompt rules

State the behavioural contract, not the implementation. A prompt that
names the exact functions to create is testing transcription.

Say what must remain compatible, and enumerate *all* of it. An earlier
version of one task listed three of the eight names a package re-exported,
which reads as permission to delete the rest.

List the edge cases explicitly, including malformed input. Agents
otherwise stop at the happy path, which is where a lot of real bugs live.

## Generating expectations from a seed

When the task is a refactor whose contract is "do not change behaviour",
generate the expectations by executing the seed rather than by reading it:

```python
# Record every public function's behaviour on a fixed input set.
golden = {f"{name}{args}": fn(*args) for name, args in inputs}
golden["_seed_sha256"] = hashlib.sha256(seed_source).hexdigest()[:16]
```

The recorded hash makes drift detectable: if the seed is edited without
regenerating, the hash no longer matches and the mismatch is a deliberate
decision rather than an accident.

`tasks/legacy-dateutil` does this and the hidden tests are generated from
the recording. Four obvious "improvements" take it from 47/47 to 35/47.

## Bugs in the harness

Bug reports about the harness are as welcome as task reports, and a
specific class of them is unusually valuable:

**A grader that scored something incorrectly.** If a broken environment,
a flaky test, or an unreliable race check produced a number that looked
like a model result, that is the worst possible failure mode. Please open
an issue with the task, the run record, and what you expected.

Twelve such bugs are listed in the README with a regression test each.

## Honesty requirements

Every task records `source.provenance`. Be accurate about how it was
made. If the code came from a real repository, say which one and at which
commit; if the bugs were seeded deliberately, say that. If a task turned
out to be easier than intended, say so in `expected_pitfalls` rather than
quietly moving it to a different difficulty label.

An overstated benchmark is worth less than no benchmark, because it costs
reviewers time and misleads the teams building on top of it.