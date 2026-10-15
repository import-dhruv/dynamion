# Task Design Handbook

A specification for creating tasks that measure long-horizon software
engineering, written by someone who has shipped three of them and got
several things wrong on the way.

This is the document I would want to read before authoring a task, and
the one I would want a reviewer to read before trusting a number my
harness produced.

---

## 1. What a task is for

A task exists to distinguish two things that a boolean score conflates:
*the model could not do this* and *the model failed in a way we did not
anticipate*. Every design decision below follows from that.

Concretely, a task should let you answer at least one of these after an
agent attempt:

- Did it fail because of a capability limit, or because it misunderstood
  the contract?
- Did it produce the right design and fail to apply it?
- Did it satisfy the grader without doing the work?
- Did the grader measure what it claims to?

A task that cannot answer any of these is a toy, however many tests it
has.

## 2. The realism gate

**Test:** would a competent engineer plausibly be asked to do this in a
real codebase?

A task must involve comprehension of existing code, span more than one
concept, and have a reason to exist beyond producing test output.

**Rejecting signal:** the task can be solved without reading the seed
workspace at all.

| Rejected | Why | Accepted instead |
|---|---|---|
| "Return the index of the first vowel" | No codebase comprehension | "Extract retry policy from the client into a transport-independent component" |
| "Implement binary search" | Nothing to understand | "Make this append-only ledger idempotent under concurrent retry" |
| "Add a `slugify` function" | No existing behaviour to preserve | "Modernise this module to typed Python without changing its output" |

The last row is worth dwelling on. "Add a function" became a hard task
only because the module already had six counterintuitive edge cases that
callers depended on. **Constraint is what makes a task hard** — not size.

## 3. The solvability gate

**Test:** have you solved it yourself, blind, under the same budget?

Record the wall-clock time. That number is the human baseline, and
without it, difficulty is an assertion rather than a measurement.

An unsolvable task is worse than a trivial one, because it looks like
evidence about model capability when it is evidence about your grader.

**Procedure**

1. Set a timer. Do not reset it.
2. Work the task as a colleague would, not as a benchmark author.
3. Record time-to-first-correct-solution and total time.
4. Note every place you nearly gave up, and what would have unblocked you.

Step 4 is the valuable part. It tells you which parts of the prompt are
under-specified, which is the most common defect in published tasks.

## 4. The anti-gaming gate

**Test:** what is the cheapest way to score 1.0 without solving the task?

Then actually try it. A grader you cannot describe an exploit for is not
finished.

### Known exploits and their counters

| Exploit | Countermeasure |
|---|---|
| Rewrite or delete the tests | Hidden tests injected *after* the agent's turn; visible tests are a copy |
| `return True`, `pytest.skip`, `@unittest.skip` | Behavioural assertions a stub cannot satisfy; gaming markers classified as `grader_gaming` |
| Patch the grader's dependencies | Hidden paths never exist in the agent's workspace |
| Detect the test environment and branch | Vary observable inputs; never branch on environment identity |
| Edit `conftest.py` or fixtures | Ship fixtures inside the hidden test file |
| Delete a public symbol to dodge an assertion | Assert the API surface explicitly |
| Delete the module under test entirely | Assert the public API still exists after the diff |

The last two were added after real runs. See §8.

### The separation that matters most

**The agent's turn must not be able to see the grader.**

This is structural, not a matter of grader cleverness. Concretely:

- Hidden test files live in `hidden/` and are copied into the workspace
  **after** the agent finishes.
- The workspace is mounted **read-only** during grading, so an agent with
  write access at grading time cannot edit the tests that judge it.
- Runners never receive the hidden paths, so no adapter can be prompted
  to optimise against them.

If the prompt describes the contract and the grader enforces it, then
satisfying the grader *is* doing the work. That is the property you want.

## 5. The calibration gate

**Test:** do two models of clearly different ability separate on it?

If a weak and a strong model score identically, the task is measuring
nothing — usually because it is too easy, or because the grader keys on a
detail that does not matter.

Weak models are also *diagnostically* useful beyond calibration. A tier
that fails in a distinctive way exposes task properties a strong model
masks. In this harness, `qwen2.5-coder:3b` invented a new file rather
than editing the named one, which is a capability gap a single tier would
never have revealed.

## 6. Writing the prompt

**State the behavioural contract, not the implementation.** A prompt that
names the exact functions to create is testing transcription.

**Enumerate everything that must remain compatible — all of it.** An
early version of one task said "keep the existing public names" and then
listed three of the eight the package re-exported. That reads as
permission to delete the rest, and an agent deleted one, breaking the
package import and making the run unscoreable.

**List edge cases explicitly, including malformed input.** Agents
otherwise stop at the happy path, which is where most real bugs live.

**Say that "improvements" are failures** when the contract is
compatibility. Without that sentence, an agent that sees a bug will fix
it and score worse than one that never looked.

## 7. Writing the grader

### Hidden tests

- **Partial credit over a boolean.** A long-horizon task almost never
  resolves all-or-nothing, and a binary signal discards the information
  that says *how close* an agent got.
- **Derive the count from per-test results**, never from the exit code.
  Exit codes collapse 9-of-10 and 0-of-10 into one bucket.
- **Map hidden files to their destination explicitly**
  (`source=destination`). Copying a hidden test to its source path means
  the command runs a file that does not exist, collects nothing, and
  reports a plausible-looking `0.00`.

### Rubric judges

- `self_consistency_k >= 3`. A single judge call is noisy enough that
  differences between agents are indistinguishable from sampling noise.
- **Anchor every criterion.** `anchor_low` and `anchor_high` say what
  each end of the scale means; without them judges drift and scores stop
  being comparable across runs.
- **The judge must not be the model under test.** A model grading its
  own work rewards its own stylistic habits rather than correctness.
- Treat malformed judge responses as failures, not as empty samples. An
  empty sample drags every criterion toward zero, turning a flaky judge
  into a *confident* low score.

### The rule that outranks all of these

**A grader failure must never be reported as a task failure.**

A missing test runner, a broken plugin, a collection error — none of
these are measurements. Score them as `0.0` and you will quietly
attribute your infrastructure bugs to model capability, and nobody will
notice until the numbers are published.

Every grader in this project raises `GraderEnvironmentError` rather than
returning a score it cannot justify. Twelve bugs of exactly this class
were found and fixed during development; the README lists each one with
the number it produced.

## 8. What went wrong, in practice

Recorded because each was invisible until an agent actually ran.

| Defect | Symptom | Fix |
|---|---|---|
| Hidden tests graded from the wrong path | Every run scored `0.00`; looked like total model failure | `source=destination` mapping |
| Collection error scored as a failing run | A broken import reported as a confident `0/2` | Detect faults *before* counting |
| Output parser matched the first count | A real 4-of-17 run reported as 4-of-5 | Match the last count; the summary prints last |
| Environment fault classified as `GAVE_UP` | Harness breakage blamed on the model | Check faults before behavioural modes |
| Three failure modes unreachable | Taxonomy looked complete; reported `UNKNOWN` always | Reachability test |
| Prompt listed 3 of 8 public names | Agent deleted the rest; package import broke | Enumerate all; assert the API surface |

The last one is the clearest lesson: **a prompt asking for preservation
is not a safeguard. A test is.**

## 9. Recording what a run found

A task ships with more than a spec and a grader. Each task directory
should answer:

- What is the agent's *known* failure mode on this task?
- Which pitfalls were anticipated, and which were not?
- Does the human baseline exist, and at what time?
- Is the baseline reproducible under a different grading mode?

The `expected_pitfalls` field exists to force this. Writing it before the
first run is worth it: it is the difference between "the model failed"
and "the model failed in the third of five ways we predicted."

## 10. Checklist

- [ ] Realism: needs codebase comprehension
- [ ] Solvability: solved blind, timed, recorded
- [ ] Prompt: contract stated, not implementation
- [ ] Prompt: all compatibility constraints enumerated
- [ ] Prompt: "improvements are failures" stated if applicable
- [ ] Hidden tests: exist, and are unreachable during the agent's turn
- [ ] Hidden paths: explicitly mapped to a destination
- [ ] Grader: partial credit from per-test counts
- [ ] Rubric: k ≥ 3, criteria anchored, judge independent
- [ ] Calibration: two model tiers separate, or the reason is recorded
- [ ] Anti-gaming: cheapest exploit tried, and closed
- [ ] Provenance: recorded, contamination checked
- [ ] `dynamion validate` clean except human baseline
- [ ] Failure taxonomy: every predicted mode recorded in `expected_pitfalls`