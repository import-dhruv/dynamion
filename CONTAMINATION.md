# Contamination report

The weakest criticism of public coding benchmarks is that a model may
have already seen a task during pretraining, and that a "solved" score
therefore measures recall rather than capability. This document is the
attempt to make that falsifiable per task rather than a matter of trust.

Read it as a claim you can check, not a reassurance.

## Method

For every task we record, in the task spec itself:

- **`source.kind`** — how the task was produced
- **`source.upstream`** / **`source.commit`** — the repository and exact
  commit, when one exists
- **`source.contamination_checked_at`** — when it was last reviewed

`dynamion validate` warns when a task has no contamination check recorded,
and when a task derived from an upstream repository fails to name it.

## Per-task summary

| Task | Kind | Upstream | Checked | Contamination risk |
|---|---|---|---|---|
| `retry-backoff` | `synthetic_feature` | none | 2026-10-14 | **Low** |
| `idempotent-ledger` | `synthetic_feature` | none | 2026-10-14 | **Low** |
| `legacy-dateutil` | `synthetic_feature` | none | 2026-10-14 | **Low** |

All three are clean-room: authored for this project, with no upstream
repository involved.

That is a real strength and also a real limitation, stated plainly below.

## Why these are low risk

- **No upstream.** There is no published repository whose code, tests or
  commit history a model could have memorised.
- **Bugs are seeded deliberately.** `idempotent-ledger` starts with named
  defects (double-counting, silent key reuse, validation ordering, no
  synchronisation). Those are not a copy of anything; they are an
  argument about what makes an implementation incorrect.
- **Hidden graders are generated after the fact.** `legacy-dateutil`'s
  expectations were produced by *executing the seed implementation*, and
  the seed's SHA-256 is recorded in `behaviour-golden.json`. The grader
  did not exist when the module was written, so it cannot have leaked.

## What this report does not prove

**A low-risk label is not a guarantee.** We cannot inspect any model's
pretraining data. What we can do is remove the obvious routes by which a
task becomes familiar, and say which routes remain open:

- **Prompt text.** A prompt could coincide with published text even when
  the code does not. We have not run a systematic search against large
  web corpora, and we do not claim one.
- **Domain familiarity.** `idempotency`, `retry backoff` and
  `compatibility refactoring` are well-trodden areas. A model may
  recognise the *shape* of the problem. That is arguably the point — the
  tasks are long-horizon *engineering*, not knowledge recall — but it
  does mean familiarity with the domain is not contamination.
- **Shared vocabulary.** The presence of a hidden test asserting
  `parse_duration("abc") is None` does not mean a model has seen it.

## How to check us

- Read `source` in any `tasks/*/task.yaml`.
- For `legacy-dateutil`, verify the seed hash:
  `sha256sum tasks/legacy-dateutil/workspace/legacyutil/__init__.py` and
  compare with `_seed_sha256` in `hidden/behaviour-golden.json`.
- Re-run any task. A hermetic run should reproduce its baseline exactly;
  CI enforces that hermetic and local grading agree.

## Known limitation of the whole approach

A benchmark built from synthetic, clean-room tasks measures a narrower
thing than one built from real repositories. Synthetic tasks are clean
but also *artificial*: real codebases are messier, and the bugs that
matter in production are rarely as cleanly separable as the ones seeded
here.

The honest path is a mix, and real-repo tasks with recorded provenance
are the obvious next step. When those are added they will appear in this
table with their upstream and commit, and with a higher stated risk. A
report claiming uniform low risk across every task would be a report not
worth reading.