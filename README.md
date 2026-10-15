# dynamion

**A long-horizon software-engineering evaluation harness for coding agents.**

Coding-agent benchmarks built on single-file algorithmic puzzles do not
measure the capability that matters: reading an unfamiliar codebase,
making a change that spans several files, and holding an implementation
consistent across a long session. `dynamion` is an open-source harness for
measuring that instead.

It provides a task specification format, deterministic environments,
pluggable agent runners, and verifiable graders — the workflow that
frontier labs use internally to build long-horizon coding tasks, packaged
so anyone can reproduce it.

> **Status: v0.1.0.** Three tasks, six agent adapters (one verified),
> local models. See [Limitations](#limitations) — the honest list is
> longer than the feature list at this stage.

---

## Why this exists

When an agent fails a long-horizon task, the interesting question is
*which* failure it was. A model that edits the wrong file, one that
misreads a test, one that gives up after two attempts, and one that
rewrites the test suite to make its own work pass are four different
systems with four different trajectories toward capability. A benchmark
that reports only a single boolean cannot tell them apart, and neither can
the team deciding where to invest next.

`dynamion` treats the failure taxonomy as a first-class output, not a
diagnostic afterthought.

## Design principles

**Depth over breadth.** Ten toy tasks measure less than three real ones.
Tasks here are multi-file, require codebase comprehension, and are built
so that passing the visible tests is insufficient.

**The grader is the hard part.** LLM judges get no credit here without
self-consistency sampling and a documented independence check from the
model under test. A grader that silently measures nothing is worse than
no grader.

**Provenance is mandatory.** Every task records how it was sourced and
when contamination was last checked. See [CONTAMINATION.md](CONTAMINATION.md).

**Infra failures are not task failures.** If the test runner is broken or
the container dies, the harness raises rather than reporting a `0.0`.
Silently converting infrastructure breakage into an agent failure
corrupts a leaderboard in a way nobody notices until the numbers are
published.

## Quickstart

```bash
git clone https://github.com/import-dhruv/dynamion
cd dynamion
pip install -e ".[dev]"

dynamion validate      # check every task spec
dynamion list          # show the task inventory
```

Run a single task against a local model, graded hermetically:

```bash
mkdir -p ~/dynamion-ws && cp -r tasks/retry-backoff/workspace ~/dynamion-ws/run
cd ~/dynamion-ws/run && git init -q && git add -A && \
  git -c user.email=a@b -c user.name=base commit -qm base

dynamion run tasks/retry-backoff/task.yaml \
    --agent aider --model ollama/qwen2.5-coder:7b \
    --env docker --allow-host host.docker.internal \
    --workspace ~/dynamion-ws/run --out results/run.json
```

Drop `--env docker` to grade on the host. The two modes are **not
comparable** — see [Isolation](#isolation) — and every run record states
which one produced it.

> Workspaces must live under `$HOME`. Docker Desktop does not share `/tmp`
> with containers by default.

Using Aider with a local Ollama model requires no API key and no network
access, which makes results reproducible offline.

## Isolation

Grading and the agent's turn get deliberately different treatment.

**Grading is hermetic.** The test command runs with `--network none` and
the workspace mounted **read-only**. Read-only matters as much as offline:
with write access at grading time, an agent could edit the hidden tests
that decide whether it passed, and the resulting score would be
indistinguishable from a real solve.

**The agent's turn is not network-isolated**, because a local model
served over HTTP has to be reachable. `--allow-host` opens a route to a
specific model server and nothing else.

Both claims are verified, not assumed. `tests/test_isolation.py` asserts
that a DNS lookup fails and a write to the workspace raises during
grading; the same tests assert the *opposite* under `--env local`, which
is the control proving the assertions are meaningful.

### Two Docker behaviours worth knowing

- **Setup commands are baked into a derived image.** `docker run --rm`
  discards its filesystem, so a `pip install` during setup is gone before
  grading starts. `prepare()` runs setup in a long-lived container and
  `docker commit`s the result.
- **`${PYTHON}` resolves per environment.** It maps to the host
  interpreter locally and to `python3` in a container. Hard-coding an
  interpreter path in a task spec breaks on one of the two.

## Architecture

```
task.yaml ──► TaskSpec (validated)  ──► Environment
                                    ├─ Docker (hermetic, offline grading)
                                    └─ Local  (no isolation, labelled)
                    │
                    ├──► Runner (aider, …)  run() → Trajectory + Patch
                    └──► Grader              hidden tests + rubric → Score
```

- **`core.py`** — the domain types. Every model is `extra="forbid"`, so a
  typo in a task spec fails loudly instead of silently dropping a field.
- **`tasks.py`** — spec loading and validation.
- **`env/`** — execution environments. `LocalEnvironment` and
  `DockerEnvironment` record `hermetic` in every run, because a score
  obtained without isolation measures something different.
- **`runners/`** — agent adapters behind one interface. Adapters never
  receive the grader's hidden paths. Six adapters ship; only Aider has
  been executed end to end here, and the leaderboard labels the rest
  unverified rather than presenting them as measurements.
- **`graders/`** — test and rubric graders. Graders see only the finished
  artefact, never the agent, so grading is reproducible offline.
- **`bench.py`** — matrix execution and leaderboard aggregation.

## Current tasks

| Task | Domain | Graders | Test baseline | Agent (qwen2.5-coder:7b) |
|---|---|---|---|---|
| `retry-backoff` | API design, error handling, typing | 17 tests (0.7) + rubric (0.3) | 4/17 → 0.17 | 5/17 → 0.21 |
| `idempotent-ledger` | Concurrency, atomicity, invariants | 16 tests (0.75) + rubric (0.25) | 9/16 → 0.42 | 9/16 → 0.42 |
| `legacy-dateutil` | Compatibility refactor, property-based testing | 47 tests (0.75) + rubric (0.25) | 47/47 → 0.75 | 47/47 → 0.75 |

`legacy-dateutil` seeds at 100% by design: its hidden expectations were
generated by executing the seed module, so the task is scored entirely on
whether the agent changes behaviour it was told to preserve. Four obvious
"improvements" — raising on unparseable input, returning a placeholder
slug, lowercasing token interiors, slicing to a negative width — take it
from 47/47 to 35/47.

## Difficulty calibration: currently failing

Running the deliberately weaker model tier (`qwen2.5-coder:3b`) against
`retry-backoff`, the two tiers **do not separate on score**:

| Model | Hidden tests | Score | Edits | Files actually touched |
|---|---|---|---|---|
| baseline (no agent) | 4/17 | 0.17 | 0 | — |
| `qwen2.5-coder:3b` | 5/17 | 0.21 | 0 | a new file with literal backticks in its name |
| `qwen2.5-coder:7b` | 5/17 | 0.21 | 1 | a new file at the repo root |

A 7B model and a 3B model score identically on a task labelled `medium`.
Under the review bar in the handbook, that means the task is not currently
demonstrating that it measures anything.

The trajectories separate clearly — the 7B produced a plausible
implementation, the 3B produced none — so the *capability* difference is
visible and the *score* is not. Two likely causes, neither yet tested:

1. **Partial credit floors too low.** Four tests pass on the untouched
   seed, so any agent that does not regress the module starts near the
   baseline. A task where doing nothing scores 17% cannot discriminate
   between doing a little and doing a lot.
2. **The file-targeting failure dominates.** Because agents are not
   editing the intended file, the score is measuring "did you avoid
   regressing the seed", not "did you solve the task".

Both are fixable and both are task-design problems, not model problems.
The honest summary: **three tasks, one model, and a calibration failure —
this benchmark is not yet ready to rank agents.**

## The one finding three tasks agree on

All three runs, same model, same configuration:

| Task | What it produced | Where it put it |
|---|---|---|
| `retry-backoff` | a transport-independent retry policy | new `retry_policy.py`, never wired into `FluxClient.request` |
| `idempotent-ledger` | a ledger with a `threading.Lock` | new `store.py` at the repo root, `ledgerpay/store.py` untouched |
| `legacy-dateutil` | a modernised, typed module | new `legacyutil.py` at the repo root, `legacyutil/__init__.py` untouched |

**Three tasks, three occurrences.** In every case the model wrote
structurally reasonable code and in every case it wrote it in the wrong
place. On two of the three the score was *exactly the baseline* — a pass
worth as much as doing nothing — while the trajectory contained work that
looked like a solution.

This is the argument for keeping trajectories, and it is a finding about
**file targeting** rather than about engineering capability. A leaderboard
shows `0.42` next to a baseline `0.42` and reads as a null result; the
trajectory shows a capable model losing points to a mechanical habit.

It is also the cheapest improvement available: an agent configuration that
names the target file would likely convert two of these null results into
partial credit. That experiment has not been run yet.

Caveat on the strength of this claim: three tasks, one model, one
configuration is a strong signal and not yet a result. It says the
benchmark is currently measuring something narrower than it intends to,
which is worth fixing before adding more tasks.

## Grader bugs found by running it

Every one of these was found by executing the harness, not by reading it.
All are pinned by regression tests. They are also the most credible part
of the project, because a grader that measures something other than what
it claims to measure is the worst failure mode a benchmark has.

| Bug | What it produced | Now |
|---|---|---|
| Hidden tests graded from the wrong path | every run collected 0 tests, reported `0.00` | `source=destination` mapping; `no tests ran` raises |
| Transport-independence test guessed module names | failed a correct solution for its filename | searches for the property the prompt states |
| `python3` hard-coded in task specs | broke on a venv with a bad plugin | `${PYTHON}`, resolved per environment |
| `docker run --rm` discarded setup | `pip install` vanished; `No module named pytest` | setup committed into a derived image |
| Case-sensitive summary parser | a fully-failing run parsed as `0/0` | case-insensitive, with an all-failed branch |
| Parser matched the *first* count | a real 4-of-17 run reported as 4-of-5 | last match; the summary is printed last |
| `classify_failure` called `.get()` on a model | crashed every single-grader run | components normalised to dicts |
| Environment fault checked after "gave up" | harness breakage blamed on the model | faults checked first |
| Collection errors counted as failed tests | broken import reported as a confident `0/2` | faults checked before counting |
| Malformed judge response counted as an empty sample | flaky judge read as a confident low score | malformed samples excluded |
| Three failure modes unreachable | taxonomy looked complete, reported `UNKNOWN` always | reachability test fails if any mode dies |
| Task loader matched any YAML/JSON | a golden fixture broke the whole load | explicit `task.yaml` naming |

## Limitations

- **Three tasks, one model tier, calibration failing.** A 7B and a 3B
  model score identically on the only task where both have been run. See
  the calibration section above. No human baselines either;
  `dynamion validate` flags them.
- **Base images are digest-pinned**, and `dynamion validate` warns when
  they are not. A task may alternatively supply a Dockerfile, built from a
  digest-pinned base.
- **The agent's turn is not network-restricted.** A local model server has
  to be reachable, so the agent could in principle fetch a reference
  implementation. Egress filtering during the agent turn needs a proxy.
- **Trajectory token counts are unreliable for local models**, which
  report no usage. Cost-per-solve is therefore unavailable locally.
- **Five of six adapters are unverified.** Written against documented CLI
  surfaces, not executed here.
- **Synthetic tasks are clean but artificial.** Real repositories are
  messier. See [CONTAMINATION.md](CONTAMINATION.md) for what this costs.

## Documentation

| Document | What it is for |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Task-authoring review bar and harness contribution |
| [docs/task-design-handbook.md](docs/task-design-handbook.md) | The specification for creating tasks; reads as a design document, not a checklist |
| [docs/datasheet.md](docs/datasheet.md) | Composition, intended use, and limitations, in datasheet form |
| [docs/grader-design-notes.md](docs/grader-design-notes.md) | What each grader measures and which attacks were attempted against it |
| [CONTAMINATION.md](CONTAMINATION.md) | Provenance per task, and what the claim does not cover |

## License

Apache-2.0.