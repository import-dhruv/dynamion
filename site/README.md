# Leaderboard

*This page is regenerated from `results/leaderboard.json` by CI. Do not
edit it by hand — edit the generator.*

A row is one agent evaluated over the full task set. Scores come from
hidden tests graded hermetically (`--network none`, workspace mounted
read-only), so a run cannot depend on anything outside its container.

## How to read this

- **Resolved** — tasks where every hidden test passed. Partial credit is
  reported separately as **Score**.
- **Score** — mean fractional credit across tasks. A long-horizon task
  almost never resolves all-or-nothing, and a binary signal discards the
  part that says *how close* an agent got.
- **Verified** — whether the adapter was actually executed against the
  agent's real CLI. Unverified rows ship so the harness is extensible,
  but their numbers are not measurements.
- **Hermetic** — grading ran with no network and a read-only workspace.
  A non-hermetic row is not comparable with a hermetic one.

## Caveats we consider load-bearing

- Three tasks, one model tier. This shows the pipeline works. It does not
  establish anything about model capability.
- Rubric subscores come from a small local judge and are excluded from
  Score; they are too noisy to rank agents on.
- Some reported failures are the grader's fault, not the agent's. Every
  such bug we have found is listed in the repository README with a
  regression test pinning it.

## Regenerating

```bash
dynamion bench --tasks-dir tasks --results-dir results
python -m dynamion.site --results results/leaderboard.json --out site
```
