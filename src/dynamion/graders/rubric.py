"""Rubric grader: LLM-as-judge over explicit criteria.

Self-consistency (``k`` independent samples, reported as mean with the
spread) rather than a single pass. A single judge call is noisy enough
that differences between agents are not distinguishable from sampling
noise, which makes an entire benchmark worthless.

Judge independence: the judge model must not be the model under test.
A model grading its own work systematically rewards its own stylistic
habits rather than correctness. :func:`check_judge_independence` makes
that violation explicit instead of silent.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from ..core import GraderResult, Patch, RubricGrader
from .base import Grader

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class JudgeUnavailable(RuntimeError):
    """The judge model could not be reached."""


def check_judge_independence(judge_model: str, evaluated_model: str) -> None:
    """Raise if the judge is the same model it is judging."""

    def strip(provider: str) -> str:
        return provider.split("/")[-1].strip()

    if strip(judge_model) == strip(evaluated_model):
        raise JudgeUnavailable(
            f"judge model {judge_model!r} is the same as the model under evaluation; "
            "self-judging is not a valid signal"
        )


class RubricGraderRunner(Grader):
    """Score a diff against weighted, anchored criteria."""

    name = "rubric"

    def __init__(self, grader_id: str, spec: RubricGrader) -> None:
        self._grader_id = grader_id
        self.spec = spec

    @property
    def grader_id(self) -> str:
        return self._grader_id

    def build_prompt(self, task_prompt: str, patch: Patch) -> str:
        criteria_block = "\n".join(
            f"- {c.id}: {c.description}\n"
            f"    low ({c.anchor_low}) .. high ({c.anchor_high}) "
            f"[weight={c.weight:g}, required={str(c.required).lower()}]"
            for c in self.spec.criteria
        )
        anti_gaming = (
            f"\n{self.spec.anti_gaming_note}\n" if self.spec.anti_gaming_note else "\n"
        )

        return (
            "You are grading a software-engineering attempt against a rubric.\n"
            "Judge only what the diff demonstrates. Do not reward intent, comments,\n"
            "or claims about work that is not present in the diff.\n"
            f"{anti_gaming}"
            f"TASK:\n{task_prompt}\n\n"
            f"DIFF:\n{patch.unified_diff or '(no changes)'}\n\n"
            f"RUBRIC CRITERIA:\n{criteria_block}\n\n"
            'Reply with JSON only, of the form:\n'
            '{"scores": {"<criterion_id>": <int 0-4>}, "reasoning": "<one paragraph>"}\n'
            "Use 0-4 per criterion. Omit nothing."
        )

    def _call_judge(self, prompt: str) -> dict:
        """Send one judging request. Overridden in tests."""
        import urllib.error
        import urllib.request

        base = self.spec.judge_base_url or "http://localhost:11434"
        model = self.spec.judge_model.split("/")[-1]

        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.7},
            }
        ).encode()

        req = urllib.request.Request(
            f"{base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JudgeUnavailable(f"judge {model} unreachable at {base}: {exc}") from exc

        raw = body.get("response", "")
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise JudgeUnavailable(f"judge returned no JSON: {raw[:200]!r}")
        return json.loads(match.group(0))

    def run(self, workdir: Path, patch: Patch) -> GraderResult:
        prompt = self.build_prompt("", patch)

        samples: list[dict[str, int]] = []
        reasoning: list[str] = []
        errors: list[str] = []

        for _ in range(self.spec.self_consistency_k):
            try:
                parsed = self._call_judge(prompt)
                raw_scores = parsed.get("scores") or {}
                if not raw_scores:
                    # Well-formed JSON with no scores is still a failed
                    # judgement. Counting it as an empty sample would
                    # dilute the real ones and pull every criterion to 0.
                    raise ValueError("judge response contained no scores")
                samples.append({k: int(v) for k, v in raw_scores.items()})
                if parsed.get("reasoning"):
                    reasoning.append(str(parsed["reasoning"]))
            except JudgeUnavailable as exc:
                errors.append(str(exc))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"malformed judge response: {exc}")

        if not samples:
            raise JudgeUnavailable(
                f"rubric grader {self.grader_id!r} produced no usable samples: {errors[:3]}"
            )

        criteria_scores: dict[str, float] = {}
        criteria_spread: dict[str, float] = {}
        for criterion in self.spec.criteria:
            values = [s[criterion.id] / 4.0 for s in samples if criterion.id in s]
            if values:
                criteria_scores[criterion.id] = statistics.fmean(values)
                criteria_spread[criterion.id] = statistics.pstdev(values)

        total_weight = sum(
            c.weight for c in self.spec.criteria if c.id in criteria_scores
        )
        score = (
            sum(
                criteria_scores[c.id] * c.weight
                for c in self.spec.criteria
                if c.id in criteria_scores
            )
            / total_weight
            if total_weight > 0
            else 0.0
        )

        required_ids = {c.id for c in self.spec.criteria if c.required}
        required_met = required_ids.issubset(
            {cid for cid, v in criteria_scores.items() if v > 0}
        )

        return GraderResult(
            grader_id=self.grader_id,
            type="rubric",
            score=score,
            weight=self.spec.weight,
            passed=required_met and score >= 0.999,
            error="; ".join(errors) if errors else None,
            detail={
                "criteria": criteria_scores,
                "spread": criteria_spread,
                "n_samples": len(samples),
                "required_met": required_met,
                "reasoning": reasoning,
            },
        )


__all__ = ["RubricGraderRunner", "JudgeUnavailable", "check_judge_independence"]
