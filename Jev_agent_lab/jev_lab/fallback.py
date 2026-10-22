"""Rule-based fallback used when the decision service is unavailable.

The fallback is deliberately conservative: it produces LOW-confidence signals so the
routing policy sends almost everything to human review, except unmistakable keyword
hits for high-risk categories (which are escalated, never auto-closed).
"""

from __future__ import annotations

import re
from typing import Mapping

from .decisions import DecisionResult, NoulSignal, SignalSource
from .providers import Questions, State, choice_from_scores, score_from_scores

_SECURITY = re.compile(r"\b(hack(ed)?|breach|leak(ed)?|phishing|unauthori[sz]ed|stolen|bị hack|lộ dữ liệu)\b", re.I)
_LEGAL = re.compile(r"\b(lawyer|lawsuit|sue|legal action|luật sư|kiện)\b", re.I)


def _text(state: State) -> str:
    if isinstance(state, str):
        return state
    return str(state)


class RuleFallbackProvider:
    """Keyword rules shaped like DecisionResult, tagged source=FALLBACK_RULES."""

    async def decide(self, state: State, questions: Questions) -> DecisionResult:
        text = _text(state)
        answers: dict = {}
        for key, spec in questions.items():
            kind = spec["type"]
            if kind == "choice":
                options = list(spec["criteria"])
                # Uniform distribution => confidence 0 => policy will not auto-act.
                answers[key] = choice_from_scores({o: 1.0 for o in options})
            elif kind == "score":
                levels = list(spec["criteria"])
                answers[key] = score_from_scores([1.0] * len(levels), [str(lv) for lv in levels])
            elif kind == "noul":
                answers[key] = NoulSignal(self._noul(key, text))
        return DecisionResult(answers=answers, source=SignalSource.FALLBACK_RULES, model="rules-v1")

    @staticmethod
    def _noul(key: str, text: str) -> float:
        if "security" in key and _SECURITY.search(text):
            return 0.9
        if "legal" in key and _LEGAL.search(text):
            return 0.9
        # 0.0 means "no rule fired", NOT "model says no". Returning 0.5 ("unknown")
        # would trip every low-threshold, high-recall gate and escalate all traffic.
        # Safe only because routing sends every fallback result to human review.
        return 0.0


async def decide_with_fallback(
    primary, fallback, state: State, questions: Questions
) -> tuple[DecisionResult, str | None]:
    """Try the primary provider; on DecisionUnavailable use the fallback.

    Returns (result, failure_reason). The reason is surfaced so callers can log it and
    so dashboards can track the fallback rate per reason.
    """
    from .decisions import DecisionUnavailable

    try:
        return await primary.decide(state, questions), None
    except DecisionUnavailable as exc:
        return await fallback.decide(state, questions), exc.reason
