"""Domain types for decision signals.

Business logic depends on these types, never on the TypeSafe SDK directly, so the
model behind them can be swapped (Jev, an LLM logprob readout, a classifier, rules)
and unit-tested with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias


class SignalSource(str, Enum):
    JEV = "jev"
    FALLBACK_RULES = "fallback_rules"
    FAKE = "fake"


@dataclass(frozen=True, slots=True)
class ChoiceSignal:
    """Answer to a Choice question: a distribution over unordered options."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float  # vendor statistic derived from probabilities (see docs/confidence)

    @property
    def top_probability(self) -> float:
        return self.probabilities[self.choice]

    @property
    def margin(self) -> float:
        """p(top) - p(runner-up): how clearly the winner beats the second option."""
        ranked = sorted(self.probabilities.values(), reverse=True)
        return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)

    def prob(self, option: str) -> float:
        return self.probabilities.get(option, 0.0)


@dataclass(frozen=True, slots=True)
class ScoreSignal:
    """Answer to a Score question: a distribution over ORDERED levels 0..n-1.

    `expected` is the probability-weighted level the API returns as `score`.
    TypeSafe warns not to interpolate magnitudes from it; prefer the argmax level
    or a tail probability such as P(level >= k).
    """

    expected: float
    probabilities: Mapping[int, float]
    legend: Mapping[int, str]
    confidence: float

    @property
    def most_likely_level(self) -> int:
        return max(self.probabilities, key=lambda k: self.probabilities[k])

    def prob_at_least(self, level: int) -> float:
        return sum(p for k, p in self.probabilities.items() if k >= level)


@dataclass(frozen=True, slots=True)
class NoulSignal:
    """Answer to a Noul (yes/no) question: P(yes). No separate confidence field."""

    p_yes: float

    @property
    def confidence(self) -> float:
        """Distance from 0.5 on a 0..1 scale (|2p - 1|), as suggested in TypeSafe docs."""
        return abs(2 * self.p_yes - 1)


Signal: TypeAlias = ChoiceSignal | ScoreSignal | NoulSignal


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """All answers for one state, plus provenance needed for audit and evaluation."""

    answers: Mapping[str, Signal]
    source: SignalSource
    model: str | None = None  # versioned id that actually answered, e.g. "jev-1.13.0"
    request_id: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def choice(self, key: str) -> ChoiceSignal:
        value = self.answers[key]
        if not isinstance(value, ChoiceSignal):
            raise TypeError(f"answer {key!r} is {type(value).__name__}, not ChoiceSignal")
        return value

    def score(self, key: str) -> ScoreSignal:
        value = self.answers[key]
        if not isinstance(value, ScoreSignal):
            raise TypeError(f"answer {key!r} is {type(value).__name__}, not ScoreSignal")
        return value

    def noul(self, key: str) -> NoulSignal:
        value = self.answers[key]
        if not isinstance(value, NoulSignal):
            raise TypeError(f"answer {key!r} is {type(value).__name__}, not NoulSignal")
        return value


class DecisionUnavailable(Exception):
    """The decision layer could not produce a trustworthy answer.

    `retryable` separates transient failures (timeout, 429, 5xx) from bugs in our
    own request (400/422) that should page an engineer rather than be retried.
    """

    def __init__(self, reason: str, *, retryable: bool, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.retryable = retryable
        self.detail = detail
