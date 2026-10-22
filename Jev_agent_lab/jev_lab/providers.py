"""Decision providers: the only module that talks to the TypeSafe SDK.

Questions are passed in the wire format documented in the OpenAPI spec
(https://api.typesafe.ai/openapi.json), e.g.

    {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}}
    {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}
    {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}

The Python SDK accepts these plain dicts directly, which keeps business logic free of
SDK imports.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .config import JevSettings
from .decisions import (
    ChoiceSignal,
    DecisionResult,
    DecisionUnavailable,
    NoulSignal,
    ScoreSignal,
    Signal,
    SignalSource,
)

log = logging.getLogger("jev_lab.provider")

QuestionSpec = Mapping[str, Any]
Questions = Mapping[str, QuestionSpec]
State = str | Mapping[str, Any] | list[Any]


@runtime_checkable
class DecisionProvider(Protocol):
    async def decide(self, state: State, questions: Questions) -> DecisionResult: ...


# --------------------------------------------------------------------------- Jev


class JevDecisionProvider:
    """Adapter around typesafe_sdk.AsyncTypeSafeClient (verified against SDK 0.7.5).

    Responsibilities: timeouts, bounded retries, error classification, response
    sanity checks, and provenance logging. It does NOT make business decisions.
    """

    def __init__(self, settings: JevSettings, client: Any | None = None) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy  # lazy: tests need no key

        self._settings = settings
        self._client = client or AsyncTypeSafeClient(
            model=settings.model,
            timeout=settings.request_timeout_s,
            retry=RetryPolicy(
                max_retries=settings.max_retries,
                timeout=settings.retry_budget_s,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "JevDecisionProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def decide(self, state: State, questions: Questions) -> DecisionResult:
        import typesafe_sdk as ts

        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.system_one(state=state, questions=dict(questions)),
                timeout=self._settings.hard_deadline_s,
            )
        # Order matters: TypeSafeAPITimeoutError subclasses TimeoutError, which IS
        # asyncio.TimeoutError on Python 3.11+, so SDK errors must be caught first.
        except ts.TypeSafeAPIConnectionError as exc:  # includes TypeSafeAPITimeoutError
            raise DecisionUnavailable("connection", retryable=True, detail=type(exc).__name__) from exc
        except asyncio.TimeoutError as exc:
            raise DecisionUnavailable("deadline_exceeded", retryable=True) from exc
        except ts.TypeSafeRateLimitError as exc:
            raise DecisionUnavailable("rate_limited", retryable=True, detail=str(exc.status)) from exc
        except (ts.TypeSafeBadRequestError, ts.TypeSafeUnprocessableEntityError) as exc:
            # Our request is malformed: a bug, not an outage. Log loudly.
            log.error("jev_request_rejected status=%s body=%s", exc.status, exc.body)
            raise DecisionUnavailable("bad_request", retryable=False, detail=str(exc.status)) from exc
        except ts.TypeSafeAuthenticationError as exc:
            raise DecisionUnavailable("auth", retryable=False) from exc
        except ts.TypeSafeAPIError as exc:  # 5xx incl. 529 overloaded
            raise DecisionUnavailable("server", retryable=True, detail=str(exc.status)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        answers = _convert_answers(response.answers, questions)
        try:  # property raises when the x-typesafe-request-id header is absent (e.g. some gateways)
            request_id = response.request_id
        except ts.TypeSafeError:
            request_id = None
        log.info(
            "jev_decision model=%s request_id=%s latency_ms=%.0f input_tokens=%s questions=%d",
            response.model, request_id, latency_ms, response.usage.input_tokens, len(questions),
        )  # never log the state: it usually contains customer PII
        return DecisionResult(
            answers=answers,
            source=SignalSource.JEV,
            model=response.model,
            request_id=request_id,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
        )


def _convert_answers(raw_answers: Mapping[str, Any], questions: Questions) -> dict[str, Signal]:
    """Map SDK answer models to domain signals and sanity-check them.

    A typed response is not a correct response, but a structurally broken one should
    never reach business logic.
    """
    out: dict[str, Signal] = {}
    for key, spec in questions.items():
        if key not in raw_answers:
            raise DecisionUnavailable("invalid_response", retryable=True, detail=f"missing {key}")
        ans = raw_answers[key]
        kind = spec["type"]
        if kind == "choice":
            probs = {str(k): float(v) for k, v in ans.probabilities.items()}
            _check_distribution(key, probs.values())
            if ans.choice not in spec["criteria"]:
                raise DecisionUnavailable("invalid_response", retryable=False, detail=f"{key}: unknown option")
            out[key] = ChoiceSignal(ans.choice, probs, float(ans.confidence))
        elif kind == "score":
            # SDK parses wire keys "0","1",... into int keys.
            probs_s = {int(k): float(v) for k, v in ans.probabilities.items()}
            _check_distribution(key, probs_s.values())
            legend = {int(k): str(v) for k, v in ans.legend.items()}
            out[key] = ScoreSignal(float(ans.score), probs_s, legend, float(ans.confidence))
        elif kind == "noul":
            p = float(ans.noul)
            if not 0.0 <= p <= 1.0:
                raise DecisionUnavailable("invalid_response", retryable=False, detail=f"{key}: p={p}")
            out[key] = NoulSignal(p)
        else:
            raise ValueError(f"unsupported question type {kind!r}")
    return out


def _check_distribution(key: str, values: Any, tol: float = 0.05) -> None:
    vals = list(values)
    total = sum(vals)
    if any(v < 0 or v > 1 or math.isnan(v) for v in vals) or abs(total - 1.0) > tol:
        raise DecisionUnavailable("invalid_response", retryable=False, detail=f"{key}: sum={total:.3f}")


# ------------------------------------------------------------------ test doubles


class ScriptedProvider:
    """Returns pre-built answers. For unit tests of business logic."""

    def __init__(
        self,
        responder: Callable[[State, Questions], Mapping[str, Signal]] | None = None,
        *,
        fail_with: DecisionUnavailable | None = None,
        model: str = "scripted",
    ) -> None:
        self._responder = responder
        self._fail_with = fail_with
        self._model = model
        self.calls: list[tuple[State, Questions]] = []

    async def decide(self, state: State, questions: Questions) -> DecisionResult:
        self.calls.append((state, questions))
        if self._fail_with is not None:
            raise self._fail_with
        assert self._responder is not None
        return DecisionResult(
            answers=dict(self._responder(state, questions)),
            source=SignalSource.FAKE,
            model=self._model,
        )


def choice_from_scores(raw: Mapping[str, float]) -> ChoiceSignal:
    """Build a ChoiceSignal from unnormalized non-negative scores (helper for fakes)."""
    total = sum(raw.values()) or 1.0
    probs = {k: v / total for k, v in raw.items()}
    top = max(probs, key=probs.__getitem__)
    n = len(probs)
    conf = (probs[top] - 1 / n) / (1 - 1 / n) if n > 1 else 1.0  # TypeSafe Choice formula
    return ChoiceSignal(top, probs, round(conf, 4))


def score_from_scores(raw: list[float], legend: list[str]) -> ScoreSignal:
    total = sum(raw) or 1.0
    probs = {i: v / total for i, v in enumerate(raw)}
    expected = sum(i * p for i, p in probs.items())
    n = len(probs)
    m = max(probs, key=probs.__getitem__)
    spread = sum(p * abs(i - m) for i, p in probs.items())
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    conf = max(0.0, 1 - spread / even) if even else 1.0  # TypeSafe Score formula
    return ScoreSignal(expected, probs, dict(enumerate(legend)), round(conf, 4))
