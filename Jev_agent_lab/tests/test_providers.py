"""Adapter tests against the REAL SDK response models (no network)."""

import asyncio
import json

import httpx2
import pytest
import typesafe_sdk as ts

from jev_lab.config import JevSettings
from jev_lab.decisions import ChoiceSignal, DecisionUnavailable, ScoreSignal
from jev_lab.providers import JevDecisionProvider, _convert_answers

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "technical": None}},
    "frustration": {"type": "score", "instructions": "How frustrated?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}

# Response body shape copied from the documented example (zuplo.com/docs/ai-gateway/jev).
BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.95},
        "dept": {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.88, "technical": 0.12}, "confidence": 0.81},
        "frustration": {"type": "score", "score": 1.05,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0, "1": 0.95, "2": 0.05}, "confidence": 0.92},
    },
    "usage": {"input_tokens": 296, "output_tokens": 20},
}


def sdk_response(body=BODY):
    return ts.SystemOneResponse.model_validate_json(json.dumps(body))


def test_convert_real_sdk_models():
    out = _convert_answers(sdk_response().answers, QUESTIONS)
    assert isinstance(out["dept"], ChoiceSignal) and out["dept"].choice == "billing"
    assert out["dept"].margin == pytest.approx(0.76)
    assert isinstance(out["frustration"], ScoreSignal)
    assert out["frustration"].most_likely_level == 1
    assert out["frustration"].prob_at_least(2) == pytest.approx(0.05)
    assert out["urgent"].p_yes == 0.95


def test_broken_distribution_is_rejected():
    body = json.loads(json.dumps(BODY))
    body["answers"]["dept"]["probabilities"] = {"billing": 0.9, "technical": 0.5}
    with pytest.raises(DecisionUnavailable) as e:
        _convert_answers(sdk_response(body).answers, QUESTIONS)
    assert e.value.reason == "invalid_response"


class FakeClient:
    def __init__(self, exc=None, delay=0.0):
        self.exc, self.delay = exc, delay

    async def system_one(self, state, questions):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return sdk_response()

    async def aclose(self):
        pass


def decide(client, settings=JevSettings()):
    return asyncio.run(JevDecisionProvider(settings, client=client).decide("x", QUESTIONS))


def test_success_carries_provenance():
    r = decide(FakeClient())
    assert r.model == "jev-1.13.0" and r.input_tokens == 296


@pytest.mark.parametrize("exc,reason,retryable", [
    (ts.TypeSafeRateLimitError(429, None, httpx2.Headers()), "rate_limited", True),
    (ts.TypeSafeAPITimeoutError(2.0), "connection", True),
    (ts.TypeSafeInternalServerError(529, None, httpx2.Headers()), "server", True),
    (ts.TypeSafeUnprocessableEntityError(422, {"detail": []}, httpx2.Headers()), "bad_request", False),
])
def test_errors_are_classified(exc, reason, retryable):
    with pytest.raises(DecisionUnavailable) as e:
        decide(FakeClient(exc=exc))
    assert (e.value.reason, e.value.retryable) == (reason, retryable)


def test_hard_deadline():
    with pytest.raises(DecisionUnavailable) as e:
        decide(FakeClient(delay=0.5), JevSettings(hard_deadline_s=0.05))
    assert e.value.reason == "deadline_exceeded"
