import asyncio

import pytest

from jev_lab.config import TicketRoutingPolicy, threshold_from_costs
from jev_lab.decisions import DecisionUnavailable, NoulSignal, SignalSource
from jev_lab.fallback import RuleFallbackProvider
from jev_lab.providers import ScriptedProvider, choice_from_scores, score_from_scores
from jev_lab.ticket_routing import (
    TEAMS, TOPICS, URGENCY_LEVELS, Action, route_ticket,
)

POLICY = TicketRoutingPolicy()


def signals(topic: dict, team: dict | None = None, urgency=(1, 3, 1, 0),
            security=0.02, legal=0.02, steering=0.02):
    """Build a full answer set; unspecified options get ~0 mass."""
    t = {k: 0.001 for k in TOPICS} | topic
    tm = {k: 0.001 for k in TEAMS} | (team or {"payments_team": 1.0})
    return lambda state, q: {
        "topic": choice_from_scores(t),
        "team": choice_from_scores(tm),
        "urgency": score_from_scores(list(urgency), URGENCY_LEVELS),
        "security_incident": NoulSignal(security),
        "legal_threat": NoulSignal(legal),
        "steering_attempt": NoulSignal(steering),
    }


def run(provider, body="I was charged twice for my subscription"):
    return asyncio.run(route_ticket("subject", body, primary=provider,
                                    fallback=RuleFallbackProvider(), policy=POLICY))


def test_threshold_from_costs():
    # misroute costs 10x a review -> act only above 0.9
    assert threshold_from_costs(30, 3) == pytest.approx(0.9)
    assert POLICY.auto_route_min_top_probability == pytest.approx(0.9)


def test_confident_billing_auto_routes():
    d = run(ScriptedProvider(signals({"billing": 0.97})))
    assert d.action is Action.AUTO_ROUTE
    assert d.team == "payments_team"


def test_uncertain_topic_goes_to_review():
    d = run(ScriptedProvider(signals({"billing": 0.6, "technical": 0.4})))
    assert d.action is Action.REVIEW
    assert any(r.startswith("topic_uncertain") for r in d.reasons)


def test_low_security_probability_still_escalates():
    # 0.35 is "probably not", but the gate is high-recall by design.
    d = run(ScriptedProvider(signals({"billing": 0.97}, security=0.35)))
    assert d.action is Action.ESCALATE and d.team == "security_ops" and d.priority == "P1"


def test_team_disagreement_is_a_review_signal():
    d = run(ScriptedProvider(signals({"billing": 0.97}, team={"platform_engineering": 1.0})))
    assert d.action is Action.REVIEW
    assert d.team == "payments_team"  # code's mapping wins, model is only a cross-check


def test_steering_attempt_blocks_auto_route():
    d = run(ScriptedProvider(signals({"billing": 0.99}, steering=0.9)))
    assert d.action is Action.REVIEW


def test_urgency_tail_probability_sets_priority():
    d = run(ScriptedProvider(signals({"billing": 0.97}, urgency=(0, 1, 1, 8))))
    assert d.priority == "P1"
    d = run(ScriptedProvider(signals({"billing": 0.97}, urgency=(1, 3, 6, 0))))
    assert d.priority == "P2"


def test_outage_falls_back_to_review():
    down = ScriptedProvider(fail_with=DecisionUnavailable("deadline_exceeded", retryable=True))
    d = run(down)
    assert d.source is SignalSource.FALLBACK_RULES
    assert d.fallback_reason == "deadline_exceeded"
    assert d.action is Action.REVIEW


def test_outage_with_security_keyword_still_escalates():
    down = ScriptedProvider(fail_with=DecisionUnavailable("server", retryable=True))
    d = run(down, body="My account was hacked and someone changed my email")
    assert d.action is Action.ESCALATE and d.team == "security_ops"
