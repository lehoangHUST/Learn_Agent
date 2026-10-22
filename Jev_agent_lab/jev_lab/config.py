"""All tunable numbers live here, in one reviewed place.

Every default below is a STARTING POINT, not a recommendation. Re-tune on your own
labeled data, per decision, per model version (pin the version you tuned against).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class JevSettings:
    # Pin a versioned id when thresholds were tuned against it; aliases move silently.
    model: str = os.environ.get("JEV_MODEL", "jev-1.13.0")
    # Per-attempt HTTP timeout (seconds). Jev latency is sub-second in published
    # measurements, so a short timeout keeps the fallback path fast.
    request_timeout_s: float = 2.0
    # Total budget for the SDK retry loop, including backoff.
    retry_budget_s: float = 3.0
    max_retries: int = 1
    # Hard wall-clock deadline enforced with asyncio.wait_for on top of the SDK.
    hard_deadline_s: float = 3.5


def threshold_from_costs(cost_wrong_auto: float, cost_review: float) -> float:
    """Minimum P(correct) at which acting automatically beats sending to review.

    Act automatically when expected cost of acting < cost of review:
        (1 - p) * cost_wrong_auto < cost_review
        p > 1 - cost_review / cost_wrong_auto
    Valid only if probabilities are calibrated ON YOUR DATA (check with evaluation.py).
    """
    if cost_wrong_auto <= 0 or cost_review < 0:
        raise ValueError("costs must be positive")
    return max(0.0, min(1.0, 1.0 - cost_review / cost_wrong_auto))


@dataclass(frozen=True, slots=True)
class TicketRoutingPolicy:
    # Cost model (arbitrary units, e.g. minutes of agent time). A misrouted ticket
    # bounces between teams; a review costs one triage look.
    cost_misroute: float = 30.0
    cost_review: float = 3.0
    # Floor below which we never auto-route even if costs say otherwise.
    min_topic_confidence: float = 0.5
    # Security/legal are high-recall gates: a LOW threshold, because missing one is
    # far worse than an extra human look. These never auto-close.
    security_incident_min: float = 0.30
    legal_threat_min: float = 0.40
    # Urgency is a Score; we gate on the tail probability P(level >= HIGH).
    high_urgency_level: int = 2
    high_urgency_min: float = 0.50
    # Injected text in a ticket that tries to steer routing.
    injection_min: float = 0.50

    @property
    def auto_route_min_top_probability(self) -> float:
        return max(
            threshold_from_costs(self.cost_misroute, self.cost_review),
            self.min_topic_confidence,
        )


@dataclass(frozen=True, slots=True)
class RagPolicy:
    # Per-passage gates (same structure as TypeSafe's "Classifying RAG passages"
    # cookbook; the numbers there were tuned for their corpus, not yours).
    injection_max: float = 0.70
    contradicts_min: float = 0.70
    relevant_min: float = 0.45
    evidence_min: float = 0.55
    # Evidence sufficiency over the accepted set.
    sufficient_min: float = 0.60
    max_retrieval_rounds: int = 2
    top_k: int = 8
    # Answer validation: every claim must be supported above this.
    claim_supported_min: float = 0.60
    max_unsupported_claims: int = 0


@dataclass(frozen=True, slots=True)
class AppConfig:
    jev: JevSettings = field(default_factory=JevSettings)
    tickets: TicketRoutingPolicy = field(default_factory=TicketRoutingPolicy)
    rag: RagPolicy = field(default_factory=RagPolicy)
