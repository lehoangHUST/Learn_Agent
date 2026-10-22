"""jev_lab: Jev (TypeSafe System One) as a decision component inside agents.

Layering (each layer only imports the ones above it):

    decisions.py   domain types (signals), no SDK import
    config.py      thresholds and costs, no SDK import
    providers.py   DecisionProvider protocol + Jev adapter + offline fakes
    fallback.py    rule-based fallback provider
    ticket_routing.py / rag_gate.py   business logic, depends only on the protocol
    evaluation.py  calibration and threshold metrics, pure Python
"""
