"""A keyword-driven FAKE decision provider so demos run without an API key.

Its probabilities are invented. Never benchmark this; it only exercises the plumbing.
"""

from __future__ import annotations

import re

from jev_lab.decisions import DecisionResult, NoulSignal, SignalSource
from jev_lab.providers import choice_from_scores, score_from_scores

KEYWORDS = {
    "billing": r"charge|refund|invoice|payment|trừ tiền|hoàn tiền|hóa đơn",
    "technical": r"error|500|bug|crash|outage|api|lỗi|sập",
    "account_access": r"login|password|2fa|locked|đăng nhập|mật khẩu",
    "security": r"hack|breach|phishing|unauthori|stolen|bị hack",
    "legal_privacy": r"gdpr|delete my data|lawyer|xóa dữ liệu",
    "feedback": r"feature|love|suggest|góp ý",
}
TEAM_FOR = {"billing": "payments_team", "technical": "platform_engineering",
            "account_access": "identity_team", "security": "security_ops",
            "legal_privacy": "legal", "feedback": "customer_success", "other": "customer_success"}


class FakeKeywordProvider:
    async def decide(self, state, questions) -> DecisionResult:
        text = str(state).lower()
        answers = {}
        for key, spec in questions.items():
            kind = spec["type"]
            if kind == "choice" and key == "topic":
                raw = {o: 0.2 for o in spec["criteria"]}
                for topic, pat in KEYWORDS.items():
                    raw[topic] += 3.0 * len(re.findall(pat, text))
                answers[key] = choice_from_scores(raw)
            elif kind == "choice" and key == "team":
                topic = answers.get("topic")
                raw = {o: 0.2 for o in spec["criteria"]}
                if topic:
                    raw[TEAM_FOR[topic.choice]] += 5.0
                answers[key] = choice_from_scores(raw)
            elif kind == "choice":
                answers[key] = choice_from_scores({o: 1.0 for o in spec["criteria"]})
            elif kind == "score":
                urgent = len(re.findall(r"asap|now|urgent|can't|cannot|every|gấp|ngay", text))
                answers[key] = score_from_scores([1, 2, 1 + urgent, urgent * 1.5],
                                                 [str(c) for c in spec["criteria"]])
            else:
                pat = {"security_incident": KEYWORDS["security"],
                       "legal_threat": r"lawyer|sue|kiện",
                       "steering_attempt": r"ignore (all|previous)|set priority|classify this as"}.get(key, "$^")
                answers[key] = NoulSignal(0.9 if re.search(pat, text) else 0.05)
        return DecisionResult(answers=answers, source=SignalSource.FAKE, model="fake-keywords")
