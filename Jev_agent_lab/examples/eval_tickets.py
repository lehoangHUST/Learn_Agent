"""Benchmark harness: run the ticket questions over a labeled JSONL file and report
prediction metrics, calibration, selective prediction, cost-tuned thresholds,
option-order stability, latency and fallback rate.

    TYPESAFE_API_KEY=... python -m examples.eval_tickets data/tickets_labeled.jsonl --live

Input lines: {"id": "...", "subject": "...", "body": "...", "topic": "<label>", "security": 0|1}
Every live response is cached in .cache/<model>.jsonl keyed by request, so re-runs and
threshold sweeps cost nothing. Split train/test BEFORE tuning thresholds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
from pathlib import Path

from jev_lab import evaluation as ev
from jev_lab.config import AppConfig
from jev_lab.decisions import DecisionUnavailable
from jev_lab.ticket_routing import TICKET_QUESTIONS, build_state


def _key(state, questions) -> str:
    return hashlib.sha256(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()


class CachedRunner:
    def __init__(self, provider, cache_path: Path) -> None:
        self.provider, self.path = provider, cache_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec

    async def run(self, state, questions):
        k = _key(state, questions)
        if k in self.cache:
            return self.cache[k]
        try:
            r = await self.provider.decide(state, questions)
            rec = {"key": k, "ok": True, "model": r.model, "latency_ms": r.latency_ms,
                   "answers": {q: _ser(a) for q, a in r.answers.items()}}
        except DecisionUnavailable as e:
            rec = {"key": k, "ok": False, "reason": e.reason}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.cache[k] = rec
        return rec


def _ser(sig):
    d = {f: getattr(sig, f) for f in sig.__slots__}
    if "probabilities" in d:
        d["probabilities"] = {str(k): v for k, v in d["probabilities"].items()}
    d.pop("legend", None)
    return d


def reversed_topic_questions():
    q = json.loads(json.dumps(TICKET_QUESTIONS))
    q["topic"]["criteria"] = dict(reversed(list(q["topic"]["criteria"].items())))
    return {"topic": q["topic"]}


async def main(path: str, live: bool, test_frac: float) -> None:
    cfg = AppConfig()
    if live:
        from jev_lab.providers import JevDecisionProvider
        provider = JevDecisionProvider(cfg.jev)
    else:
        from examples.offline_fake import FakeKeywordProvider
        provider = FakeKeywordProvider()
    runner = CachedRunner(provider, Path(".cache") / f"{cfg.jev.model if live else 'fake'}.jsonl")

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    random.Random(0).shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac))
    test, train = rows[:n_test], rows[n_test:]

    async def score(split):
        out = []
        for r in split:
            state = build_state(r["subject"], r["body"])
            main_rec = await runner.run(state, TICKET_QUESTIONS)
            flip_rec = await runner.run(state, reversed_topic_questions())
            out.append((r, main_rec, flip_rec))
        return out

    tr, te = await score(train), await score(test)
    ok = [(r, m, f) for r, m, f in te if m["ok"]]
    print(f"rows train={len(train)} test={len(test)}  fallback_rate={1 - len(ok) / len(te):.1%}")

    topic = [m["answers"]["topic"] for _, m, _ in ok]
    gold = [r["topic"] for r, _, _ in ok]
    top_p = [t["probabilities"][t["choice"]] for t in topic]
    correct = [t["choice"] == g for t, g in zip(topic, gold)]
    print(f"topic accuracy={sum(correct) / len(correct):.3f}  ECE={ev.ece(top_p, correct):.3f}  "
          f"Brier={ev.brier_multiclass([t['probabilities'] for t in topic], gold):.3f}  "
          f"logloss={ev.log_loss([t['probabilities'].get(g, 0) for t, g in zip(topic, gold)]):.3f}")
    for cov in (1.0, 0.8, 0.5):
        print(f"  accuracy@{cov:.0%} coverage = {ev.accuracy_at_coverage(top_p, correct, cov):.3f}")
    print(f"  AURC={ev.aurc(top_p, correct):.3f}")
    for b in ev.reliability_bins(top_p, correct, n_bins=10):
        print(f"  bin [{b.lo:.1f},{b.hi:.1f}) n={b.count:<4} conf={b.mean_confidence:.2f} acc={b.accuracy:.2f}")

    flips = [m["answers"]["topic"]["choice"] != f["answers"]["topic"]["choice"]
             for _, m, f in ok if f["ok"]]
    print(f"option-order flip rate={sum(flips) / max(1, len(flips)):.1%}")

    # Security: tune threshold on TRAIN under a cost model, report on TEST.
    tr_ok = [(r, m) for r, m, _ in tr if m["ok"] and "security" in r]
    if tr_ok:
        p_tr = [m["answers"]["security_incident"]["p_yes"] for _, m in tr_ok]
        y_tr = [bool(r["security"]) for r, _ in tr_ok]
        t, _ = ev.best_threshold(p_tr, y_tr, cost_fp=1, cost_fn=50)
        p_te = [m["answers"]["security_incident"]["p_yes"] for r, m, _ in ok if "security" in r]
        y_te = [bool(r["security"]) for r, _, _ in ok if "security" in r]
        if p_te:
            pr, rc, f1 = ev.prf([p >= t for p in p_te], y_te)
            print(f"security threshold(train, FN=50xFP)={t:.2f} -> test P={pr:.2f} R={rc:.2f} F1={f1:.2f}")

    lat = [m["latency_ms"] for _, m, _ in ok if m.get("latency_ms")]
    if len(lat) >= 2:
        q = statistics.quantiles(lat, n=20)
        print(f"latency p50={statistics.median(lat):.0f}ms p95={q[18]:.0f}ms")
    if live:
        await provider.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.5)
    a = ap.parse_args()
    asyncio.run(main(a.path, a.live, a.test_frac))
