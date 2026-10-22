"""Metrics for judging decision probabilities on YOUR labeled data. Pure Python.

Prediction metrics (is the probability good?):
  accuracy, precision/recall/F1, Brier, log loss, ECE, reliability bins,
  selective prediction (risk-coverage, AURC), negation coherence.
Decision metrics (is the policy good?):
  expected cost of a threshold, best threshold under a cost model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def brier_binary(p_yes: Sequence[float], y: Sequence[int]) -> float:
    return sum((p - t) ** 2 for p, t in zip(p_yes, y)) / len(y)


def brier_multiclass(probs: Sequence[Mapping[str, float]], labels: Sequence[str]) -> float:
    total = 0.0
    for dist, lab in zip(probs, labels):
        total += sum((dist.get(k, 0.0) - (1.0 if k == lab else 0.0)) ** 2
                     for k in set(dist) | {lab})
    return total / len(labels)


def log_loss(p_true: Sequence[float], eps: float = 1e-4) -> float:
    """Mean -log P(true outcome). eps matters: Jev returns values on a 0.01 grid."""
    return -sum(math.log(min(1 - eps, max(eps, p))) for p in p_true) / len(p_true)


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> list[Bin]:
    """Equal-width bins of stated probability vs observed frequency (reliability diagram)."""
    bins: list[Bin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if idx:
            bins.append(Bin(lo, hi, len(idx),
                            sum(conf[i] for i in idx) / len(idx),
                            sum(correct[i] for i in idx) / len(idx)))
    return bins


def ece(conf: Sequence[float], correct: Sequence[bool], n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
    n = len(conf)
    return sum(b.count / n * abs(b.accuracy - b.mean_confidence)
               for b in reliability_bins(conf, correct, n_bins))


def selective_curve(conf: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float]]:
    """(coverage, accuracy) when keeping only the most confident fraction."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    out, hits = [], 0
    for rank, i in enumerate(order, start=1):
        hits += correct[i]
        out.append((rank / len(order), hits / rank))
    return out


def aurc(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    curve = selective_curve(conf, correct)
    return sum(1 - acc for _, acc in curve) / len(curve)


def accuracy_at_coverage(conf: Sequence[float], correct: Sequence[bool], coverage: float) -> float:
    curve = selective_curve(conf, correct)
    k = max(1, round(coverage * len(curve)))
    return curve[k - 1][1]


def prf(pred: Sequence[bool], y: Sequence[bool]) -> tuple[float, float, float]:
    tp = sum(p and t for p, t in zip(pred, y))
    fp = sum(p and not t for p, t in zip(pred, y))
    fn = sum((not p) and t for p, t in zip(pred, y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def expected_cost(p_yes: Sequence[float], y: Sequence[bool], threshold: float,
                  cost_fp: float, cost_fn: float) -> float:
    cost = 0.0
    for p, t in zip(p_yes, y):
        pred = p >= threshold
        cost += cost_fp if (pred and not t) else cost_fn if ((not pred) and t) else 0.0
    return cost / len(y)


def best_threshold(p_yes: Sequence[float], y: Sequence[bool], cost_fp: float,
                   cost_fn: float, grid: Iterable[float] | None = None) -> tuple[float, float]:
    """Threshold minimizing expected cost. Tune on a TRAIN split, report on a TEST split."""
    grid = list(grid) if grid is not None else [i / 100 for i in range(1, 100)]
    return min(((t, expected_cost(p_yes, y, t, cost_fp, cost_fn)) for t in grid),
               key=lambda tc: (tc[1], tc[0]))


def negation_violation(s: Sequence[float], n: Sequence[float]) -> tuple[float, float, float]:
    """Coherence check that needs no labels: P(X) + P(not X) should equal 1.

    Returns (mean |s+n-1|, mean signed s+n-1, share of pairs on the same side of 0.5).
    """
    pairs = list(zip(s, n))
    mad = sum(abs(a + b - 1) for a, b in pairs) / len(pairs)
    signed = sum(a + b - 1 for a, b in pairs) / len(pairs)
    same = sum((a - 0.5) * (b - 0.5) > 0 for a, b in pairs) / len(pairs)
    return mad, signed, same


def bootstrap_ci(values: Sequence[float], stat=lambda v: sum(v) / len(v), n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    k = len(values)
    reps = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return reps[int(alpha / 2 * n)], reps[int((1 - alpha / 2) * n) - 1]
