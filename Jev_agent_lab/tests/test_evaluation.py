import random

import pytest

from jev_lab import evaluation as ev


def test_perfectly_calibrated_has_low_ece():
    rng = random.Random(0)
    conf = [rng.random() for _ in range(20000)]
    correct = [rng.random() < c for c in conf]
    assert ev.ece(conf, correct) < 0.02


def test_overconfident_model_has_high_ece():
    conf = [0.95] * 1000
    correct = [i % 10 < 7 for i in range(1000)]  # 70% accurate
    assert ev.ece(conf, correct) == pytest.approx(0.25, abs=1e-9)


def test_brier_and_logloss():
    assert ev.brier_binary([1.0, 0.0], [1, 0]) == 0
    assert ev.log_loss([0.5]) == pytest.approx(0.6931, abs=1e-3)


def test_selective_prediction_improves_accuracy_when_confidence_is_informative():
    conf = [0.9, 0.8, 0.7, 0.3, 0.2]
    correct = [True, True, True, False, False]
    assert ev.accuracy_at_coverage(conf, correct, 0.6) == 1.0
    assert ev.aurc(conf, correct) < 0.3


def test_best_threshold_follows_costs():
    p = [0.1, 0.3, 0.45, 0.55, 0.7, 0.9]
    y = [False, False, True, False, True, True]
    t_cheap_fn, _ = ev.best_threshold(p, y, cost_fp=1, cost_fn=10)   # missing is expensive
    t_cheap_fp, _ = ev.best_threshold(p, y, cost_fp=10, cost_fn=1)   # false alarm is expensive
    assert t_cheap_fn < t_cheap_fp


def test_negation_violation():
    # pair 1: 0.7 + 0.6 = 1.3 (says yes to both X and not-X); pair 2: coherent
    mad, signed, same = ev.negation_violation([0.7, 0.4], [0.6, 0.6])
    assert mad == pytest.approx(0.15) and signed == pytest.approx(0.15)
    assert same == pytest.approx(0.5)
