import numpy as np

from malsnif.utils.metrics import binary_metrics, choose_threshold


def _bruteforce_best_f1(y, s):
    cand = sorted(set([0.5] + list(s) + [min(s) - 1e-6, max(s) + 1e-6]))
    mids = [(a + b) / 2 for a, b in zip(sorted(set(s))[:-1], sorted(set(s))[1:])]
    cand = sorted(set([min(1.0, max(0.0, x)) for x in cand + mids]))
    best = (-1.0, 0.0, 0.0, 0.5)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    for t in cand:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if (f1, t) > (best[0], best[3]):
            best = (f1, prec, rec, t)
    return best


def test_binary_metrics_best_f1_matches_small_bruteforce():
    y = [0, 1, 0, 1, 1, 0]
    s = [0.2, 0.8, 0.5, 0.6, 0.3, 0.1]
    brute_f1, brute_prec, brute_rec, _ = _bruteforce_best_f1(y, s)
    md = binary_metrics(y, s, threshold=0.5).to_dict()
    assert abs(md["best_f1"] - brute_f1) < 1e-12
    assert abs(md["best_f1_precision"] - brute_prec) < 1e-12
    assert abs(md["best_f1_recall"] - brute_rec) < 1e-12


def test_choose_threshold_handles_many_scores_without_quadratic_loop():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=20000)
    s = rng.random(20000)
    threshold, diag = choose_threshold(y, s, metric="f1_min_recall", min_recall=0.6)
    assert 0.0 <= threshold <= 1.0
    assert diag["num_thresholds"] <= 2 * len(np.unique(s)) + 3
    assert diag["metrics_at_threshold"]["recall"] >= 0.6 or diag["metrics_at_threshold"]["recall"] == 1.0
