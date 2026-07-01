import numpy as np

from malsnif.utils.metrics import choose_threshold, binary_metrics


def brute_choose_threshold(y, s, metric='f1_min_recall', min_recall=0.85):
    # Keep this intentionally simple and small; it verifies the vectorized path
    # against the same public binary_metrics interface without causing the old
    # quadratic blow-up on real validation splits.
    from malsnif.utils.metrics import _threshold_candidates
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    best = (-1e18, -1, -1.0, -1.0, -1.0, 0.5, None)
    for t in _threshold_candidates(s):
        m = binary_metrics(y, s, float(t)).to_dict()
        if metric.endswith('_min_recall'):
            base = metric[:-len('_min_recall')] or 'f1'
            key = {'balanced': 'balanced_accuracy', 'bal': 'balanced_accuracy'}.get(base, base)
            if (m.get('recall') or 0) < min_recall:
                score = -1e9 + (m.get('recall') or 0)
            else:
                score = m.get(key) if m.get(key) is not None else m.get('f1')
        else:
            score = m.get(metric) if m.get(metric) is not None else m.get('f1')
        pred_rate = m.get('predicted_positive_rate') or 0.0
        collapse = 1 if pred_rate in {0.0, 1.0} else 0
        score = float(score) - (1e-6 if collapse else 0.0)
        current = (score, -collapse, float(t), float(m['precision']), float(m['recall']), float(t), m)
        if current > best:
            best = current
    return best[5], best[6]


def test_vectorized_choose_threshold_matches_bruteforce_small():
    y = [1, 0, 1, 0, 1, 0, 0, 1]
    s = [0.9, 0.8, 0.65, 0.62, 0.4, 0.3, 0.2, 0.1]
    tv, dv = choose_threshold(y, s, metric='f1_min_recall', min_recall=0.5)
    tb, db = brute_choose_threshold(y, s, metric='f1_min_recall', min_recall=0.5)
    assert tv == tb
    assert dv['metrics_at_threshold']['f1'] == db['f1']


def test_vectorized_choose_threshold_handles_large_validation_split():
    rng = np.random.default_rng(123)
    y = (rng.random(5000) < 0.3).astype(int)
    s = rng.random(5000)
    t, diag = choose_threshold(y, s, metric='f1_min_recall', min_recall=0.8)
    assert 0.0 <= t <= 1.0
    assert diag['num_threshold_candidates'] > 1000
