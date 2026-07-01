from malsnif.utils.metrics import binary_metrics, choose_threshold


def test_metrics_exposes_all_positive_collapse():
    y = [0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.6]
    m = binary_metrics(y, scores, threshold=0.5).to_dict()
    assert m["predicted_positive_rate"] == 1.0
    assert m["specificity"] == 0.0
    assert m["balanced_accuracy"] == 0.5
    assert m["mcc"] == 0.0


def test_choose_threshold_can_optimize_mcc():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    th, diag = choose_threshold(y, scores, metric="mcc")
    assert 0.2 <= th <= 0.8
    assert diag["metrics_at_threshold"]["mcc"] == 1.0
