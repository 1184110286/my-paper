from malsnif.utils.metrics import choose_threshold, binary_metrics


def test_f1_min_recall_threshold_respects_recall_constraint():
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.45, 0.7, 0.2, 0.1]
    th, diag = choose_threshold(y, s, metric="f1_min_recall", min_recall=2/3)
    m = binary_metrics(y, s, th).to_dict()
    assert m["recall"] >= 2/3
    assert diag["metric"] == "f1_min_recall"
