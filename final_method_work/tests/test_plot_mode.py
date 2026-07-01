from malsnif.utils.plot import plot_history, plot_score_histogram


def test_plot_history_essential_only_combined(tmp_path):
    hist = [
        {"epoch": 1, "loss": 1.0, "val_f1": 0.1, "val_mcc": 0.0, "val_average_precision": 0.2, "val_threshold": 0.5},
        {"epoch": 2, "loss": 0.5, "val_f1": 0.2, "val_mcc": 0.1, "val_average_precision": 0.3, "val_threshold": 0.6},
    ]
    out = tmp_path / "plots" / "history.png"
    plot_history(hist, out, mode="essential")
    assert out.exists()
    assert not (tmp_path / "plots" / "f1.png").exists()
    assert not (tmp_path / "plots" / "val_f1.png").exists()


def test_plot_mode_none_disables_score_plot(tmp_path):
    out = tmp_path / "plots" / "scores_test.png"
    plot_score_histogram([0, 1], [0.1, 0.9], out, threshold=0.5, mode="none")
    assert not out.exists()
