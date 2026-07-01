from malsnif.config import Config
from malsnif.train import _selection_tuple, _selection_is_better


def test_checkpoint_selection_uses_ap_tie_breaker_for_equal_f1():
    cfg = Config(model_selection_metric='val_f1')
    # Later checkpoint has identical F1 but substantially better AP and should be selected.
    old = _selection_tuple({'epoch': 2, 'loss': 0.39}, {'f1': 0.99, 'average_precision': 0.94, 'mcc': 0.98, 'balanced_accuracy': 0.98}, {}, cfg)
    new = _selection_tuple({'epoch': 6, 'loss': 0.13}, {'f1': 0.99, 'average_precision': 0.998, 'mcc': 0.98, 'balanced_accuracy': 0.98}, {}, cfg)
    assert _selection_is_better(old, None, cfg)
    assert _selection_is_better(new, old, cfg)


def test_checkpoint_selection_primary_metric_still_dominates():
    cfg = Config(model_selection_metric='val_f1')
    old = _selection_tuple({'epoch': 2, 'loss': 0.39}, {'f1': 0.991, 'average_precision': 0.94, 'mcc': 0.98, 'balanced_accuracy': 0.98}, {}, cfg)
    new = _selection_tuple({'epoch': 6, 'loss': 0.13}, {'f1': 0.990, 'average_precision': 0.999, 'mcc': 0.99, 'balanced_accuracy': 0.99}, {}, cfg)
    assert not _selection_is_better(new, old, cfg)
