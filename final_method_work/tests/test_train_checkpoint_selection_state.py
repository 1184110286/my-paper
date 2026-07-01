from pathlib import Path
import json

from malsnif.config import Config
from malsnif.data.dataset import preprocess
from malsnif.train import train


def _cdm(t, payload):
    return {"datum": {f"com.bbn.tc.schema.avro.cdm18.{t}": payload}}


def _uuid(i: int) -> str:
    return f"00000000-0000-0000-0000-{i:012d}"


def test_train_initializes_checkpoint_selection_tuple_with_val_f1(tmp_path: Path):
    """Regression for v3.0.0: first checkpoint selection must not read an unbound local.

    The long CADETS run failed after epoch 1 because best_selection_tuple was
    only assigned inside the update branch but never initialized before the
    first _selection_is_better() call.  This tiny end-to-end run uses the same
    val_f1 + tie-breaker selection path as the GraphSAGE scripts.
    """
    raw_dir = tmp_path / "raw"
    label_dir = tmp_path / "labels"
    processed = tmp_path / "processed"
    run_dir = tmp_path / "run"
    raw_dir.mkdir()
    label_dir.mkdir()

    bad_proc = _uuid(1)
    good_proc = _uuid(2)
    bad_file = _uuid(3)
    good_file = _uuid(4)
    rows = [
        _cdm("Subject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": bad_proc}, "cmdLine": {"string": "/bin/bad"}}),
        _cdm("Subject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": good_proc}, "cmdLine": {"string": "/bin/good"}}),
        _cdm("FileObject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": bad_file}, "filename": {"string": "/tmp/payload"}}),
        _cdm("FileObject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": good_file}, "filename": {"string": "/tmp/benign"}}),
    ]
    # Three windows; each contains one malicious-process event and one benign-process event.
    ts = 1
    for w in range(3):
        rows.append(_cdm("Event", {
            "uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": _uuid(100 + 2*w)},
            "type": "EVENT_WRITE",
            "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": bad_proc},
            "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": bad_file},
            "timestampNanos": {"long": ts},
        }))
        ts += 1
        rows.append(_cdm("Event", {
            "uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": _uuid(101 + 2*w)},
            "type": "EVENT_READ",
            "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": good_proc},
            "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": good_file},
            "timestampNanos": {"long": ts},
        }))
        ts += 1
    (raw_dir / "toy.json").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (label_dir / "malicious_uuids.txt").write_text(bad_file + "\n", encoding="utf-8")

    cfg = Config(
        raw_dir=str(raw_dir), label_dir=str(label_dir), processed_dir=str(processed), run_dir=str(run_dir),
        input_format="cdm_json", raw_glob="*.json", window_events=2, max_events=None,
        split_ratio=(1/3, 1/3, 1/3), skipgram_epochs=0, word_dim=8, hidden_dim=8, semantic_dim=8,
        behavior_dim=8, gcn_layers=1, epochs=1, model_selection_metric="val_f1",
        model_selection_tie_breakers="val_average_precision,val_mcc,val_balanced_accuracy",
        threshold_strategy="fixed", allow_unlabeled_training=False,
        simplify_graph=False, reduce_sequences=False, show_progress=False, train_progress=False,
        use_amp=False, graph_encoder="graphsage", max_events_per_node=8, max_events_per_edge=4,
    )
    preprocess(cfg)
    summary = train(cfg, "cpu")
    assert summary["best_epoch"] == 1
    assert summary["best_selection_tuple"] is not None
    assert (run_dir / "checkpoints" / "best.pt").exists()
