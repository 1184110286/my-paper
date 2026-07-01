from pathlib import Path
import json

from malsnif.data.cdm import iter_cdm_events
from malsnif.config import Config
from malsnif.data.dataset import preprocess


def _cdm(t, payload):
    return {"datum": {f"com.bbn.tc.schema.avro.cdm18.{t}": payload}}


def test_iter_cdm_events_with_uuid_label(tmp_path: Path):
    subj = "11111111-1111-1111-1111-111111111111"
    file = "22222222-2222-2222-2222-222222222222"
    evu = "33333333-3333-3333-3333-333333333333"
    raw = tmp_path / "ta1-cadets-e3-official-1.json"
    records = [
        _cdm("Subject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "cmdLine": {"string": "/usr/sbin/nginx"}}),
        _cdm("FileObject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "filename": {"string": "/var/log/drakon"}}),
        _cdm("Event", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": evu}, "type": "EVENT_WRITE", "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "timestampNanos": {"long": 10}}),
    ]
    raw.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    lab = tmp_path / "labels"
    lab.mkdir()
    (lab / "malicious_uuids.txt").write_text(file + "\n", encoding="utf-8")

    events = list(iter_cdm_events([raw], label_dir=lab))
    assert len(events) == 1
    assert events[0].edge_type == "EVENT_WRITE"
    assert events[0].src_type == "PROCESS"
    assert events[0].dst_type == "FILE"
    assert events[0].tag == 1
    assert "drakon" in events[0].raw["dst_display"]


def test_preprocess_toy_cdm(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw" / "darpa_tc" / "cadets" / "e3" / "cdm"
    label_dir = tmp_path / "data" / "raw" / "darpa_tc" / "cadets" / "e3" / "labels"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    subj = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    file = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    raw = raw_dir / "cadets_e3_01.json"
    records = [
        _cdm("Subject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "cmdLine": {"string": "/bin/bash"}}),
        _cdm("FileObject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "filename": {"string": "/tmp/payload.sh"}}),
        _cdm("Event", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"}, "type": "EVENT_READ", "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "timestampNanos": {"long": 1}}),
        _cdm("Event", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": "ffffffff-ffff-ffff-ffff-ffffffffffff"}, "type": "EVENT_WRITE", "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "timestampNanos": {"long": 2}}),
    ]
    raw.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    (label_dir / "malicious_paths.txt").write_text("/tmp/payload.sh\n", encoding="utf-8")
    cfg = Config(
        raw_dir=str(raw_dir), label_dir=str(label_dir), processed_dir=str(out_dir),
        input_format="cdm_json", window_events=10, max_events=None,
        skipgram_epochs=0, word_dim=8, max_nodes_per_graph=None, max_edges_per_graph=None,
    )
    meta = preprocess(cfg)
    assert meta["num_graphs"] == 1
    assert meta["parse_stats"]["raw_events_consumed"] == 2
    assert meta["parse_stats"]["labeled_events_consumed"] == 2
    assert (out_dir / "metadata.json").exists()


def test_iter_cdm_events_extracts_nested_object_attributes(tmp_path: Path):
    subj = "11111111-1111-1111-1111-111111111111"
    file = "22222222-2222-2222-2222-222222222222"
    evu = "33333333-3333-3333-3333-333333333333"
    raw = tmp_path / "ta1-cadets-e3-official.json"
    records = [
        _cdm("Subject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "baseObject": {"properties": {"map": {"cmdLine": {"string": "/usr/bin/python /tmp/implant.py"}}}}}),
        _cdm("FileObject", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "baseObject": {"properties": {"map": {"path": {"string": "/home/admin/payload.bin"}}}}}),
        _cdm("Event", {"uuid": {"com.bbn.tc.schema.avro.cdm18.UUID": evu}, "type": "EVENT_READ", "subject": {"com.bbn.tc.schema.avro.cdm18.UUID": subj}, "predicateObject": {"com.bbn.tc.schema.avro.cdm18.UUID": file}, "timestampNanos": {"long": 10}}),
    ]
    raw.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    events = list(iter_cdm_events([raw], label_dir=tmp_path / "labels"))
    assert len(events) == 1
    assert "implant.py" in events[0].raw["dst_display"] or "implant.py" in events[0].raw["src_display"]
    assert "payload.bin" in events[0].raw["semantic_object_display"]


def test_event_semantic_prefers_predicate_object_path_even_when_object_has_uuid_fallback():
    from malsnif.data.cdm import CdmObject, CdmLabeler, event_from_cdm

    obj_uuid = "11111111-2222-3333-4444-555555555555"
    subj_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    event = {
        "uuid": "99999999-2222-3333-4444-555555555555",
        "type": "EVENT_READ",
        "subject": subj_uuid,
        "predicateObject": obj_uuid,
        "predicateObjectPath": "/tmp/drakon_payload.sh",
        "timestampNanos": 1,
    }
    objects = {
        subj_uuid: CdmObject(subj_uuid, "Subject", "PROCESS", f"subject:{subj_uuid}"),
        obj_uuid: CdmObject(obj_uuid, "FileObject", "FILE", f"file:{obj_uuid}"),
    }
    ev = event_from_cdm(event, objects, CdmLabeler(), information_flow=True)
    assert ev is not None
    assert ev.raw["semantic_object_display"] == "/tmp/drakon_payload.sh"
