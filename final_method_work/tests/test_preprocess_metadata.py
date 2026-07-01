from malsnif.data.dataset import split_indices
from malsnif.data.cdm import CdmLabeler


def test_split_indices_keeps_validation_for_four_windows():
    split = split_indices(4, (0.6, 0.2, 0.2))
    assert split == {"train": [0, 1], "val": [2], "test": [3]}


def test_labeler_loads_raw_cadets_json_uuid_list(tmp_path):
    raw = tmp_path / "_raw"
    raw.mkdir()
    (raw / "cadets.json").write_text('["123e4567-e89b-12d3-a456-426614174000"]', encoding="utf-8")
    lab = CdmLabeler.from_dir(tmp_path)
    assert lab.has_labels
    assert "123e4567-e89b-12d3-a456-426614174000" in lab.uuids
    assert lab.summary()["uuid_count"] == 1
