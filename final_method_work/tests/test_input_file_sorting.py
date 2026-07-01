from pathlib import Path

from malsnif.data.parsers import discover_input_files


def test_cdm_shards_are_sorted_in_stream_order(tmp_path: Path):
    names = [
        "ta1-cadets-e3-official-1.json.2",
        "ta1-cadets-e3-official-1.json",
        "ta1-cadets-e3-official.json.2",
        "ta1-cadets-e3-official-2.json.1",
        "ta1-cadets-e3-official.json",
        "ta1-cadets-e3-official-1.json.1",
        "ta1-cadets-e3-official.json.1",
        "ta1-cadets-e3-official-2.json",
        "ta1-cadets-e3-official.json.txt",
    ]
    for n in names:
        (tmp_path / n).write_text("{}\n")
    got = [p.name for p in discover_input_files(tmp_path, raw_glob="*.json*")]
    assert got == [
        "ta1-cadets-e3-official.json",
        "ta1-cadets-e3-official.json.1",
        "ta1-cadets-e3-official.json.2",
        "ta1-cadets-e3-official-1.json",
        "ta1-cadets-e3-official-1.json.1",
        "ta1-cadets-e3-official-1.json.2",
        "ta1-cadets-e3-official-2.json",
        "ta1-cadets-e3-official-2.json.1",
    ]
