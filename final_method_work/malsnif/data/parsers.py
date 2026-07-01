from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from malsnif.data.cdm import open_text_auto, iter_json_records, iter_cdm_events

from malsnif.constants import NODE_TYPE_NORMALIZATION, SELECTED_EVENTS
from malsnif.data.events import EventRecord
from malsnif.data.sanitize import infer_dst_type, normalize_path


def _first(row: Dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    lower = {str(k).lower().replace(" ", "_"): k for k in row.keys()}
    for c in candidates:
        key = c.lower().replace(" ", "_")
        if key in lower:
            v = row[lower[key]]
            if v is not None and str(v) != "":
                return v
    return default


def _norm_type(t: Any) -> str:
    if t is None:
        return "UNKNOWN"
    key = str(t).strip().lower().replace(" ", "")
    return NODE_TYPE_NORMALIZATION.get(key, str(t).strip().upper() or "UNKNOWN")


def _to_int_tag(v: Any) -> int:
    if v is None or v == "":
        return 0
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "mal", "malicious", "attack", "yes", "y"}:
        return 1
    try:
        return int(float(s) > 0)
    except Exception:
        return 0


def row_to_event(row: Dict[str, Any], sanitize: bool = True) -> EventRecord:
    # Procmon fields: Time of Day, Process Name, PID, Operation, Path, Result, Detail
    proc_name = _first(row, ["Process Name", "process_name", "process", "processname", "Image", "exe"])
    pid = _first(row, ["PID", "pid", "process_id"], "")
    operation = _first(row, ["Operation", "operation", "EdgeType", "edge_type", "event_type", "type", "predicate"], "UNKNOWN_EVENT")
    path = _first(row, ["Path", "path", "DstId", "dst_id", "object", "object_id", "target", "dst", "file", "registry", "socket"], "")

    src_id = _first(row, ["SrcId", "src_id", "source", "source_id", "subject", "subject_id", "src"])
    if src_id is None:
        src_id = f"{proc_name or 'process'}:{pid}" if pid != "" else str(proc_name or "<process>")
    dst_id = _first(row, ["DstId", "dst_id", "target", "target_id", "object", "object_id", "dst"], path)
    src_type = _norm_type(_first(row, ["SrcType", "src_type", "source_type", "subject_type"], "PROCESS"))
    dst_type = _norm_type(_first(row, ["DstType", "dst_type", "target_type", "object_type"], None))
    if dst_type == "UNKNOWN":
        dst_type = infer_dst_type(str(dst_id), str(operation))
    tag = _to_int_tag(_first(row, ["Tag", "tag", "label", "malicious", "is_malicious", "attack", "y"], 0))
    time = _first(row, ["Time", "time", "timestamp", "Time of Day", "datetime"], None)
    if sanitize:
        src_id = normalize_path(str(src_id)) if src_type != "PROCESS" else str(src_id).lower()
        dst_id = normalize_path(str(dst_id))
    return EventRecord(
        src_id=str(src_id), src_type=src_type, dst_id=str(dst_id), dst_type=dst_type,
        edge_type=str(operation), time=time, tag=tag, raw=row
    )


def _iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text_auto(path) as f:
        # Try to let csv sniff; fallback to comma.
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            yield row


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text_auto(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    # Handles normal JSON arrays/objects and DARPA-style line-delimited JSON.
    yield from iter_json_records(path)


def iter_event_rows(path: str | Path, input_format: str = "auto") -> Iterator[Dict[str, Any]]:
    path = Path(path)
    fmt = input_format.lower()
    # CDM rows are converted directly to EventRecord by iter_events/read_events.
    if fmt in {"cdm", "cdm_json", "darpa_cdm", "darpa_cdm_json"}:
        raise ValueError("iter_event_rows does not emit raw CDM rows; use iter_events/read_events instead")
    if fmt == "auto":
        name = path.name.lower()
        ext = path.suffix.lower()
        if name.endswith((".jsonl", ".jsonl.gz", ".jsonl.bz2", ".ndjson", ".ndjson.gz", ".ndjson.bz2")):
            fmt = "jsonl"
        elif ".json" in name or ext == ".json":
            fmt = "json"
        else:
            fmt = "csv"
    if fmt == "csv":
        yield from _iter_csv(path)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path)
    elif fmt == "json":
        yield from _iter_json(path)
    else:
        raise ValueError(f"Unsupported input_format={input_format}")


def iter_events(
    paths: Iterable[str | Path],
    input_format: str = "auto",
    sanitize: bool = True,
    filter_selected_events: bool = False,
    label_dir: str | Path | None = None,
    cdm_information_flow: bool = True,
) -> Iterator[EventRecord]:
    fmt = input_format.lower()
    path_list = list(paths)
    if fmt in {"cdm", "cdm_json", "darpa_cdm", "darpa_cdm_json"}:
        yield from iter_cdm_events(path_list, label_dir=label_dir, sanitize=sanitize, information_flow=cdm_information_flow)
        return
    for path in path_list:
        for row in iter_event_rows(path, input_format=input_format):
            ev = row_to_event(row, sanitize=sanitize)
            if filter_selected_events and ev.edge_type not in SELECTED_EVENTS:
                continue
            yield ev


def read_events(
    paths: Iterable[str | Path],
    input_format: str = "auto",
    sanitize: bool = True,
    filter_selected_events: bool = False,
    max_events: int | None = None,
    label_dir: str | Path | None = None,
    cdm_information_flow: bool = True,
) -> list[EventRecord]:
    events: list[EventRecord] = []
    for ev in iter_events(
        paths, input_format=input_format, sanitize=sanitize,
        filter_selected_events=filter_selected_events, label_dir=label_dir,
        cdm_information_flow=cdm_information_flow,
    ):
        if filter_selected_events and ev.edge_type not in SELECTED_EVENTS:
            continue
        events.append(ev)
        if max_events is not None and len(events) >= max_events:
            return events
    return events



_CDM_SHARD_RE = re.compile(
    r"^(?P<prefix>.*?)(?:-(?P<group>\d+))?\.json(?P<part>\.\d+)?(?P<compression>\.(?:gz|bz2|bzip2|xz|lzma|zst|zstd))?$",
    re.IGNORECASE,
)
_SPLIT_NUM_RE = re.compile(r"(\d+)")


def _natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower() for x in _SPLIT_NUM_RE.split(text)]


def _cdm_shard_sort_key(path: Path):
    """Sort DARPA CDM shards in their intended stream order.

    Windows/extracted DARPA files are often named like::

        ta1-cadets-e3-official.json
        ta1-cadets-e3-official.json.1
        ta1-cadets-e3-official-1.json
        ta1-cadets-e3-official-1.json.1
        ta1-cadets-e3-official-2.json

    Plain lexicographic sorting incorrectly places ``official-1`` before the
    unnumbered ``official`` base shard.  That changes chronology and can move
    object declarations or attack windows into the wrong train/val/test split.
    This key treats the unnumbered base as group 0, then -1, -2, ...; within
    each group it sorts .json, .json.1, .json.2, ... .
    """
    name = path.name
    low = name.lower()
    m = _CDM_SHARD_RE.match(low)
    if m:
        prefix = m.group("prefix")
        group = int(m.group("group") or 0)
        part = m.group("part")
        part_no = int(part[1:]) if part else 0
        # Put compressed form of the same logical shard after the uncompressed
        # file if both are accidentally present; normally only one exists.
        compressed = 1 if m.group("compression") else 0
        return (str(path.parent).lower(), _natural_key(prefix), group, part_no, compressed, low)
    return (str(path.parent).lower(), _natural_key(low), 0, 0, 0, low)


def _sort_input_files(files: list[Path], sort_mode: str | None = "auto") -> list[Path]:
    mode = (sort_mode or "auto").lower()
    if mode in {"name", "lexicographic", "lex"}:
        return sorted(files, key=lambda p: str(p).lower())
    if mode in {"mtime", "modified"}:
        return sorted(files, key=lambda p: (p.stat().st_mtime, str(p).lower()))
    # auto/natural/cdm_shards: CDM-aware natural order. It is also safe for
    # ordinary CSV/JSON names because non-CDM names fall back to natural order.
    return sorted(files, key=_cdm_shard_sort_key)


def _looks_like_auxiliary_text_file(path: Path) -> bool:
    name = path.name.lower()
    # Windows users often keep ta1-*.json.txt summaries beside the real 4GB
    # shards.  raw_glob="*.json*" would otherwise pick them up and silently
    # pollute preprocessing.
    return name.endswith(".txt") and (".json" in name or "ta1-" in name)


def discover_input_files(raw_dir: str | Path, raw_glob: str | None = None, sort_mode: str | None = "auto") -> list[Path]:
    raw_dir = Path(raw_dir)
    if raw_dir.is_file():
        return [] if _looks_like_auxiliary_text_file(raw_dir) else [raw_dir]
    if raw_glob:
        files = [p for p in raw_dir.glob(raw_glob) if p.is_file() and not _looks_like_auxiliary_text_file(p)]
        if files:
            return _sort_input_files(files, sort_mode)
    allowed_suffixes = (
        ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
        ".csv.gz", ".tsv.gz", ".json.gz", ".jsonl.gz", ".ndjson.gz",
        ".csv.bz2", ".tsv.bz2", ".json.bz2", ".jsonl.bz2", ".ndjson.bz2",
        ".csv.xz", ".json.xz", ".jsonl.xz", ".zst", ".zstd",
    )
    files: list[Path] = []
    for p in raw_dir.rglob("*"):
        if not p.is_file() or _looks_like_auxiliary_text_file(p):
            continue
        name = p.name.lower()
        # CDM shards are sometimes named *.json.1, *.json.2, etc.
        if name.endswith(allowed_suffixes) or ".json." in name or ".jsonl." in name:
            files.append(p)
    return _sort_input_files(files, sort_mode)
