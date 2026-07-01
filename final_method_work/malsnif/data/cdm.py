from __future__ import annotations

"""DARPA Transparent Computing CDM/JSON preprocessing utilities.

The DARPA TC releases usually store records as Avro-style JSON objects.  In
that representation, a record is often wrapped like

    {"datum": {"com.bbn.tc.schema.avro.cdm18.Event": {...}}}

and nullable/union fields are wrapped as {"string": "..."}, {"long": 1}, or
{"com.bbn.tc.schema.avro.cdm18.UUID": "..."}.  This module intentionally
implements a permissive parser because different TC exports and converted
JSON dumps use slightly different wrappers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
import bz2
import csv
import gzip
import io
import json
import lzma
import re

from malsnif.constants import NODE_TYPE_NORMALIZATION
from malsnif.data.events import EventRecord
from malsnif.data.sanitize import normalize_path

# Event types whose information flow is normally object -> subject.
# For the remaining events we keep the CDM subject -> predicate-object order.
READ_LIKE_EVENTS = {
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_MMAP",
    "EVENT_LOADLIBRARY",
    "EVENT_OPEN",  # Conservative: opening a file gives the subject access to the object.
}

PROCESS_CREATE_EVENTS = {
    "EVENT_FORK",
    "EVENT_CLONE",
    "EVENT_EXECUTE",
    "EVENT_UNIT",
}

CDM_ENTITY_TO_NODE_TYPE = {
    "Subject": "PROCESS",
    "Principal": "PROCESS",
    "FileObject": "FILE",
    "UnnamedPipeObject": "FILE",
    "RegistryKeyObject": "REGISTRY",
    "NetFlowObject": "NETWORK",
    "SrcSinkObject": "NETWORK",
    "IpcObject": "NETWORK",
    "MemoryObject": "FILE",
}

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")
PATHISH_RE = re.compile(r"(?:/|\\|[A-Za-z]:\\|[A-Za-z]:/)")
POSITIVE_CONTEXT_RE = re.compile(r"mal|attack|ioc|indicator|compromise|ground.?truth|truth|positive|apt|drakon|backdoor", re.I)
UUID_CONTEXT_RE = re.compile(r"uuid|guid|id|node|entity|subject|object|event|src|dst|source|target", re.I)
PATH_CONTEXT_RE = re.compile(r"path|file|filename|process|cmd|command|image|exe|name|predicate", re.I)
EVENT_CONTEXT_RE = re.compile(r"event.?type|operation|predicate|type", re.I)


@dataclass
class CdmObject:
    uuid: str
    cdm_type: str
    node_type: str
    display: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CdmParseStats:
    records_seen: int = 0
    objects_seen: int = 0
    events_seen: int = 0
    events_emitted: int = 0
    events_missing_subject: int = 0
    events_missing_object: int = 0
    events_without_label: int = 0
    events_with_predicate_path: int = 0
    events_semantic_object_from_predicate_path: int = 0


@dataclass
class TimeRangeRule:
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None
    uuid: str = ""
    path: str = ""
    event_type: str = ""
    description: str = ""


class CdmLabeler:
    """Flexible label loader for raw DARPA CDM/JSON data.

    Supported files under label_dir:
    - malicious_uuids.txt: one UUID per line. Comments start with #.
    - malicious_paths.txt: one substring per line, or regex when prefixed by re:.
    - malicious_event_types.txt: one event type per line, e.g. EVENT_WRITE.
    - malicious_time_ranges.csv: start_ns,end_ns[,uuid,path,event_type,description].
    - malicious_events.csv: flexible CSV; UUID/path/time columns are consumed when present.
    """

    def __init__(self) -> None:
        self.uuids: set[str] = set()
        self.path_substrings: list[str] = []
        self.path_regexes: list[re.Pattern[str]] = []
        self.event_types: set[str] = set()
        self.time_rules: list[TimeRangeRule] = []
        self.loaded_files: list[str] = []

    @property
    def has_labels(self) -> bool:
        return bool(self.uuids or self.path_substrings or self.path_regexes or self.event_types or self.time_rules)

    @classmethod
    def from_dir(cls, label_dir: str | Path | None) -> "CdmLabeler":
        lab = cls()
        if not label_dir:
            return lab
        root = Path(label_dir)
        if not root.exists():
            return lab
        lab._load_txt(root / "malicious_uuids.txt", kind="uuid")
        lab._load_txt(root / "malicious_paths.txt", kind="path")
        lab._load_txt(root / "malicious_event_types.txt", kind="event_type")
        lab._load_time_ranges(root / "malicious_time_ranges.csv")
        lab._load_malicious_events(root / "malicious_events.csv")
        # Dataset-specific ground-truth helper files are often named after the
        # host/source rather than converted to malicious_*.txt.  Examples seen in
        # local DARPA TC reproductions include labels/theia.txt, labels/cadets.txt
        # and labels/trace.txt.  Treat these as positive indicator lists and parse
        # UUIDs, paths and EVENT_* strings permissively.  Files that are likely to
        # be notes or generated manifests are ignored.  This keeps canonical
        # malicious_*.txt/csv semantics unchanged while allowing the user's raw GT
        # filename to work without manual renaming.
        for candidate in sorted(root.glob("*.txt")):
            low = candidate.name.lower()
            if low.startswith("malicious_") or low in {"readme.txt", "manifest.txt"} or low.startswith("manifest."):
                continue
            lab._load_generic_positive_txt(candidate)
        # Many DARPA reproductions keep the original ground-truth helper file as
        # labels/_raw/cadets.json.  Load it permissively so users do not have to
        # manually convert common UUID/path JSON lists before validating the CDM
        # parser.  Explicit malicious_*.txt/csv files still take precedence and
        # are less ambiguous for strict experiments.
        for candidate in sorted(list(root.glob("*.json")) + list((root / "_raw").glob("*.json")) if (root / "_raw").exists() else list(root.glob("*.json"))):
            if candidate.name.lower() in {"metadata.json", "config.json"}:
                continue
            lab._load_json_labels(candidate)
        for candidate in sorted(list(root.glob("*.jsonl")) + list((root / "_raw").glob("*.jsonl")) if (root / "_raw").exists() else list(root.glob("*.jsonl"))):
            lab._load_json_labels(candidate)
        return lab

    def summary(self) -> dict[str, Any]:
        return {
            "has_labels": self.has_labels,
            "loaded_files": self.loaded_files,
            "uuid_count": len(self.uuids),
            "path_substring_count": len(self.path_substrings),
            "path_regex_count": len(self.path_regexes),
            "event_type_count": len(self.event_types),
            "time_rule_count": len(self.time_rules),
        }

    def _iter_clean_lines(self, path: Path) -> Iterator[str]:
        if not path.exists():
            return
        self.loaded_files.append(str(path))
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line

    def _load_txt(self, path: Path, kind: str) -> None:
        for line in self._iter_clean_lines(path) or []:
            if kind == "uuid":
                self.uuids.add(line.lower())
            elif kind == "event_type":
                self.event_types.add(line.upper())
            elif kind == "path":
                if line.startswith("re:"):
                    self.path_regexes.append(re.compile(line[3:], re.IGNORECASE))
                else:
                    self.path_substrings.append(normalize_path(line).lower())

    def _load_generic_positive_txt(self, path: Path) -> None:
        """Load a dataset-specific ground-truth .txt as positive indicators.

        This is intentionally more permissive than malicious_uuids.txt because
        raw GT files may contain comments, prose, UUIDs, paths, event types, or
        whitespace/CSV-like mixtures.  We parse UUIDs anywhere in the line, treat
        path-like tokens as malicious path substrings, and consume EVENT_* tokens.
        If a line contains no UUID/path/event-type but looks like a simple token,
        it is stored as a path substring because many GT helper files list process
        names or file fragments.
        """
        for line in self._iter_clean_lines(path) or []:
            # Remove common inline comment separators while preserving paths.
            raw = re.split(r"\s+#|\s+//", line, maxsplit=1)[0].strip()
            if not raw:
                continue
            self._consume_label_string(raw, key_context=path.stem, positive_context=True)
            # Additional robust tokenization for CSV/table/prose style GT files.
            tokens = re.split(r"[\s,;|]+", raw)
            consumed_specific = bool(UUID_RE.search(raw) or PATHISH_RE.search(raw) or "EVENT_" in raw.upper())
            for tok in tokens:
                tok = tok.strip().strip('"\'[](){}<>')
                if not tok or len(tok) < 2:
                    continue
                for m in UUID_RE.findall(tok):
                    self.uuids.add(m.lower())
                    consumed_specific = True
                if tok.upper().startswith("EVENT_"):
                    self.event_types.add(tok.split(".")[-1].upper())
                    consumed_specific = True
                if PATHISH_RE.search(tok):
                    self.path_substrings.append(normalize_path(tok).lower())
                    consumed_specific = True
            if not consumed_specific and len(raw) <= 512:
                # Many hand-written GT files list short command/process/path
                # fragments without a slash, e.g. drakon or firefox.
                self.path_substrings.append(normalize_path(raw).lower())

    def _load_time_ranges(self, path: Path) -> None:
        if not path.exists():
            return
        self.loaded_files.append(str(path))
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rule = TimeRangeRule(
                    start_ns=_to_int_or_none(_first(row, ["start_ns", "start", "begin_ns", "begin"])),
                    end_ns=_to_int_or_none(_first(row, ["end_ns", "end", "finish_ns", "finish"])),
                    uuid=str(_first(row, ["uuid", "subject_uuid", "object_uuid", "event_uuid"], "") or "").lower(),
                    path=normalize_path(str(_first(row, ["path", "file", "process", "cmd", "dst_path"], "") or "")).lower(),
                    event_type=str(_first(row, ["event_type", "type", "operation"], "") or "").upper(),
                    description=str(_first(row, ["description", "attack", "note"], "") or ""),
                )
                self.time_rules.append(rule)

    def _load_malicious_events(self, path: Path) -> None:
        if not path.exists():
            return
        self.loaded_files.append(str(path))
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ["uuid", "event_uuid", "subject_uuid", "object_uuid", "src_uuid", "dst_uuid"]:
                    v = row.get(key)
                    if v:
                        self.uuids.add(str(v).strip().lower())
                path_val = _first(row, ["path", "file", "process", "cmd", "dst_path", "predicateObjectPath"])
                if path_val:
                    self.path_substrings.append(normalize_path(str(path_val)).lower())
                event_type = _first(row, ["event_type", "type", "operation"])
                if event_type:
                    self.event_types.add(str(event_type).upper())
                if _first(row, ["start_ns", "start", "begin_ns"]) or _first(row, ["end_ns", "end", "finish_ns"]):
                    self.time_rules.append(TimeRangeRule(
                        start_ns=_to_int_or_none(_first(row, ["start_ns", "start", "begin_ns"])),
                        end_ns=_to_int_or_none(_first(row, ["end_ns", "end", "finish_ns"])),
                        uuid=str(_first(row, ["uuid", "event_uuid", "subject_uuid", "object_uuid"], "") or "").lower(),
                        path=normalize_path(str(path_val or "")).lower(),
                        event_type=str(event_type or "").upper(),
                        description=str(_first(row, ["description", "attack", "note"], "") or ""),
                    ))

    def _load_json_labels(self, path: Path) -> None:
        if not path.exists():
            return
        self.loaded_files.append(str(path))
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if not text:
                return
            try:
                obj = json.loads(text)
                self._consume_label_json(obj, key_context=path.stem, positive_context=bool(POSITIVE_CONTEXT_RE.search(path.stem)))
            except json.JSONDecodeError:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._consume_label_json(json.loads(line), key_context=path.stem, positive_context=bool(POSITIVE_CONTEXT_RE.search(path.stem)))
                    except json.JSONDecodeError:
                        self._consume_label_string(line, key_context=path.stem, positive_context=True)
        except Exception:
            # Labels are optional; malformed helper files should not prevent raw
            # CDM parsing.  The absence of loaded labels is surfaced in metadata.
            return

    def _consume_label_json(self, obj: Any, key_context: str = "", positive_context: bool = False) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            self._consume_label_string(obj, key_context=key_context, positive_context=positive_context or not key_context)
            return
        if isinstance(obj, (int, float, bool)):
            return
        if isinstance(obj, list):
            # A bare list in a label JSON is normally a positive list of UUIDs,
            # paths, or event objects.
            child_positive = positive_context or bool(POSITIVE_CONTEXT_RE.search(key_context)) or not key_context
            for item in obj:
                self._consume_label_json(item, key_context=key_context, positive_context=child_positive)
            return
        if not isinstance(obj, dict):
            return

        explicit_label = _first(obj, ["label", "tag", "malicious", "is_malicious", "attack", "is_attack", "y"], None)
        if explicit_label is not None:
            positive_context = _json_positive(explicit_label)
        positive_context = positive_context or bool(POSITIVE_CONTEXT_RE.search(key_context))

        # Time range row support.
        start = _to_int_or_none(_first(obj, ["start_ns", "start", "begin_ns", "begin", "startTimeNanos"]))
        end = _to_int_or_none(_first(obj, ["end_ns", "end", "finish_ns", "finish", "endTimeNanos"]))
        if positive_context and (start is not None or end is not None):
            path_val = _first(obj, ["path", "file", "filename", "process", "cmd", "command", "predicateObjectPath"], "")
            event_type = _first(obj, ["event_type", "eventType", "type", "operation"], "")
            uuid_val = _first(obj, ["uuid", "event_uuid", "subject_uuid", "object_uuid", "src_uuid", "dst_uuid", "id"], "")
            self.time_rules.append(TimeRangeRule(
                start_ns=start,
                end_ns=end,
                uuid=str(uuid_val or "").lower(),
                path=normalize_path(str(path_val or "")).lower() if path_val else "",
                event_type=str(event_type or "").split(".")[-1].upper(),
                description=str(_first(obj, ["description", "attack", "note"], "") or ""),
            ))

        for k, v in obj.items():
            k_str = str(k)
            child_positive = positive_context or bool(POSITIVE_CONTEXT_RE.search(k_str))
            if isinstance(v, str):
                # Add UUID-like strings when the key or surrounding context says
                # this is a label/indicator.  This covers common cadets.json
                # ground-truth files without blindly treating arbitrary text as a
                # malicious indicator.
                if child_positive or UUID_CONTEXT_RE.search(k_str):
                    for m in UUID_RE.findall(v):
                        self.uuids.add(m.lower())
                if child_positive and PATH_CONTEXT_RE.search(k_str) and PATHISH_RE.search(v):
                    self.path_substrings.append(normalize_path(v).lower())
                if child_positive and EVENT_CONTEXT_RE.search(k_str) and str(v).upper().startswith("EVENT_"):
                    self.event_types.add(str(v).split(".")[-1].upper())
            else:
                self._consume_label_json(v, key_context=k_str, positive_context=child_positive)

    def _consume_label_string(self, value: str, key_context: str = "", positive_context: bool = False) -> None:
        if not value:
            return
        for m in UUID_RE.findall(value):
            self.uuids.add(m.lower())
        if positive_context and PATHISH_RE.search(value):
            self.path_substrings.append(normalize_path(value).lower())
        if positive_context and value.upper().startswith("EVENT_"):
            self.event_types.add(value.split(".")[-1].upper())

    def match(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Return detailed label-match information for one CDM event.

        The older reproduction only returned a binary event tag and then marked
        both endpoints as malicious.  That is too aggressive for DARPA ground
        truth files that are entity-UUID lists: a benign process that merely
        touches a known malicious file/socket becomes a positive process node.
        This method records which UUID fields actually matched so graph building
        can label only the matched endpoint when requested.
        """
        out: dict[str, Any] = {
            "matched": False,
            "uuid_fields": [],
            "matched_uuid_values": [],
            "event_type": False,
            "path": False,
            "time_rule": False,
            "fallback_event_label": False,
        }
        if not self.has_labels:
            return out
        event_type = str(meta.get("event_type") or "").upper()
        uuid_fields = ["event_uuid", "src_uuid", "dst_uuid", "subject_uuid", "object_uuid"]
        for field_name in uuid_fields:
            value = meta.get(field_name)
            if value and str(value).lower() in self.uuids:
                out["uuid_fields"].append(field_name)
                out["matched_uuid_values"].append(str(value).lower())
        paths = [normalize_path(str(x)).lower() for x in [
            meta.get("src_display"), meta.get("dst_display"), meta.get("predicate_path")
        ] if x]
        timestamp_ns = _to_int_or_none(meta.get("timestamp_ns"))
        if event_type and event_type in self.event_types:
            out["event_type"] = True
        for p in paths:
            if any(s and s in p for s in self.path_substrings) or any(r.search(p) for r in self.path_regexes):
                out["path"] = True
                break
        for r in self.time_rules:
            if r.event_type and r.event_type != event_type:
                continue
            if timestamp_ns is not None:
                if r.start_ns is not None and timestamp_ns < r.start_ns:
                    continue
                if r.end_ns is not None and timestamp_ns > r.end_ns:
                    continue
            elif r.start_ns is not None or r.end_ns is not None:
                continue
            event_uuids = {str(meta.get(k) or "").lower() for k in uuid_fields if meta.get(k)}
            if r.uuid and r.uuid not in event_uuids:
                continue
            if r.path and not any(r.path in p for p in paths):
                continue
            out["time_rule"] = True
            break
        out["fallback_event_label"] = bool(out["event_type"] or out["path"] or out["time_rule"] or ("event_uuid" in out["uuid_fields"]))
        out["matched"] = bool(out["uuid_fields"] or out["event_type"] or out["path"] or out["time_rule"])
        return out

    def tag(self, meta: dict[str, Any]) -> int:
        return int(bool(self.match(meta).get("matched")))


def _json_positive(v: Any) -> bool:
    v = unwrap_avro(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > 0
    s = str(v).strip().lower()
    return s in {"1", "true", "mal", "malicious", "attack", "yes", "y", "positive"}


def _to_int_or_none(v: Any) -> Optional[int]:
    v = unwrap_avro(v)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _first(row: Dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    lower = {str(k).lower(): k for k in row.keys()}
    for c in candidates:
        if c.lower() in lower:
            v = row[lower[c.lower()]]
            if v is not None and str(v) != "":
                return v
    return default


def open_text_auto(path: str | Path):
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if name.endswith(".bz2") or name.endswith(".bzip2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    if name.endswith(".xz") or name.endswith(".lzma"):
        return lzma.open(path, "rt", encoding="utf-8", errors="ignore")
    if name.endswith(".zst") or name.endswith(".zstd"):
        try:
            import zstandard as zstd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Reading .zst files requires: pip install zstandard") from exc
        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def unwrap_avro(value: Any) -> Any:
    """Unwrap common Avro JSON union wrappers while preserving normal dicts."""
    if isinstance(value, dict) and len(value) == 1:
        k, v = next(iter(value.items()))
        lk = str(k).lower()
        if lk in {"string", "int", "long", "float", "double", "boolean", "bytes", "null"}:
            return unwrap_avro(v)
        if str(k).startswith("com.bbn.tc.schema"):
            return unwrap_avro(v)
        if lk in {"array", "map"}:
            return unwrap_avro(v)
    return value


def cdm_record_type(obj: dict[str, Any]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Return (short_type, record_payload) for a CDM JSON record."""
    if not isinstance(obj, dict):
        return None, None
    datum = obj.get("datum", obj)
    # Do this before generic union unwrapping so the schema class name is not lost.
    if isinstance(datum, dict) and len(datum) == 1:
        k, v = next(iter(datum.items()))
        if str(k).startswith("com.bbn.tc.schema"):
            payload = unwrap_avro(v)
            return str(k).split(".")[-1], payload if isinstance(payload, dict) else None
    datum = unwrap_avro(datum)
    if not isinstance(datum, dict):
        return None, None
    # Some converters flatten records but preserve a type name.
    for type_key in ["cdm_type", "record_type", "_type", "schema_type"]:
        if type_key in datum:
            return str(datum[type_key]).split(".")[-1], datum
    # Heuristic fallback.
    if "subject" in datum and ("predicateObject" in datum or "timestampNanos" in datum):
        return "Event", datum
    if "cmdLine" in datum or "parentSubject" in datum:
        return "Subject", datum
    if "filename" in datum:
        return "FileObject", datum
    if "remoteAddress" in datum or "localAddress" in datum:
        return "NetFlowObject", datum
    return None, datum


def get_uuid(value: Any) -> str:
    value = unwrap_avro(value)
    if value is None:
        return ""
    if isinstance(value, dict):
        # UUID may appear as {"uuid": ...} or {"string": ...} after partial unwrapping.
        for k in ["uuid", "UUID", "id", "value"]:
            if k in value:
                return str(unwrap_avro(value[k]))
        if len(value) == 1:
            return str(unwrap_avro(next(iter(value.values()))))
    return str(value)


def get_field(record: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    lower = {str(k).lower(): k for k in record.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            v = unwrap_avro(record[key])
            if v is not None and str(v) != "":
                return v
    return default


def _flatten_scalar_strings(value: Any, max_depth: int = 6) -> list[str]:
    """Return scalar strings from possibly nested Avro JSON structures.

    DARPA CDM converters differ in how they expose object attributes.  Some put
    fields such as filename/cmdLine/localAddress at the top level; others keep
    them below baseObject.properties.map or Avro union wrappers.  This helper is
    intentionally conservative and is only used for display/token extraction,
    not for identity or labels.
    """
    out: list[str] = []

    def rec(x: Any, depth: int) -> None:
        if depth < 0 or x is None:
            return
        x = unwrap_avro(x)
        if isinstance(x, (str, int, float, bool)):
            sx = str(x)
            if sx and sx.lower() != "none":
                out.append(sx)
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v, depth - 1)
        elif isinstance(x, list):
            for v in x:
                rec(v, depth - 1)

    rec(value, max_depth)
    return out


def get_field_deep(record: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    """Find a field by key at any shallow depth in a CDM object/event record.

    CDM JSON converters differ substantially.  Some expose fields directly
    (``predicateObjectPath``), while others keep values inside Avro maps or
    property lists such as ``[{"key": "filename", "value": ...}]``.
    This function is deliberately used only for display/token extraction, not
    for identity or label decisions, so permissive recursive matching is safe
    and helps preserve the semantic path/command tokens required by MalSnif.
    """
    wanted = {str(n).lower() for n in names}
    best: Any = None

    def non_empty(v: Any) -> bool:
        v = unwrap_avro(v)
        return v is not None and str(v) != ""

    def maybe_property_pair(x: dict[str, Any]) -> Any:
        # Common representations: {key: "path", value: "/tmp/a"},
        # {name: "cmdLine", string: "..."}, or nested Avro union variants.
        key_candidates = ["key", "name", "field", "property", "propertyName"]
        val_candidates = ["value", "stringValue", "intValue", "longValue", "floatValue", "doubleValue", "bytesValue", "val"]
        k = None
        for kk in key_candidates:
            if kk in x:
                k = unwrap_avro(x.get(kk))
                break
        if k is None:
            return None
        lk = str(k).lower()
        if lk not in wanted:
            return None
        for vk in val_candidates:
            if vk in x and non_empty(x.get(vk)):
                return unwrap_avro(x.get(vk))
        # If the matched property has only one non-key value, use it.
        for kk, vv in x.items():
            if kk not in key_candidates and non_empty(vv):
                return unwrap_avro(vv)
        return None

    def rec(x: Any, depth: int) -> None:
        nonlocal best
        if best is not None or depth < 0 or x is None:
            return
        x = unwrap_avro(x)
        if isinstance(x, dict):
            pair_val = maybe_property_pair(x)
            if non_empty(pair_val):
                best = pair_val
                return
            for k, v in x.items():
                if str(k).lower() in wanted:
                    vv = unwrap_avro(v)
                    if non_empty(vv):
                        best = vv
                        return
            # Prefer explicit properties/map containers before arbitrary raw fields.
            ordered = []
            for key in ["predicateObjectPath", "properties", "map", "baseObject", "propertiesMap", "propertyMap"]:
                if key in x:
                    ordered.append(x[key])
            ordered.extend(v for k, v in x.items() if k not in {"predicateObjectPath", "properties", "map", "baseObject", "propertiesMap", "propertyMap"})
            for v in ordered:
                rec(v, depth - 1)
                if best is not None:
                    return
        elif isinstance(x, list):
            for v in x:
                rec(v, depth - 1)
                if best is not None:
                    return

    rec(record, 8)
    return default if best is None else best

def _display_value(value: Any) -> str:
    value = unwrap_avro(value)
    if isinstance(value, list):
        return " ".join(str(unwrap_avro(x)) for x in value if unwrap_avro(x) not in {None, ""})
    if isinstance(value, dict):
        # Avro maps sometimes remain as nested dicts. Pick the most path-like or
        # command-like scalar instead of serialising the whole JSON blob.
        vals = _flatten_scalar_strings(value)
        if vals:
            pathish = [v for v in vals if PATHISH_RE.search(v) or ":" in v or "." in v]
            return str((pathish or vals)[0])
    return str(value or "")


def _norm_node_type(cdm_type: str, record: dict[str, Any] | None = None) -> str:
    base = CDM_ENTITY_TO_NODE_TYPE.get(cdm_type)
    if base:
        return base
    raw_type = get_field(record or {}, ["type"], "")
    key = str(raw_type or cdm_type).strip().lower().replace("_", "").replace(" ", "")
    return NODE_TYPE_NORMALIZATION.get(key, "UNKNOWN")


def _display_for_object(cdm_type: str, record: dict[str, Any], uuid: str) -> str:
    if cdm_type == "Subject":
        cmd = get_field_deep(record, ["cmdLine", "commandLine", "cmd", "exec", "path", "name", "programName", "image"], None)
        return _display_value(cmd) or f"subject:{uuid}"
    if cdm_type == "FileObject":
        v = get_field_deep(record, ["filename", "fileName", "filepath", "filePath", "path", "name", "objectPath"], None)
        return _display_value(v) or f"file:{uuid}"
    if cdm_type == "RegistryKeyObject":
        v = get_field_deep(record, ["key", "keyName", "path", "name", "objectPath"], None)
        return _display_value(v) or f"registry:{uuid}"
    if cdm_type == "NetFlowObject":
        local_addr = _display_value(get_field_deep(record, ["localAddress", "srcAddress", "sourceAddress", "localHost", "sourceHost"], ""))
        local_port = _display_value(get_field_deep(record, ["localPort", "srcPort", "sourcePort"], ""))
        remote_addr = _display_value(get_field_deep(record, ["remoteAddress", "dstAddress", "destinationAddress", "remoteHost", "destinationHost"], ""))
        remote_port = _display_value(get_field_deep(record, ["remotePort", "dstPort", "destinationPort"], ""))
        proto = _display_value(get_field_deep(record, ["ipProtocol", "protocol"], ""))
        left = f"{local_addr}:{local_port}" if local_addr or local_port else "local"
        right = f"{remote_addr}:{remote_port}" if remote_addr or remote_port else "remote"
        return f"{proto}:{left}->{right}" if proto else f"{left}->{right}"
    if cdm_type in {"SrcSinkObject", "IpcObject", "UnnamedPipeObject"}:
        v = get_field_deep(record, ["name", "path", "source", "sink", "objectPath"], None)
        return _display_value(v) or f"{cdm_type}:{uuid}"
    v = get_field_deep(record, ["name", "path", "value", "objectPath"], None)
    return _display_value(v) or f"{cdm_type}:{uuid}"


def object_from_record(cdm_type: str, record: dict[str, Any]) -> Optional[CdmObject]:
    uuid = get_uuid(get_field(record, ["uuid", "id", "UUID"]))
    if not uuid and isinstance(record.get("baseObject"), dict):
        uuid = get_uuid(get_field(record["baseObject"], ["uuid", "id", "UUID"]))
    if not uuid:
        return None
    node_type = _norm_node_type(cdm_type, record)
    return CdmObject(uuid=uuid, cdm_type=cdm_type, node_type=node_type, display=_display_for_object(cdm_type, record, uuid), raw=record)


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate JSONL, NDJSON, one JSON array, or one JSON object."""
    with open_text_auto(path) as f:
        # Most DARPA CDM JSON exports are line-delimited; stream first to avoid loading huge files.
        first = ""
        while True:
            pos = f.tell() if hasattr(f, "tell") else None
            line = f.readline()
            if not line:
                return
            stripped = line.lstrip()
            if stripped:
                first = stripped[:1]
                if pos is not None:
                    f.seek(pos)
                else:  # e.g. non-seekable zstd stream: process this line and continue.
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, dict):
                                    yield item
                        elif isinstance(obj, dict):
                            yield obj
                    except json.JSONDecodeError:
                        pass
                break
        if first == "[":
            try:
                arr = json.load(f)
                for item in arr:
                    if isinstance(item, dict):
                        yield item
                return
            except Exception:
                f.seek(0)
        # JSONL / concatenated line-oriented records.
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(obj, dict):
                # Wrapper with {"events": [...]} also supported.
                emitted = False
                for key in ["events", "data", "records", "rows"]:
                    if key in obj and isinstance(obj[key], list):
                        for item in obj[key]:
                            if isinstance(item, dict):
                                yield item
                        emitted = True
                        break
                if not emitted:
                    yield obj


def event_from_cdm(
    record: dict[str, Any],
    objects: dict[str, CdmObject],
    labeler: CdmLabeler,
    sanitize: bool = True,
    information_flow: bool = True,
) -> Optional[EventRecord]:
    event_uuid = get_uuid(get_field(record, ["uuid", "id", "event_uuid"]))
    event_type = str(get_field(record, ["type", "eventType", "operation"], "EVENT_UNKNOWN") or "EVENT_UNKNOWN")
    event_type = event_type.split(".")[-1].upper()
    subject_uuid = get_uuid(get_field(record, ["subject", "src", "source", "sourceUuid", "subjectUuid"]))
    predicate_uuid = get_uuid(get_field(record, ["predicateObject", "object", "dst", "target", "predicateObjectUuid", "objectUuid"]))
    predicate2_uuid = get_uuid(get_field(record, ["predicateObject2", "object2", "target2"]))
    predicate_path = _display_value(get_field_deep(record, ["predicateObjectPath", "path", "objectPath", "filePath", "filename", "fileName"], ""))
    timestamp_ns = _to_int_or_none(get_field(record, ["timestampNanos", "timestamp", "time", "sequence"]))

    subj = objects.get(subject_uuid)
    obj_uuid = predicate_uuid or predicate2_uuid
    obj = objects.get(obj_uuid) if obj_uuid else None

    src_uuid = subject_uuid
    dst_uuid = obj_uuid
    src_type = subj.node_type if subj else "PROCESS"
    dst_type = obj.node_type if obj else _infer_node_type_from_event(event_type, predicate_path)
    src_display = subj.display if subj else (f"subject:{subject_uuid}" if subject_uuid else "<missing-subject>")
    obj_display = obj.display if obj else ""
    fallback_display = (not obj_display) or bool(obj_uuid and obj_display.lower() in {f"file:{obj_uuid}".lower(), f"subject:{obj_uuid}".lower(), f"registry:{obj_uuid}".lower(), f"netflowobject:{obj_uuid}".lower(), f"object:{obj_uuid}".lower()})
    dst_display = obj_display if obj_display and not fallback_display else (predicate_path or obj_display or (f"object:{obj_uuid}" if obj_uuid else f"event:{event_uuid or event_type}"))
    # Semantic object is always the event predicate object before optional
    # information-flow reversal.  Prefer predicateObjectPath whenever present,
    # because DARPA CDM FileObject/Subject records often only carry UUIDs while
    # the per-event path contains the semantic token MalSnif needs.
    semantic_object_display = predicate_path or dst_display

    # Preserve structural identity with UUIDs, but keep display strings for NLP tokens.
    src_id = f"{src_type.lower()}:{src_uuid or normalize_path(src_display)}"
    dst_id = f"{dst_type.lower()}:{dst_uuid or normalize_path(dst_display)}"
    if information_flow and event_type in READ_LIKE_EVENTS and dst_uuid:
        src_uuid, dst_uuid = dst_uuid, subject_uuid
        src_type, dst_type = dst_type, src_type
        src_display, dst_display = dst_display, src_display
        src_id, dst_id = dst_id, src_id

    meta = {
        "event_uuid": event_uuid,
        "event_type": event_type,
        "subject_uuid": subject_uuid,
        "object_uuid": obj_uuid,
        "src_uuid": src_uuid,
        "dst_uuid": dst_uuid,
        "src_display": src_display,
        "dst_display": dst_display,
        "predicate_path": predicate_path,
        "semantic_object_display": semantic_object_display,
        "timestamp_ns": timestamp_ns,
    }
    label_match = labeler.match(meta)
    tag = int(bool(label_match.get("matched")))
    if sanitize:
        # Keep UUID part stable; only normalize synthetic fallback components if no UUID exists.
        if not src_uuid:
            src_id = normalize_path(src_id)
        if not dst_uuid:
            dst_id = normalize_path(dst_id)
    return EventRecord(
        src_id=str(src_id),
        src_type=src_type,
        dst_id=str(dst_id),
        dst_type=dst_type,
        edge_type=event_type,
        time=timestamp_ns,
        tag=tag,
        raw={"cdm": record, **meta, "label_match": label_match},
    )


def _infer_node_type_from_event(event_type: str, path: str) -> str:
    e = event_type.upper()
    p = (path or "").lower()
    if any(x in e for x in ["SEND", "RECV", "CONNECT", "ACCEPT"]) or re.search(r"\d+\.\d+\.\d+\.\d+", p):
        return "NETWORK"
    if any(x in e for x in ["FORK", "CLONE", "EXECUTE", "UNIT"]):
        return "PROCESS"
    if "REG" in e or "registry" in p:
        return "REGISTRY"
    return "FILE"


def iter_cdm_events(
    paths: Iterable[str | Path],
    label_dir: str | Path | None = None,
    sanitize: bool = True,
    information_flow: bool = True,
    stats: Optional[CdmParseStats] = None,
    labeler: Optional[CdmLabeler] = None,
) -> Iterator[EventRecord]:
    objects: dict[str, CdmObject] = {}
    labeler = labeler or CdmLabeler.from_dir(label_dir)
    local_stats = stats or CdmParseStats()
    for path in paths:
        for obj in iter_json_records(path):
            local_stats.records_seen += 1
            cdm_type, record = cdm_record_type(obj)
            if not cdm_type or not isinstance(record, dict):
                continue
            if cdm_type == "Event":
                local_stats.events_seen += 1
                ev = event_from_cdm(record, objects, labeler, sanitize=sanitize, information_flow=information_flow)
                if ev is None:
                    continue
                if ev.raw and ev.raw.get("predicate_path"):
                    local_stats.events_with_predicate_path += 1
                if ev.raw and ev.raw.get("semantic_object_display") and ev.raw.get("predicate_path") and ev.raw.get("semantic_object_display") == ev.raw.get("predicate_path"):
                    local_stats.events_semantic_object_from_predicate_path += 1
                if ev.raw and str(ev.raw.get("subject_uuid") or "") and str(ev.raw.get("subject_uuid")) not in objects:
                    local_stats.events_missing_subject += 1
                if ev.raw and str(ev.raw.get("object_uuid") or "") and str(ev.raw.get("object_uuid")) not in objects:
                    local_stats.events_missing_object += 1
                if ev.tag == 0:
                    local_stats.events_without_label += 1
                local_stats.events_emitted += 1
                yield ev
            else:
                cdm_obj = object_from_record(cdm_type, record)
                if cdm_obj:
                    objects[cdm_obj.uuid] = cdm_obj
                    local_stats.objects_seen += 1


def is_probably_cdm_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(x in name for x in ["cdm", "ta1", "cadets", "theia", "trace", "fivedirections", "clearscope"])
