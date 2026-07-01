from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List

# A practical open reproduction of the paper's 80+28 regex idea.
# The full list was not printed in the paper; the README explains this limitation.
DEFAULT_REGEX_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\\", "/"),
    (r"^[a-zA-Z]:", ""),
    (r"s-1-5-[0-9]{1,2}(?:(?:-[0-9]{9,10}){3}-[0-9]{3,4})?", "<sid>"),
    (r"\{?[a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}\}?", "<guid>"),
    (r"/users/[^/]+", "/users/<user>"),
    (r"/home/[^/]+", "/home/<user>"),
    (r"/tmp/[A-Za-z0-9_.-]+", "/tmp/<tmp>"),
    (r"/var/tmp/[A-Za-z0-9_.-]+", "/var/tmp/<tmp>"),
    (r"/windows/system32/drivers", "[system32_drivers]"),
    (r"/windows/system32", "[system32]"),
    (r"/windows/syswow64", "[syswow64]"),
    (r"/windows", "[windows]"),
    (r"/program files \(x86\)", "[program_files_x86]"),
    (r"/program files", "[program_files]"),
    (r"/users/<user>/documents", "[documents]"),
    (r"/users/<user>/downloads", "[downloads]"),
    (r"/users/<user>/desktop", "[desktop]"),
    (r"/appdata/local/temp", "[temp]"),
    (r"/appdata/roaming", "[appdata_roaming]"),
    (r"/appdata/local", "[appdata_local]"),
    (r"/registry/(?:machine|lm)/software/microsoft/windows/currentversion", "[registry_lm_currentversion]"),
    (r"/registry/(?:user|cu)/software/microsoft/windows", "[registry_cu_windows]"),
    (r"/registry/(?:machine|lm)/software/wow6432node/microsoft/windows", "[registry_lm_wow64_windows]"),
    (r"\b[0-9a-fA-F]{16,}\b", "<hex>"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>"),
    (r"\b\d+\b", "<num>"),
]

_SPLIT_RE = re.compile(r"[^A-Za-z0-9_<>\[\].-]+")


def normalize_path(path: str | None, lowercase: bool = True) -> str:
    if path is None:
        return "<none>"
    s = str(path).strip().replace("\\", "/")
    if lowercase:
        s = s.lower()
    for pattern, repl in DEFAULT_REGEX_REPLACEMENTS:
        flags = re.IGNORECASE if lowercase else 0
        s = re.sub(pattern, repl, s, flags=flags)
    s = re.sub(r"/+", "/", s)
    return s or "<empty>"


def suffix_of(path: str | None) -> str:
    if not path:
        return "<nosuffix>"
    p = str(path).split("?")[0].rstrip("/")
    name = p.rsplit("/", 1)[-1]
    if "." in name:
        suffix = name.rsplit(".", 1)[-1]
        if 0 < len(suffix) <= 8:
            return "." + suffix.lower()
    return "<nosuffix>"


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def looks_meaningless(token: str) -> bool:
    """A small Nostril-like fallback for filtering hashes and random strings."""
    t = token.strip("._- ")
    if len(t) >= 16 and re.fullmatch(r"[a-f0-9]+", t, re.I):
        return True
    if len(t) >= 20 and shannon_entropy(t) > 3.8:
        return True
    if len(set(t)) <= 2 and len(t) >= 8:
        return True
    return False


def tokenize_path(path: str | None, lowercase: bool = True, drop_meaningless: bool = True) -> list[str]:
    s = normalize_path(path, lowercase=lowercase)
    parts = [p for p in _SPLIT_RE.split(s) if p]
    out: list[str] = []
    for p in parts:
        p = p.strip("/._-")
        if not p:
            continue
        if drop_meaningless and looks_meaningless(p):
            continue
        out.append(p)
    return out or ["<empty>"]


def event_to_tokens(edge_type: str, dst_path: str, lowercase: bool = True, max_tokens: int | None = None) -> list[str]:
    edge = (edge_type or "<event>").strip()
    if lowercase:
        edge = edge.lower()
    tokens = [edge, suffix_of(dst_path)] + tokenize_path(dst_path, lowercase=lowercase)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def infer_dst_type(path: str | None, edge_type: str | None = None) -> str:
    s = normalize_path(path, lowercase=True)
    e = (edge_type or "").lower()
    if "reg" in e or "registry" in s or s.startswith("hk"):
        return "REGISTRY"
    if "tcp" in e or "udp" in e or "socket" in s or re.search(r"<ip>|:\d+", s):
        return "NETWORK"
    if "process" in e or "thread" in e:
        return "PROCESS"
    return "FILE"
