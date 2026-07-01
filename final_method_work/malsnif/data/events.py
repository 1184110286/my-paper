from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EventRecord:
    src_id: str
    src_type: str
    dst_id: str
    dst_type: str
    edge_type: str
    time: str | float | int | None = None
    tag: int = 0
    raw: Optional[Dict[str, Any]] = None

    def key(self) -> str:
        return f"{self.src_id}_{self.dst_id}"
