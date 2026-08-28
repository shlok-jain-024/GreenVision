"""JSONL memory store — the system's record of every question it asked,
what it answered, what really happened, and the lesson it drew.

This is the "learning" substrate: nothing here (or anywhere else) updates
the model's weights. Instead, the most relevant past records are re-injected
into each new prompt so the model can pattern-match against its own prior
hits and misses (in-context learning).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def build_feature_vector(inputs: dict[str, float], stats: dict[str, dict[str, float]]) -> list[float]:
    """Compact signature of a prediction task, used for similarity retrieval:
    z-scored latest levels + slopes for each metric, plus season phase."""
    vec: list[float] = []
    for m in ("traffic", "aqi", "ndvi"):
        s = stats[m]
        vec.append((inputs[f"{m}_latest"] - s["mean"]) / s["std"])
        vec.append(inputs[f"{m}_slope"] * 12.0 / s["std"])  # slope per year, scaled
    season = inputs.get("season", 0)
    vec.append(math.sin(2 * math.pi * season / 12))
    vec.append(math.cos(2 * math.pi * season / 12))
    return [round(float(v), 4) for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[Record] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[Record]:
        return self._records

    def append(self, record: Record) -> None:
        record = {"id": len(self._records), "ts": datetime.now(timezone.utc).isoformat(), **record}
        self._records.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve(self, zone: str, feat: list[float], k: int) -> list[Record]:
        """Top-k records by: same-zone bonus + feature cosine similarity +
        mild recency preference. Returned oldest-first for prompt ordering."""
        if not self._records or k <= 0:
            return []
        n = len(self._records)
        scored = []
        for i, r in enumerate(self._records):
            score = (
                (1.5 if r.get("zone") == zone else 0.0)
                + _cosine(feat, r.get("feat", []))
                + 0.25 * (i / n)
            )
            scored.append((score, i))
        scored.sort(reverse=True)
        top = sorted(i for _, i in scored[:k])
        return [self._records[i] for i in top]
