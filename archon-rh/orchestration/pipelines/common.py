from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


GLOBAL_SEEDS = {
    "sft": 42,
    "rl": 1234,
    "funsearch": 0,
    "numerics": 7,
}


@dataclass
class PipelineResult:
    name: str
    payload: Dict[str, float]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": self.name, "payload": self.payload}, indent=2), encoding="utf8")
