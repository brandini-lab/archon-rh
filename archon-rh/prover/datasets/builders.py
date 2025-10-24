from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class DatasetConfig:
    source: str
    output_dir: str = "artifacts/datasets"
    limit: Optional[int] = None


@dataclass
class DatasetArtifacts:
    jsonl: Path
    parquet: Path


def _ensure_sample_dataset(path: Path) -> None:
    sample = [
        {
            "id": "toy-1",
            "theorem": "toyLemma",
            "goal": "? Nat.succ 0 = 1",
            "tactic": "simp",
            "result_goal": "? True",
        },
        {
            "id": "toy-2",
            "theorem": "toyLemma",
            "goal": "? Nat.succ 0 = 1",
            "tactic": "rfl",
            "result_goal": "? True",
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in sample), encoding="utf8")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_dataset(config: DatasetConfig) -> DatasetArtifacts:
    source = Path(config.source)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _ensure_sample_dataset(source)

    rows: List[dict] = []
    for idx, record in enumerate(_read_jsonl(source)):
        rows.append(
            {
                "id": record.get("id", f"sample-{idx}"),
                "theorem": record.get("theorem", "unknown"),
                "goal": record.get("goal"),
                "tactic": record.get("tactic"),
                "result_goal": record.get("result_goal"),
            }
        )
        if config.limit is not None and len(rows) >= config.limit:
            break

    if not rows:
        raise ValueError(f"No rows found in {source}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "lean_tactics.jsonl"
    parquet_path = output_dir / "lean_tactics.parquet"

    with jsonl_path.open("w", encoding="utf8") as writer:
        for row in rows:
            writer.write(json.dumps(row) + "\n")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    return DatasetArtifacts(jsonl=jsonl_path, parquet=parquet_path)
