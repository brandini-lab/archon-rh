"""Extracts (goal -> tactic) pairs from Mathlib using LeanDojo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from libs.rh_common import get_logger, ensure_dir, write_json
from .schemas import DatasetShard, TacticExample

logger = get_logger("datasets.mathlib")

OUTPUT_JSONL = Path("artifacts/data/mini_mathlib.jsonl")
OUTPUT_PARQUET = Path("artifacts/data/mini_mathlib.parquet")


def _fallback_examples() -> DatasetShard:
    logger.warning(
        "LeanDojo unavailable; emitting fallback shard with trivial examples only."
    )
    examples = [
        TacticExample(
            goal="⊢ True",
            tactic="trivial",
            hypotheses={},
            metadata={"source": "fallback"},
        ),
        TacticExample(
            goal="⊢ Nat.zero + 1 = 1",
            tactic="simp",
            hypotheses={},
            metadata={"source": "fallback"},
        ),
    ]
    return DatasetShard(examples)


def _collect_from_leandojo(limit: int = 32) -> DatasetShard:
    try:
        from leandojo.dataset import TacticStateDataset  # type: ignore
    except Exception:  # pragma: no cover
        return _fallback_examples()

    dataset = TacticStateDataset.from_prebuilt("mathlib")  # type: ignore
    examples: List[TacticExample] = []
    for item in dataset:  # type: ignore
        examples.append(
            TacticExample(
                goal=item.goal.pp,
                tactic=item.tactic,
                hypotheses={ctx.local_decl.name: ctx.local_decl.type for ctx in item.context},
                metadata={
                    "file": item.theorem,
                    "idx": str(item.idx),
                },
            )
        )
        if len(examples) >= limit:
            break
    if not examples:
        return _fallback_examples()
    return DatasetShard(examples)


def dump_jsonl(shard: DatasetShard, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf8") as handle:
        for row in shard.as_dicts():
            handle.write(json.dumps(row) + "\n")


def dump_parquet(shard: DatasetShard, path: Path) -> None:
    ensure_dir(path.parent)
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        logger.warning("pandas not available, skipping parquet export.")
        return
    frame = pd.DataFrame(shard.as_dicts())
    frame.to_parquet(path, index=False)


def build(limit: int = 32) -> DatasetShard:
    shard = _collect_from_leandojo(limit=limit)
    dump_jsonl(shard, OUTPUT_JSONL)
    dump_parquet(shard, OUTPUT_PARQUET)
    return shard


if __name__ == "__main__":
    shard = build()
    logger.info("Wrote %d tactic examples to %s", len(shard.examples), OUTPUT_JSONL)
