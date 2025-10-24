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
        "LeanDojo unavailable; emitting fallback shard with diverse training examples."
    )
    examples = [
        # Basic logic
        TacticExample(goal="⊢ True", tactic="trivial", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ True ∧ True", tactic="exact ⟨trivial, trivial⟩", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ True ∨ False", tactic="left; trivial", hypotheses={}, metadata={"source": "fallback"}),
        
        # Natural number arithmetic
        TacticExample(goal="⊢ Nat.zero + 1 = 1", tactic="rfl", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ 1 + 1 = 2", tactic="rfl", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n : ℕ ⊢ 0 + n = n", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n : ℕ ⊢ n + 0 = n", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="a b : ℕ ⊢ a + b = b + a", tactic="ring", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n : ℕ ⊢ n * 0 = 0", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n : ℕ ⊢ 0 * n = 0", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        
        # Inequalities
        TacticExample(goal="⊢ 0 < 1", tactic="norm_num", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ 1 ≤ 2", tactic="norm_num", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n : ℕ ⊢ 0 ≤ n", tactic="exact Nat.zero_le n", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="n m : ℕ, h : n ≤ m ⊢ n ≤ m + 1", tactic="omega", hypotheses={}, metadata={"source": "fallback"}),
        
        # List operations
        TacticExample(goal="α : Type, l : List α ⊢ [] ++ l = l", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="α : Type, l : List α ⊢ l ++ [] = l", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="α : Type ⊢ List.length ([] : List α) = 0", tactic="rfl", hypotheses={}, metadata={"source": "fallback"}),
        
        # Function composition
        TacticExample(goal="α : Type, x : α ⊢ id x = x", tactic="rfl", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="α β : Type, f : α → β, x : α ⊢ (fun y => f y) x = f x", tactic="rfl", hypotheses={}, metadata={"source": "fallback"}),
        
        # Set theory basics
        TacticExample(goal="α : Type, s : Set α, x : α ⊢ x ∈ s ∪ s ↔ x ∈ s", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="α : Type, x : α ⊢ x ∈ (∅ : Set α) ↔ False", tactic="simp", hypotheses={}, metadata={"source": "fallback"}),
        
        # Universal quantifiers
        TacticExample(goal="⊢ ∀ n : ℕ, 0 ≤ n", tactic="intro n; exact Nat.zero_le n", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ ∀ n : ℕ, n = n", tactic="intro n; rfl", hypotheses={}, metadata={"source": "fallback"}),
        
        # Existential quantifiers
        TacticExample(goal="⊢ ∃ n : ℕ, n = 0", tactic="exact ⟨0, rfl⟩", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ ∃ n : ℕ, 0 < n", tactic="exact ⟨1, Nat.one_pos⟩", hypotheses={}, metadata={"source": "fallback"}),
        
        # Real numbers (if available)
        TacticExample(goal="x : ℝ ⊢ x + 0 = x", tactic="ring", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="x y : ℝ ⊢ x + y = y + x", tactic="ring", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="x : ℝ ⊢ x * 1 = x", tactic="ring", hypotheses={}, metadata={"source": "fallback"}),
        
        # Divisibility
        TacticExample(goal="n : ℕ ⊢ n ∣ n", tactic="exact dvd_refl n", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="⊢ 2 ∣ 4", tactic="norm_num", hypotheses={}, metadata={"source": "fallback"}),
        
        # Implications
        TacticExample(goal="p q : Prop, h : p ⊢ p", tactic="exact h", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="p q : Prop, h1 : p, h2 : p → q ⊢ q", tactic="exact h2 h1", hypotheses={}, metadata={"source": "fallback"}),
        
        # Negations
        TacticExample(goal="⊢ ¬False", tactic="trivial", hypotheses={}, metadata={"source": "fallback"}),
        TacticExample(goal="p : Prop, h1 : p, h2 : ¬p ⊢ False", tactic="exact h2 h1", hypotheses={}, metadata={"source": "fallback"}),
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
    try:
        frame = pd.DataFrame(shard.as_dicts())
        frame.to_parquet(path, index=False)
    except Exception as e:  # pragma: no cover
        logger.info(f"Parquet export skipped ({type(e).__name__}). JSONL format is sufficient for training.")


def build(limit: int = 32) -> DatasetShard:
    shard = _collect_from_leandojo(limit=limit)
    dump_jsonl(shard, OUTPUT_JSONL)
    dump_parquet(shard, OUTPUT_PARQUET)
    return shard


if __name__ == "__main__":
    shard = build()
    logger.info("Wrote %d tactic examples to %s", len(shard.examples), OUTPUT_JSONL)
