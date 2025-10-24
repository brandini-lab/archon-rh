from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TacticExample:
    goal: str
    tactic: str
    hypotheses: Dict[str, str]
    metadata: Dict[str, str]


@dataclass
class DatasetShard:
    examples: List[TacticExample]

    def as_dicts(self) -> List[dict]:
        return [
            {
                "goal": ex.goal,
                "tactic": ex.tactic,
                "hypotheses": ex.hypotheses,
                "metadata": ex.metadata,
            }
            for ex in self.examples
        ]
