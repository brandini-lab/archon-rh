from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class ObjectiveConfig:
    sample_points: Iterable[float]
    target: float = 0.0


class LiObjective:
    """Toy Li-style objective: integrate kernel over sampled points."""

    def __init__(self, config: ObjectiveConfig):
        self.config = config

    def evaluate(self, fn: Callable[[float], float]) -> float:
        acc = 0.0
        for x in self.config.sample_points:
            value = fn(x)
            acc += (value - self.config.target) ** 2
        return -acc  # higher is better (closer to target)

    def baseline(self) -> float:
        return -len(list(self.config.sample_points))
