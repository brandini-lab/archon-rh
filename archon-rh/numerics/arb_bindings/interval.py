from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from mpmath import iv


@dataclass
class Interval:
    lo: float
    hi: float

    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def to_iv(self) -> iv.mpf:
        return iv.mpf([self.lo, self.hi])


def interval(lo: float, hi: float) -> Interval:
    if lo > hi:
        raise ValueError("Lower bound must not exceed upper bound.")
    return Interval(lo, hi)


def verify_certified_zero_count(target: Interval, count: int) -> Tuple[bool, str]:
    if count < 0:
        return False, "Count must be non-negative."
    if target.width() <= 0:
        return False, "Degenerate interval."
    return True, "Interval certificate superficially valid."
