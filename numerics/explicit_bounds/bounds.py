from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ZeroFreeRegion:
    sigma: float
    height: float


def zero_free_bound(height: float) -> ZeroFreeRegion:
    sigma = 1 - 1 / math.log(height + math.e)
    return ZeroFreeRegion(sigma=sigma, height=height)
