from __future__ import annotations

import math
from typing import Iterable, Sequence


def pass_at_k(successes: int, total: int, k: int) -> float:
    if total == 0 or k == 0:
        return 0.0
    k = min(k, total)
    prod = 1.0
    for i in range(k):
        prod *= (total - successes - i) / (total - i)
    return 1.0 - prod


def proof_accept_rate(accepted: Iterable[bool]) -> float:
    accepted_list = list(accepted)
    if not accepted_list:
        return 0.0
    return sum(1 for flag in accepted_list if flag) / len(accepted_list)


def length_normalized_score(lengths: Sequence[int], rewards: Sequence[float]) -> float:
    if not lengths or not rewards or len(lengths) != len(rewards):
        return 0.0
    weighted = [
        reward / math.log(max(length, 2))
        for reward, length in zip(rewards, lengths)
    ]
    return sum(weighted) / len(weighted)
