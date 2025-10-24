from __future__ import annotations

from typing import Callable, Dict

from conjecture.objectives import LiObjective, ObjectiveConfig


def default_registry() -> Dict[str, Callable[[], LiObjective]]:
    return {
        "li_criterion": lambda: LiObjective(
            ObjectiveConfig(sample_points=[0.1, 0.5, 0.9, 1.3], target=1.0)
        )
    }
