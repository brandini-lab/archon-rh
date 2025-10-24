from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .bridge import LeanGoalState


class ToyLeanSession:
    def __init__(self, theorem_path: str):
        self.theorem_path = theorem_path
        self._solved = False

    def _make_state(self, goal_text: str, solved: bool, step: int, session_id: str):
        from .bridge import LeanGoalState

        return LeanGoalState(
            goal_text=goal_text,
            hypotheses={"h": "Nat"},
            metavariables={"?m": "Nat"},
            solved=solved,
            step=step,
            session_id=session_id,
        )

    def get_goal_state(self, step: int, session_id: str):
        goal = "? Nat.succ 0 = 1" if not self._solved else "? True"
        return self._make_state(goal, self._solved, step, session_id)

    def apply_tactic(self, tactic: str, step: int, session_id: str):
        normalized = tactic.strip().lower()
        if normalized in {"rfl", "simp", "dec_trivial"}:
            self._solved = True
            goal = "? True"
        else:
            goal = "? Nat.succ 0 = 1"
        return self._make_state(goal, self._solved, step, session_id)
