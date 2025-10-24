from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover
    from .bridge import LeanGoalState


class ToyLeanSession:
    """Deterministic Lean-like environment for tests when Lean is unavailable."""

    def __init__(self, theorem_path: str):
        self.theorem_path = theorem_path
        self._step = 0
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
        goal = "⊢ Nat.succ 0 = 1"
        if self._solved:
            goal = "⊢ True"
        return self._make_state(goal, self._solved, step, session_id)

    def apply_tactic(self, tactic: str, step: int, session_id: str):
        normalized = tactic.strip().lower()
        valid: Dict[str, str] = {
            "rfl": "⊢ True",
            "simp": "⊢ True",
            "dec_trivial": "⊢ True",
        }
        if normalized in valid:
            self._solved = True
            goal = valid[normalized]
        else:
            goal = "⊢ Nat.succ 0 = 1"
        self._step = step
        return self._make_state(goal, self._solved, step, session_id)
