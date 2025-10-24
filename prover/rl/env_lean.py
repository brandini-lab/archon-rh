from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from formal.leandojo_bridge import MockLean, LeanDojoBridge, default_project


@dataclass
class StepResult:
    goal: str
    reward: float
    done: bool
    info: dict


class LeanEnv:
    """Environment that matches the OpenAI Gym API but without the dependency."""

    def __init__(self, backend: str = "mock", timeout: float = 5.0):
        self.backend = backend
        if backend == "leandojo":
            self.client = LeanDojoBridge(default_project(), timeout=timeout)
        else:
            self.client = MockLean()
        self.tactic_history: List[str] = []

    def reset(self) -> str:
        if isinstance(self.client, LeanDojoBridge):
            self.client.close()
            self.client.start()
            state = self.client.get_goal_state()
            self.tactic_history.clear()
            return state.goal_text
        self.client = MockLean()
        self.tactic_history.clear()
        return self.client.get_goal_state().goal

    def step(self, tactic: str) -> StepResult:
        state = self.client.apply_tactic(tactic)
        self.tactic_history.append(tactic)
        reward = 1.0 if state.solved else -0.05
        return StepResult(goal=state.goal_text, reward=reward, done=state.solved, info={})

    def goal_satisfied(self) -> bool:
        return self.client.goal_satisfied()
