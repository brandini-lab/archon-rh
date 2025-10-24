"""LeanDojo-backed client for interacting with a real Lean server.

The implementation gracefully degrades when LeanDojo is unavailable so that
the rest of the codebase (and CI) continues to run on CPU-only machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .bridge import LeanGoalState


try:  # pragma: no cover - optional dependency
    from leandojo import LeanCloudSession  # type: ignore
    from leandojo.dataset import register_lean_project  # type: ignore
except Exception:  # pragma: no cover
    LeanCloudSession = None  # type: ignore
    register_lean_project = None  # type: ignore


@dataclass
class LeanProjectConfig:
    root: Path
    lakefile: Path
    toolchain: Path
    main_module: str = "Mathlib"


class LeanDojoBridge:
    """Wrapper that mirrors the MockLean interface but talks to LeanDojo.

    When `leandojo` is not installed the bridge raises a clear error message.
    """

    def __init__(self, project: LeanProjectConfig, timeout: float = 10.0) -> None:
        if LeanCloudSession is None:  # pragma: no cover
            raise RuntimeError(
                "LeanDojo is not installed. Install `leandojo` and ensure Lean binaries "
                "are available to use the LeanDojoBridge."
            )
        self.project = project
        self.timeout = timeout
        self._register_project()
        self._session: Optional[LeanCloudSession] = None
        self._goal_state: Optional[LeanGoalState] = None

    def _register_project(self) -> None:
        if register_lean_project is None:  # pragma: no cover
            return
        register_lean_project(
            repo_path=str(self.project.root),
            main_module=self.project.main_module,
            lean_toolchain=str(self.project.toolchain),
        )

    def start(self) -> None:
        if self._session is not None:
            return
        assert LeanCloudSession is not None  # mypy hint
        self._session = LeanCloudSession(str(self.project.root), timeout=self.timeout)
        state = self._session.get_goal_state()
        self._goal_state = LeanGoalState(
            goal_text=state.goal.pp,
            hypotheses={h.name: h.type for h in state.hyps},
            metavariables={},  # LeanDojo currently exposes a flattened goal
            solved=state.is_goal_proved,
            step=0,
            session_id=str(state.id),
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._goal_state = None

    def __enter__(self) -> "LeanDojoBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_goal_state(self) -> LeanGoalState:
        if self._goal_state is None:
            self.start()
        assert self._goal_state is not None
        return self._goal_state

    def apply_tactic(self, tactic: str) -> LeanGoalState:
        if self._session is None or self._goal_state is None:
            self.start()
        assert self._session is not None
        assert self._goal_state is not None
        result = self._session.run_tactic(tactic)
        self._goal_state = LeanGoalState(
            goal_text=result.goal.pp,
            hypotheses={h.name: h.type for h in result.hyps},
            metavariables={},
            solved=result.is_goal_proved,
            step=self._goal_state.step + 1,
            session_id=self._goal_state.session_id,
        )
        return self._goal_state

    def goal_satisfied(self) -> bool:
        if self._goal_state is None:
            self.start()
        assert self._goal_state is not None
        return bool(self._goal_state.solved)

    def run_script(self, tactics: Iterable[str]) -> LeanGoalState:
        state = None
        for tactic in tactics:
            state = self.apply_tactic(tactic)
            if state.solved:
                break
        assert state is not None
        return state


def default_project() -> LeanProjectConfig:
    root = Path("formal/lean_project")
    return LeanProjectConfig(
        root=root,
        lakefile=root / "lakefile.lean",
        toolchain=root / "lean-toolchain",
        main_module="Mathlib",
    )
