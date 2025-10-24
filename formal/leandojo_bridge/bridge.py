from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .toy import ToyLeanSession

try:
    from leandojo import LeanProofSession  # type: ignore
    from leandojo.environment import LeanGoal  # type: ignore
except ImportError:  # pragma: no cover - LeanDojo optional
    LeanProofSession = None  # type: ignore
    LeanGoal = Any  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class LeanBridgeConfig:
    """Configuration for Lean bridge sessions."""

    theorem_path: str
    tactic_timelimit: float = 5.0
    log_dir: str = "artifacts/logs/lean_bridge"
    lean_project_path: Optional[str] = None
    allow_toy_fallback: bool = True

    def ensure_dirs(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class LeanGoalState:
    """Structured Lean goal representation."""

    goal_text: str
    hypotheses: Dict[str, str] = field(default_factory=dict)
    metavariables: Dict[str, str] = field(default_factory=dict)
    solved: bool = False
    step: int = 0
    session_id: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "goal_text": self.goal_text,
            "hypotheses": self.hypotheses,
            "metavariables": self.metavariables,
            "solved": self.solved,
            "step": self.step,
            "session_id": self.session_id,
        }


class LeanBridge:
    """Wrapper around LeanDojo LeanProofSession with structured logging."""

    def __init__(self, config: LeanBridgeConfig):
        self.config = config
        self.config.ensure_dirs()
        self._session: Optional[Any] = None
        self._step = 0
        self._session_id = str(uuid.uuid4())
        self._log_path = Path(self.config.log_dir, f"{self._session_id}.jsonl")
        LOGGER.debug("Initialized LeanBridge session %s", self._session_id)

    def __enter__(self) -> "LeanBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Start the Lean session."""
        if self._session is not None:
            return

        theorem_path = Path(self.config.theorem_path)
        if LeanProofSession is not None:
            LOGGER.info("Starting LeanProofSession for %s", theorem_path)
            self._session = LeanProofSession(
                self.config.lean_project_path or theorem_path.parent,
                theorem_path,
                timeout=self.config.tactic_timelimit,
            )
            self._session.__enter__()
        elif self.config.allow_toy_fallback:
            LOGGER.warning("LeanDojo not available, using ToyLeanSession fallback.")
            self._session = ToyLeanSession(str(theorem_path))
        else:  # pragma: no cover - environment misconfigured
            raise RuntimeError(
                "LeanDojo not available and toy fallback not allowed. "
                "Install leandojo or enable fallback."
            )

    def close(self) -> None:
        if isinstance(self._session, LeanProofSession):
            self._session.__exit__(None, None, None)  # type: ignore[call-arg]
        self._session = None

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        structured = {
            "ts": time.time(),
            "event": event,
            "session_id": self._session_id,
            "step": self._step,
            **payload,
        }
        with self._log_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(structured) + "\n")

    def get_goal_state(self) -> LeanGoalState:
        """Return the current goal state."""
        self.start()
        assert self._session is not None, "Session must be started before reading state."

        if LeanProofSession is not None and isinstance(self._session, LeanProofSession):
            goal: LeanGoal = self._session.get_goal()  # type: ignore[assignment]
            state = LeanGoalState(
                goal_text=str(goal.state.pp),
                hypotheses={hp.name: str(hp.type.pp) for hp in goal.state.local_context},
                metavariables={m.name: str(m.type.pp) for m in goal.state.goals},
                solved=goal.is_goal_finished,
                step=self._step,
                session_id=self._session_id,
            )
        else:
            state = self._session.get_goal_state(step=self._step, session_id=self._session_id)  # type: ignore[attr-defined]

        self._log("goal_state", state.to_json())
        return state

    def apply_tactic(self, tactic: str) -> LeanGoalState:
        """Apply a tactic and return updated state."""
        self.start()
        assert self._session is not None, "Session must be started before issuing tactics."
        self._step += 1

        if LeanProofSession is not None and isinstance(self._session, LeanProofSession):
            result = self._session.apply_tactic(tactic)
            state = LeanGoalState(
                goal_text=str(result.state.pp),
                hypotheses={hp.name: str(hp.type.pp) for hp in result.state.local_context},
                metavariables={m.name: str(m.type.pp) for m in result.state.goals},
                solved=result.is_goal_finished,
                step=self._step,
                session_id=self._session_id,
            )
        else:
            state = self._session.apply_tactic(  # type: ignore[attr-defined]
                tactic, step=self._step, session_id=self._session_id
            )

        self._log("apply_tactic", {"tactic": tactic, **state.to_json()})
        return state

    def goal_satisfied(self) -> bool:
        """Return True when the current goal is solved."""
        state = self.get_goal_state()
        return state.solved
