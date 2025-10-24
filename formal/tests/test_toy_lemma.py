from pathlib import Path

from formal.leandojo_bridge import LeanBridge, LeanBridgeConfig


def test_toy_lemma(tmp_path):
    theorem_path = Path("formal/tests/data/toy.lean")
    config = LeanBridgeConfig(
        theorem_path=str(theorem_path),
        log_dir=str(tmp_path / "logs"),
        allow_toy_fallback=True,
    )
    with LeanBridge(config) as bridge:
        state = bridge.get_goal_state()
        assert not state.solved
        state = bridge.apply_tactic("simp")
        assert state.solved
        assert bridge.goal_satisfied()
