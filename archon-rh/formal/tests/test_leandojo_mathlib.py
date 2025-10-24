import pytest

try:
    import leandojo  # type: ignore
except Exception:  # pragma: no cover
    leandojo = None

from formal.leandojo_bridge.client import LeanDojoBridge, default_project

pytestmark = pytest.mark.skipif(leandojo is None, reason="LeanDojo not installed")


def test_leandojo_mathlib_trivial():
    project = default_project()
    bridge = LeanDojoBridge(project, timeout=5.0)
    with bridge as session:
        state = session.get_goal_state()
        assert state.goal_text
        result = session.run_script(["simp", "exact?" ])
        assert result.solved
