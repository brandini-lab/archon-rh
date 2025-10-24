from __future__ import annotations

import json
from pathlib import Path

from orchestration.pipelines.common import PipelineResult


def run(best_path: str = "artifacts/conjecture/best.py") -> PipelineResult:
    path = Path(best_path)
    if not path.exists():
        raise FileNotFoundError(f"No candidate file at {path}")
    code = path.read_text(encoding="utf8")
    lean_lines = [
        "import Mathlib",
        "",
        "namespace ArchonRiemann",
        "",
        "/-- Candidate produced by FunSearch. -/",
        "def funSearchCandidate (x : ℝ) : ℝ :=",
    ]
    for line in code.splitlines():
        if line.strip().startswith("return"):
            lean_lines.append(f"  {line.strip().removeprefix('return ').strip()}")
    lean_lines.append("")
    lean_lines.append("end ArchonRiemann")
    target = Path("formal/leanlib/ArchonRiemann/Candidates.lean")
    target.write_text("\n".join(lean_lines), encoding="utf8")
    return PipelineResult(name="formalize_candidate", payload={"length": float(len(code))})


def main() -> None:  # pragma: no cover
    result = run()
    result.write(Path("artifacts/pipelines/formalize_candidate.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
