from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml

from conjecture.funsearch.objective_registry import default_registry


@dataclass
class FunsearchConfig:
    iterations: int = 20
    objectives: List[str] = field(default_factory=lambda: ["li_criterion"])
    export_dir: str = "artifacts/conjecture"
    lean_export: str = "formal/leanlib/ArchonRiemann/Generated.lean"
    seed: int = 0


def load_config(path: str) -> FunsearchConfig:
    with open(path, "r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle)
    return FunsearchConfig(
        iterations=payload.get("iterations", 20),
        objectives=payload.get("objectives", ["li_criterion"]),
        export_dir=payload.get("export_dir", "artifacts/conjecture"),
        lean_export=payload.get("lean_export", "formal/leanlib/ArchonRiemann/Generated.lean"),
        seed=payload.get("seed", 0),
    )


def generate_candidate() -> Tuple[str, Callable[[float], float]]:
    coeffs = [random.uniform(-2.0, 2.0) for _ in range(3)]
    body = " + ".join(f"{c:.4f} * x**{i+1}" for i, c in enumerate(coeffs))
    code = f"def kernel(x: float) -> float:\n    return {body}\n"
    namespace: Dict[str, Callable[[float], float]] = {}
    exec(code, {}, namespace)
    return code, namespace["kernel"]


def run_loop(cfg: FunsearchConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    reg = default_registry()
    objectives = {name: reg[name]() for name in cfg.objectives}
    archive: Dict[str, Tuple[float, str]] = {name: (obj.baseline(), "") for name, obj in objectives.items()}
    history: List[Dict[str, float]] = []

    for _ in range(cfg.iterations):
        code, fn = generate_candidate()
        scores: Dict[str, float] = {}
        for name, objective in objectives.items():
            score = objective.evaluate(fn)
            scores[name] = score
            if score > archive[name][0]:
                archive[name] = (score, code)
        history.append(scores)

    export_dir = Path(cfg.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    Path(export_dir, "scores.jsonl").write_text(
        "\n".join(json.dumps(item) for item in history),
        encoding="utf8",
    )
    best_code = max(archive.values(), key=lambda item: item[0])[1]
    if best_code:
        Path(export_dir, "best.py").write_text(best_code, encoding="utf8")
        lean_content = [
            "import Mathlib",
            "",
            "namespace ArchonRiemann",
            "",
            "/-- Auto-generated conjecture kernel. -/",
            "def generatedKernel (x : R) : R :=",
        ]
        for line in best_code.splitlines()[1:]:
            lean_content.append(f"  {line.replace('return', '').strip()}")
        lean_content.append("")
        lean_content.append("end ArchonRiemann")
        Path(cfg.lean_export).write_text("\n".join(lean_content), encoding="utf8")

    summary = {name: score for name, (score, _) in archive.items()}
    Path(export_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FunSearch-style loop for conjecture discovery.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_loop(cfg)
    print("FunSearch summary:", summary)


if __name__ == "__main__":
    main()
