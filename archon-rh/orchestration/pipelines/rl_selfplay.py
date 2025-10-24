from __future__ import annotations

from pathlib import Path

from orchestration.pipelines.common import GLOBAL_SEEDS, PipelineResult
from prover.rl.run_selfplay import load_config, run_selfplay


def run(config_path: str, walltime: int = 60) -> PipelineResult:
    cfg = load_config(config_path)
    cfg.seed = GLOBAL_SEEDS["rl"]
    run_selfplay(cfg, walltime=walltime)
    return PipelineResult(name="rl_selfplay", payload={"walltime": float(walltime)})


def main() -> None:  # pragma: no cover
    result = run("orchestration/configs/rl_tiny.yaml", walltime=60)
    result.write(Path("artifacts/pipelines/rl_selfplay.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
