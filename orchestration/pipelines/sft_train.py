from __future__ import annotations

from pathlib import Path

from orchestration.pipelines.common import GLOBAL_SEEDS, PipelineResult
from prover.train import load_config, train


def run(config_path: str, max_steps: int = 100) -> PipelineResult:
    cfg = load_config(config_path)
    cfg.seed = GLOBAL_SEEDS["sft"]
    train(cfg, max_steps=max_steps)
    return PipelineResult(name="sft_train", payload={"final_step": float(max_steps)})


def main() -> None:  # pragma: no cover - CLI glue
    result = run("orchestration/configs/sft_tiny.yaml", max_steps=100)
    result.write(Path("artifacts/pipelines/sft_train.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
