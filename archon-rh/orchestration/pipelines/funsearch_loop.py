from __future__ import annotations

from pathlib import Path

from conjecture.funsearch.funsearch_loop import load_config, run_loop
from orchestration.pipelines.common import GLOBAL_SEEDS, PipelineResult


def run(config_path: str) -> PipelineResult:
    cfg = load_config(config_path)
    cfg.seed = GLOBAL_SEEDS["funsearch"]
    summary = run_loop(cfg)
    return PipelineResult(name="funsearch_loop", payload=summary)


def main() -> None:  # pragma: no cover
    result = run("orchestration/configs/funsearch_loop.yaml")
    result.write(Path("artifacts/pipelines/funsearch.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
