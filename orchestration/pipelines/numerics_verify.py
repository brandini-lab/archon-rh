from __future__ import annotations

from pathlib import Path

from numerics.zeros.run_zero_checks import load_config, run_zero_checks
from orchestration.pipelines.common import GLOBAL_SEEDS, PipelineResult


def run(config_path: str) -> PipelineResult:
    cfg = load_config(config_path)
    # deterministic sampling based on seed by adjusting output path
    cfg.output = str(Path(cfg.output).with_suffix(f".seed{GLOBAL_SEEDS['numerics']}.json"))
    certificates = run_zero_checks(cfg)
    return PipelineResult(
        name="numerics_verify",
        payload={"certificates": float(len(certificates))},
    )


def main() -> None:  # pragma: no cover
    result = run("orchestration/configs/numeric_verify.yaml")
    result.write(Path("artifacts/pipelines/numerics.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
