from pathlib import Path

from prover.datasets import build_mathlib_dataset
from prover.train import sft_hf
from prover.rl import ppo_loop


def test_hf_sft_and_ppo_pipeline(tmp_path):
    shard = build_mathlib_dataset.build(limit=4)
    assert shard.examples
    sft_hf.main("orchestration/configs/sft_hf_tiny.yaml")
    assert Path("artifacts/checkpoints/sft_hf_tiny").exists()
    ppo_loop.train("orchestration/configs/rl_ppo_tiny.yaml")
    assert Path("artifacts/checkpoints/rl_ppo_tiny/ppo_policy.pt").exists()
